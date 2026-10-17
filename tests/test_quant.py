from datetime import date

import numpy as np
import pandas as pd
import pytest

from gagcap.forecasting.backtesting import (next_policy_decision, point_in_time_vintage,
    resolve_calendar_horizon, walk_forward, walk_forward_with_vintages)
from gagcap.forecasting.metrics import interval_coverage, pinball_loss
from gagcap.forecasting.metrics import mae, rmse, directional_accuracy
from gagcap.forecasting.baselines import seasonal_naive
from gagcap.intelligence.geo_risk import GeoEvent, event_study, events_available_at_origin
from gagcap.intelligence.qwen import ExtractedEvent, extract_event
from gagcap.quant.macro import TaylorInputs, rate_scenario, taylor_rate
from gagcap.quant.capm import beta, rolling_beta
from gagcap.quant.econometrics import arimax_forecast, garch11_forecast
from gagcap.quant.models import Forecast, calendar_horizon
from gagcap.quant.risk import historical_var_es, monte_carlo_normal
from gagcap.quant.returns import log_returns, realized_volatility
from gagcap.quant.volatility import historical_volatility, ewma_volatility
from gagcap.forecasting.scenarios import MacroScenario, validate_scenario_set


def test_forecast_horizon_metadata_and_interval_contract():
    calendar = pd.date_range("2025-01-01", periods=100, freq="B", tz="UTC")
    origin = pd.Timestamp("2025-01-03", tz="UTC")
    horizon = calendar_horizon(30, session_calendar=calendar, forecast_origin=origin)
    assert horizon.sessions == resolve_calendar_horizon(calendar, origin, 30)[0]
    with pytest.raises(ValueError):
        Forecast("m", date(2025, 1, 2), date(2025, 1, 1), 7, 5, 1.0, None, 2.0,
                 0.95, "RUB", "v1", "assumption")


def test_risk_metrics_reject_missing_and_report_loss_units():
    with pytest.raises(ValueError):
        historical_var_es(pd.Series([0.01, np.nan, -0.02]))
    result = historical_var_es(pd.Series([0.01, -0.02, -0.03, 0.0] * 5), confidence=0.75)
    assert result.unit == "return_fraction"
    assert result.expected_shortfall_loss >= result.var_loss


def test_returns_and_volatility_reject_non_positive_or_missing_prices():
    prices = pd.Series([100.0, 101.0, 99.0], index=pd.date_range("2025-01-01", periods=3))
    returns = log_returns(prices)
    assert len(returns) == 2
    assert len(realized_volatility(returns, window=2).dropna()) == 1
    assert len(historical_volatility(returns, window=2).dropna()) == 1
    assert ewma_volatility(returns).notna().all()
    with pytest.raises(ValueError):
        log_returns(pd.Series([10.0, 0.0]))


def test_capm_and_rolling_beta_require_aligned_complete_returns():
    index = pd.date_range("2025-01-01", periods=5)
    market = pd.Series([.01, -.01, .02, .0, .01], index=index)
    asset = market * 1.5
    assert beta(asset, market) == pytest.approx(1.5)
    assert rolling_beta(asset, market, window=3).dropna().iloc[-1] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        beta(asset.iloc[:1], market.iloc[:1])


def test_arimax_rejects_future_macro_release_before_model_fit():
    index = pd.date_range("2025-01-01", periods=3)
    future_index = pd.date_range("2025-01-04", periods=1)
    y = pd.Series([1., 2., 3.], index=index)
    exog = pd.DataFrame({"macro": [1., 2., 3.]}, index=index)
    future = pd.DataFrame({"macro": [4.]}, index=future_index)
    released = pd.Series([pd.Timestamp("2025-01-04T00:00:00Z")], index=future_index)
    with pytest.raises(ValueError):
        arimax_forecast(y, exog, future, forecast_origin=pd.Timestamp("2025-01-03T00:00:00Z"),
                        future_exog_available_at=released)


def test_garch_rejects_insufficient_history_before_model_fit():
    with pytest.raises(ValueError):
        garch11_forecast(pd.Series([0.01, -0.02]), horizon=1)


def test_monte_carlo_is_reproducible_under_explicit_normal_assumption():
    first = monte_carlo_normal(mean_return=0, volatility=0.01, horizon_sessions=5, seed=4)
    second = monte_carlo_normal(mean_return=0, volatility=0.01, horizon_sessions=5, seed=4)
    assert first == second
    assert first["distribution"] == "normal_iid_log_returns"


def test_walk_forward_excludes_late_published_observation():
    index = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    values = pd.Series(range(8), index=index, dtype=float)
    available = pd.Series(index, index=index)
    available.iloc[2] = pd.Timestamp("2025-01-06", tz="UTC")
    seen = []

    def predictor(train, horizon):
        seen.append(train.index.tolist())
        return [float(train.iloc[-1])] * horizon

    records = walk_forward(values, horizon=2, min_train=3, predictor=predictor, available_at=available)
    assert records
    assert index[2] not in seen[0]


def test_rolling_window_uses_only_origin_visible_tail():
    index = pd.date_range("2025-01-01", periods=9, freq="D", tz="UTC")
    values = pd.Series(range(9), index=index, dtype=float)
    available = pd.Series(index, index=index)
    available.iloc[4] = pd.Timestamp("2025-01-08", tz="UTC")
    seen = []

    def predictor(train, horizon):
        seen.append(train.index.tolist())
        return [float(train.iloc[-1])] * horizon

    walk_forward(values, horizon=1, min_train=3, predictor=predictor,
                 available_at=available, window="rolling", rolling_window=3)
    assert all(len(window) <= 3 for window in seen)
    assert all(index[4] not in window for window in seen if window[-1] < index[7])


def test_backtest_rejects_unsorted_and_duplicate_indexes():
    values = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-03", "2025-01-01", "2025-01-02"]))
    with pytest.raises(ValueError):
        walk_forward(values, horizon=1, min_train=2, predictor=lambda train, h: [0.0])
    duplicate = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]))
    with pytest.raises(ValueError):
        walk_forward(duplicate, horizon=1, min_train=2, predictor=lambda train, h: [0.0])


def test_point_in_time_vintages_choose_only_latest_available_revision():
    data = pd.DataFrame([
        {"source": "fred", "series_id": "CPI", "observation_date": date(2025, 1, 1),
         "value": 100.0, "available_at": "2025-02-01T00:00:00Z", "vintage_date": date(2025, 2, 1)},
        {"source": "fred", "series_id": "CPI", "observation_date": date(2025, 1, 1),
         "value": 101.0, "available_at": "2025-03-01T00:00:00Z", "vintage_date": date(2025, 3, 1)},
    ])
    selected = point_in_time_vintage(data, pd.Timestamp("2025-02-15T00:00:00Z"))
    assert selected.value.tolist() == [100.0]


def test_revision_aware_walk_forward_hides_later_vintage_at_each_origin():
    idx = pd.date_range("2025-02-01", periods=6, freq="D", tz="UTC")
    target = pd.Series([1., 2., 3., 4., 5., 6.], index=idx)
    revisions = pd.DataFrame([
        {"source": "fred", "series_id": "CPI", "observation_date": date(2025, 1, 1),
         "value": 10., "available_at": "2025-02-01T00:00:00Z", "vintage_date": date(2025, 2, 1)},
        {"source": "fred", "series_id": "CPI", "observation_date": date(2025, 1, 1),
         "value": 11., "available_at": "2025-02-05T00:00:00Z", "vintage_date": date(2025, 2, 5)},
    ])
    observed = []

    def predictor(train, macro, horizon):
        observed.append(macro.value.tolist())
        return [float(train.iloc[-1])] * horizon

    walk_forward_with_vintages(target, revisions, horizon=1, min_train=2, predictor=predictor)
    assert observed[0] == [10.]
    assert observed[-1] == [11.]


def test_qwen_absent_model_returns_unavailable_without_fake_event(tmp_path):
    result = extract_event("Source text", source_url="https://www.cbr.ru",
                           model_path=str(tmp_path / "missing.gguf"))
    assert result.status == "QWEN_UNAVAILABLE"
    assert result.event is None


def test_event_requires_evidence_and_source():
    with pytest.raises(ValueError):
        GeoEvent("e1", "sanctions", "http://insecure", date.today(), date.today(),
                 ("RU",), (), (), "document excerpt", "manual")


def test_geo_events_published_after_forecast_origin_are_excluded():
    from datetime import datetime, timezone
    event = GeoEvent("e2", "sanctions", "https://example.org/report",
                     datetime(2025, 2, 2, tzinfo=timezone.utc), date(2025, 2, 1),
                     ("RU",), (), (), "source evidence", "extractor")
    selected = events_available_at_origin([event], datetime(2025, 2, 1, tzinfo=timezone.utc))
    assert selected == []


def test_scenario_math_rejects_infinite_values():
    with pytest.raises(ValueError):
        taylor_rate(TaylorInputs(1, 2, 2, float("inf")))
    with pytest.raises(ValueError):
        rate_scenario(10, float("inf"))


def test_interval_coverage_and_pinball_contract():
    assert interval_coverage([1, 2], [0, 2], [2, 3]) == 1
    assert pinball_loss([1, 2], [0, 2], quantile=0.5) == 0.25
    assert mae([1, 2], [0, 2]) == 0.5
    assert rmse([1, 2], [0, 2]) == pytest.approx(2 ** -0.5)
    assert directional_accuracy([2, 0], [3, 1], origin=1) == 0.5


def test_interval_coverage_rejects_inverted_bounds():
    with pytest.raises(ValueError):
        interval_coverage([1], [2], [0])


def test_event_study_requires_sufficient_history():
    returns = pd.Series(np.zeros(20), index=pd.date_range("2025-01-01", periods=20))
    with pytest.raises(ValueError):
        event_study(returns, date(2025, 1, 10), estimation_window=120)


def test_calendar_horizon_uses_actual_sessions_not_fixed_252_approximation():
    sessions = pd.DatetimeIndex(["2025-01-03", "2025-01-06", "2025-01-08", "2025-02-03",
                                 "2025-02-04", "2025-02-05"], tz="UTC")
    steps, target = resolve_calendar_horizon(sessions, pd.Timestamp("2025-01-03", tz="UTC"), 30)
    assert target == pd.Timestamp("2025-02-03", tz="UTC")
    assert steps == 3


def test_policy_forecast_uses_separate_decision_calendar():
    decisions = pd.DatetimeIndex(["2025-03-21", "2025-04-25"], tz="UTC")
    assert next_policy_decision(decisions, pd.Timestamp("2025-03-22", tz="UTC")) == decisions[1]


def test_calendar_horizon_rejects_insufficient_calendar():
    sessions = pd.date_range("2025-01-01", periods=3, freq="B", tz="UTC")
    with pytest.raises(ValueError):
        resolve_calendar_horizon(sessions, sessions[0], 90)


def test_walk_forward_overlapping_forecasts_keep_origin_and_target_distinct():
    index = pd.date_range("2025-01-01", periods=8, freq="D", tz="UTC")
    values = pd.Series(range(8), index=index, dtype=float)
    records = walk_forward(values, horizon=3, min_train=2,
                           predictor=lambda train, h: [float(train.iloc[-1])] * h)
    assert any(a.target == b.target and a.origin != b.origin for a in records for b in records)


def test_walk_forward_rejects_irregular_or_ambiguous_frequency_when_requested():
    # A true session calendar may be irregular; an explicitly declared expected frequency must match.
    from gagcap.forecasting.backtesting import validate_regular_frequency
    irregular = pd.DatetimeIndex(["2025-01-01", "2025-01-02", "2025-01-04"])
    with pytest.raises(ValueError):
        validate_regular_frequency(irregular, "D")


def test_qwen_schema_rejects_extra_fields():
    with pytest.raises(Exception):
        ExtractedEvent.model_validate({"untrusted": "field"})


def test_scenario_set_requires_all_names_and_probability_basis():
    scenarios = [MacroScenario(name, {"inflation": value}, ())
                 for name, value in [("BASE", 4.), ("BULL", 3.), ("BEAR", 6.)]]
    validate_scenario_set(scenarios)
    with pytest.raises(ValueError):
        seasonal_naive(pd.Series([1., 2.]), 3, season_length=4)
