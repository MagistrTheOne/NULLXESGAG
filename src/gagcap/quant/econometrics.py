"""Lazy statsmodels and arch integrations for time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ModelOutcome:
    status: Literal["AVAILABLE_UNVERIFIED", "CONVERGENCE_FAILED", "UNSTABLE_PARAMETERS", "MODEL_ERROR"]
    forecast: pd.DataFrame | None
    diagnostics: dict[str, Any]


def arimax_forecast(y: pd.Series, exog: pd.DataFrame, future_exog: pd.DataFrame,
                    *, forecast_origin: pd.Timestamp,
                    future_exog_available_at: pd.Series,
                    order: tuple[int, int, int] = (1, 0, 1), alpha: float = 0.05) -> ModelOutcome:
    if (y.empty or y.isna().any() or exog.isna().any() or future_exog.isna().any()
            or not np.isfinite(y.to_numpy(dtype=float)).all()
            or not np.isfinite(exog.to_numpy(dtype=float)).all()
            or not np.isfinite(future_exog.to_numpy(dtype=float)).all()):
        raise ValueError("ARIMAX inputs must be non-empty and complete")
    if not y.index.equals(exog.index) or not exog.columns.equals(future_exog.columns):
        raise ValueError("Endogenous and exogenous indexes/schema must align")
    if (not isinstance(y.index, pd.DatetimeIndex) or not y.index.is_monotonic_increasing
            or not y.index.is_unique or not isinstance(future_exog.index, pd.DatetimeIndex)
            or not future_exog.index.is_monotonic_increasing or not future_exog.index.is_unique
            or future_exog.index[0] <= y.index[-1]):
        raise ValueError("ARIMAX requires ordered unique dates and strictly future exog rows")
    if not future_exog_available_at.index.equals(future_exog.index):
        raise ValueError("Future exogenous availability timestamps must align")
    if not isinstance(future_exog_available_at.dtype, pd.DatetimeTZDtype):
        raise ValueError("Future exogenous availability timestamps must be timezone-aware")
    if forecast_origin.tzinfo is None or (future_exog_available_at > forecast_origin).any():
        raise ValueError("Forecast origin must be aware and all future exog must be available by origin")
    if not 0 < alpha < 1 or len(future_exog) < 1 or any(value < 0 for value in order):
        raise ValueError("Invalid alpha or empty forecast exog")
    if len(y) < max(20, sum(order) * 5):
        raise ValueError("SARIMAX requires at least 20 observations and adequate order-dependent history")
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.stattools import adfuller
    stationarity_series = y.diff().dropna() if order[1] > 0 else y
    if len(stationarity_series) < 15:
        raise ValueError("Insufficient observations for stationarity diagnostic")
    try:
        adf_statistic, adf_pvalue, *_ = adfuller(stationarity_series.to_numpy(dtype=float), autolag="AIC")
    except Exception as exc:
        return ModelOutcome("MODEL_ERROR", None,
                            {"error_type": type(exc).__name__, "message": f"ADF diagnostic failed: {exc}"})
    if not np.isfinite([adf_statistic, adf_pvalue]).all():
        return ModelOutcome("UNSTABLE_PARAMETERS", None, {"reason": "non-finite stationarity diagnostic"})
    if adf_pvalue > 0.10:
        return ModelOutcome("UNSTABLE_PARAMETERS", None,
                            {"reason": "differenced endogenous series not stationary at 10% threshold",
                             "adf_statistic": float(adf_statistic), "adf_pvalue": float(adf_pvalue)})
    try:
        result = SARIMAX(y.astype(float), exog=exog.astype(float), order=order,
                         enforce_stationarity=True, enforce_invertibility=True).fit(disp=False)
        converged = bool(result.mle_retvals.get("converged", False))
        if not converged:
            return ModelOutcome("CONVERGENCE_FAILED", None,
                                {"converged": False, "iterations": result.mle_retvals.get("iterations")})
        if not np.isfinite(result.params.to_numpy(dtype=float)).all():
            return ModelOutcome("UNSTABLE_PARAMETERS", None, {"reason": "non-finite parameters"})
        forecast = result.get_forecast(steps=len(future_exog), exog=future_exog.astype(float))
        bounds = forecast.conf_int(alpha=alpha)
        frame = pd.DataFrame({"point_estimate": forecast.predicted_mean.to_numpy(dtype=float),
                              "interval_lower": bounds.iloc[:, 0].to_numpy(dtype=float),
                              "interval_upper": bounds.iloc[:, 1].to_numpy(dtype=float)},
                             index=future_exog.index)
        if (not np.isfinite(frame.to_numpy(dtype=float)).all()
                or (frame["interval_lower"] > frame["interval_upper"]).any()):
            return ModelOutcome("UNSTABLE_PARAMETERS", None, {"reason": "invalid prediction interval"})
        return ModelOutcome("AVAILABLE_UNVERIFIED", frame,
                            {"converged": True, "aic": float(result.aic), "alpha": alpha,
                             "adf_statistic": float(adf_statistic), "adf_pvalue": float(adf_pvalue)})
    except Exception as exc:
        return ModelOutcome("MODEL_ERROR", None, {"error_type": type(exc).__name__, "message": str(exc)})


def garch11_forecast(returns: pd.Series, *, horizon: int = 1) -> ModelOutcome:
    if (len(returns) < 30 or returns.isna().any()
            or not np.isfinite(returns.to_numpy(dtype=float)).all() or horizon < 1
            or not returns.index.is_monotonic_increasing or not returns.index.is_unique):
        raise ValueError("GARCH(1,1) requires >=30 complete returns and positive horizon")
    from arch import arch_model

    model = arch_model(returns.astype(float) * 100, mean="Constant", vol="GARCH", p=1, q=1,
                       dist="normal", rescale=False)
    try:
        fitted = model.fit(disp="off", show_warning=False)
        convergence_flag = int(getattr(fitted, "convergence_flag", 1))
        params = fitted.params
        persistence = float(params.get("alpha[1]", np.inf) + params.get("beta[1]", np.inf))
        if convergence_flag != 0:
            return ModelOutcome("CONVERGENCE_FAILED", None, {"convergence_flag": convergence_flag})
        if not np.isfinite(params.to_numpy(dtype=float)).all() or not 0 <= persistence < 1:
            return ModelOutcome("UNSTABLE_PARAMETERS", None,
                                {"persistence": persistence, "reason": "nonstationary or non-finite fit"})
        forecast = fitted.forecast(horizon=horizon, reindex=False)
        variance = forecast.variance.iloc[-1].to_numpy(dtype=float) / 10_000
        if not np.isfinite(variance).all() or (variance < 0).any():
            return ModelOutcome("UNSTABLE_PARAMETERS", None, {"reason": "invalid variance forecast"})
        frame = pd.DataFrame({"variance": variance}, index=pd.RangeIndex(1, horizon + 1, name="step"))
        return ModelOutcome("AVAILABLE_UNVERIFIED", frame,
                            {"convergence_flag": convergence_flag, "persistence": persistence,
                             "assumption": "Gaussian innovations; estimated from supplied sample"})
    except Exception as exc:
        return ModelOutcome("MODEL_ERROR", None, {"error_type": type(exc).__name__, "message": str(exc)})
