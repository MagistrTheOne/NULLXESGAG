# NULLXES GAG CAP VC

CPU-first, auditable quantitative financial research system. Its forecasts are experimental outputs, not investment advice. Brokerage and trading execution are not implemented.

## Execution boundary

Source changes and static inspection only were performed for this phase. No Python, CLI, pytest, notebook, model, service, API, package installation, or dataset was run locally. The provided Colab notebook is prepared but has not been executed. Do not describe the code as operational or validated until Colab output and data artifacts have been reviewed.

## Architecture

- `src/gagcap/ingestion`: CBR, MOEX ISS, FRED/ALFRED HTTP adapters.
- `src/gagcap/storage`: append/merge observation Parquet lake, DuckDB query access, forecast persistence.
- `src/gagcap/quant`: returns, CAPM/rolling beta, historical/EWMA volatility, SARIMAX/ARIMAX, GARCH(1,1), historical VaR/ES, normal-IID Monte Carlo, Taylor rule and explicit rate shocks.
- `src/gagcap/forecasting`: baseline forecasts, expanding/rolling multi-step walk-forward, point-in-time vintage selection, forecast metrics and experiment persistence.
- `src/gagcap/intelligence`: optional CPU Qwen structured extraction, geopolitical event records/studies, factor association and explicit stress scenarios.
- `app.py`: Streamlit research terminal. It reads persisted records only and reports `DATA_UNAVAILABLE` when they are absent.
- `notebooks/colab_acceptance.ipynb`: Phase 02.5 Colab CPU acceptance flow. It records PASS/FAIL/SKIPPED for each gate and does not install pytest. `notebooks/colab_cpu.ipynb` is the earlier Phase 02 draft and is not the current acceptance path.

## Static audit and corrections

The initial source had no registered `gagcap` command; Parquet writes replaced the prior batch; the DuckDB view used unsafe string interpolation; MOEX parsing trusted response shapes; and backtesting supported only one step without a mandatory datetime index. These were corrected. The lake now merges/deduplicates on source, series, observation date and vintage, writes by temporary-file replacement, and limits SQL to one SELECT/DESCRIBE/SHOW statement. Adapter response contracts and point-in-time checks are explicit.

Static inspection also exposed constraints that remain material: provider adapters have not been exercised against current live schemas; provider publication timestamps are incomplete (especially CBR history); 7/30/90 horizons now require a real session calendar and must not use an annualized approximation; the MOEX candle close is not total return or dividend adjusted; and a fitted model, unit tests, or API reachability alone do not establish forecast value.

## Colab first run

1. Copy this reviewed project folder to `MyDrive/NULLXES/GAG-CAP-VC/project`.
2. Open `notebooks/colab_acceptance.ipynb` in hosted Google Colab and select a CPU runtime. Do not connect a local runtime. Drive mounting prompts for explicit Google authorization.
3. Run the notebook cells in order. Drive mounting asks for Google permission. Put the FRED key in Colab Secrets as `FRED_API_KEY`; do not paste it into source or notebook output.
4. Review the MOEX data license and confirm instrument/board permissions before ingestion. The notebook explicitly checks CBR, MOEX and FRED reachability, then independently ingests available real observations. It stops acceptance progression unless one real MOEX series and one real macro series are stored.
5. The acceptance notebook runs pytest only if it is already present in the Colab runtime; it does not install pytest. Missing pytest is recorded as SKIPPED. It maps each 7/30/90 calendar horizon to the actual next session in the ingested trading calendar. Preserve the generated acceptance report and Parquet artifacts.
6. Qwen remains SKIPPED until `QWEN_GGUF_PATH` points to the official GGUF inside Colab and reviewed source text plus its exact HTTPS citation are supplied. No model is fetched by default. The runtime must use `n_gpu_layers=0`.
7. The Phase 02.5 acceptance notebook does not start Streamlit. Terminal launch remains a separate Colab deployment task and must not be treated as part of acceptance evidence.

The notebook's dependencies and model can require substantial Colab storage and setup time. If any provider, build, test, schema, or extraction step fails, keep that component marked unavailable and retain the exact error output.

## Explicit limitations

- No numerical forecast model has been demonstrated against real data. ARIMAX and GARCH implementations are coded but not fitted or validated here.
- The multi-horizon walk-forward accepts only actual ordered exchange sessions; no built-in source of official calendars is shipped. The caller must provide the applicable session calendar and prevent the training data from including records published after each origin.
- Historical ALFRED vintages are request-budgeted and retrieved only when invoked with an API key. Vintage availability is conservatively represented at the end of the vintage date; it does not establish the exact intraday release time.
- `point_in_time_vintage` requires trustworthy publication/availability timestamps; records without one cannot be safely used for PIT research.
- Qwen output validation checks JSON schema, exact cited spans and that listed entities occur in cited text. It cannot establish semantic truth; confidence stays `UNCALIBRATED` until independently labeled calibration and human review.
- GEO event studies and factor regressions are descriptive associations. No causal identification or robust exposure database is included.
- BASE/BULL/BEAR contracts support unweighted scenarios or explicitly labeled empirical/expert probabilities; no scenario probabilities are inferred automatically.
- Streamlit currently renders observation views, forecast records and source inventory. Macro modeling panels, central-bank event history, GEO events, risk reports, backtest registry and model registry remain `DATA_UNAVAILABLE` until their validated records are persisted and their pages are implemented.
- Historical VaR/ES, normal-IID log-return Monte Carlo, rate-rule estimates, and additive return shocks are research calculations with stated assumptions, not loss guarantees or policy predictions.
- Tests are written but unexecuted. Colab test/integration output, provider freshness, model download, latency, and acceptance remain unverified.

## Commands (Colab only)

The notebook installs `.[terminal,calibration,dev]`; the Qwen section separately installs the CPU `llama-cpp-python` dependency. For explicit manual use inside the Colab runtime after setup:

```bash
python -m pytest -q
gagcap cbr --start 2020-01-01 --end 2026-10-17
gagcap moex --secid SBER --unit RUB_per_share --start 2020-01-01 --end 2026-10-17
```

These commands are documentation for Colab only and were not executed locally. Add `--unit <documented-series-unit>` to MOEX and `--unit <documented-series-unit>` to FRED; FRED also has a separate API `--transform` option. FRED requires `FRED_API_KEY`. MOEX data licensing and security selection remain operator responsibilities.
# NULLXESGAG
