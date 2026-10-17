# NULLXES GAG CAP VC — Phase 02 static handoff

Audit boundary: source files were read and edited only. No project code, tests, package manager, provider API, notebook, model, Streamlit process, or Colab session was executed.

| Area | Status | Evidence boundary |
| --- | --- | --- |
| Typed observation contracts and explicit missing-value quality | IMPLEMENTED | Source only; provider payload behavior UNVERIFIED |
| CBR, MOEX ISS, FRED/ALFRED adapters | IMPLEMENTED | Adapters and opt-in integration tests written; live schema, coverage, licensing UNVERIFIED |
| Parquet lake and DuckDB query layer | IMPLEMENTED | Static code only; persistence/query behavior UNVERIFIED |
| Returns, volatility, CAPM, SARIMAX/ARIMAX, GARCH, VaR/ES, Monte Carlo, Taylor rule | IMPLEMENTED | Mathematical implementation written; numerical behavior UNVERIFIED |
| Forecast metadata and 7/30/90 calendar-day approximation | IMPLEMENTED | Conversion uses 252/365.25 scaling; exchange-calendar correctness UNVERIFIED |
| Expanding/rolling multi-step backtest and PIT vintage selector | IMPLEMENTED | Unit tests written; vintage-aware code accepts multiple snapshots, but historical PIT coverage is UNVERIFIED and notebook fetches one current snapshot only |
| Metrics, seasonal naive and transaction cost accounting | IMPLEMENTED | Source/tests written; strategy research performance UNVERIFIED |
| Qwen GGUF CPU path and structured extraction | IMPLEMENTED | Path, schema, citations, entity checks written; model download/inference UNVERIFIED |
| Qwen confidence calibration | IMPLEMENTED | Calibration fitter requires labeled examples; calibration data and quality UNAVAILABLE |
| GEO event records, event study, factor association, stress shocks | IMPLEMENTED | User-sourced events required; source coverage and estimates UNAVAILABLE |
| BASE/BULL/BEAR scenario contracts | IMPLEMENTED | Probability labels validated in code; scenario estimates UNAVAILABLE |
| Streamlit shell and real-data display | IMPLEMENTED | Sections exist; only observations, forecast records and source inventory have views |
| Macro/GEO/risk/backtest/model-registry terminal pages | UNAVAILABLE | No validated persisted records or page implementations |
| Unit and provider integration tests | IMPLEMENTED | Test files written only; all test results UNVERIFIED |
| Google Colab CPU workflow | IMPLEMENTED | Notebook authored only; installation, API calls, dataset, tests, model and UI UNVERIFIED |
| Quantitative acceptance on real market plus macro series | UNAVAILABLE | No data has been ingested or backtested in this handoff |

## Known limits

- FRED API access requires a Colab Secret named `FRED_API_KEY` and a documented measurement unit supplied by the operator.
- MOEX board/instrument selection and commercial or redistribution rights require operator review.
- CBR history does not provide complete macro release timestamps; observations lacking trustworthy availability times cannot be used for strict PIT evaluation.
- Calendar-day horizons map to approximate session counts, not a verified exchange calendar.
- Qwen cannot validate the truth of its own classifications. Exact quotes and entity-to-quote matching reduce unsupported output but do not replace human review.
- Event studies and factor regressions are associations and do not establish causality.
- The Data Foundation acceptance gate and all runtime behavior remain unverified until Colab produces reviewable logs and persisted artifacts.
