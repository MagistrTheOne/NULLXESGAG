# NULLXES GAG CAP VC

CPU-first quantitative financial research system. Forecasts are experimental research outputs. Brokerage and trading execution are not implemented.

## Colab CPU run

1. Push this repository to `MagistrTheOne/NULLXESGAG` and open [`notebooks/colab_cpu.ipynb`](notebooks/colab_cpu.ipynb) in hosted Google Colab with a CPU runtime. Do not connect a local runtime.
2. Run notebook cells in order. The notebook checks out the project directly from GitHub and asks Google to authorize a Drive mount for persistent data and reports.
3. The notebook checks core dependencies using `importlib.util.find_spec` and installed versions. It installs only absent or below-minimum packages. Streamlit and Qwen dependencies are handled as optional components and report their availability.
4. To enable FRED, add `FRED_API_KEY` to Colab Secrets. Without it, the notebook continues with available CBR and MOEX series and records FRED as unavailable.
5. Review MOEX licensing and access permissions for selected instruments before ingestion. The notebook stores provider observations, provenance, forecast records, risk metrics, macro context, GEO reports, and the analytical report under `MyDrive/NULLXES/GAG-CAP-VC`.
6. Forecast horizons are 7, 30, and 90 calendar days mapped through the Moscow exchange calendar. The reported point forecast is the random-walk baseline; the interval uses historical IID normal log-return Monte Carlo and is explicitly uncalibrated. Macro observations are context, not claimed causal predictors. Qwen extraction remains unverified and source-linked reports remain marked unverified.
7. The final notebook cell opens the Streamlit research terminal through Colab's authenticated port proxy. It does not create a public tunnel.

## Project modules

- `src/gagcap/ingestion`: CBR, MOEX ISS, and FRED/ALFRED adapters.
- `src/gagcap/storage`: Parquet observation lake, DuckDB queries, and forecast persistence.
- `src/gagcap/quant`: returns, CAPM/rolling beta, volatility, SARIMAX/ARIMAX, GARCH, risk, Monte Carlo, and rate scenarios.
- `src/gagcap/forecasting`: point-in-time handling, multi-horizon walk-forward logic, baselines, and evaluation metrics.
- `src/gagcap/intelligence`: optional CPU Qwen extraction and GEO event analysis.
- `app.py`: Streamlit research terminal reading persisted project artifacts.

## Readiness boundary

This repository and notebook have not been executed in Colab. Provider reachability, package compatibility, model download and inference, artifact generation, and the terminal session remain unverified until the user runs the notebook and reviews its output. Unavailable data or integrations are reported as unavailable; the workflow does not create synthetic financial observations.

MOEX close prices are not dividend-adjusted total returns. Historical risk and Monte Carlo intervals depend on their stated assumptions and are not guarantees. Qwen confidence is not a statistical event probability, and extracted GEO claims remain unverified reports unless independently confirmed.
