# Pipeline reference

The build is a make dependency graph over parquet layers. External fetches are manual targets that never run as a side effect of a local build; every stage that can be computed from files already on disk is wired into the graph and rebuilds when its inputs change.

## Data flow

```mermaid
flowchart TD
    subgraph raw["Raw inputs"]
        CSV["Investing.com and exchange CSVs<br/>(committed under data/raw/)"]
        NET["Binance, Yahoo, FRED, World Bank<br/>(network fetches)"]
    end
    subgraph bronze["Bronze: data/parquet"]
        INV["investing_manual.parquet"]
        EQ["equity.parquet"]
        FX["fx.parquet"]
        CTRL["controls.parquet"]
        KAO["kaopen.parquet"]
        BTC1["btcusdt / YYYY-MM.parquet"]
        ETH1["ethusdt / YYYY-MM.parquet"]
    end
    subgraph silver["Silver: data/parquet/clean"]
        EQ2["equity.parquet"]
        FX2["fx.parquet"]
        BTC2["btc_1min.parquet"]
        ETH2["eth_1min.parquet"]
    end
    subgraph gold["Gold: data/parquet/gold"]
        RV["btc_rv_daily.parquet<br/>eth_rv_daily.parquet"]
        PNL["panel.parquet"]
        MEM["sample_membership.parquet<br/>crisis_flags.parquet"]
        SPILL["spillovers.parquet<br/>spillovers_w60.parquet<br/>spillovers_eth.parquet"]
        SYS3["spillover_components.parquet<br/>spillover_3var.parquet<br/>spillovers_stability.parquet"]
        STEP2["step2_analysis.parquet<br/>step2_panel_coefs.parquet"]
        ROB["step2_robustness_coefs.parquet<br/>decomp_coefs.parquet<br/>stability_comparison.parquet<br/>step2_leaveout_coefs.parquet<br/>horse_race_coefs.parquet<br/>dev_horse_race_coefs.parquet"]
    end
    subgraph out["Generated artifacts (local, not committed)"]
        TABLES["thesis/tables"]
        APPENDIX["thesis/appendix"]
        FIGURES["thesis/figures"]
    end

    CSV --> INV
    NET --> EQ
    NET --> FX
    NET --> CTRL
    NET --> KAO
    NET --> BTC1
    NET --> ETH1

    INV --> EQ2
    EQ --> EQ2
    FX --> FX2
    BTC1 --> BTC2
    ETH1 --> ETH2

    BTC2 --> RV
    ETH2 --> RV
    EQ2 --> PNL
    FX2 --> PNL
    CTRL --> PNL
    KAO --> PNL
    RV --> PNL

    EQ2 --> MEM
    FX2 --> MEM

    PNL --> SPILL
    PNL --> SYS3
    PNL --> SPILL
    SPILL --> STEP2
    MEM --> STEP2
    SPILL --> ROB
    STEP2 --> ROB
    SYS3 --> ROB
    SPILL --> SYS3

    STEP2 --> TABLES
    MEM --> APPENDIX
    STEP2 --> FIGURES
    ROB --> FIGURES
    SYS3 --> FIGURES
```

The generated tables, appendices, and figures land in a local `thesis/` folder that stays out of version control. The scripts that write them are committed (`scripts/make_tables.py`, `scripts/make_appendix_*.py`, `scripts/make_figures.py`), so every reported number remains reproducible from the gold parquets.

## Targets

| Target | What it does | Network |
|---|---|---|
| `fetch-crypto` | Binance 1-minute BTCUSDT and ETHUSDT klines, one file per month, skips months already on disk | yes |
| `fetch-markets` | Yahoo equity indices, FX rates, VIX, S&P 500, Brent; FRED DTWEXBGS | yes |
| `fetch-investing` | Parses the committed Investing.com and exchange CSVs into `investing_manual.parquet` | no |
| `bronze` | All three bronze stages above | yes |
| `clean` | Silver layer: deduplicates, sorts, applies the glitch rule, validates schemas | no |
| `err_float.csv` rule | Regenerates the regime table from `config/countries.yaml`; uncurated countries get a -1 sentinel and fail loudly downstream | no |
| `kaopen` rule | Parses the committed Chinn-Ito workbook into `kaopen.parquet` | no |
| `features` | Realized variance for BTC and ETH, daily returns, and the country-day panel | no |
| `sample` | Applies the frozen data-quality rule in `config/sample_rule.yaml`; writes `sample_membership.parquet` | no |
| `crisis` | Flags country-years with CPI inflation of at least 25% or depreciation of at least 30% (World Bank CPI) | yes |
| `spillovers` | Rolling bivariate VAR(1) plus generalized FEVD for every country and channel, 200-day window | no |
| `panel` | Panel preparation and the Step 2 regressions (pooled OLS with Driscoll-Kraay standard errors, two-way fixed effects, H2 and H3 specifications) | no |
| `robustness` | Winsorized log-odds outcome and the 60-day window variant | no |
| `decomposition` | Variance-component diagnostics and the trivariate system with the broad dollar index | no |
| `stability` | Records each rolling window's companion-matrix spectral radius | no |
| `stability-coefs` | Re-runs the headline results excluding non-stationary windows, next to an unfiltered baseline | no |
| `leaveouts` | Crisis-country drop, Ethereum as transmitter, and the trivariate specification | no |
| `horse-race` | Global-risk and development horse races (openness, VIX, dollar index, GDP per capita, private credit all interacted with standardized crypto stress) | no |
| `h3-extra` | Extended H3 diagnostics: the two-way fixed-effects rung and the equity placebo | no |
| `fetch-dev` | World Bank development controls (GDP per capita, private credit, 2019-2023 means) | yes |
| `tables` | Generates the regression tables into a local `thesis/tables/` | no |
| `appendix` | Generates the sample and variable appendices into a local `thesis/appendix/` | no |
| `figures` | Generates the result figures into a local `thesis/figures/` | no |
| `diagrams` | Renders the conceptual framework diagram from its Graphviz source (requires the `dot` binary) | no |
| `check` | Runs the test suite (190 tests) | no |
| `all` | Every local stage above, in dependency order | no |

## Conventions

- A target that talks to an external service is manual and marked in the table above. `make all` never hits the network.
- Every parquet write passes a Pandera schema contract before the file is accepted by the next stage.
- Missing observations stay missing at every layer. There is no forward-fill, interpolation, or imputation anywhere in the pipeline.
- All FX rates are local currency per USD, so an increase always means depreciation.
- The frozen sample binds the regressions in `panel_prep`: spillover shares for channels that failed the data-quality rule are set to NaN rather than dropped, so the regression sample is auditable.
- Tables, appendices, and figures are generated exclusively from the gold parquets and the config files. Nothing is hand-transcribed.

## Typical runtimes

| Stage | Approximate time |
|---|---|
| `fetch-crypto` | under an hour for the full 2019-2025 grid at polite request rates |
| `spillovers` | about 10 minutes per variant (baseline, 60-day, Ethereum) |
| `stability` | about 7 minutes |
| `check` | about 2 minutes |
| `all` | roughly an hour on a modern laptop |
