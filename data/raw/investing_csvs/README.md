# Investing.com CSV Drops

Place manually downloaded Investing.com historical data CSVs in this directory.

## Naming Convention

```
{COUNTRY_ID}_{data_type}.csv
```

- `COUNTRY_ID`: ISO3 country code (uppercase)
- `data_type`: `equity` or `fx`

### Examples
```
CZE_equity.csv     # PX Index (Prague SE); yfinance can't resolve regional ^-prefixed tickers
PER_equity.csv     # SPBLPGPT
POL_equity.csv     # WIG20 (blue-chip benchmark); yfinance can't resolve regional ^-prefixed tickers
COL_equity.csv     # ^COLCAP unreliable on Yahoo; download CSV from Investing.com
EGY_equity.csv     # EGX30 (Cairo & Alexandria Stock Exchanges); ^CASE30 returns no data on yfinance
NGA_equity.csv     # ^NGSEINDX has poor coverage on Yahoo
TUR_equity.csv     # BIST 100; yfinance can't resolve regional ^-prefixed tickers

GHA_equity.csv     # https://www.african-markets.com/index.php/en/stock-markets/gse

MUS_equity.csv     # https://www.stockexchangeofmauritius.com/products-market-data/indices
DZA_equity.csv     # https://www.sikafinance.com/marches/historiques/DZIDX.dz
ECU_equity.csv     # https://seekingalpha.com/symbol/BVQA/historical-price-quotes
KGZ_equity.csv     # https://countryeconomy.com/stock-exchange/kyrgyzstan?dr=2026-01 
LAO_equity.csv     # https://seekingalpha.com/symbol/LSXC/historical-price-quotes
MDV_equity.csv     # https://stockexchange.mv/insights
MNG_equity.csv     # https://mse.mn/index
RUS_equity.csv     # https://www.moex.com/en/index/IMOEX/archive?from=2019-01-01&till=2025-12-31&sort=TRADEDATE&order=desc
TZA_equity.csv     # https://africanfinancials.com/pt-index/tz-xtzall/
UKR_equity.csv     # https://pfts.ua/en/1-market-data/1-pfts-index/1-index-data
BHR_equity.csv     # https://bahrainbourse.com/en/Quotes%20and%20Market/Indices/bahrain-index
BRB_equity.csv     # https://www.sase.ba/v1/en-us/Market/General-Information/Indices
BWA_equity.csv     # https://africanfinancials.com/pt-index/bw-xbwdci/
BGR_equity.csv     # https://www.infostock.bg/infostock/control/trading/index/graphics/SOFIX
HRV_equity.csv     # https://zse.hr/en/indeks-366/365?isin=HRZB00ICBEX6&tab=index_history&date_from=2019-01-01&date_to=2025-12-31
PAN_equity.csv     # https://countryeconomy.com/stock-exchange/panama?dr=2026-
SVK_equity.csv     # https://www.bsse.sk/bcpb/en/indices/sax-index/
SVN_equity.csv     # https://ljse.si/en/indeks-366/365?isin=SI0026109882&tab=index_history&date_from=2023-05-10&date_to=2025-12-31
UGA_equity.csv     # https://africanfinancials.com/pt-index/ug-xugall/

KGZ_fx.csv         # https://www.investing.com/currencies/usd-kgs-historical-data
MNG_fx.csv         # https://www.investing.com/currencies/usd-mnt-historical-data

MAR_equity.csv     → Morocco MASI index (missing Volume)
TUN_equity.csv     → Tunisia Tunindex
BGD_equity.csv     → Bangladesh DSEX (missing Volume)
DZA_fx.csv         → Algeria DZD/USD FX rate (missing Volume)
TUN_fx.csv         → Tunisia TND/USD FX rate (missing Volume)
PAK_equity.csv     → Pakistan KSE100
VNM_equity.csv     → Vietnam VN-Index
NGA_equity.csv     → Nigeria NGX All-Share (= NSE All Share / NGSEINDEX — rebranded 2021) 
KEN_equity.csv     → Kenya NASI (Nairobi All Share Index) (missing High, low, Volume) "https://www.african-markets.com/en/stock-markets/nse"
```

### Dropped Countries
- **Ghana (GHA)**: GSE Composite not available on Investing.com — cut per cut-order item 3
- **Bolivia (BOL)**: Bolsa Boliviana de Valores not available on Investing.com — cut per cut-order item 3

## Countries Needing Investing.com CSVs

| Country | ID | Data needed | Investing.com URL hint |
|---------|-----|-------------|------------------------|
| Morocco | MAR | equity (MASI) | Search "MASI historical data" |
| Tunisia | TUN | equity (Tunindex) + FX | Search "Tunindex historical data" |
| Algeria | DZA | FX only (DZD/USD) | Search "USD/DZD historical data" |
| Bangladesh | BGD | equity (DSEX) | Search "DSEX historical data" |
| Pakistan | PAK | equity (KSE100) | Search "KSE 100 historical data" |
| Vietnam | VNM | equity (VN-Index) | Search "VN-Index historical data" |
| Nigeria | NGA | equity (NGX All-Share / NGSEINDEX) | ✅ Downloaded — NSE All Share = NGX All-Share (rebranded 2021) |
| Kenya | KEN | equity (NASI) | Search "Nairobi All Share historical data" — use NASI, NOT NSE20 |

## How to Download

1. Navigate to the historical data page for the index on Investing.com
2. Click the calendar icon and set range: **Jan 01, 2019 → Dec 31, 2025**
3. Click **"Download Data"** (blue button, top-right of the table)
4. Save the file and rename to match the convention above
5. Drop it here

## Expected CSV Format

Investing.com downloads look like this:
```csv
"Date","Price","Open","High","Low","Vol.","Change %"
"Nov 01, 2023","12,345.67","12,300.00","12,400.00","12,200.00","1.5M","0.37%"
"Oct 31, 2023","12,300.00","12,250.00","12,350.00","12,100.00","1.3M","-0.12%"
```

The `investing_csv_to_bronze.py` Spark job auto-detects this format.

## Notes

- Files are sorted newest-first in Investing.com downloads — the parser handles this
- Volume column uses shorthand notation (`1.5M`, `230K`) — stored as string in Bronze, parsed in Silver
- `Change %` column is a cross-check only — returns are always recomputed from prices
- FX CSVs from Investing.com typically only have `Date` and `Price` columns — that is fine

## Verification After Ingestion

After running `spark_investing_csv_to_bronze.py`, verify with:
```python
spark.read.format("delta").load("/opt/data/bronze/investing_manual") \
    .groupBy("country_id", "data_type") \
    .count() \
    .orderBy("country_id") \
    .show()
```
