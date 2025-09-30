# ai-asset-screener

## General info

`ai-asset-screener` is a CLI tool for fast, fundamentals-driven equity screening with LLM-assisted reading of 10-K filings. It pulls core metrics from Yahoo Finance, enriches them with fair-value EV adjustments and insurance float extracted from 10-Ks, and builds a segment-level SOTP model. The tool issues BUY/SELL/UNSURE signals per metric and per peer group using robust, outlier-aware rules, then aggregates them into a final verdict. Configuration is done via environment variables (`.env` supported): `LLM_MODEL` (required), plus either an OpenAI-compatible `LLM_ENDPOINT` or `LLM_OPENAI_API_KEY`. Results are printed as readable console reports with group details and a summary table; intermediate data and 10-K parses are cached under `cache/`. Logging goes to `asset.log`.

## Features list

* **Data foundation**

  * Yahoo Finance fetch (market cap, EV, debt, cash, FCF TTM, EBIT/EBITDA TTM, Forward P/E) with local JSON caching per ticker.
  * EDGAR 10-K ingestion (via `edgartools`) to analyze notes and segment disclosures; large-file chunking and caching.

* **Fair-value EV adjustments (10-K notes => EV)**

  * LLM extracts candidate items with signed deltas (USD **millions**): e.g., lease liabilities, pension deficits, TRA payables, contingent consideration, AROs, environmental liabilities, FV uplifts/discounts.
  * Canonicalization + **whitelist/blacklist** filters (drops obligations without recognized liabilities, operating-lease *obligations* schedules, VIE “exposures”, deferred taxes, etc.).
  * **Sanity caps** by category and as a fraction of EV; duplicates merged; EV and EV/EBITDA recomputed.

* **Insurance float model (insurers only)**

  * LLM reads balance/notes to assemble float components (USD **millions**):

    * Unpaid losses & LAE (net), Unearned premiums (net), Future policy benefits, Policyholders’ account balances, Other insurance liabilities;
      − Reinsurance recoverable, (Re)insurance balances receivable, Deferred charge assets for retroactive reinsurance.
  * Strict **netting rules** (prefer NET where disclosed), scale sanity checks, and Float/EV ratio computed.

* **SOTP (Sum-of-the-Parts) from operating segments**

  * LLM extracts **operating segments** (ASC 280) for the latest year and parses a clean table: `Segment | Revenue | Operating income`.
  * Segment valuation via peer **median EV/EBIT**; fallback: **EV/EBITDA × 1.25** when EV/EBIT is sparse.
  * Loss-making segments: insurers => EV≈0 (conservative); others => fallback **EV/Sales ≈ 1×** if revenue is available.
  * Outputs per-segment implied EVs, peer bucket used, multiples applied, and a **premium/discount vs current EV**.

* **Signals & robustness**

  * **Metric signals (IQR rule):**

    * Forward P/E, P/FCF, EV/EBITDA => BUY if value < Q1, SELL if > Q3, else UNSURE (with caps for outliers).
    * **Float/EV** signal for insurers uses peer quartiles **within subsector** (P\&C / Life / Reins / Conglom).
  * **SOTP signal:** discount/premium threshold ±10% => BUY/SELL; otherwise UNSURE.
  * **Majority vote** across available signals with tie-break priority: EV/EBITDA > P/FCF > Forward P/E.

* **Peer groups & auto-scoping**

  * Built-in universes (e.g., BIG\_TECH, SEMI, CLOUD\_SOFTWARE, INTERNET\_ADS, ECOMMERCE, AUTO\_EV, ASSET\_MANAGERS, CONGLOMERATES, INSURANCE, CRYPTO miners/exchanges).
  * Auto-selection of relevant groups for a ticker or explicit filtering via `--group`.

* **DX & ops**

  * `.env` support; clear stdout reports (per group + consolidated table).
  * Persistent caching for Yahoo snapshots and 10-K-derived artifacts.
  * Structured logging to `asset.log`.


## How to run:

```bash
pip install -e .
ai-asset-screener --ticker=ADBE --group=BIG_TECH_CORE  --use-cache
```

.env file example:
```toml
LLM_ENDPOINT = "http://localhost:1234/v1"
LLM_MODEL = "openai/gpt-oss-20b"
```
Add LLM_OPENAI_API_KEY if you use commercial OpenAI API and remove LLM_ENDPOINT