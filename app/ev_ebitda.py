import logging

import pandas as pd
from app.yahoo import yahoo

logger = logging.getLogger(__name__)

def estimate_ev_ebitda(rows):
    df = pd.DataFrame(rows)

    df_sorted = df.sort_values(
        by="EV/EBITDA",
        key=lambda s: pd.to_numeric(s, errors="coerce").fillna(float("inf"))
    ).reset_index(drop=True)

    result = []
    result.append("EV/EBITDA (Enterprise Value / EBITDA; EV may include fair-value adjustments) by sector:")
    view_cols = ["Ticker", "Company", "EV/EBITDA", "EV", "EBITDA_TTM", "Market Cap", "Debt", "Cash"]
    result.append(df_sorted.loc[:, view_cols].to_string(index=False))

    ratio_series = df["EV/EBITDA"].dropna()
    raw = pd.to_numeric(df["EV/EBITDA"], errors="coerce")
    ratio_series = raw[(raw > 0) & (raw < 200)].dropna()
    total = raw.notna().sum()
    if len(ratio_series) == 0:
        result.append("\nMedian EV/EBITDA: N/A")
        result.append("Most Undervalued/Overvalued: N/A")
    else:
        median_ratio = ratio_series.median()

        q1 = ratio_series.quantile(0.25)
        q3 = ratio_series.quantile(0.75)
        iqr = q3 - q1

        df_valid = df.loc[ratio_series.index]
        min_row = df_valid.loc[df_valid["EV/EBITDA"].idxmin()]
        max_row = df_valid.loc[df_valid["EV/EBITDA"].idxmax()]

        result.append(f"\nMedian EV/EBITDA from available data: {median_ratio:.2f}")
        result.append(f"\nIQR (Q3 - Q1, 25%/75%): {iqr:.2f} (Q1={q1:.2f}, Q3={q3:.2f})")
        result.append(f"\nCoverage: {len(ratio_series)} valid of {total} available")

        result.append("\nMost \"undervalued\" (minimum EV/EBITDA):")
        result.append(f"- {min_row['Ticker']} — {min_row['Company']}: {min_row['EV/EBITDA']:.2f}")

        result.append("\nСамая «переоценённая» (максимальный EV/EBITDA):")
        result.append(f"- {max_row['Ticker']} — {max_row['Company']}: {max_row['EV/EBITDA']:.2f}")

        result.append("\nMost \"overvalued\" (maximum EV/EBITDA):")
        result.append(f"- {max_row['Ticker']} — {max_row['Company']}: {max_row['EV/EBITDA']:.2f}")

    return "\n".join(result)

