import pandas as pd
from app.yahoo import yahoo


def estimate(rows):
    df = pd.DataFrame(rows)

    df_sorted = df.sort_values(by="P/FCF", key=lambda s: s.fillna(float("inf"))).reset_index(drop=True)

    parts = []
    parts.append("P/FCF (Price to Free Cash Flow = market capitalization / free cash flow) by sector of the economy:\n")
    view_cols = ["Ticker", "Company", "P/FCF", "Market Cap", "FCF_TTM"]
    parts.append(df_sorted.loc[:, view_cols].to_string(index=False))

    ratio_series = df["P/FCF"].dropna()

    if ratio_series.empty:
        parts.append("\nMedian P/FCF: no data")
        parts.append("IQR (Q3 - Q1, 25%/75%): no data")
        parts.append("Most Undervalued/Overvalued: no data")
        return "\n".join(parts)

    median_ratio = ratio_series.median()
    q1 = ratio_series.quantile(0.25)
    q3 = ratio_series.quantile(0.75)
    iqr = q3 - q1

    min_row = df.loc[df["P/FCF"].idxmin()]
    max_row = df.loc[df["P/FCF"].idxmax()]

    parts.append(f"\nMedian P/FCF from available data: {median_ratio:.2f}")
    parts.append(f"IQR (Q3 - Q1, 25%/75%): {iqr:.2f}  (Q1={q1:.2f}, Q3={q3:.2f})")

    parts.append("\nMost \"undervalued\" (minimum P/FCF):")
    parts.append(f"- {min_row['Ticker']} — {min_row['Company']}: {min_row['P/FCF']:.2f}")

    parts.append("\nMost \"overvalued\" (maximum P/FCF):")
    parts.append(f"- {max_row['Ticker']} — {max_row['Company']}: {max_row['P/FCF']:.2f}")

    return "\n".join(parts)

