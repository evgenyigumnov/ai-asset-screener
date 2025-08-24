import pandas as pd
from app.yahoo import yahoo

def _pe_sort_key(s: pd.Series):
    x = pd.to_numeric(s, errors="coerce")
    x = x.where((x > 0) & (x < 200), float("inf"))  # мусор в конец
    return x

def estimate(rows):
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by="Forward P/E", key=_pe_sort_key).reset_index(drop=True)

    result = ""
    result += "Forward P/E (Price-to-Earnings = цена акции / прибыль на акцию) по сектору экономики:\n"
    view_cols = ["Ticker", "Company", "Forward P/E"]
    result += df_sorted.loc[:, view_cols].to_string(index=False) + "\n"

    raw = pd.to_numeric(df["Forward P/E"], errors="coerce")
    valid = raw[(raw > 0) & (raw < 200)].dropna()
    total = raw.notna().sum()

    if valid.empty:
        result += "\nMedian Forward P/E: no data\n"
        result += "Most Undervalued/Overvalued: no data\n"
        return result

    median_pe = valid.median()
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1

    df_valid = df.loc[valid.index]
    min_row = df_valid.loc[df_valid["Forward P/E"].idxmin()]
    max_row = df_valid.loc[df_valid["Forward P/E"].idxmax()]

    result += f"\nMedian Forward P/E based on available data: {median_pe:.2f}\n"
    result += f"IQR (Q3 - Q1, 25%/75%): {iqr:.2f} (Q1={q1:.2f}, Q3={q3:.2f})\n"
    result += f"Coverage: {len(valid)} valid out of {total} available.\n"
    result += "\nMost \"undervalued\" (minimum Forward P/E):\n"
    result += f"- {min_row['Ticker']} — {min_row['Company']}: {min_row['Forward P/E']:.2f}\n"
    result += "\nMost \"overvalued\" (maximum Forward P/E):\n"
    result += f"- {max_row['Ticker']} — {max_row['Company']}: {max_row['Forward P/E']:.2f}\n"
    return result


