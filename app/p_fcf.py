import pandas as pd
from yahoo import yahoo


def estimate(rows):
    """Принимает массив по тикерам и возвращает строку с анализом P/FCF (медиана + IQR)."""
    df = pd.DataFrame(rows)

    # Сортируем по P/FCF (меньше — лучше). NaN уводим в конец.
    df_sorted = df.sort_values(by="P/FCF", key=lambda s: s.fillna(float("inf"))).reset_index(drop=True)

    parts = []
    parts.append("P/FCF (Price to Free Cash Flow = рыночная капитализация / свободный денежный поток) по сектору экономики:\n")
    view_cols = ["Ticker", "Company", "P/FCF", "Market Cap", "FCF_TTM"]
    parts.append(df_sorted.loc[:, view_cols].to_string(index=False))

    ratio_series = df["P/FCF"].dropna()

    if ratio_series.empty:
        parts.append("\nМедианный P/FCF: нет данных")
        parts.append("IQR (Q3 - Q1, 25%/75%): нет данных")
        parts.append("Самая недооценённая/переоценённая: нет данных")
        return "\n".join(parts)

    # --- ключевые метрики на выборке ---
    median_ratio = ratio_series.median()
    q1 = ratio_series.quantile(0.25)   # 25-й перцентиль
    q3 = ratio_series.quantile(0.75)   # 75-й перцентиль
    iqr = q3 - q1

    # Для справок по тикерам (минимум/максимум P/FCF)
    min_row = df.loc[df["P/FCF"].idxmin()]
    max_row = df.loc[df["P/FCF"].idxmax()]

    parts.append(f"\nМедианный P/FCF по доступным данным: {median_ratio:.2f}")
    parts.append(f"IQR (Q3 - Q1, 25%/75%): {iqr:.2f}  (Q1={q1:.2f}, Q3={q3:.2f})")

    parts.append("\nСамая «недооценённая» (минимальный P/FCF):")
    parts.append(f"- {min_row['Ticker']} — {min_row['Company']}: {min_row['P/FCF']:.2f}")

    parts.append("\nСамая «переоценённая» (максимальный P/FCF):")
    parts.append(f"- {max_row['Ticker']} — {max_row['Company']}: {max_row['P/FCF']:.2f}")

    return "\n".join(parts)


if __name__ == "__main__":
    TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    rows = yahoo(TICKERS)
    report = estimate(rows)
    print(report)