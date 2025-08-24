import re
import math
import pandas as pd

from app.edgar_extractor import extract_operating_segments, parse_markdown_table
from app.yahoo import yahoo

_PEER_GROUPS = {
    "software":   ["MSFT","ORCL","IBM","NOW","CRM","ADBE","SNOW","MDB","DDOG","NET","ZS","OKTA"],
    "semis":      ["NVDA","AVGO","AMD","INTC","QCOM","TXN","MU","ARM","ASML","ADI","AMAT","LRCX"],
    "ads":        ["GOOGL","META","NFLX","TTD","SNAP","PINS","RBLX"],
    "auto":       ["TSLA","GM","F","STLA","RIVN","LCID","NIO","LI","XPEV"],
    "ecom":       ["AMZN","SHOP","MELI","EBAY","ETSY","SE","WMT","COST","TGT"],
    "rail":       ["UNP","CSX","NSC","CNI","CP"],
    "pc_ins":     ["CB","PGR","TRV","ALL","HIG"],
    "life_ins":   ["MET","PRU","AIG","LNC","EQH"],
    "reins":      ["RNR","EG","RGA"],
    "utilities":  ["NEE","DUK","SO","AEP","XEL","PCG"],
    "industrial": ["HON","GE","ITW","EMR","MMM","CAT"],
    "bigtech":    ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","AVGO","ORCL","ADBE","CRM"],  # дефолт
}

_SEGMENT_MAP = [
    (re.compile(r'\breinsur', re.I), "reins"),
    (re.compile(r'\blife|annuities?\b', re.I), "life_ins"),
    (re.compile(r'\binsur|underwrit|general re|geico\b', re.I), "pc_ins"),
    (re.compile(r'\brail|railroad|bnsf\b', re.I), "rail"),
    (re.compile(r'energy|utility|power|grid|transmission', re.I), "utilities"),
    (re.compile(r'manufactur|industrial|parts|components', re.I), "industrial"),
    (re.compile(r'e-?commerce|retail|marketplace', re.I), "ecom"),
    (re.compile(r'cloud|software|services|saas|platform|subscription', re.I), "software"),
    (re.compile(r'semi|chip|foundry|network|connect|datacenter|broadband|gpu|gaming', re.I), "semis"),
    (re.compile(r'\b(ad|advert|marketing|commerce|content)\b', re.I), "ads"),
    (re.compile(r'auto|vehicle|mobility|\bev\b', re.I), "auto"),
]

_SEGMENT_OVERRIDES = {
    "BRK.B": [
        (re.compile(r'\bBNSF\b', re.I),                             "rail"),
        (re.compile(r'Berkshire Hathaway Energy', re.I),            "utilities"),
        (re.compile(r'Manufacturing.*Retailing', re.I),             "industrial"),
        (re.compile(r'GEICO|Primary Group', re.I),                  "pc_ins"),
        (re.compile(r'Berkshire Hathaway Reinsurance Group', re.I), "reins"),
    ],
}


_peer_median_cache = {}

def _peer_multiple_ev_ebit_or_ev_ebitda(tickers):
    rows = yahoo(tickers)

    s_ebit = pd.to_numeric(pd.Series([r.get("EV/EBIT") for r in rows]), errors="coerce")
    s_ebit = s_ebit[(s_ebit > 0) & (s_ebit < 200)].dropna()
    if len(s_ebit) >= 3:
        return float(s_ebit.median()), "EV/EBIT"

    s_ebitda = pd.to_numeric(pd.Series([r.get("EV/EBITDA") for r in rows]), errors="coerce")
    s_ebitda = s_ebitda[(s_ebitda > 0) & (s_ebitda < 200)].dropna()
    if len(s_ebitda) >= 3:
        return float(s_ebitda.median()) * 1.25, "EV/EBITDA×1.25"

    return None, ""

def _median_ev_ebit(tickers):
    rows = yahoo(tickers)
    s = pd.to_numeric(pd.Series([r.get("EV/EBIT") for r in rows]), errors="coerce")
    s = s[(s > 0) & (s < 200)].dropna()
    return float(s.median()) if len(s) else None

def _peer_multiple_for_segment(ticker: str, seg_name: str) -> tuple[str, float, str]:
    rules = _SEGMENT_OVERRIDES.get(ticker, [])
    for rx, key in rules:
        if rx.search(seg_name or ""):
            if key not in _peer_median_cache:
                mult, lbl = _peer_multiple_ev_ebit_or_ev_ebitda(_PEER_GROUPS[key])
                _peer_median_cache[key] = (mult, lbl)
            return key, *_peer_median_cache[key]

    group_key = "bigtech"
    for rx, key in _SEGMENT_MAP:
        if rx.search(seg_name or ""):
            group_key = key
            break

    if group_key not in _peer_median_cache:
        mult, lbl = _peer_multiple_ev_ebit_or_ev_ebitda(_PEER_GROUPS[group_key])
        _peer_median_cache[group_key] = (mult, lbl)

    mult, lbl = _peer_median_cache[group_key]
    return group_key, mult, lbl


def _fmt_b(n):
    if n is None or not math.isfinite(n):
        return ""
    return f"{n/1e9:.2f}"

def _coerce_num(x):
    try:
        if x is None or x == "":
            return None
        v = float(str(x).strip())
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None

def _normalize_units_df(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    def _num(x):
        try:
            return float(str(x).replace(',', '').strip())
        except Exception:
            return None

    sample = []
    for c in df2.columns:
        sample.extend([_num(v) for v in df2[c].values[:]])
    if any(isinstance(v, (int, float)) and v and v > 100_000 for v in sample):
        for c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce") / 1000.0
    return df2


def calculate_sotp_for_ticker(ticker: str, LLM_MODEL: str, LLM_ENDPOINT: str, LLM_OPENAI_API_KEY: str):
    seg_md = extract_operating_segments(ticker, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
    if not seg_md.strip():
        return {"md": "_Not enough data on operating segments (or 10-K only provides geography); SOTP is not calculated._",
                "total_implied": None, "premium_pct": None}
    df = parse_markdown_table(seg_md)
    if df is None or df.empty:
        return {"md": "_Failed to parse the operational segment table; SOTP is not calculated._",
                "total_implied": None, "premium_pct": None}
    df = _normalize_units_df(df)

    cols = {c.lower(): c for c in df.columns}
    seg_col = cols.get("segment") or next((c for c in df.columns if "segment" in c.lower()), None)
    rev_col = next((c for c in df.columns if "revenue" in c.lower() or "net sales" in c.lower()), None)
    op_col  = next((c for c in df.columns
                    if re.search(r'\boperat(ing)?\b', c, re.I)
                    or re.search(r'underwrit', c, re.I)
                    or re.search(r'pre[-\s]?tax', c, re.I)
                    or re.search(r'segment\s+profit', c, re.I)), None)
    if seg_col is None:
        return {"md": "_There is no Segment column in the table; SOTP is not calculated._",
                "total_implied": None, "premium_pct": None}

    rows = yahoo([ticker])
    cur_ev = rows[0].get("EV")

    out_rows, implied_values = [], []

    for _, r in df.iterrows():
        seg = str(r.get(seg_col, "")).strip()
        if not seg:
            continue  # пропускаем пустые
        if re.search(r'eliminat|intersegment', seg, re.I):
            continue

        rev = None
        opi = None
        try:
            rev = float(str(r.get(rev_col)).replace(',', '').strip()) if rev_col else None
        except Exception:
            pass
        try:
            opi = float(str(r.get(op_col)).replace(',', '').strip()) if op_col else None
        except Exception:
            pass

        peer_key, mult, mult_label = _peer_multiple_for_segment(ticker, seg)
        implied_ev = None
        note = ""

        if opi is not None and mult is not None and math.isfinite(mult):
            if opi > 0:
                implied_ev = opi * 1e6 * mult
                note = f"{mult_label or 'EV/EBIT'}≈{mult:.2f} ({peer_key})"
            elif opi < 0:
                if peer_key in ("pc_ins", "life_ins", "reins"):
                    implied_ev = 0.0
                    note = "loss (insurance): assume 0 (conservative)"
                elif rev is not None and rev > 0:
                    implied_ev = rev * 1e6 * 1.0
                    note = "loss: EV/Sales≈1.0× (fallback)"
                else:
                    implied_ev = 0.0
                    note = "loss: assume 0 (conservative)"
        else:
            note = "no Operating income or multiplier - skip"

        implied_values.append(implied_ev if implied_ev is not None else 0.0)
        out_rows.append({
            "Segment": seg,
            "Revenue (m)": f"{rev:.0f}" if isinstance(rev, (int, float)) and math.isfinite(rev) else "",
            "Operating income (m)": f"{opi:.0f}" if isinstance(opi, (int, float)) and math.isfinite(opi) else "",
            "Peers": peer_key,
            "Multiple": (f"{mult:.2f}×" if mult else ""),
            "Implied EV (B$)": f"{(implied_ev/1e9):.2f}" if implied_ev is not None else "",
            "Note": note,
        })

    total_implied = sum(v for v in implied_values if v)
    premium = None
    if cur_ev and total_implied:
        premium = (total_implied / cur_ev - 1.0) * 100.0

    df_out = pd.DataFrame(out_rows, columns=[
        "Segment","Revenue (m)","Operating income (m)","Peers","Multiple","Implied EV (B$)","Note"
    ])

    md = []
    md.append("SOTP - valuation by operating segments (EV/EBIT via median EV/EBITDA of companies × 1.25):")
    md.append(df_out.to_markdown(index=False))
    md.append("")
    md.append(f"**Total by segments (B$):** {total_implied / 1e9:.2f}" if total_implied else "**Total by segments:** —")
    if isinstance(cur_ev, (int,float)) and cur_ev > 0:
        md.append(f"**Current EV (B$):** {cur_ev / 1e9:.2f}")
    if premium is not None and math.isfinite(premium):
        md.append(f"**Premium/discount to SOTP:** {premium:+.1f}%")
    else:
        md.append("_Premium/discount cannot be calculated (no EV or SOTP)._")

    return {"md": "\n".join(md), "total_implied": total_implied, "premium_pct": premium}


def estimate(rows):
    total_str = ""
    for row in rows:
        ticker = row["Ticker"]
        md = row.get("SOTP_MD", "") or row.get("SOTP", "")
        total_str += f"#{ticker}\n{md}\n\n"
    total_str += "\n\nNote: SOTP counts ONLY operating segments (ASC 280), counts only reportable segments (ASC 280). If the issuer's operating segments are defined by geography (like Apple), they are taken into account. Tables purely on \"sales by geography\" without operating profit are ignored."
    return total_str


def add_sotp(rows, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY):
    for row in rows:
        ticker = row["Ticker"]
        res = calculate_sotp_for_ticker(ticker, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
        if isinstance(res, dict):
            row["SOTP_MD"] = res.get("md", "")
            row["SOTP_TOTAL"] = res.get("total_implied")
            row["SOTP_PREMIUM_PCT"] = res.get("premium_pct")
        else:
            row["SOTP_MD"] = str(res or "")
            row["SOTP_TOTAL"] = None
            row["SOTP_PREMIUM_PCT"] = None
    return rows


