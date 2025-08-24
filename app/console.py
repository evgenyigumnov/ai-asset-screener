import re
import argparse
from typing import List, Tuple, Dict
import pandas as pd

import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


from app.float_value import add_float_value
from app.float_value import estimate as estimate_float_value
from app.ev_ebitda import estimate_ev_ebitda
from app.ev_fair_value import add_ev_fair_value

from app.forward_p_e import estimate as estimate_fpe
from app.ev_fair_value import estimate as estimate_ev_fair_value
from app.p_fcf import estimate as estimate_pfcf
from app.sotp import estimate as estimate_sotp, add_sotp
from app.yahoo import yahoo
from app.llm_util import init_llm

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asset.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)



def _get_env(name: str, default=None):
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if v == "" or v.lower() in {"none", "null"}:
        return default
    return v

LLM_MODEL = _get_env("LLM_MODEL", None)
if LLM_MODEL == None:
    raise RuntimeError("Set LLM_MODEL")

LLM_ENDPOINT = _get_env("LLM_ENDPOINT", None)

LLM_OPENAI_API_KEY = _get_env("LLM_OPENAI_API_KEY", "")

if LLM_OPENAI_API_KEY == "":
    print("LLM_OPENAI_API_KEY not set - you work with local OpenAI-compatible API")
    LLM_OPENAI_API_KEY  = "fake_api_key"

if LLM_ENDPOINT is None and LLM_OPENAI_API_KEY == "":
    raise RuntimeError("Set LLM_ENDPOINT")

init_llm(LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)

CAP_PE = 200.0
CAP_PFCF = 300.0
CAP_EVEBITDA = 200.0
CAP_FLOATSHARE = 4.0

_ANS_BUY = {"BUY"}
_ANS_SELL = {"SELL"}

def _float_signal_from_rows(metric_name: str, rows: List[Dict], asset_name: str) -> Tuple[str, str]:
    df = pd.DataFrame(rows)

    df["Subsector"] = df["Ticker"].map(INSURANCE_SUBSECTOR)

    s_raw = pd.to_numeric(df.get("FloatShare"), errors="coerce")

    logger.info("Determining peer group for float signal")
    asset_sub = df.loc[df["Ticker"] == asset_name, "Subsector"]
    if not asset_sub.empty and pd.notna(asset_sub.iloc[0]):
        peer_mask = (df["Subsector"] == asset_sub.iloc[0])
        logger.info(f"Using subsector: {asset_sub.iloc[0]}")
    else:
        peer_mask = pd.Series(True, index=df.index)
        logger.info("Using full peer group (no subsector found)")

    logger.info("Filtering valid peer observations")
    mask_all = (s_raw > 0) & (s_raw < CAP_FLOATSHARE)
    peer_valid = s_raw.where(mask_all & peer_mask).dropna()
    logger.info(f"Found {len(peer_valid)} valid peer observations")

    row = df[df["Ticker"] == asset_name].head(1)
    if row.empty:
        return "UNCERTAIN", f"{asset_name}: no data."

    val = pd.to_numeric(row["FloatShare"].iloc[0], errors="coerce")
    if row.index.size == 0:
        return "UNCERTAIN", f"For {asset_name} by {metric_name} no valid value."
    if pd.isna(val) or not bool(mask_all.iloc[row.index[0]]):
        return "UNCERTAIN", f"For {asset_name} by {metric_name} no valid value."

    if len(peer_valid) < 3:
        peer_valid = s_raw.where(mask_all).dropna()
        if len(peer_valid) < 3:
            return "UNCERTAIN", f"Too few valid observations (subsector={int(len(peer_valid))})."

    q1 = peer_valid.quantile(0.25)
    med = peer_valid.quantile(0.50)
    q3 = peer_valid.quantile(0.75)

    if val > q3:
        ans = "BUY"
    elif val < q1:
        ans = "SELL"
    else:
        ans = "UNCERTAIN"

    reason = (
        f"{metric_name} {asset_name} = {val:.2f}; "
        f"Q1={q1:.2f}, Median={med:.2f}, Q3={q3:.2f}. "
        f"Rule IQR => >Q3=BUY, <Q1=SELL, else UNCERTAIN. "
        f"(peer-group: {asset_sub.iloc[0] if not asset_sub.empty else 'all group'})"
    )
    return ans, reason



def _parse_signal(text: str) -> Tuple[str, str, str]:
    ans = "НЕОПРЕДЕЛЁННО"
    reason = ""
    t = (text or "").strip()

    # reason
    m = re.search(r"<REASON>(.*?)</REASON>", t, flags=re.S | re.I)
    if m:
        reason = m.group(1).strip()

    # answer
    m = re.search(r"<ANSWER>(.*?)</ANSWER>", t, flags=re.S | re.I)
    if m:
        raw = m.group(1).strip().upper()
        if any(tok in raw for tok in _ANS_BUY):
            ans = "КУПИ"
        elif any(tok in raw for tok in _ANS_SELL):
            ans = "ПРОДАЙ"
    else:
        upper = t.upper()
        if "SELL" in upper:
            ans = "SELL"
        if "BUY" in upper and ans != "SELL":
            ans = "BUY"

    return ans, reason, t

def _metric_signal_from_rows(metric_name: str, rows: List[Dict], asset_name: str) -> Tuple[str, str]:

    logger.info(f"Calculating {metric_name} signal")
    df = pd.DataFrame(rows)
    col = {"Forward P/E": "Forward P/E", "P/FCF": "P/FCF", "EV/EBITDA": "EV/EBITDA"}[metric_name]
    s_raw = pd.to_numeric(df[col], errors="coerce")
    logger.info(f"Completed {metric_name} signal calculation")

    if metric_name == "Forward P/E":
        mask = (s_raw > 0) & (s_raw < CAP_PE)
    elif metric_name == "P/FCF":
        mask = (s_raw > 0) & (s_raw < CAP_PFCF)
    else:  # EV/EBITDA
        mask = (s_raw > 0) & (s_raw < CAP_EVEBITDA)

    s = s_raw.where(mask)
    valid = s.dropna()

    row = df[df["Ticker"] == asset_name].head(1)
    if row.empty:
        return "UNCERTAIN", f"{asset_name}: no data."
    val = pd.to_numeric(row[col].iloc[0], errors="coerce")
    if pd.isna(val) or not mask.iloc[row.index[0]]:
        return "UNCERTAIN", f"For {asset_name} by {metric_name} no valid value."

    if len(valid) < 3:
        return "UNCERTAIN", f"Too few valid observations ({len(valid)})."

    q1 = valid.quantile(0.25)
    med = valid.quantile(0.50)
    q3 = valid.quantile(0.75)

    if val < q1:
        ans = "BUY"
    elif val > q3:
        ans = "SELL"
    else:
        ans = "UNCERTAIN"

    reason = (f"{metric_name} {asset_name} = {val:.2f}; Q1={q1:.2f}, Median={med:.2f}, Q3={q3:.2f}. "
              f"Rule IQR => <Q1=BUY, >Q3=SELL, else UNCERTAIN.")
    return ans, reason

def _build_question(metric_name: str, analysis_text: str, asset_name: str) -> str:
    return (
        f"{analysis_text}\n\n"
        f"Stock under consideration: {asset_name}.\n\n"
        f"Based on THIS analysis for the {metric_name} parameter, provide the result strictly in the format "
        f"<ANSWER>BUY</ANSWER> or <ANSWER>SELL</ANSWER> or <ANSWER>UNCERTAIN</ANSWER>.\n"
        f"Provide an explanation in Russian strictly in the format "
        f"<REASON>a brief explanation of why for {asset_name} according to the {metric_name} metric</REASON>.\n"
        f"Do not add anything outside these tags."
    )



def _majority_vote(signals: List[Tuple[str, str, str]]) -> Tuple[str, int, int, int]:
    buy_c = sum(1 for _, a, _ in signals if a == "BUY")
    sell_c = sum(1 for _, a, _ in signals if a == "SELL")
    unsure_c = sum(1 for _, a, _ in signals if a == "UNCERTAIN")
    if buy_c > sell_c:
        final = "BUY"
    elif sell_c > buy_c:
        final = "SELL"
    else:
        priority = ["EV/EBITDA", "P/FCF", "Forward P/E"]
        final = "UNCERTAIN"
        for pr in priority:
            for m, a, _ in signals:
                if m == pr and a in ("BUY", "SELL"):
                    final = a
                    break
            else:
                continue
            break

    return final, buy_c, sell_c, unsure_c


CRYPTO_MINERS_EXCHANGES = [
    "MSTR",
    "MARA",
    "RIOT",
    "BSTR",
    "BLSH",
    "COIN",
    "GLXY"
]

BIG_TECH_CORE = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    "AVGO","ORCL","ADBE","CRM"
]

BIG_TECH_EXPANDED = [
    "AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA",
    "AVGO","ORCL","ADBE","CRM","CSCO","IBM","INTC","AMD",
    "QCOM","TXN","MU","ARM","ASML","ADI","AMAT","LRCX",
    "NOW","SNOW","MDB","DDOG","NET","CRWD","PANW","FTNT",
    "NFLX","TTD","SNAP","PINS","RBLX"
]

SEMI_PEERS = [
    "NVDA","AVGO","AMD","INTC","QCOM","TXN","MU","ARM","ASML","ADI","AMAT","LRCX"
]

CLOUD_SOFTWARE = [
    "MSFT","GOOGL","ORCL","IBM","NOW","CRM","ADBE","SNOW","MDB","DDOG","NET","ZS","OKTA"
]

INTERNET_ADS = [
    "GOOGL","META","NFLX","TTD","SNAP","PINS","RBLX"
]

ECOMMERCE_RETAIL = [
    "AMZN","SHOP","MELI","EBAY","ETSY","SE","WMT","COST","TGT"
]

AUTO_EV = [
    "TSLA","NIO","LI","XPEV","RIVN","LCID","GM","F","STLA"
]

CONGLOMERATES_CORE = [
    "BRK.B",
    "SFTBY",
]

CONGLOMERATES = [
    "BRK.B",
    "SFTBY",
    "IAC",
    "LBRDA",
    "MC.PA",
    "6501.T",
]


ASSET_MANAGERS_CORE = [
    "BLK",
    "BX",
    "KKR",
    "APO",
]

ASSET_MANAGERS = [
    "BLK",
    "BX",
    "KKR",
    "APO",
    "CG",
    "ARES",
    "TPG",
    "BN",
    "BAM",
    "IVZ",
    "TROW",
    "BEN",
    "STT",
    "SCHW",
    "AB",
    "JHG"
]

INSURANCE = [
    "BRK.B",
    "CB",
    "PGR",
    "TRV",
    "ALL",
    "AIG",
    "MET",
    "PRU",
    "HIG",
    "EG",
    "RNR",
]

INSURANCE_SUBSECTOR = {
    "CB": "P&C", "PGR": "P&C", "TRV": "P&C", "ALL": "P&C", "HIG": "P&C",
    "MET": "LIFE", "PRU": "LIFE", "AIG": "LIFE",
    "RNR": "REINS", "EG": "REINS",
    "BRK.B": "CONGLOM",
}


def _print_divider(title: str = "", char: str = "="):
    line = char * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}")


def _print_group_header(name: str, tickers: List[str], asset_name: str):
    _print_divider(f"GROUP: {name}")
    print(f"Tickers ({len(tickers)}): {', '.join(tickers)}")
    print(f"The stock in question: {asset_name}\n")


def _print_metric_vote(signals: List[Tuple[str, str, str]]):
    print("VOTE BY METRICS:")
    for metric, ans, rsn in signals:
        print(f"- {metric} -> Signal: {ans}")
        print(f"  Reason: {rsn}")


def _format_row(cols: List[str], widths: List[int]) -> str:
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    return " | ".join(parts)


def _print_sector_summary_table(rows: List[Dict[str, str]]):
    headers = ["Group", "BUY", "SELL","UNCERTAIN", "Group summary"]
    widths = [28, 6, 8, 16, 14]

    print()
    print(_format_row(headers, widths))
    print(_format_row(["-"*w for w in widths], widths))
    for r in rows:
        print(_format_row([
            r["group"], str(r["buy"]), str(r["sell"]), str(r["unsure"]), r["final"]
        ], widths))


def analyze_group(group_name: str, tickers: List[str], asset_name: str) -> Dict:

    if asset_name not in tickers:
        print(f"Skip group '{group_name}': {asset_name} absent in group.")
        return {"group": group_name, "signals": [], "final": "N/A", "buy": 0, "sell": 0}

    _print_group_header(group_name, tickers, asset_name)

    logger.info(f"Starting yahoo() for {len(tickers)} tickers")
    rows = yahoo(tickers)
    logger.info("Completed yahoo()")
    
    logger.info("Starting add_ev_fair_value")
    rows = add_ev_fair_value(rows)
    logger.info("Completed add_ev_fair_value")
    
    logger.info("Starting estimate_ev_fair_value")
    report_ev_fair_value = estimate_ev_fair_value(rows)
    logger.info("Completed estimate_ev_fair_value")
    
    report_float = None
    if asset_name in INSURANCE:
        logger.info("Starting add_float_value for insurance asset")
        rows = add_float_value(rows)
        logger.info("Completed add_float_value for insurance asset")

        logger.info("Starting estimate_float_value for insurance asset")
        report_float = estimate_float_value(rows)
        logger.info("Completed estimate_float_value for insurance asset")

    logger.info("Starting estimate_fpe")
    report_fpe = estimate_fpe(rows)
    logger.info("Completed estimate_fpe")
    report_pfcf = None
    report_ev = None


    if asset_name not in INSURANCE:
        logger.info("Starting estimate_pfcf")
        report_pfcf = estimate_pfcf(rows)
        logger.info("Completed estimate_pfcf")
        
        logger.info("Starting estimate_ev_ebitda")
        report_ev = estimate_ev_ebitda(rows)
        logger.info("Completed estimate_ev_ebitda")

    logger.info("Starting add_sotp")
    rows = add_sotp(rows)
    logger.info("Completed add_sotp")
    
    logger.info("Starting estimate_sotp")
    report_sotp = estimate_sotp(rows)
    logger.info("Completed estimate_sotp")

    print("ANALYSIS DETAILS (by group):")
    if asset_name in INSURANCE and report_float:
        print(report_float)

    print(report_ev_fair_value)
    print()

    print(report_fpe)
    print()
    if asset_name not in INSURANCE and report_pfcf:
        print(report_pfcf)
        print()
    if asset_name not in INSURANCE and report_ev:
        print(report_ev)
        print()

    print(report_sotp)
    print()

    a_fpe, reason_fpe = _metric_signal_from_rows("Forward P/E", rows, asset_name)
    if asset_name not in INSURANCE:
        a_pfcf, reason_pfcf = _metric_signal_from_rows("P/FCF", rows, asset_name)

    a_float, reason_float = ("UNCERTAIN", "Float does not apply to this group.")
    if asset_name in INSURANCE:
        a_float, reason_float = _float_signal_from_rows("Float/EV", rows, asset_name)

    row_me = next((r for r in rows if r.get("Ticker") == asset_name), {})
    pct = row_me.get("SOTP_PREMIUM_PCT")
    if pct is None or not pd.notna(pct):
        a_sotp, reason_sotp = "UNCERTAIN", "No SOTP numeric rating (or segment table not recognized)."
    else:
        thr = 10.0
        if pct > thr:
            a_sotp = "BUY"
        elif pct < -thr:
            a_sotp = "SELL"
        else:
            a_sotp = "UNCERTAIN"
        lbl = "discount" if pct > 0 else "premium"
        reason_sotp = f"SOTP to market: {pct:+.1f}% ({lbl} relative to EV; threshold ±{thr}%)."

    group_signals = [
        ("Forward P/E", a_fpe, reason_fpe),
    ]
    if asset_name not in INSURANCE:
        group_signals.append(("P/FCF", a_pfcf, reason_pfcf))
    if asset_name not in INSURANCE:
        a_ev, reason_ev = _metric_signal_from_rows("EV/EBITDA", rows, asset_name)
        group_signals.append(("EV/EBITDA", a_ev, reason_ev))

    if asset_name in INSURANCE:
        group_signals.append(("Float/EV", a_float, reason_float))

    group_signals.append(("SOTP", a_sotp, reason_sotp))

    print()
    _print_metric_vote(group_signals)

    group_final, buy_c, sell_c, unsure_c = _majority_vote(group_signals)

    print("\nGROUP SCORE:")
    print(f"BUY: {buy_c} | SELL: {sell_c} | UNCERTAIN: {unsure_c}")

    print("\nGROUP TOTAL:")
    print(f"Signal: {group_final}")

    return {
        "group": group_name,
        "signals": group_signals,
        "final": group_final,
        "buy": buy_c,
        "sell": sell_c,
        "unsure": unsure_c
    }


def _add_if_absent(asset_name: str, tickers: List[str]) -> List[str]:
    if asset_name not in tickers:
        tickers.append(asset_name)
    return tickers

def _group_key(name: str) -> str:
    return re.split(r"\s|\(", name, maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument(
        '--group',
        help = ("List of group keys separated by commas (for example: INSURANCE,BIG_TECH_CORE). "
                "If not specified, groups are selected automatically by ticker.")
    )
    args = parser.parse_args()

    asset_name = args.ticker.upper()

    full_group_specs = [
        ("CRYPTO_MINERS_EXCHANGES", CRYPTO_MINERS_EXCHANGES),
        ("BIG_TECH_CORE", BIG_TECH_CORE),
        ("BIG_TECH_EXPANDED", BIG_TECH_EXPANDED),
        ("SEMI_PEERS (полупроводники)", SEMI_PEERS),
        ("CLOUD_SOFTWARE (облако/enterprise/SaaS)", CLOUD_SOFTWARE),
        ("INTERNET_ADS (интернет/реклама/контент)", INTERNET_ADS),
        ("ECOMMERCE_RETAIL (e-com/ритейл)", ECOMMERCE_RETAIL),
        ("AUTO_EV (авто/EV)", AUTO_EV),
        ("ASSET_MANAGERS_CORE", ASSET_MANAGERS_CORE),
        ("ASSET_MANAGERS", ASSET_MANAGERS),
        ("CONGLOMERATES", CONGLOMERATES),
        ("CONGLOMERATES_CORE", CONGLOMERATES_CORE),
        ("INSURANCE", INSURANCE),
    ]

    key_to_spec = { _group_key(name).upper(): (name, tickers) for name, tickers in full_group_specs }
    all_keys_str = ", ".join(sorted(key_to_spec.keys()))

    if args.group:
        requested_keys = [k.strip().upper() for k in args.group.split(",") if k.strip()]
        unknown = [k for k in requested_keys if k not in key_to_spec]
        if unknown:
            _print_divider("Error: unknown group keys")
            print("Unknown:", ", ".join(unknown))
            print("Available group keys:", all_keys_str)
            return

        group_specs = [key_to_spec[k] for k in requested_keys]
    else:
        auto_specs = [(name, tickers) for name, tickers in full_group_specs if asset_name in tickers]

        if len(auto_specs) == 0:
            _print_divider("No matching groups")
            print(f"{asset_name} is not in any group. Analysis stopped.")
            return
        elif len(auto_specs) == 1:
            group_specs = auto_specs
        else:
            _print_divider("group(s) must be specified")
            human_list = ", ".join([f"{name} (key: {_group_key(name)})" for name, _ in auto_specs])
            example_all = ",".join([_group_key(name) for name, _ in auto_specs])
            example_one = _group_key(auto_specs[0][0])

            print(f"Ticker {asset_name} found in multiple groups at once:")
            print(human_list)
            print("\nSpecify the --group parameter with one or more keys separated by commas. Examples:")
            print(f"  --group={example_one}")
            print(f"  --group={example_all}")
            print("\nList of all possible group keys:", all_keys_str)
            return

    eligible_specs = [(name, tickers) for name, tickers in group_specs if asset_name in tickers]
    skipped_specs = [(name, tickers) for name, tickers in group_specs if asset_name not in tickers]

    if skipped_specs:
        _print_divider("Skipping irrelevant groups", char="-")
        for name, _ in skipped_specs:
            print(f"Skipping group '{name}': {asset_name} is not in the group.")

    if not eligible_specs:
        _print_divider("No matching groups")
        print(f"{asset_name} is not in any of the specified groups. Analysis terminated.")
        return

    all_group_results: List[Dict] = []
    all_signals_flat: List[Tuple[str, str, str]] = []

    for name, tickers in eligible_specs:
        res = analyze_group(name, tickers, asset_name)
        all_group_results.append(res)
        all_signals_flat.extend(res["signals"])
        _print_divider(char="-")

    _print_divider("SUMMARY TABLE BY GROUPS (sector account)")
    table_rows = [
        {"group": r["group"], "buy": r["buy"], "sell": r["sell"], "unsure": r["unsure"], "final": r["final"]}
        for r in all_group_results
    ]
    _print_sector_summary_table(table_rows)

    if all_signals_flat:
        final_all, buy_all, sell_all, unsure_all = _majority_vote(all_signals_flat)
        print("\nTOTAL SCORE FOR ALL RELEVANT GROUPS (by metrics):")
        print(f"BUY: {buy_all} | SELL: {sell_all} | UNCERTAIN: {unsure_all}")
        print("\nTOTAL FINAL DECISION:")
        print(f"Signal: {final_all}")
    else:
        print("\nThere are no signals for any relevant group - no total calculated.")

if __name__ == "__main__":
    main()
