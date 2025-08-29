from __future__ import annotations

import json
import logging
import math
import re

from typing import Any, Dict, List, Optional, Tuple

from edgar import Company, set_identity

from app.float_value import chunk_text
from app.llm_util import ask_llm
from app import cache

logger = logging.getLogger(__name__)


try:
    set_identity("igumnovnsk@gmail.com")
except Exception:
    pass


def _ticker_key(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(ticker).upper())




def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


_WHITELIST_RULES = [
    (lambda n: "lease" in n and "liab" in n and "operating lease obligation" not in n, "lease liability"),
    (lambda n: ("pension" in n or "postretirement" in n or "opeb" in n) and ("deficit" in n or "obligation" in n or "liab" in n), "pension deficit"),
    (lambda n: ("tax receivable" in n or " tra " in (" " + n + " ")) and ("obligation" in n or "liab" in n or "payable" in n), "tax receivable agreement payable"),
    (lambda n: "contingent consideration" in n and ("liab" in n or "payable" in n), "contingent consideration liability"),
    (lambda n: ("asset retirement" in n) and ("obligation" in n or "liab" in n), "asset retirement obligation"),
    (lambda n: ("environmental" in n) and ("obligation" in n or "liab" in n), "environmental liability"),
]

# Страховые/полисные и пр. — в EV НЕ добавляем
_INSURANCE_DROP_KWS = [
    "interest-sensitive contract", "future policy benefits", "fabn", "fhlb",
    "funding agreement", "market risk benefit", "universal life", "annuity",
    "policyholder", "insurance", "global atlantic", "athene",
]

_BLACKLIST_KWS = [
    "operating lease obligations", "unfunded commitment", "capital commitment",
    "investment commitment", "purchase obligation", "guarantee", "unsettled",
    "accounts payable", "accrued compensation", "taxes payable", "interest payable",
    "deferred tax", "deferred income tax", "deferred revenue", "impairment",
    "fair value uplift", "fv↑", "vie", "variable interest entity",
    "potential clawback", "clawback potential", "exposure", "potential",
]

def _norm_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\s\-–—_/]+", " ", s)
    s = re.sub(r"[^\w\s\.]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _canonicalize_name(raw: str) -> str:
    n = _norm_name(raw)
    # унификация часто встречающихся вариантов
    n = n.replace("liabilities", "liability").replace("obligations", "obligation")
    n = n.replace("agreements", "agreement")
    n = n.replace("tax receivable agreement", "tax receivable")
    return n


_INSURANCE_DROP_KWS_RAW = [
    "interest sensitive contract",  # без дефиса!
    "future policy benefit", "future policy benefits",
    "funding agreement", "fabn", "fhlb",
    "market risk benefit", "market risk benefits",
    "universal life", "annuity",
    "policyholder", "insurance",
    "global atlantic", "athene",
]
_INSURANCE_DROP_KWS = [_norm_name(s) for s in _INSURANCE_DROP_KWS_RAW]

_BLACKLIST_KWS_RAW = [
    "operating lease obligation", "operating lease obligations",
    "unfunded commitment", "capital commitment", "investment commitment",
    "purchase obligation", "guarantee", "unsettled",
    "accounts payable", "accrued compensation", "taxes payable", "interest payable",
    "deferred tax", "deferred income tax", "deferred revenue",
    "impairment",
    "fair value uplift", "fv uplift", "unrealized", "mark to market", "mark-to-market",
    "vie", "variable interest entity",
    "potential clawback", "clawback potential", "exposure", "potential",
]
_BLACKLIST_KWS = [_norm_name(s) for s in _BLACKLIST_KWS_RAW]


def _classify_item(what: str, delta_musd: float) -> Tuple[str, str, str]:
    n = _canonicalize_name(what)
    if any(kw in n for kw in _INSURANCE_DROP_KWS):
        return "drop", what, "insurance/policy liability — exclude from EV"
    if any(kw in n for kw in _BLACKLIST_KWS) and ("tax receivable" not in n):
        return "drop", what, "blacklisted/non-debt-like/obligation-not-liability"

    if "obligation" in n and "liab" not in n:
        if not ("asset retirement" in n or "environmental" in n):
            return "drop", what, "obligation total (undiscounted) — double count risk"

    for pred, canon in _WHITELIST_RULES:
        if pred(n):
            return "keep", canon, "whitelisted debt-like"

    if "liab" in n or "payable" in n:
        return "review", what, "unknown liability — manual check"
    return "drop", what, "not debt-like for EV"

def _dedup_merge(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    for it in items:
        key = _canonicalize_name(it["what"])
        acc[key] = acc.get(key, 0.0) + float(it["delta"])
    out = [{"what": k, "delta": round(v, 6)} for k, v in acc.items()]
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out

_CATEGORY_REVIEW_CAP_USD = {
    _norm_name("contingent consideration liability"): 5_000_000_000.0,  # $5B
    _norm_name("lease liability"): 20_000_000_000.0,                    # $20B
    _norm_name("tax receivable agreement payable"): 10_000_000_000.0,   # $10B
    _norm_name("pension deficit"): 10_000_000_000.0,                    # $10B
}

def _sanity_partition(kept, ev_before_usd):
    if not kept:
        return [], []
    limit_usd = 20_000_000_000.0
    if isinstance(ev_before_usd, (int, float)) and math.isfinite(ev_before_usd) and ev_before_usd > 0:
        limit_usd = max(limit_usd, 0.4 * ev_before_usd)

    ok, review = [], []
    for it in kept:
        v_usd = float(it["delta"]) * 1_000_000.0
        name_norm = _norm_name(it["what"])
        too_big = abs(v_usd) > limit_usd

        cap = _CATEGORY_REVIEW_CAP_USD.get(name_norm)
        if cap and abs(v_usd) > cap:
            too_big = True

        (review if too_big else ok).append(it)
    return ok, review

def _filter_classify(items: List[Dict[str, Any]], ev_before_usd: Optional[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    keep_raw, review_raw, drop_raw = [], [], []
    for it in items:
        cls, canon, reason = _classify_item(it["what"], float(it["delta"]))
        rec = {"what": canon, "delta": float(it["delta"]), "reason": reason}
        if cls == "keep":
            keep_raw.append(rec)
        elif cls == "review":
            review_raw.append(rec)
        else:
            drop_raw.append(rec)

    kept = _dedup_merge(keep_raw)
    for it in kept:
        nm = _norm_name(it["what"])
        if "lease liability" in nm and it["delta"] < 0:
            it["delta"] = abs(it["delta"])
        if "pension deficit" in nm and it["delta"] < 0:
            it["delta"] = abs(it["delta"])
    ok, too_big = _sanity_partition(kept, ev_before_usd)

    review_all = review_raw + [{"what": it["what"], "delta": it["delta"], "reason": "sanity/scale"} for it in too_big]

    ok.sort(key=lambda r: abs(r["delta"]), reverse=True)
    review_all.sort(key=lambda r: abs(r["delta"]), reverse=True)
    drop_raw.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return ok, review_all, drop_raw



def _ensure_json_array(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    m = re.search(r"\[.*\]", text, flags=re.S)
    raw = m.group(0) if m else text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        raw = re.sub(r"^[^\[]+", "", text, flags=re.S)
        raw = re.sub(r"[^\]]+$", "", raw, flags=re.S)
        try:
            data = json.loads(raw)
        except Exception:
            return []

    if not isinstance(data, list):
        return []

    cleaned: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        what = str(item.get("what", "")).strip()
        delta = _to_float(item.get("delta"))
        if what and (delta is not None):
            cleaned.append({"what": what, "delta": float(delta)})
    return cleaned


def _merge_adjustments(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    for it in items:
        w = it["what"].strip()
        d = float(it["delta"])
        acc[w] = acc.get(w, 0.0) + d
    out = [{"what": w, "delta": round(d, 6)} for w, d in acc.items()]
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out


def _fetch_10k_markdown(ticker: str) -> Optional[str]:
    try:
        co = Company(ticker)
        filings = co.get_filings(form="10-K")
        filing = next(
            (f for f in filings if isinstance(getattr(f, "form", None), str) and f.form.upper() == "10-K"),
            None,
        )
        if filing is None:
            filing = next((f for f in filings if "/A" not in getattr(f, "form", "")), None)
        if filing is None:
            return None
        return filing.markdown()
    except Exception:
        return None


def _build_prompt_for_chunk(chunk: str) -> str:
    return f"""
    {chunk}
    
    Here is a 10-K fragment (markdown). Analyze ONLY this fragment—notes to the financial statements and disclosures.
    
    We need to adjust Enterprise Value (EV) to fair value—find mentions of hidden assets/liabilities in the notes:
    
    - Operating leases (lease obligations)
    - Pension and social obligations (plan deficits, mandatory payments)
    - Litigation and contingencies (contingent liabilities)
    - Investments and joint ventures (JV)—FV uplift/discount
    - Intangible assets (brands, licenses, patents)
    - Fair value instruments (FV measurements, Levels 1–3) where carrying value differs materially from market value
    
    Output rules:
    - Answer ONLY as a JSON array of objects with no text outside.
    - Each object: {{"what": "...", "delta": number}}
    - Signs: delta > 0 — INCREASES EV (debt-like items/hidden obligations).
             delta < 0 — DECREASES EV (FV uplift of non-operating assets, etc.).
    - Units: ALL delta in USD millions (if the text says "$2.5 billion", write 2500).
    
    Examples are allowed, but return only factual/estimated items from this fragment.
    
    Return the JSON array immediately with no explanations:
    [
      {{"what": "Pension deficit", "delta": 1500}},
      {{"what": "FV uplift of investments (JV delta)", "delta": -2500}}
    ]
    
    If nothing is found, return an empty array: []
    """.strip()


def extract_ev_adjustments_json(
    ticker: str,  force_refresh: bool = False
) -> List[Dict[str, Any]]:
    key = _ticker_key(ticker)
    cache_name = f"ev_fair_value/{key}.ev_fair_value.json"  # НОВОЕ

    if not force_refresh:
        cached = cache.read_text(cache_name)
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    return _merge_adjustments(_ensure_json_array(json.dumps(data)))
            except Exception:
                pass

    md = _fetch_10k_markdown(ticker)
    if not md or len(md.strip()) == 0:
        cache.write_text(cache_name, "[]")
        return []

    chunks = chunk_text(md, max_chars=50000, overlap=1000)
    all_items: List[Dict[str, Any]] = []

    for ch in chunks:
        prompt = _build_prompt_for_chunk(ch)
        logger.info(f"Starting ask_llm  ticker={ticker}")
        ret = ask_llm(prompt)
        logger.info(f"Finished ask_llm ticker={ticker}")
        items = _ensure_json_array(ret)
        if items:
            all_items.extend(items)

    merged = _merge_adjustments(all_items)
    cache.write_text(cache_name, json.dumps(merged, ensure_ascii=False))
    return merged


def _print_adjustments_stdout(
    ticker: str,
    ev_before_usd: Optional[float],
    items: List[Dict[str, Any]],
    ev_after_usd: Optional[float],
    review_items: Optional[List[Dict[str, Any]]] = None
) -> str:
    lines: List[str] = []
    lines.append(f"\n{ticker} - EV Fair Value Adjustments")

    if ev_before_usd is None:
        lines.append("EV (Yahoo): no data")
    else:
        lines.append(f"EV (Yahoo): {ev_before_usd/1e9:.2f} B$")

    if not items:
        lines.append("No adjustments found (or filtered out).")
    else:
        lines.append("Details (USD million; + increases EV, - decreases EV):")
        for it in items:
            sign = "+" if it["delta"] >= 0 else ""
            lines.append(f" - {it['what']}: {sign}{it['delta']:.2f}")
        total_m = sum(it["delta"] for it in items)
        lines.append(f"Total adjustment: {total_m:+.2f} million USD")
    if review_items:
        m_review = sum(it["delta"] for it in review_items)
        if m_review != 0:
            lines.append(f"(!) For review (not taken into account): {m_review:+.2f} million USD")

    if ev_after_usd is None:
        lines.append("Final EV: —")
    else:
        lines.append(f"Final EV: {ev_after_usd / 1e9:.2f} B$")
    return "\n".join(lines)

def estimate(rows):
    result = ""
    for row in rows:
        ev_after = row["EV"]
        ev_before = row["EV_ORIG"]
        items = row["EV_FV_Adjustments"]
        review = row.get("EV_FV_Adjustments_Review") or []
        tk = row.get("Ticker")
        result += _print_adjustments_stdout(tk, ev_before, items, ev_after, review_items=review)
        result += "\n"
    return result

def add_ev_fair_value(rows: List[Dict[str, Any]], force_refresh: bool = False) -> List[Dict[str, Any]]:

    for row in rows:
        tk = row.get("Ticker")
        if not tk:
            continue

        ev_before = row.get("EV")
        raw_items = extract_ev_adjustments_json(tk, force_refresh=force_refresh)

        kept, review, dropped = _filter_classify(raw_items, ev_before)

        total_adj_usd = sum(it["delta"] for it in kept) * 1_000_000.0

        if isinstance(ev_before, (int, float)) and math.isfinite(ev_before):
            ev_after = ev_before + total_adj_usd
            row["EV"] = ev_after

            ebitda = row.get("EBITDA_TTM")
            if isinstance(ebitda, (int, float)) and ebitda and ebitda > 0 and math.isfinite(ebitda):
                row["EV/EBITDA"] = ev_after / ebitda

        row["EV_ORIG"] = ev_before
        row["EV_FV_Adjustments_RAW"] = raw_items
        row["EV_FV_Adjustments"] = kept
        row["EV_FV_Adjustments_Review"] = review
        row["EV_FV_Adjustments_Dropped"] = dropped

    return rows

