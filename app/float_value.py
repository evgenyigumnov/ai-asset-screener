from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from edgar import Company, set_identity
from app.llm_util import ask_llm

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger(__name__)


try:
    set_identity("igumnovnsk@gmail.com")
except Exception:
    pass


def _ticker_key(ticker: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(ticker).upper())

def _read_cache_text(fname: str) -> Optional[str]:
    p = CACHE_DIR / fname
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None

def _write_cache_text(fname: str, s: str) -> None:
    p = CACHE_DIR / fname
    try:
        p.write_text(s, encoding="utf-8")
    except Exception:
        pass

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

    out: List[Dict[str, Any]] = []
    for it in data:
        if isinstance(it, dict) and "what" in it and "delta" in it:
            try:
                what = str(it["what"]).strip()
                delta = float(it["delta"])
                if what and math.isfinite(delta):
                    rec = {"what": what, "delta": float(delta)}
                    # опциональные поля:
                    if isinstance(it.get("source"), str):
                        rec["source"] = it["source"].strip().lower()
                    if isinstance(it.get("as_of"), (int, float)):
                        rec["as_of"] = int(it["as_of"])
                    out.append(rec)
            except Exception:
                continue
    return out



def _merge_same(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[str, float] = {}
    label: Dict[str, str] = {}
    for it in items:
        key = re.sub(r"\s+", " ", it["what"].strip().lower())
        v = float(it["delta"])
        if key not in best or abs(v) > abs(best[key]):
            best[key] = v
            label[key] = it["what"].strip()
    out = [{"what": label[k], "delta": round(v, 6)} for k, v in best.items()]
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out

def _filter_best_scope(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not items:
        return items
    years = [it.get("as_of") for it in items if isinstance(it.get("as_of"), int)]
    target_year = max(years) if years else None
    def ok(it):
        sy = (it.get("source") == "balance")
        yr = (target_year is None or it.get("as_of") == target_year)
        return sy and yr
    best = [it for it in items if ok(it)]
    if best:
        return best
    if target_year is not None:
        best = [it for it in items if it.get("as_of") == target_year]
        if best:
            return best
    return items

def chunk_text(s: str, max_chars: int = 50000, overlap: int = 1000) -> List[str]:
    s = s or ""
    if len(s) <= max_chars:
        return [s]
    chunks, i = [], 0
    while i < len(s):
        j = min(len(s), i + max_chars)
        chunks.append(s[i:j])
        if j == len(s):
            break
        i = max(0, j - overlap)
    return chunks

def fetch_10k_markdown(ticker: str) -> Optional[str]:
    try:
        co = Company(ticker)
        filings = co.get_filings(form="10-K")
        filing = next((f for f in filings if isinstance(getattr(f, "form", None), str)
                       and f.form.upper() == "10-K"), None)
        if filing is None:
            filing = next((f for f in filings if "/A" not in getattr(f, "form", "")), None)
        if filing is None:
            return None
        return filing.markdown()
    except Exception:
        return None


def _build_prompt_for_chunk(chunk: str) -> str:
    head = f"{chunk}\n\n"

    body = """
    10-K fragment (analyze ONLY this text):

    You are a financial analyst. A 10-K excerpt (markdown) is GIVEN. We need the INSURANCE FLOAT AS OF THE END OF THE MOST RECENT YEAR.

    DATA SOURCE — ONLY the balance sheet (Consolidated Balance Sheet / Statement of Financial Position) and the NOTES that explicitly disclose balances AS OF THE REPORTING DATE. Ignore limits, stress tests, payments/flows during the period, and prior years.

    Float composition (+ = liabilities, − = assets):
      + Unpaid losses & LAE, Unearned premiums, Future policy benefits, Policyholders’ account balances, Other insurance liabilities
      − Reinsurance recoverable (including on unpaid losses / on policy benefits), Insurance/Reinsurance balances receivable, Prepaid reinsurance premiums, Deferred charge assets for retroactive reinsurance

    Rules:
    - If there is NET — use NET and DO NOT add the corresponding gross/recoverable separately.
    - If there is NO NET: unpaid = gross − reinsurance recoverable (on unpaid losses); unearned = unearned − prepaid reinsurance premiums.
    - Use ONLY the latest available year. Convert everything to USD MILLIONS.

    Return ONLY a JSON array. Object format:
    {
      "what": "<string label>",
      "delta": <number in USD millions>,
      "source": "balance" | "note",
      "as_of": <year, e.g., 2024 or 2023>
    }

    EXAMPLE:
    [
      {{"what":"Unpaid losses & LAE (net)","delta": 32500,"source":"balance","as_of":2023}},
      {{"what":"Unearned premiums (net)","delta": 14200,"source":"balance","as_of":2023}},
      {{"what":"Reinsurance recoverable","delta": -9800,"source":"note","as_of":2023}}
    ]

    If nothing is found, return an empty array: []
    """.strip()
    return head + body


_DROP_RE = re.compile(
    r'(wildfire|hurricane|catastroph|payments?|incurred|modeled|max(imum)?|limit|tripa?|tria|program|'
    r'prior\s+period\s+development|favorable|unfavorable|withdrawals|'
    r'separate\s+account\s+liabilit|deferred\s+acquisition|[^a-z]dac[^a-z]|loss\s+reserve\s+discount|mrb|repurchase)',
    re.I)

_UNPAID_NET_RE   = re.compile(r'(unpaid.*loss|loss(es)?\s+and\s+loss\s+(adjustment|expenses?)|(al)?ae).*?\bnet\b', re.I)
_UNPAID_GROSS_RE = re.compile(r'(unpaid.*loss|loss(es)?\s+and\s+loss\s+(adjustment|expenses?))(?!.*\bnet\b)', re.I)
_REINS_UNPAID_RE = re.compile(r'reinsurance\s+recoverable.*(unpaid.*loss|loss(es)?\s+and\s+loss\s+(adjustment|expenses?))', re.I)

_UNEARNED_NET_RE = re.compile(r'\bunearned\s+premium(s)?\b.*\bnet\b', re.I)
_UNEARNED_GRO_RE = re.compile(r'\bunearned\s+premium(s)?\b(?!.*\bnet\b)', re.I)
_PREPAID_REINS_RE= re.compile(r'prepaid\s+reinsurance\s+premium', re.I)

_FPB_RE          = re.compile(r'(future\s+policy\s+benefit|long[-\s]?duration\s+(insurance\s+)?liabilit)', re.I)
_PH_ACC_RE       = re.compile(r'policyholders?[’\']?\s+account\s+balances?', re.I)
_OTHER_INS_L_RE  = re.compile(r'other\s+insurance\s+liabilit', re.I)

_REINS_GENERIC_RE= re.compile(r'\breinsurance\s+recoverable\b', re.I)
_INS_BAL_REC_RE  = re.compile(r'(insurance|reinsurance)\s+balances?\s+receivable|premiums?\s+receivable', re.I)
_DCH_RETRO_RE    = re.compile(r'deferred\s+charge\s+assets?\s+for\s+retroactive\s+reinsurance', re.I)


def _pick_max(items, pred):
    vals = []
    for it in items:
        name = it.get('what', '')
        matched = False
        if hasattr(pred, 'search'):
            matched = bool(pred.search(name))
        elif callable(pred):
            matched = bool(pred(name))
        if matched:
            try:
                vals.append(float(it['delta']))
            except Exception:
                pass
    if not vals:
        return 0.0
    return max(vals, key=lambda x: abs(x))

def _cap_core(name: str, val_musd: float, ev_usd: Optional[float]) -> float:
    if not (isinstance(ev_usd, (int, float)) and ev_usd > 0):
        return val_musd
    limit_musd = 1.2 * ev_usd / 1_000_000.0
    return val_musd if abs(val_musd) <= limit_musd else 0.0

def _postprocess_float_items(items_raw: List[Dict[str, Any]], ev_usd: Optional[float]) -> Tuple[List[Dict[str, Any]], float]:
    clean = [it for it in items_raw if not _DROP_RE.search(it['what'])]

    unpaid_net = _pick_max(clean, _UNPAID_NET_RE)
    unpaid_gross = _pick_max(clean, _UNPAID_GROSS_RE) if unpaid_net == 0 else 0.0
    reins_unpaid = _pick_max(clean, _REINS_UNPAID_RE) if unpaid_net == 0 else 0.0

    unearned_net = _pick_max(clean, _UNEARNED_NET_RE)
    unearned_gro = _pick_max(clean, _UNEARNED_GRO_RE) if unearned_net == 0 else 0.0
    prepaid_rein = _pick_max(clean, _PREPAID_REINS_RE) if unearned_net == 0 else 0.0

    fpb = _pick_max(clean, _FPB_RE)
    ph_acc = _pick_max(clean, _PH_ACC_RE)
    other_ins_l = _pick_max(clean, _OTHER_INS_L_RE)

    reins_gen = _pick_max(clean, _REINS_GENERIC_RE)
    ins_bal_rec = _pick_max(clean, _INS_BAL_REC_RE)
    dch_retro = _pick_max(clean, _DCH_RETRO_RE)

    unpaid   = unpaid_net if unpaid_net > 0 else max(0.0, unpaid_gross - reins_unpaid)
    unearned = unearned_net if unearned_net > 0 else max(0.0, unearned_gro - prepaid_rein)

    comps = []

    unpaid = _cap_core("Unpaid", unpaid, ev_usd)
    unearned = _cap_core("Unearned", unearned, ev_usd)
    fpb = _cap_core("FPB", fpb, ev_usd)
    ph_acc = _cap_core("PH_ACC", ph_acc, ev_usd)

    if unpaid   > 0: comps.append({"what":"Unpaid losses & LAE (net)","delta":+unpaid})
    if unearned > 0: comps.append({"what":"Unearned premiums (net)","delta":+unearned})
    if fpb      > 0: comps.append({"what":"Future policy benefits","delta":+fpb})
    if ph_acc   > 0: comps.append({"what":"Policyholders’ account balances","delta":+ph_acc})
    if other_ins_l > 0:
        def _same(a, b):
            return (a > 0 and b > 0 and abs(a - b) / max(a, b) <= 0.01)

        if not (_same(other_ins_l, unpaid) or _same(other_ins_l, unearned)):
            comps.append({"what": "Other insurance liabilities", "delta": +other_ins_l})

    use_reins_generic = (unpaid_net == 0 or unearned_net == 0) or (fpb > 0 or ph_acc > 0)
    if use_reins_generic and reins_gen > 0:
        comps.append({"what":"Reinsurance recoverable","delta":-reins_gen})
    if ins_bal_rec > 0:
        comps.append({"what":"Insurance/reinsurance balances receivable","delta":-ins_bal_rec})
    if dch_retro > 0:
        comps.append({"what":"Deferred charge assets for retroactive reinsurance","delta":-dch_retro})

    CORE_KEYS = (
        "Unpaid losses & LAE (net)",
        "Unearned premiums (net)",
        "Future policy benefits",
        "Policyholders’ account balances",
        "Insurance float (per 10-K)",
    )

    if isinstance(ev_usd, (int, float)) and ev_usd > 0:
        cap_usd = max(2.0 * ev_usd, 250_000_000_000.0)  # 2× EV или $250B
        kept = []
        for c in comps:
            if any(c["what"].startswith(k) for k in CORE_KEYS):
                kept.append(c)
            else:
                if abs(c["delta"] * 1_000_000.0) <= cap_usd:
                    kept.append(c)
        comps = kept

    total_musd = sum(c["delta"] for c in comps)
    comps.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return comps, total_musd


_FLOAT_DIRECT_RE = re.compile(r'\binsurance\s+float\b.*?(\d+(?:\.\d+)?)\s*(billion|million)', re.I)

def _try_direct_float(items_raw: List[Dict[str, Any]]) -> Optional[float]:
    cand = [it for it in items_raw if 'float' in it['what'].lower()]
    if not cand:
        return None
    best = max(cand, key=lambda it: abs(float(it['delta'])))
    v = float(best['delta'])
    return v if 50_000 <= abs(v) <= 300_000 else None


def extract_float_components_json(ticker: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
    key = _ticker_key(ticker)
    cache_name = f"{key}.float.json"

    if not force_refresh:
        cached = _read_cache_text(cache_name)
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    return data
            except Exception:
                pass

    md = fetch_10k_markdown(ticker)
    if not md or not md.strip():
        _write_cache_text(cache_name, "[]")
        return []

    chunks = chunk_text(md, max_chars=50000, overlap=1000)
    all_items: List[Dict[str, Any]] = []

    for ch in chunks:
        prompt = _build_prompt_for_chunk(ch)
        logger.info("Starting ask_llm   ticker=%s",  ticker)
        ret = ask_llm(prompt)
        logger.info("Finished ask_llm   ticker=%s",  ticker)
        items = _ensure_json_array(ret)
        if items:
            all_items.extend(items)

    merged = _merge_same(all_items)
    merged = _filter_best_scope(merged)

    _write_cache_text(cache_name, json.dumps(merged, ensure_ascii=False))
    return merged


def add_float_value(rows: List[Dict[str, Any]], force_refresh: bool = False) -> List[Dict[str, Any]]:
    for row in rows:
        tk = row.get("Ticker")
        if not tk:
            continue

        ev = row.get("EV")
        items_raw = extract_float_components_json(tk, force_refresh=force_refresh) or []

        direct_musd = _try_direct_float(items_raw)
        comps_musd = []
        total_musd = None
        if direct_musd is not None:
            comps_musd = [{"what": "Insurance float (per 10-K)", "delta": direct_musd}]
            total_musd = direct_musd
        else:
            comps_musd, total_musd = _postprocess_float_items(items_raw, ev)

        float_usd = (total_musd * 1_000_000.0) if (total_musd is not None) else None
        row["FLOAT_ITEMS"] = comps_musd
        row["FLOAT_USD"] = float_usd

        if isinstance(ev, (int, float)) and ev > 0 and float_usd:
            row["FloatShare"] = float_usd / ev
        else:
            row["FloatShare"] = None

    return rows



def _fmt_b(x: Optional[float]) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "—"
    return f"{x/1e9:.2f} B$"

def _fmt_pct(x: Optional[float]) -> str:
    try:
        if x is None or not math.isfinite(x):
            return "—"
        return f"{x*100:.1f}%"
    except Exception:
        return "—"

def estimate(rows: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for row in rows:
        tk = row.get("Ticker")
        ev = row.get("EV")
        fv = row.get("FLOAT_USD")
        share = row.get("FloatShare")
        items = row.get("FLOAT_ITEMS") or []

        lines.append(f"{tk} — Insurance Float")
        lines.append(f"EV: {_fmt_b(ev)}")
        lines.append(f"Float: {_fmt_b(fv)}  (Float/EV: {_fmt_pct(share)})")

        if not items:
            lines.append("Components: no data (or not found in 10-K).")
        else:
            lines.append("Components (USD million; + adds to float, - decreases):")
            for it in items:
                sign = "+" if it['delta'] >= 0 else ""
                lines.append(f" - {it['what']}: {sign}{it['delta']:.2f}")
        lines.append("")
    return "\n".join(lines)


