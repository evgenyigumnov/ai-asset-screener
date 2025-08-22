from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from edgar import Company, set_identity

from app.float_value import chunk_text
from app.llm_util import ask_llm
from app.yahoo import yahoo

logger = logging.getLogger(__name__)


# Единый кэш как и в других модулях
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Можно указать свою почту для edgartools (как в edgar_extractor)
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


def _to_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        pass
    return None


# ===== Правила отбора (whitelist/blacklist) и нормализация =====

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

# Явный чёрный список (контингенты/обязательства без признанной liability, оборотка, p&l и т.д.)
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

# === Явный blacklist: P&L/обязательства без признанной liability и оборотка ===
_BLACKLIST_KWS_RAW = [
    "operating lease obligation", "operating lease obligations",
    "unfunded commitment", "capital commitment", "investment commitment",
    "purchase obligation", "guarantee", "unsettled",
    "accounts payable", "accrued compensation", "taxes payable", "interest payable",
    "deferred tax", "deferred income tax", "deferred revenue",
    "impairment",
    # FV/mark-to-market/нереализованные — не EV
    "fair value uplift", "fv uplift", "unrealized", "mark to market", "mark-to-market",
    # VIE и прочие раскрытия без явной debt-like
    "vie", "variable interest entity",
    "potential clawback", "clawback potential", "exposure", "potential",
]
_BLACKLIST_KWS = [_norm_name(s) for s in _BLACKLIST_KWS_RAW]


def _classify_item(what: str, delta_musd: float) -> Tuple[str, str, str]:
    """
    Классификация строки:
    returns: ("keep"|"drop"|"review", canonical_name, reason)
    """
    # 0) мусор/страховые/контингенты/оборотка
    n = _canonicalize_name(what)
    if any(kw in n for kw in _INSURANCE_DROP_KWS):
        return "drop", what, "insurance/policy liability — exclude from EV"
    if any(kw in n for kw in _BLACKLIST_KWS) and ("tax receivable" not in n):
        return "drop", what, "blacklisted/non-debt-like/obligation-not-liability"

    # «обligation» без «liability» (кроме whitelisted категорий ниже) — исключаем
    if "obligation" in n and "liab" not in n:
        # кроме asset retirement / environmental — они специально whitelisted
        if not ("asset retirement" in n or "environmental" in n):
            return "drop", what, "obligation total (undiscounted) — double count risk"

    # 1) whitelist
    for pred, canon in _WHITELIST_RULES:
        if pred(n):
            return "keep", canon, "whitelisted debt-like"

    # 2) по умолчанию — review, если похоже на liability debt-like
    if "liab" in n or "payable" in n:
        return "review", what, "unknown liability — manual check"
    # Иначе — drop
    return "drop", what, "not debt-like for EV"

def _dedup_merge(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acc: Dict[str, float] = {}
    for it in items:
        key = _canonicalize_name(it["what"])
        acc[key] = acc.get(key, 0.0) + float(it["delta"])
    out = [{"what": k, "delta": round(v, 6)} for k, v in acc.items()]
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out

# USD-пороги для автоматического review по категориям (после каноникализации)
_CATEGORY_REVIEW_CAP_USD = {
    _norm_name("contingent consideration liability"): 5_000_000_000.0,  # $5B
    _norm_name("lease liability"): 20_000_000_000.0,                    # $20B
    _norm_name("tax receivable agreement payable"): 10_000_000_000.0,   # $10B
    _norm_name("pension deficit"): 10_000_000_000.0,                    # $10B
}

def _sanity_partition(kept, ev_before_usd):
    if not kept:
        return [], []
    # общий порог масштаба: 40% EV или $20B (раньше было $50B — слишком щедро)
    limit_usd = 20_000_000_000.0
    if isinstance(ev_before_usd, (int, float)) and math.isfinite(ev_before_usd) and ev_before_usd > 0:
        limit_usd = max(limit_usd, 0.4 * ev_before_usd)

    ok, review = [], []
    for it in kept:
        v_usd = float(it["delta"]) * 1_000_000.0
        name_norm = _norm_name(it["what"])
        too_big = abs(v_usd) > limit_usd

        # Персональные капы по категориям
        cap = _CATEGORY_REVIEW_CAP_USD.get(name_norm)
        if cap and abs(v_usd) > cap:
            too_big = True

        (review if too_big else ok).append(it)
    return ok, review

def _filter_classify(items: List[Dict[str, Any]], ev_before_usd: Optional[float]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Возвращает (kept, review, dropped) — все в млн USD, уже дедупнутые.
    """
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
    # sanity-чек масштаба
    ok, too_big = _sanity_partition(kept, ev_before_usd)

    # всё, что ушло в too_big, переносим в review
    review_all = review_raw + [{"what": it["what"], "delta": it["delta"], "reason": "sanity/scale"} for it in too_big]

    # финальные списки отсортированы по |delta|
    ok.sort(key=lambda r: abs(r["delta"]), reverse=True)
    review_all.sort(key=lambda r: abs(r["delta"]), reverse=True)
    drop_raw.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return ok, review_all, drop_raw



def _ensure_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Достаём из ответа массив JSON (между первой '[' и последней ']'), парсим,
    фильтруем на объекты вида {"what": str, "delta": number}.
    """
    if not text:
        return []
    # вырезаем ровно JSON-массив
    m = re.search(r"\[.*\]", text, flags=re.S)
    raw = m.group(0) if m else text.strip()
    try:
        data = json.loads(raw)
    except Exception:
        # попытка очистить хвосты и повторить
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
    """
    Сливаем одинаковые 'what' (строгое равенство) суммируя delta (в USD млн).
    """
    acc: Dict[str, float] = {}
    for it in items:
        w = it["what"].strip()
        d = float(it["delta"])
        acc[w] = acc.get(w, 0.0) + d
    out = [{"what": w, "delta": round(d, 6)} for w, d in acc.items()]
    # сортировка: убывание по абсолютной величине
    out.sort(key=lambda r: abs(r["delta"]), reverse=True)
    return out


def _fetch_10k_markdown(ticker: str) -> Optional[str]:
    """
    Текст последнего 10-K (без /A). Возвращает markdown.
    """
    try:
        co = Company(ticker)
        filings = co.get_filings(form="10-K")
        filing = next(
            (f for f in filings if isinstance(getattr(f, "form", None), str) and f.form.upper() == "10-K"),
            None,
        )
        if filing is None:
            # fallback: любой 10-K, игнорируем 10-K/A
            filing = next((f for f in filings if "/A" not in getattr(f, "form", "")), None)
        if filing is None:
            return None
        return filing.markdown()
    except Exception:
        return None


def _build_prompt_for_chunk(chunk: str) -> str:
    """
    Инструкция для ИИ: только JSON, единицы — USD миллионы, знаки по правилам.
    """
    return f"""
{chunk}

Вот фрагмент 10-K (markdown). Проанализируй ТОЛЬКО этот фрагмент — примечания к отчётности и раскрытия.

Требуется корректировка Enterprise Value (EV) до fair value — найди упоминания о скрытых активах/обязательствах в примечаниях:

- Операционная аренда (lease obligations)
- Пенсионные и социальные обязательства (дефицит фондов, обязательные выплаты)
- Судебные иски и условные обязательства (contingent liabilities)
- Инвестиции и совместные предприятия (JV) — FV аплифт/дисконт
- Нематериальные активы (бренды, лицензии, патенты)
- Инструменты fair value (FV measurements, уровни 1–3), где книжная цена сильно отличается от рыночной

Правила вывода:
- Ответ ТОЛЬКО в формате JSON-массива объектов без какого-либо текста снаружи.
- Каждый объект: {{"what": "...", "delta": number}}
- Знаки: delta > 0 — УВЕЛИЧИВАЕТ EV (долгоподобные вещи/скрытые обязательства).
          delta < 0 — УМЕНЬШАЕТ EV (FV-аплифт неоперационных активов и т.п.).
- Единицы: ВСЕ delta в USD миллионы (если в тексте «$2.5 billion», запиши 2500).

Примеры допустимы, но верни только фактические/оценочные позиции из этого фрагмента.

Верни сразу JSON-массив без пояснений:
[
  {{"what": "Pension deficit", "delta": 1500}},
  {{"what": "FV↑ инвестиций (JV delta)", "delta": -2500}}
]
""".strip()


def extract_ev_adjustments_json(
    ticker: str, model: str, endpoint: str, api_key,  force_refresh: bool = False
) -> List[Dict[str, Any]]:
    """
    Возвращает сводный список корректировок EV для тикера (USD млн, знаки по правилам).
    Использует кэш cache/<TICKER>.ev_fair_value.json.
    """
    key = _ticker_key(ticker)
    cache_name = f"{key}.ev_fair_value.json"

    if not force_refresh:
        cached = _read_cache_text(cache_name)
        if cached:
            try:
                data = json.loads(cached)
                if isinstance(data, list):
                    return _merge_adjustments(_ensure_json_array(json.dumps(data)))
            except Exception:
                pass

    md = _fetch_10k_markdown(ticker)
    if not md or len(md.strip()) == 0:
        # нет 10-K — пусто
        _write_cache_text(cache_name, "[]")
        return []

    chunks = chunk_text(md, max_chars=50000, overlap=1000)
    all_items: List[Dict[str, Any]] = []

    for ch in chunks:
        prompt = _build_prompt_for_chunk(ch)
        logger.info(f"Starting ask_llm with model={model} ticker={ticker}")
        ret = ask_llm(prompt, model, endpoint, api_key)
        logger.info(f"Finished ask_llm with model={model} ticker={ticker}")
        items = _ensure_json_array(ret)
        if items:
            all_items.extend(items)

    merged = _merge_adjustments(all_items)
    _write_cache_text(cache_name, json.dumps(merged, ensure_ascii=False))
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
        lines.append("EV (Yahoo): нет данных")
    else:
        lines.append(f"EV (Yahoo): {ev_before_usd/1e9:.2f} B$")

    if not items:
        lines.append("Корректировок не найдено (или отфильтрованы).")
    else:
        lines.append("Детали (USD млн; + увеличивает EV, − уменьшает EV):")
        for it in items:
            sign = "+" if it["delta"] >= 0 else ""
            lines.append(f" - {it['what']}: {sign}{it['delta']:.2f}")
        total_m = sum(it["delta"] for it in items)
        lines.append(f"Итого корректировка: {total_m:+.2f} млн USD")

    # Примечание о «на проверку»
    # if review_items:
    #     m_review = sum(it["delta"] for it in review_items)
    #     if m_review != 0:
    #         lines.append(f"(!) На проверку (не учтены): {m_review:+.2f} млн USD")

    if ev_after_usd is None:
        lines.append("Итоговый EV: —")
    else:
        lines.append(f"Итоговый EV: {ev_after_usd/1e9:.2f} B$")

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

def add_ev_fair_value(rows: List[Dict[str, Any]], model: str, endpoint: str, api_key: str, force_refresh: bool = False) -> List[Dict[str, Any]]:
    """
    Корректирует rows по месту:
      - Правит row['EV'] с учётом fair-value корректировок (delta в млн USD) ТОЛЬКО из белого списка.
      - Пересчитывает row['EV/EBITDA'], если есть EBITDA_TTM > 0.
      - Складывает детали в:
          row['EV_FV_Adjustments']          — учтённые (kept)
          row['EV_FV_Adjustments_Review']   — на проверку (review)
          row['EV_FV_Adjustments_Dropped']  — исключённые (drop)
      - Не трогает «обычный» долг — считаем, что он уже в EV.

    Возвращает те же rows (для удобной композиции).
    """
    for row in rows:
        tk = row.get("Ticker")
        if not tk:
            continue

        ev_before = row.get("EV")  # базовый EV; обычно берёшь из Yahoo/своего расчёта
        raw_items = extract_ev_adjustments_json(tk, model, endpoint, api_key, force_refresh=force_refresh)

        # Классификация/фильтрация
        kept, review, dropped = _filter_classify(raw_items, ev_before)

        # сумма в USD (дельты в миллионах)
        total_adj_usd = sum(it["delta"] for it in kept) * 1_000_000.0

        ev_after = None
        if isinstance(ev_before, (int, float)) and math.isfinite(ev_before):
            ev_after = ev_before + total_adj_usd
            row["EV"] = ev_after  # правим EV в rows

            # Пересчёт EV/EBITDA
            ebitda = row.get("EBITDA_TTM")
            if isinstance(ebitda, (int, float)) and ebitda and ebitda > 0 and math.isfinite(ebitda):
                row["EV/EBITDA"] = ev_after / ebitda

        # cохраняем детали
        row["EV_ORIG"] = ev_before
        row["EV_FV_Adjustments_RAW"] = raw_items
        row["EV_FV_Adjustments"] = kept
        row["EV_FV_Adjustments_Review"] = review
        row["EV_FV_Adjustments_Dropped"] = dropped

    return rows


if __name__ == "__main__":
    # Простой тест: python -m app.ev_fair_value
    # Настройки LLM (совпадают с константами в console.py по умолчанию)
    LLM_ENDPOINT = "http://localhost:1234/v1"
    LLM_MODEL = "openai/gpt-oss-20b"


    rows = yahoo(["TSLA"])
    rows = add_ev_fair_value(rows, LLM_MODEL, LLM_ENDPOINT)
    result = estimate(rows)
    print(result)
