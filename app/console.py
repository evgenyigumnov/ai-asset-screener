import re
import argparse
from typing import List, Tuple, Dict
import pandas as pd

import os
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()  # подхватит переменные из .env, если файл есть
except Exception:
    pass  # если пакета нет — просто берём из окружения


from app.float_value import add_float_value
from app.float_value import estimate as estimate_float_value
from app.ev_ebitda import estimate_ev_ebitda
from app.ev_fair_value import add_ev_fair_value

from app.forward_p_e import estimate as estimate_fpe
from app.ev_fair_value import estimate as estimate_ev_fair_value
from app.p_fcf import estimate as estimate_pfcf
from app.sotp import estimate as estimate_sotp, add_sotp
from app.yahoo import yahoo

import logging

# ===== НАСТРОЙКА ЛОГИРОВАНИЯ =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('asset.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# ===== ЗАГРУЗКА КОНФИГА ИЗ .env/ОКРУЖЕНИЯ =====

def _get_env(name: str, default=None):
    """Берёт str из окружения, пустые/None/'null'/'none' трактует как default."""
    v = os.getenv(name)
    if v is None:
        return default
    v = v.strip()
    if v == "" or v.lower() in {"none", "null"}:
        return default
    return v

# Модель: по умолчанию как раньше
LLM_MODEL = _get_env("LLM_MODEL", None)
if LLM_MODEL == None:
    raise RuntimeError("Задайте LLM_MODEL")

# Endpoint: если не задан — строго None
LLM_ENDPOINT = _get_env("LLM_ENDPOINT", None)

# Ключ OpenAI: если не задан — пустая строка и сообщение в stdout
LLM_OPENAI_API_KEY = _get_env("LLM_OPENAI_API_KEY", "")

if LLM_OPENAI_API_KEY == "":
    print("LLM_OPENAI_API_KEY не задан — считаем, что вы работаете с локальным OpenAI-совместимым API.")
    LLM_OPENAI_API_KEY  = "fake_api_key"

# Жёсткая проверка endpoint'а
if LLM_ENDPOINT is None and LLM_OPENAI_API_KEY == "":
    raise RuntimeError("Задайте LLM_ENDPOINT")

# Пороговые фильтры против «мусора»/выбросов (простые, но практичные)
CAP_PE = 200.0       # Forward P/E > 200 — выкидываем
CAP_PFCF = 300.0     # P/FCF > 300 — выкидываем
CAP_EVEBITDA = 200.0 # EV/EBITDA > 200 — выкидываем
CAP_FLOATSHARE = 4.0   # Float/EV > 4 (400%) — выкидываем как выброс/шум

# --- словари для робастного парсинга ответа ---
_ANS_BUY = {"КУПИ", "ПОКУПАТЬ", "BUY", "ПОКУПКА"}
_ANS_SELL = {"ПРОДАЙ", "ПРОДАВАТЬ", "SELL", "ПРОДАЖА"}

def _float_signal_from_rows(metric_name: str, rows: List[Dict], asset_name: str) -> Tuple[str, str]:
    df = pd.DataFrame(rows)

    # добавим подсектор (только для страховщиков; остальным None)
    df["Subsector"] = df["Ticker"].map(INSURANCE_SUBSECTOR)

    # столбец с долей флоута
    s_raw = pd.to_numeric(df.get("FloatShare"), errors="coerce")

    # peers: тот же подсектор, если есть; иначе вся группа
    logger.info("Determining peer group for float signal")
    asset_sub = df.loc[df["Ticker"] == asset_name, "Subsector"]
    if not asset_sub.empty and pd.notna(asset_sub.iloc[0]):
        peer_mask = (df["Subsector"] == asset_sub.iloc[0])
        logger.info(f"Using subsector: {asset_sub.iloc[0]}")
    else:
        peer_mask = pd.Series(True, index=df.index)
        logger.info("Using full peer group (no subsector found)")

    # фильтр валидных наблюдений среди пиров
    logger.info("Filtering valid peer observations")
    mask_all = (s_raw > 0) & (s_raw < CAP_FLOATSHARE)
    peer_valid = s_raw.where(mask_all & peer_mask).dropna()
    logger.info(f"Found {len(peer_valid)} valid peer observations")

    # значение по искомой бумаге
    row = df[df["Ticker"] == asset_name].head(1)
    if row.empty:
        return "НЕОПРЕДЕЛЁННО", f"{asset_name}: нет строки с данными."

    val = pd.to_numeric(row["FloatShare"].iloc[0], errors="coerce")
    if row.index.size == 0:
        return "НЕОПРЕДЕЛЁННО", f"Для {asset_name} по {metric_name} нет валидного значения."
    if pd.isna(val) or not bool(mask_all.iloc[row.index[0]]):
        return "НЕОПРЕДЕЛЁННО", f"Для {asset_name} по {metric_name} нет валидного значения."

    if len(peer_valid) < 3:
        # fallback: вся группа без подсектора
        peer_valid = s_raw.where(mask_all).dropna()
        if len(peer_valid) < 3:
            return "НЕОПРЕДЕЛЁННО", f"Слишком мало валидных наблюдений (подсектор={int(len(peer_valid))})."

    q1 = peer_valid.quantile(0.25)
    med = peer_valid.quantile(0.50)
    q3 = peer_valid.quantile(0.75)

    if val > q3:
        ans = "КУПИ"
    elif val < q1:
        ans = "ПРОДАЙ"
    else:
        ans = "НЕОПРЕДЕЛЁННО"

    reason = (
        f"{metric_name} {asset_name} = {val:.2f}; "
        f"Q1={q1:.2f}, Median={med:.2f}, Q3={q3:.2f}. "
        f"Правило IQR ⇒ >Q3=КУПИ, <Q1=ПРОДАЙ, иначе нейтрально. "
        f"(peer-группа: {asset_sub.iloc[0] if not asset_sub.empty else 'вся группа'})"
    )
    return ans, reason



def _parse_signal(text: str) -> Tuple[str, str, str]:
    """
    Возвращает кортеж: (answer, reason, raw)
      answer ∈ {"КУПИ", "ПРОДАЙ", "НЕОПРЕДЕЛЁННО"}
      reason — строка (может быть пустой)
      raw — исходный текст (на всякий)
    """
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
        # хедж, если модель забыла про теги
        upper = t.upper()
        if "ПРОДА" in upper or "SELL" in upper:
            ans = "ПРОДАЙ"
        if ("КУП" in upper or "BUY" in upper) and ans != "ПРОДАЙ":
            ans = "КУПИ"

    return ans, reason, t

def _metric_signal_from_rows(metric_name: str, rows: List[Dict], asset_name: str) -> Tuple[str, str]:
    """
    Делаем сигнал без LLM по правилу IQR:
      Buy:   value < Q1
      Sell:  value > Q3
      Else:  НЕОПРЕДЕЛЁННО
    С явными фильтрами по «мусорным» значениям.
    """
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

    # значение по искомой бумаге
    row = df[df["Ticker"] == asset_name].head(1)
    if row.empty:
        return "НЕОПРЕДЕЛЁННО", f"{asset_name}: нет строки с данными."
    val = pd.to_numeric(row[col].iloc[0], errors="coerce")
    # свой же фильтр применяем и к бумаге
    if pd.isna(val) or not mask.iloc[row.index[0]]:
        return "НЕОПРЕДЕЛЁННО", f"Для {asset_name} по {metric_name} нет валидного значения."

    if len(valid) < 3:
        return "НЕОПРЕДЕЛЁННО", f"Слишком мало валидных наблюдений ({len(valid)})."

    q1 = valid.quantile(0.25)
    med = valid.quantile(0.50)
    q3 = valid.quantile(0.75)

    if val < q1:
        ans = "КУПИ"
    elif val > q3:
        ans = "ПРОДАЙ"
    else:
        ans = "НЕОПРЕДЕЛЁННО"

    reason = (f"{metric_name} {asset_name} = {val:.2f}; Q1={q1:.2f}, Median={med:.2f}, Q3={q3:.2f}. "
              f"Правило IQR ⇒ <Q1=КУПИ, >Q3=ПРОДАЙ, иначе нейтрально.")
    return ans, reason

def _build_question(metric_name: str, analysis_text: str, asset_name: str) -> str:
    return (
        f"{analysis_text}\n\n"
        f"Рассматриваемая акция: {asset_name}.\n\n"
        f"На основе ЭТОГО анализа по параметру {metric_name} дай итог строго в формате "
        f"<ANSWER>КУПИ</ANSWER> или <ANSWER>ПРОДАЙ</ANSWER> или <ANSWER>НЕОПРЕДЕЛЁННО</ANSWER>.\n"
        f"Дай объяснение на русском строго в формате "
        f"<REASON>краткое объяснение почему для {asset_name} по метрике {metric_name}</REASON>.\n"
        f"Не добавляй ничего вне этих тегов."
    )


def _majority_vote(signals: List[Tuple[str, str, str]]) -> Tuple[str, int, int, int]:
    """
    signals: список кортежей (metric, answer, reason)
    Возвращает финальное решение "КУПИ"/"ПРОДАЙ" + подробности голосования.
    Ничья: тай-брейкер по приоритету метрик: EV/EBITDA > P/FCF > Forward P/E.
    """
    buy_c = sum(1 for _, a, _ in signals if a == "КУПИ")
    sell_c = sum(1 for _, a, _ in signals if a == "ПРОДАЙ")
    unsure_c = sum(1 for _, a, _ in signals if a == "НЕОПРЕДЕЛЁННО")
    if buy_c > sell_c:
        final = "КУПИ"
    elif sell_c > buy_c:
        final = "ПРОДАЙ"
    else:
        priority = ["EV/EBITDA", "P/FCF", "Forward P/E"]
        final = "НЕОПРЕДЕЛЁННО"  # дефолт
        for pr in priority:
            for m, a, _ in signals:
                if m == pr and a in ("КУПИ", "ПРОДАЙ"):
                    final = a
                    break
            else:
                continue
            break

    return final, buy_c, sell_c, unsure_c


CRYPTO_MINERS_EXCHANGES = [
    # Майнеры и компании с большими BTC-резервами
    "MSTR",  # MicroStrategy — крупнейший держатель BTC
    "MARA",  # Marathon Digital — майнинг + BTC
    "RIOT",  # Riot Platforms — майнинг + BTC
    "BSTR",  # Bitcoin Standard Treasury — BTC
    # Биржи
    "BLSH",  # Bullish — криптобиржа (IPO 2025)
    "COIN",  # Coinbase — крупнейшая биржа в США
    "GLXY"   # Galaxy Digital — брокеридж, биржевые услуги, трейдинг
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
    "BRK.B",  # Berkshire Hathaway — конгломерат Баффета
    "SFTBY",  # SoftBank Group — конгломерат, венчурные инвестиции
]

CONGLOMERATES = [
    "BRK.B",   # Berkshire Hathaway
    "SFTBY",   # SoftBank Group
    "IAC",     # IAC Inc.
    "LBRDA",   # Liberty Broadband
    "MC.PA",   # LVMH (европейский конгломерат)
    "6501.T",  # Hitachi (Япония)
]


ASSET_MANAGERS_CORE = [
    "BLK",   # BlackRock — крупнейший в мире управляющий активами
    "BX",    # Blackstone — private equity, asset management
    "KKR",   # KKR — альтернативные инвестиции
    "APO",   # Apollo Global Management
]

ASSET_MANAGERS = [
    "BLK",   # BlackRock
    "BX",    # Blackstone
    "KKR",   # KKR
    "APO",   # Apollo
    "CG",    # Carlyle Group
    "ARES",  # Ares Management
    "TPG",   # TPG Inc.
    "BN",    # Brookfield Corporation
    "BAM",   # Brookfield Asset Management
    "IVZ",   # Invesco
    "TROW",  # T. Rowe Price
    "BEN",   # Franklin Templeton
    "STT",   # State Street
    "SCHW",  # Charles Schwab
    "AB",    # AllianceBernstein
    "JHG"    # Janus Henderson
]

INSURANCE = [
    "BRK.B",  # Berkshire Hathaway (конгломерат, страховой сегмент: GEICO, Gen Re и др.)
    "CB",     # Chubb Limited (P&C)
    "PGR",    # Progressive (P&C авто)
    "TRV",    # Travelers (P&C)
    "ALL",    # Allstate (P&C)
    "AIG",    # AIG (mixed, life-heavy)
    "MET",    # MetLife (Life & annuities)
    "PRU",    # Prudential Financial (Life & retirement)
    "HIG",    # The Hartford (P&C + группы)
    "EG",     # Everest Group (перестрахование; ранее тикер RE)
    "RNR",    # RenaissanceRe (перестрахование)
]

# подсектор для корректных квартилей в страховом флоуте
INSURANCE_SUBSECTOR = {
    "CB": "P&C", "PGR": "P&C", "TRV": "P&C", "ALL": "P&C", "HIG": "P&C",
    "MET": "LIFE", "PRU": "LIFE", "AIG": "LIFE",
    "RNR": "REINS", "EG": "REINS",
    "BRK.B": "CONGLOM",
}

# --------------- ВСПОМОГАТЕЛЬНЫЕ ПРОЦЕДУРЫ ПЕЧАТИ/ФОРМАТА ---------------

def _print_divider(title: str = "", char: str = "="):
    line = char * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}")


def _print_group_header(name: str, tickers: List[str], asset_name: str):
    _print_divider(f"ГРУППА: {name}")
    print(f"Тикеры ({len(tickers)}): {', '.join(tickers)}")
    print(f"Рассматриваемая акция: {asset_name}\n")


def _print_metric_vote(signals: List[Tuple[str, str, str]]):
    print("ГОЛОСОВАНИЕ ПО МЕТРИКАМ:")
    for metric, ans, rsn in signals:
        print(f"- {metric} -> Сигнал: {ans}")
        print(f"  Причина:{rsn}")


def _format_row(cols: List[str], widths: List[int]) -> str:
    parts = []
    for c, w in zip(cols, widths):
        parts.append(str(c).ljust(w))
    return " | ".join(parts)


def _print_sector_summary_table(rows: List[Dict[str, str]]):
    headers = ["Группа", "КУПИ", "ПРОДАЙ","НЕОПРЕДЕЛЁННО", "Итог группы"]
    widths = [28, 6, 8, 16, 14]  # было 4, надо 5

    print()
    print(_format_row(headers, widths))
    print(_format_row(["-"*w for w in widths], widths))
    for r in rows:
        print(_format_row([
            r["group"], str(r["buy"]), str(r["sell"]), str(r["unsure"]), r["final"]
        ], widths))


# --------------- ОСНОВНАЯ ЛОГИКА ПО ГРУППАМ ---------------

def analyze_group(group_name: str, tickers: List[str], asset_name: str) -> Dict:
    """
    Делает:
      - подтягивает данные по тикерам
      - строит 3 отчёта (Forward P/E, P/FCF, EV/EBITDA)
      - задаёт LLM 3 вопроса (по каждой метрике), парсит ответы
      - печатает детальные отчёты по группе + голосование + итог
      - возвращает структуру для сводной таблицы
    """
    if asset_name not in tickers:
        # На всякий случай: быстро выходим, если кто-то вызвал без фильтра
        print(f"Пропуск группы '{group_name}': {asset_name} отсутствует в группе.")
        return {"group": group_name, "signals": [], "final": "N/A", "buy": 0, "sell": 0}

    _print_group_header(group_name, tickers, asset_name)

    logger.info(f"Starting yahoo() for {len(tickers)} tickers")
    rows = yahoo(tickers)
    logger.info("Completed yahoo()")
    
    logger.info("Starting add_ev_fair_value")
    rows = add_ev_fair_value(rows, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
    logger.info("Completed add_ev_fair_value")
    
    logger.info("Starting estimate_ev_fair_value")
    report_ev_fair_value = estimate_ev_fair_value(rows)
    logger.info("Completed estimate_ev_fair_value")
    
    report_float = None
    if asset_name in INSURANCE:
        logger.info("Starting add_float_value for insurance asset")
        rows = add_float_value(rows, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
        logger.info("Completed add_float_value for insurance asset")

        logger.info("Starting estimate_float_value for insurance asset")
        report_float = estimate_float_value(rows)
        logger.info("Completed estimate_float_value for insurance asset")

    logger.info("Starting estimate_fpe")
    report_fpe = estimate_fpe(rows)
    logger.info("Completed estimate_fpe")
    report_pfcf = None
    report_ev = None


    # для не-страховых считаем P/FCF и EV/EBITDA
    if asset_name not in INSURANCE:
        logger.info("Starting estimate_pfcf")
        report_pfcf = estimate_pfcf(rows)
        logger.info("Completed estimate_pfcf")
        
        logger.info("Starting estimate_ev_ebitda")
        report_ev = estimate_ev_ebitda(rows)
        logger.info("Completed estimate_ev_ebitda")

    logger.info("Starting add_sotp")
    rows = add_sotp(rows, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
    logger.info("Completed add_sotp")
    
    logger.info("Starting estimate_sotp")
    report_sotp = estimate_sotp(rows)
    logger.info("Completed estimate_sotp")

    print("ДЕТАЛИ АНАЛИЗА (по группе):")
    # 1) спец-блоки по группе
    if asset_name in INSURANCE and report_float:
        print(report_float)

    # 2) fair-value EV — всегда
    print(report_ev_fair_value)
    print()

    # 3) классические мультипликаторы
    print(report_fpe)  # Forward P/E показываем всегда
    print()
    if asset_name not in INSURANCE and report_pfcf:
        print(report_pfcf)
        print()
    if asset_name not in INSURANCE and report_ev:
        print(report_ev)
        print()

    # 4) SOTP — всегда
    print(report_sotp)
    print()

    a_fpe, reason_fpe = _metric_signal_from_rows("Forward P/E", rows, asset_name)
    if asset_name not in INSURANCE:
        a_pfcf, reason_pfcf = _metric_signal_from_rows("P/FCF", rows, asset_name)

    # Новый сигнал по флоуту — только для страховщиков
    a_float, reason_float = ("НЕОПРЕДЕЛЁННО", "Флоут не применяется для этой группы.")
    if asset_name in INSURANCE:
        a_float, reason_float = _float_signal_from_rows("Float/EV", rows, asset_name)

    # --- SOTP: детерминированный сигнал по знаку дисконта/премии ---
    row_me = next((r for r in rows if r.get("Ticker") == asset_name), {})
    pct = row_me.get("SOTP_PREMIUM_PCT")
    if pct is None or not pd.notna(pct):
        a_sotp, reason_sotp = "НЕОПРЕДЕЛЁННО", "Нет числовой оценки SOTP (или не распознана таблица сегментов)."
    else:
        # трактовка:
        #  pct > 0  → SOTP выше EV (дисконт к рынку) → КУПИ
        #  pct < 0  → SOTP ниже EV (премия к рынку) → ПРОДАЙ
        thr = 10.0  # зона неопределённости ±10%
        if pct > thr:
            a_sotp = "КУПИ"
        elif pct < -thr:
            a_sotp = "ПРОДАЙ"
        else:
            a_sotp = "НЕОПРЕДЕЛЁННО"
        lbl = "дисконт" if pct > 0 else "премия"
        reason_sotp = f"SOTP к рынку: {pct:+.1f}% ({lbl} по отношению к EV; порог ±{thr}%)."

    group_signals = [
        ("Forward P/E", a_fpe, reason_fpe),
    ]
    if asset_name not in INSURANCE:
        group_signals.append(("P/FCF", a_pfcf, reason_pfcf))
    if asset_name not in INSURANCE:
        a_ev, reason_ev = _metric_signal_from_rows("EV/EBITDA", rows, asset_name)
        group_signals.append(("EV/EBITDA", a_ev, reason_ev))

    # Вставляем флоут в список сигналов, если релевантно
    if asset_name in INSURANCE:
        group_signals.append(("Float/EV", a_float, reason_float))

    group_signals.append(("SOTP", a_sotp, reason_sotp))

    print()
    _print_metric_vote(group_signals)

    group_final, buy_c, sell_c, unsure_c = _majority_vote(group_signals)

    print("\nСЧЁТ ПО ГРУППЕ:")
    print(f"КУПИ: {buy_c} | ПРОДАЙ: {sell_c} | НЕОПРЕДЕЛЁННО: {unsure_c}")

    print("\nИТОГ ГРУППЫ:")
    print(f"Сигнал: {group_final}")

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
    """Возвращает канонический ключ группы.
    Пример: 'SEMI_PEERS (полупроводники)' -> 'SEMI_PEERS'."""
    return re.split(r"\s|\(", name, maxsplit=1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticker', required=True)
    parser.add_argument(
        '--group',
        help=("Список ключей групп через запятую (например: INSURANCE,BIG_TECH_CORE). "
              "Если не указан — группы выбираются автоматически по тикеру.")
    )
    args = parser.parse_args()

    # кого оцениваем в контексте каждой группы (ТИКЕР!)
    asset_name = args.ticker.upper()

    # --- Полный реестр групп (как и было, просто в локальную переменную) ---
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

    # Ключ -> (display_name, tickers)
    key_to_spec = { _group_key(name).upper(): (name, tickers) for name, tickers in full_group_specs }
    all_keys_str = ", ".join(sorted(key_to_spec.keys()))

    # --- Определяем список групп к прогону ---
    if args.group:
        requested_keys = [k.strip().upper() for k in args.group.split(",") if k.strip()]
        unknown = [k for k in requested_keys if k not in key_to_spec]
        if unknown:
            _print_divider("Ошибка: неизвестные ключи групп")
            print("Неизвестные:", ", ".join(unknown))
            print("Доступные ключи групп:", all_keys_str)
            return

        # Только то, что пользователь явно попросил
        group_specs = [key_to_spec[k] for k in requested_keys]
    else:
        # Автоподбор: куда входит тикер
        auto_specs = [(name, tickers) for name, tickers in full_group_specs if asset_name in tickers]

        if len(auto_specs) == 0:
            _print_divider("Нет подходящих групп")
            print(f"{asset_name} не входит ни в одну группу. Анализ прекращён.")
            return
        elif len(auto_specs) == 1:
            group_specs = auto_specs
        else:
            # Требуем явно указать --group
            _print_divider("Требуется указать группу(ы)")
            human_list = ", ".join([f"{name} (ключ: {_group_key(name)})" for name, _ in auto_specs])
            example_all = ",".join([_group_key(name) for name, _ in auto_specs])
            example_one = _group_key(auto_specs[0][0])

            print(f"Тикер {asset_name} найден сразу в нескольких группах:")
            print(human_list)
            print("\nУкажи параметр --group с одним или несколькими ключами через запятую. Примеры:")
            print(f"  --group={example_one}")
            print(f"  --group={example_all}")
            print("\nСписок всех возможных ключей групп:", all_keys_str)
            return

    # --- Дальше логика остаётся прежней, но работает только по выбранным group_specs ---
    eligible_specs = [(name, tickers) for name, tickers in group_specs if asset_name in tickers]
    skipped_specs = [(name, tickers) for name, tickers in group_specs if asset_name not in tickers]

    if skipped_specs:
        _print_divider("Пропуск нерелевантных групп", char="-")
        for name, _ in skipped_specs:
            print(f"Пропускаю группу '{name}': {asset_name} не входит в состав группы.")

    if not eligible_specs:
        _print_divider("Нет подходящих групп")
        print(f"{asset_name} не входит ни в одну из указанных групп. Анализ прекращён.")
        return

    all_group_results: List[Dict] = []
    all_signals_flat: List[Tuple[str, str, str]] = []

    # детальные блоки по релевантным группам
    for name, tickers in eligible_specs:
        res = analyze_group(name, tickers, asset_name)
        all_group_results.append(res)
        all_signals_flat.extend(res["signals"])
        _print_divider(char="-")

    # сводная таблица
    _print_divider("ИТОГОВАЯ ТАБЛИЦА ПО ГРУППАМ (секторный счёт)")
    table_rows = [
        {"group": r["group"], "buy": r["buy"], "sell": r["sell"], "unsure": r["unsure"], "final": r["final"]}
        for r in all_group_results
    ]
    _print_sector_summary_table(table_rows)

    # общий финальный вердикт
    if all_signals_flat:
        final_all, buy_all, sell_all, unsure_all = _majority_vote(all_signals_flat)
        print("\nОБЩИЙ СЧЁТ ПО ВСЕМ РЕЛЕВАНТНЫМ ГРУППАМ (по метрикам):")
        print(f"КУПИ: {buy_all} | ПРОДАЙ: {sell_all} | НЕОПРЕДЕЛЕННО: {unsure_all}")
        print("\nОБЩЕЕ ИТОГОВОЕ РЕШЕНИЕ:")
        print(f"Сигнал: {final_all}")
    else:
        print("\nНет сигналов ни по одной релевантной группе — итог не рассчитывается.")

if __name__ == "__main__":
    main()
