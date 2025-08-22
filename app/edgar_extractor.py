import logging
import re
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd

from edgar import *

from app.float_value import fetch_10k_markdown, chunk_text
from app.llm_util import ask_llm
from app.yahoo import yahoo
from pathlib import Path

logger = logging.getLogger(__name__)

set_identity("igumnovnsk@gmail.com")


# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================


CACHE_DIR = Path("cache")


# Плоские заголовки без '#'
_PLAIN_NOTES_TITLE = re.compile(
    r'^\s*notes?\s+to\s+(?:the\s+)?consolidated\s+financial\s+statements\b', re.I
)

_PLAIN_SEGMENT_TITLE = re.compile(
    r'^\s*(?:\d{1,3}\s*[.\-:—–]\s*)?segment\s+information\b.*$', re.I
)

# Варианты "segment and geographic ..." / "segment information and geographic data"
_PLAIN_SEGMENT_GEO_TITLE = re.compile(
    r'^\s*(?:\d{1,3}\s*[.\-:—–]\s*)?(?:segment|segments?)\s+(?:information|reporting|and)\b.*\bgeograph\w*\b.*$', re.I
)

_SEGMENT_OR_GEO_IN_TITLE = re.compile(
    r'\b('
    r'segment\s+information'
    r'|segment\s+and\s+geographic'
    r'|geographic\s+data'
    r'|segment\s+information\s+and\s+geographic\s+data'
    r')\b',
    re.I
)

# 'NOTE 18 — …' (с любым тире/двоеточием), допускаем любые уровни #, регистр игнорируем
_NOTE_HEADING_WIDE = re.compile(
    r'^\s{0,3}#{1,6}\s*(?:note)\s+\d+\b.*$', re.I
)

# На случай, если markdown-конвертер не проставил '#', ловим строку-заголовок "NOTE 18 — ..."
_PLAIN_NOTE_LINE = re.compile(
    r'^\s*(?:note)\s+\d+\b.*$', re.I
)

# --- ДОБАВЬ/ЗАМЕНИ КЛЮЧИ ВВЕРХУ МОДУЛЯ ---
ALLOWED_SEC_TITLE = re.compile(
    r'\b('
    r'segment|segments?|segment information|'       # сегменты
    r'revenue|revenues|net\s+sales|disaggregation|'  # выручка
    r'geograph|americas|europe|asia\s+pacific|'      # география
    r'operating\s+income|operating\s+loss|segment\s+assets|total\s+assets'
    r')\b',
    re.I
)

KEYWORDS_IN_TABLE = re.compile(
    r'\b('
    r'segment|operating\s+income|operating\s+loss|segment\s+assets|total\s+assets|'
    r'net\s+sales|net\s+revenue|revenue|revenues|'
    r'geograph|americas|europe|asia\s+pacific|disaggregation'
    r')\b',
    re.I
)

# ДОБАВЬ ИМПОРТ
import json

# --- ХЕЛПЕРЫ ДЛЯ НОВОЙ ЛОГИКИ ---

def _assemble_batch_text(batch: List[Dict[str, Any]], start_index: int = 0) -> str:
    """
    Склеивает до 10 таблиц в один текст для LLM-промпта.
    """
    parts = []
    for k, t in enumerate(batch, start=start_index + 1):
        sec = t.get("section") or "(unknown / plain segment area)"
        parts.append(f"[{k}] Section: {sec}\n{t['markdown']}")
    return "\n\n".join(parts)

def _is_valid_llm_table(md: str) -> bool:
    """
    Грубая проверка, что LLM вернул markdown-таблицу с минимум одной строкой данных.
    """
    s = md.strip()
    if not s:
        return False
    lines = [ln for ln in s.splitlines() if ln.strip().startswith('|')]
    if len(lines) < 3:  # шапка + минимум одна строка
        return False
    header = lines[0].lower()
    # Хедер должен намекать на правильный смысл
    looks_right = (('region' in header or 'segment' in header)
                   and (('net' in header and 'sale' in header) or 'revenue' in header))
    return looks_right or len(lines) >= 3


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _ticker_key(ticker: str) -> str:
    """
    Нормализуем тикер для имени файла: NVDA, BRK_B, RDSA, и т.п.
    """
    return re.sub(r'[^A-Za-z0-9_-]+', '_', ticker.upper())

def _read_cache(fname: str) -> Optional[str]:
    p = CACHE_DIR / fname
    if p.exists():
        try:
            return p.read_text(encoding="utf-8")
        except Exception:
            return None
    return None

def _write_cache(fname: str, content: str) -> None:
    _ensure_cache_dir()
    p = CACHE_DIR / fname
    try:
        p.write_text(content, encoding="utf-8")
    except Exception:
        # при желании можно залогировать, но молча пропускаем
        pass



# === РЕГЕКСЫ (замени старые) ===
_HDR_RE = re.compile(r'^\s{0,3}(#{2,6})\s*(.+?)\s*$')
_NOTES_TITLE_RE = re.compile(
    r'^\s{0,3}#{2,6}\s*Notes\s+to\s+(?:the\s+)?Consolidated\s+Financial\s+Statements\b', re.I
)
# раньше требовал обязательно "Note", из-за чего AVGO мимо
_NOTE_N_RE = re.compile(r'^\s{0,3}#{2,6}\s*(?:Note\s+)?\d+\b', re.I)
# отдельный паттерн под "цифра-точка" в заголовке
_NUMERIC_NOTE_RE = re.compile(r'^\s{0,3}#{2,6}\s*\d+(?:[.:]|[–—-])\s+', re.I)
_SEGMENT_IN_TITLE = re.compile(r'\bsegment\s+information\b', re.I)

def _is_table_header_line(line: str) -> bool:
    """
    Первая строка таблицы markdown.
    Допускаем как форму с ведущей/замыкающей '|' так и без замыкающей.
    """
    s = line.rstrip()
    if '|' not in s:
        return False
    # либо начинается с '|', либо имеет как минимум два столбца вида "a | b"
    return s.lstrip().startswith('|') or bool(re.search(r'[^|]\|[^|]', s))

def _is_table_divider_line(line: str) -> bool:
    """
    Вторая строка таблицы: только :, -, |, пробелы/табы и возможные юникод-тире.
    Делаем tolerant к пробелам и неразрывным пробелам.
    """
    s = line.strip()
    # Обычно divider тоже начинается с '|' (но не всегда заканчивается им)
    if '|' not in s:
        return False
    # нормализуем тире и убираем пробельные символы
    core = s.replace('—', '-').replace('–', '-')
    core = re.sub(r'[ \t\u00A0\u202F]', '', core)  # обычные/неразрывные пробелы, узкий пробел
    # Оставшиеся символы должны быть только из набора ':-|'
    if not core or set(core) - set(':-|'):
        return False
    # Должна быть хотя бы одна «ячейка» с тремя дефисами подряд
    return '---' in core

def _clean_cell(x: str) -> str:
    return x.replace('\u00A0', ' ').strip()

def _to_number_if_possible(x: str) -> Any:
    if x is None:
        return x
    s = str(x).strip()
    if s == '' or s in {'—', '— ', '—–', '— —'}:
        return None
    neg = s.startswith('(') and s.endswith(')')
    if neg:
        s = s[1:-1]
    # нормализуем пробелы-разделители тысяч и тире
    s = (s.replace('$','')
           .replace(',', '')
           .replace('\u00A0','')
           .replace('\u202F','')
           .replace('—','-').replace('–','-')
           .replace(' ', ''))
    try:
        val = float(s) if '.' in s else int(s)
        return -val if neg else val
    except Exception:
        return x

def parse_md_table_to_df(lines: List[str]) -> Optional['pd.DataFrame']:
    if pd is None or len(lines) < 2:
        return None

    # ищем разделитель глубже (на случай сильно рваных шапок)
    d = None
    for k in range(1, min(len(lines), 12)):  # было 6
        if _is_table_divider_line(lines[k]):
            d = k
            break
    if d is None:
        return None

    header_line = ' '.join(s.strip() for s in lines[0:d] if s.strip())
    header = [_clean_cell(c) for c in header_line.strip().strip('|').split('|')]

    body = []
    for row in lines[d+1:]:
        if '|' not in row:
            break
        cells = [_clean_cell(c) for c in row.strip().strip('|').split('|')]
        if len(cells) < len(header):
            cells += [''] * (len(header) - len(cells))
        elif len(cells) > len(header):
            cells = cells[:len(header)]
        body.append(cells)

    if not body:
        return None

    df = pd.DataFrame(body, columns=header)
    for col in df.columns:
        df[col] = df[col].map(_to_number_if_possible)
    return df

def find_pipe_block_tables(lines: List[str]) -> List[Tuple[int, int]]:
    """
    Fallback: находим блоки из >=3 подряд идущих строк, содержащих минимум по 2 '|'.
    Полезно для 10-K, где нет markdown-разделителя '---'.
    """
    res = []
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if line.count('|') >= 2:
            start = i
            j = i + 1
            rows = 1
            while j < n and lines[j].count('|') >= 2:
                rows += 1
                j += 1
            if rows >= 3:
                res.append((start, j - 1))
            i = j
        else:
            i += 1
    return res

def find_markdown_tables(lines: List[str]) -> List[Tuple[int, int]]:
    tables = []
    n = len(lines)
    i = 0
    while i < n:
        if _is_table_header_line(lines[i]):
            d = None
            lookahead = min(i + 12, n)
            for k in range(i + 1, lookahead):
                if _is_table_divider_line(lines[k]):
                    d = k
                    break
            if d is not None:
                j = d + 1
                while j < n and '|' in lines[j]:
                    j += 1
                tables.append((i, j - 1))
                i = j
                continue
        i += 1

    # если классических markdown-таблиц нет — пробуем пайповые блоки
    if not tables:
        tables = find_pipe_block_tables(lines)

    return tables
def mark_context(lines: List[str]) -> Dict[int, Dict[str, Any]]:
    ctx = {}
    inside_notes = False
    notes_level = None
    current_section_title = None
    current_header_level = None

    for idx, line in enumerate(lines):
        m = _HDR_RE.match(line)
        if m:
            hashes, title = m.groups()
            level = len(hashes)
            title_stripped = title.strip()

            is_notes_title = bool(_NOTES_TITLE_RE.match(line))
            is_note_heading = bool(_NOTE_N_RE.match(line) or _NUMERIC_NOTE_RE.match(line)
                                   or _NOTE_HEADING_WIDE.match(line))
            is_segment_like = bool(_SEGMENT_IN_TITLE.search(title_stripped))

            if is_notes_title or is_note_heading or is_segment_like:
                # считаем это «внутри примечаний/сегментов»
                inside_notes = True
                notes_level = level
            else:
                # выход из note-блока при встрече заголовка того же/высшего уровня
                if inside_notes and notes_level is not None and level <= notes_level:
                    inside_notes = False
                    notes_level = None

            current_section_title = title_stripped
            current_header_level = level

        else:
            # --- НОВОЕ: «плоские» заголовки без '#'
            if _PLAIN_NOTES_TITLE.match(line):
                inside_notes = True
                notes_level = 6
                current_section_title = line.strip()
                current_header_level = 6

            elif _PLAIN_SEGMENT_TITLE.match(line) or _PLAIN_SEGMENT_GEO_TITLE.match(line):
                # Раздел «Segment Information» часто идёт как "13. SEGMENT INFORMATION"
                inside_notes = True  # трактуем как нужный нам раздел
                notes_level = 6
                current_section_title = line.strip()
                current_header_level = 6

            # fallback: встретили «NOTE 18 — …» строку без '#'
            elif _PLAIN_NOTE_LINE.match(line):
                inside_notes = True
                notes_level = 6
                current_section_title = line.strip()
                current_header_level = 6

        ctx[idx] = {
            "inside_notes": inside_notes,
            "section_title": current_section_title,
            "header_level": current_header_level,
        }
    return ctx

def _nearby_has_segment_marker(lines: List[str], i: int, window: int = 30) -> bool:
    start = max(0, i - window)
    chunk = "\n".join(lines[start:i])
    return (
        _PLAIN_SEGMENT_TITLE.search(chunk) is not None
        or _PLAIN_SEGMENT_GEO_TITLE.search(chunk) is not None
        or _PLAIN_NOTES_TITLE.search(chunk) is not None
        or _SEGMENT_IN_TITLE.search(chunk) is not None
    )

def extract_relevant_tables(md_text: str) -> List[Dict[str, Any]]:
    lines = md_text.splitlines()
    ctx = mark_context(lines)
    ranges = find_markdown_tables(lines)

    out = []
    for (i, j) in ranges:
        meta = ctx.get(i, {})
        sec_title = (meta.get("section_title") or "").strip()
        inside_notes = bool(meta.get("inside_notes"))
        is_segment = bool(_SEGMENT_IN_TITLE.search(sec_title))

        tbl_block = lines[i:j+1]
        raw = "\n".join(tbl_block)

        # основной фильтр
        keep = inside_notes or is_segment

        # доп. признак по ключевым словам
        if not keep:
            if ALLOWED_SEC_TITLE.search(sec_title) or KEYWORDS_IN_TABLE.search(raw):
                # если заголовок не распознан — проверим «близость» к сегментным/примечательным маркерам
                if _nearby_has_segment_marker(lines, i):
                    keep = True

        if not keep:
            continue

        df = parse_md_table_to_df(tbl_block)
        out.append({
            "section": sec_title or "(unknown / plain segment area)",
            "start_line": i,
            "end_line": j,
            "markdown": raw,
            "df": df,
        })
    return out


def ten_k_tables(ticker: str) -> List[Dict[str, Any]]:
    """
    Извлекает таблицы из последнего 10-K (не включая 10-K/A).
    Возвращает список словарей: {section, start_line, end_line, markdown}.
    Кэш хранится в JSON.
    """
    key = _ticker_key(ticker)
    cache_name = f"{key}.10k"

    # 1) попытка прочитать JSON-кэш
    cached = _read_cache(cache_name)
    if cached is not None and cached.strip():
        try:
            data = json.loads(cached)
            if isinstance(data, list) and data and isinstance(data[0], dict) and "markdown" in data[0]:
                return data
        except Exception:
            pass  # старый формат кэша — игнорируем и перегенерим

    # 2) грузим 10-K без "/A"
    company = Company(ticker)
    filings = company.get_filings(form="10-K")

    # берём первый "чистый" 10-K
    filing = next((f for f in filings if isinstance(getattr(f, "form", None), str) and f.form.upper() == "10-K"), None)
    if filing is None:
        # fallback: старый способ, но отфильтровать "/A"
        try:
            filing = next(f for f in filings if "/A" not in getattr(f, "form", ""))
        except Exception:
            # print(f"!!!!!!!!!!!!!!Failed to get 10-K for {ticker}")
            return []

    text = filing.markdown()
    # print(text)
    # 3) вытаскиваем таблицы
    tables = extract_relevant_tables(text)

    # 4) делаем сериализуемый список (без DataFrame)
    serializable = []
    for t in tables:
        serializable.append({
            "section": t.get("section"),
            "start_line": t.get("start_line"),
            "end_line": t.get("end_line"),
            "markdown": t.get("markdown"),
        })

    # 5) сохраняем JSON в кэш
    try:
        _write_cache(cache_name, json.dumps(serializable, ensure_ascii=False))
    except Exception:
        pass

    return serializable




# ===== НОВОЕ: валидация и парсинг таблицы с операционными сегментами =====

def _is_valid_segment_table(md: str) -> bool:
    s = md.strip()
    if not s:
        return False
    lines = [ln for ln in s.splitlines() if ln.strip().startswith('|')]
    if len(lines) < 3:
        return False
    header = lines[0].lower()
    # ищем и "segment", и "operating" в шапке
    return ('segment' in header) and ('operating' in header)

def parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """Упрощённый парсер одной markdown-таблицы -> DataFrame (с приведением чисел)."""
    if not md.strip():
        return None
    lines = [ln for ln in md.splitlines() if '|' in ln]
    if len(lines) < 3:
        return None
    df = parse_md_table_to_df(lines)  # используем уже имеющийся парсер
    if isinstance(df, pd.DataFrame) and not df.empty:
        # нормализуем заголовки
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None

def extract_operating_segments(ticker: str, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY) -> str:
    """
    Ищет в 10-K таблицу с операционными сегментами (НЕ география).
    Возвращает markdown-таблицу:
    | Segment | Revenue | Operating income |
    Значения — как в отчёте (если в заголовке 'In millions' — числа без домножения).
    """

    key = _ticker_key(ticker)
    cache_name = f"{key}.operating_segments"

    # 1) кэш готовой таблицы
    cached = _read_cache(cache_name)
    if cached is not None:
        return cached
    total_ret = ""
    # tables = ten_k_tables(ticker)
    # # print("111")
    # if not tables:
    #     tables = []
    #
    # # print("111")
    # batch_size = 10
    # for offset in range(0, len(tables), batch_size):
    #     batch = tables[offset:offset + batch_size]
    #     chunk_text = _assemble_batch_text(batch, start_index=offset)


    md = fetch_10k_markdown(ticker)
    if not md or not md.strip():
        _write_cache(cache_name, "")
        return ""

    chunks = chunk_text(md, max_chars=50000, overlap=1000)
    for ch in chunks:
        prompt = f"""
{ch}

Фрагмент 10-K (analyze ONLY this text):

Вытащи ИМЕННО ОПЕРАЦИОННЫЕ СЕГМЕНТЫ (ASC 280), НЕ географию.
Дай данные ТОЛЬКО за последний год.
ВЕРНИ ВСЕ ЧИСЛА СТРОГО В МИЛЛИОНАХ USD (если в тексте 'in thousands' — раздели на 1000).
Никаких знаков валют и запятых.

Верни одну markdown-таблицу со столбцами строго:
| Segment | Revenue | Operating income |

Если операционных сегментов нет (только география) — верни пустой ответ (никакой таблицы).
"""
        # print(len(prompt))
        # print("...")
        logger.info(f"Starting ask_llm with model={LLM_MODEL} for ticker={ticker}")
        ret = ask_llm(prompt, LLM_MODEL, LLM_ENDPOINT, LLM_OPENAI_API_KEY)
        logger.info(f"Completed ask_llm with model={LLM_MODEL} for ticker={ticker}")
        # оставляем только строки таблицы, чистим знаки валют/разделители
        ret = re.sub(r'^(?!\|).+', '', ret, flags=re.MULTILINE)
        ret = re.sub(r'\$', '', ret)
        ret = re.sub(r',', '', ret)
        # print(ret)
        if _is_valid_segment_table(ret) and total_ret == "":
            total_ret += ret+ "\n\n"
        if total_ret != "":
            break

    _write_cache(cache_name, total_ret)
    return total_ret


if __name__ == "__main__":
    LLM_ENDPOINT = "http://localhost:1234/v1"
    LLM_MODEL = "openai/gpt-oss-20b"  # при желании поменяй
    # tables = ten_k_tables("GOOGL")
    # for t in tables:
    #     print(t)
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    # print(extract_net_sales("MSFT", LLM_MODEL ,LLM_ENDPOINT))
    # TICKERS = ["TSLA"]
    # rows = yahoo(TICKERS)
    print(extract_operating_segments("TSLA", LLM_MODEL, LLM_ENDPOINT))
