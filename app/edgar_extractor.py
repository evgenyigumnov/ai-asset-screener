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



CACHE_DIR = Path("cache")


_PLAIN_NOTES_TITLE = re.compile(
    r'^\s*notes?\s+to\s+(?:the\s+)?consolidated\s+financial\s+statements\b', re.I
)

_PLAIN_SEGMENT_TITLE = re.compile(
    r'^\s*(?:\d{1,3}\s*[.\-:—–]\s*)?segment\s+information\b.*$', re.I
)

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

_NOTE_HEADING_WIDE = re.compile(
    r'^\s{0,3}#{1,6}\s*(?:note)\s+\d+\b.*$', re.I
)

_PLAIN_NOTE_LINE = re.compile(
    r'^\s*(?:note)\s+\d+\b.*$', re.I
)

ALLOWED_SEC_TITLE = re.compile(
    r'\b('
    r'segment|segments?|segment information|'       
    r'revenue|revenues|net\s+sales|disaggregation|' 
    r'geograph|americas|europe|asia\s+pacific|'     
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

import json


def _assemble_batch_text(batch: List[Dict[str, Any]], start_index: int = 0) -> str:
    parts = []
    for k, t in enumerate(batch, start=start_index + 1):
        sec = t.get("section") or "(unknown / plain segment area)"
        parts.append(f"[{k}] Section: {sec}\n{t['markdown']}")
    return "\n\n".join(parts)

def _is_valid_llm_table(md: str) -> bool:
    s = md.strip()
    if not s:
        return False
    lines = [ln for ln in s.splitlines() if ln.strip().startswith('|')]
    if len(lines) < 3:
        return False
    header = lines[0].lower()
    looks_right = (('region' in header or 'segment' in header)
                   and (('net' in header and 'sale' in header) or 'revenue' in header))
    return looks_right or len(lines) >= 3


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _ticker_key(ticker: str) -> str:
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
        pass



_HDR_RE = re.compile(r'^\s{0,3}(#{2,6})\s*(.+?)\s*$')
_NOTES_TITLE_RE = re.compile(
    r'^\s{0,3}#{2,6}\s*Notes\s+to\s+(?:the\s+)?Consolidated\s+Financial\s+Statements\b', re.I
)
_NOTE_N_RE = re.compile(r'^\s{0,3}#{2,6}\s*(?:Note\s+)?\d+\b', re.I)
_NUMERIC_NOTE_RE = re.compile(r'^\s{0,3}#{2,6}\s*\d+(?:[.:]|[–—-])\s+', re.I)
_SEGMENT_IN_TITLE = re.compile(r'\bsegment\s+information\b', re.I)

def _is_table_header_line(line: str) -> bool:
    s = line.rstrip()
    if '|' not in s:
        return False
    return s.lstrip().startswith('|') or bool(re.search(r'[^|]\|[^|]', s))

def _is_table_divider_line(line: str) -> bool:
    s = line.strip()
    if '|' not in s:
        return False
    core = s.replace('—', '-').replace('–', '-')
    core = re.sub(r'[ \t\u00A0\u202F]', '', core)  # обычные/неразрывные пробелы, узкий пробел
    if not core or set(core) - set(':-|'):
        return False
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
            if _PLAIN_NOTES_TITLE.match(line):
                inside_notes = True
                notes_level = 6
                current_section_title = line.strip()
                current_header_level = 6

            elif _PLAIN_SEGMENT_TITLE.match(line) or _PLAIN_SEGMENT_GEO_TITLE.match(line):
                inside_notes = True
                notes_level = 6
                current_section_title = line.strip()
                current_header_level = 6

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

        keep = inside_notes or is_segment

        if not keep:
            if ALLOWED_SEC_TITLE.search(sec_title) or KEYWORDS_IN_TABLE.search(raw):
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
    key = _ticker_key(ticker)
    cache_name = f"{key}.10k"

    cached = _read_cache(cache_name)
    if cached is not None and cached.strip():
        try:
            data = json.loads(cached)
            if isinstance(data, list) and data and isinstance(data[0], dict) and "markdown" in data[0]:
                return data
        except Exception:
            pass

    company = Company(ticker)
    filings = company.get_filings(form="10-K")

    filing = next((f for f in filings if isinstance(getattr(f, "form", None), str) and f.form.upper() == "10-K"), None)
    if filing is None:
        try:
            filing = next(f for f in filings if "/A" not in getattr(f, "form", ""))
        except Exception:
            return []

    text = filing.markdown()
    tables = extract_relevant_tables(text)

    serializable = []
    for t in tables:
        serializable.append({
            "section": t.get("section"),
            "start_line": t.get("start_line"),
            "end_line": t.get("end_line"),
            "markdown": t.get("markdown"),
        })

    try:
        _write_cache(cache_name, json.dumps(serializable, ensure_ascii=False))
    except Exception:
        pass

    return serializable




def _is_valid_segment_table(md: str) -> bool:
    s = md.strip()
    if not s:
        return False
    lines = [ln for ln in s.splitlines() if ln.strip().startswith('|')]
    if len(lines) < 3:
        return False
    header = lines[0].lower()
    return ('segment' in header) and ('operating' in header)

def parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    if not md.strip():
        return None
    lines = [ln for ln in md.splitlines() if '|' in ln]
    if len(lines) < 3:
        return None
    df = parse_md_table_to_df(lines)
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None

def extract_operating_segments(ticker: str) -> str:
    key = _ticker_key(ticker)
    cache_name = f"{key}.operating_segments"

    cached = _read_cache(cache_name)
    if cached is not None:
        return cached
    total_ret = ""


    md = fetch_10k_markdown(ticker)
    if not md or not md.strip():
        _write_cache(cache_name, "")
        return ""

    chunks = chunk_text(md, max_chars=50000, overlap=1000)
    for ch in chunks:
        prompt = f"""
        {ch}

        10-K excerpt (analyze ONLY this text):

        Extract EXACTLY the OPERATING SEGMENTS (ASC 280), NOT geography.
        Provide data ONLY for the most recent year.
        RETURN ALL NUMBERS STRICTLY IN MILLIONS OF USD (if the text says 'in thousands' — divide by 1,000).
        No currency symbols or commas.

        Return a single markdown table with the columns exactly:
        | Segment | Revenue | Operating income |

        If there are no operating segments (only geography) — return an empty response (no table).
        """

        logger.info(f"Starting ask_llm for ticker={ticker}")
        ret = ask_llm(prompt)
        logger.info(f"Completed ask_llm for ticker={ticker}")
        ret = re.sub(r'^(?!\|).+', '', ret, flags=re.MULTILINE)
        ret = re.sub(r'\$', '', ret)
        ret = re.sub(r',', '', ret)
        if _is_valid_segment_table(ret) and total_ret == "":
            total_ret += ret+ "\n\n"
        if total_ret != "":
            break

    _write_cache(cache_name, total_ret)
    return total_ret


