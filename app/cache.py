from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
import json, shutil

class _Cfg:
    base_dir: Path = Path("cache")
    enabled: bool = False

def init_cache(cache_dir: str | Path = "cache", use_cache: bool = False) -> None:
    _Cfg.base_dir = Path(cache_dir)
    _Cfg.enabled = bool(use_cache)
    if _Cfg.enabled:
        _Cfg.base_dir.mkdir(parents=True, exist_ok=True)

def is_enabled() -> bool:
    return _Cfg.enabled

def base_path() -> Path:
    return _Cfg.base_dir

def clean() -> None:
    p = _Cfg.base_dir
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)
    p.mkdir(parents=True, exist_ok=True)

def path(*parts: str) -> Path:
    return _Cfg.base_dir.joinpath(*parts)

def read_text(relpath: str) -> Optional[str]:
    if not _Cfg.enabled:
        return None
    p = path(relpath)
    if not p.exists():
        return None
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None

def write_text(relpath: str, content: str) -> None:
    if not _Cfg.enabled:
        return
    p = path(relpath)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(content, encoding="utf-8")
    except Exception:
        pass

def read_json(relpath: str) -> Optional[Any]:
    if not _Cfg.enabled:
        return None
    s = read_text(relpath)
    if s is None:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None

def write_json(relpath: str, obj: Any) -> None:
    if not _Cfg.enabled:
        return
    try:
        write_text(relpath, json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        pass
