"""XDG config loader for mokuro-fast.

Reads from $XDG_CONFIG_HOME/mokuro-fast/config.toml
(or ~/.config/mokuro-fast/config.toml if XDG_CONFIG_HOME is unset).

Expected format:

    precision = "bf16"  # or "f32" (default)

    [batch]
    detector = 30
    ocr = 60

Missing file or invalid TOML is silently ignored.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

try:
    import tomllib  # stdlib since 3.11
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment,misc]

if sys.platform != "linux" and sys.platform != "darwin" and not sys.platform.startswith("freebsd"):
    if sys.platform == "win32":
        _non_unix = True
    else:
        # Unknown unix-like – best effort
        _non_unix = False
else:
    _non_unix = False

# Sentinel used by callers to detect "no config value present".
_UNSET = object()


def _default_config_path() -> Optional[Path]:
    """Return the default XDG config path, or None if we can't determine it."""
    if _non_unix:
        return None
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        base = Path(xdg)
    else:
        base = Path.home() / ".config"
    return base / "mokuro-fast" / "config.toml"


def _config_path(explicit: Optional[str] = None) -> Optional[Path]:
    """Return the config path to use.

    If *explicit* is given, use it directly (even on non-Unix).  Otherwise
    fall back to the default XDG location.
    """
    if explicit is not None:
        return Path(explicit)
    return _default_config_path()


def _load_raw(explicit: Optional[str] = None) -> dict:
    """Load and return the raw config dict (may be empty)."""
    path = _config_path(explicit)
    if path is None or tomllib is None:
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError):
        return {}
    except Exception:  # bad TOML – silently ignore
        return {}


def get_config(explicit: Optional[str] = None) -> dict:
    """Return the parsed config dict (always succeeds, may be empty).

    If *explicit* is given, load from that path instead of the default XDG location.
    """
    return _load_raw(explicit)


def get_precision(explicit: Optional[str] = None) -> object:
    """Return the configured precision string, or _UNSET if not configured.

    Valid values: "bf16", "f32".
    """
    cfg = get_config(explicit)
    val = cfg.get("precision")
    if val in ("bf16", "f32"):
        return val
    return _UNSET


def get_batch_detector(explicit: Optional[str] = None) -> object:
    """Return the configured detector batch size, or _UNSET if not configured."""
    cfg = get_config(explicit)
    val = cfg.get("batch", {}).get("detector")
    if isinstance(val, int) and val > 0:
        return val
    return _UNSET


def get_batch_ocr(explicit: Optional[str] = None) -> object:
    """Return the configured OCR batch size, or _UNSET if not configured."""
    cfg = get_config(explicit)
    val = cfg.get("batch", {}).get("ocr")
    if isinstance(val, int) and val > 0:
        return val
    return _UNSET
