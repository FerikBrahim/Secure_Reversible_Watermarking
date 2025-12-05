"""
config.py

Simple loader for config.yaml. Tries to use PyYAML when available; otherwise falls
back to a minimal parser for simple key: value YAML files (sufficient for our config.yaml).

Function:
- load_config(path="config.yaml") -> dict

Note: for complex YAML features, install pyyaml:
    pip install pyyaml
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _simple_yaml_parse(text: str) -> Dict[str, Any]:
    """
    Minimal YAML parser for basic nested dictionaries and lists used in our config.yaml.
    Supports:
      - top-level keys
      - nested keys via indentation (two-space indent)
      - simple lists: [a, b, c]
    This is intentionally tiny and NOT a full YAML implementation.
    """
    out: Dict[str, Any] = {}
    cur_parent = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line and not line.startswith("-"):
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if not raw.startswith("  "):  # top-level
                # new parent if next lines indented
                if val == "":
                    out[key] = {}
                    cur_parent = key
                else:
                    # parse simple list or scalar
                    if val.startswith("[") and val.endswith("]"):
                        # list
                        items = [v.strip().strip('"').strip("'") for v in val[1:-1].split(",") if v.strip()]
                        # try convert numeric
                        parsed_items = []
                        for it in items:
                            try:
                                parsed_items.append(int(it))
                            except Exception:
                                try:
                                    parsed_items.append(float(it))
                                except Exception:
                                    parsed_items.append(it)
                        out[key] = parsed_items
                        cur_parent = None
                    else:
                        # scalar
                        v = val.strip().strip('"').strip("'")
                        # convert numeric or bool
                        if v.lower() in ("true", "false"):
                            out[key] = v.lower() == "true"
                        else:
                            try:
                                out[key] = int(v)
                            except Exception:
                                try:
                                    out[key] = float(v)
                                except Exception:
                                    out[key] = v
                        cur_parent = None
            else:
                # nested key under current parent
                if cur_parent is None:
                    continue
                if val == "":
                    out[cur_parent][key] = {}
                else:
                    vv = val.strip().strip('"').strip("'")
                    # convert
                    if vv.lower() in ("true", "false"):
                        out[cur_parent][key] = vv.lower() == "true"
                    else:
                        try:
                            out[cur_parent][key] = int(vv)
                        except Exception:
                            try:
                                out[cur_parent][key] = float(vv)
                            except Exception:
                                # list?
                                if vv.startswith("[") and vv.endswith("]"):
                                    items = [v.strip().strip('"').strip("'") for v in vv[1:-1].split(",") if v.strip()]
                                    # try numeric
                                    parsed_items = []
                                    for it in items:
                                        try:
                                            parsed_items.append(int(it))
                                        except Exception:
                                            try:
                                                parsed_items.append(float(it))
                                            except Exception:
                                                parsed_items.append(it)
                                    out[cur_parent][key] = parsed_items
                                else:
                                    out[cur_parent][key] = vv
    return out


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    Returns a dict with parsed values. If file not found, returns an empty dict.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("Config file %s not found; using defaults.", path)
        return {}

    # Try PyYAML first
    try:
        import yaml  # type: ignore
        with p.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            return {}
        return cfg
    except Exception:
        logger.info("PyYAML not available or failed; using fallback parser for %s", path)
        text = p.read_text(encoding="utf-8")
        return _simple_yaml_parse(text)


if __name__ == "__main__":  # pragma: no cover - quick smoke
    import pprint
    logging.basicConfig(level=logging.INFO)
    cfg = load_config()
    pprint.pprint(cfg)
