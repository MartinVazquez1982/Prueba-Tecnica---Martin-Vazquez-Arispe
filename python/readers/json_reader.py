import json
from typing import Any
from pathlib import Path


def _flatten(obj: Any) -> str:
    parts = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            label = key.replace("_", " ").capitalize()
            if isinstance(value, (dict, list)):
                parts.append(f"{label}:")
                parts.append(_flatten(value))
            else:
                parts.append(f"{label}: {value}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                parts.append(_flatten(item))
                parts.append("")
            else:
                parts.append(f"- {item}")
    else:
        parts.append(str(obj))
    return "\n".join(parts)


def read_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return _flatten(data)