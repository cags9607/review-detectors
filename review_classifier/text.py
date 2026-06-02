from typing import Any


def coerce_text(value: Any) -> str:
    if value is None:
        return ""

    return str(value).strip()


def make_text(text: Any) -> str:
    return coerce_text(text)
