"""Small, deterministic JSON contracts shared by the offline and online paths."""

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path


class ContractError(ValueError):
    """A caller violated an explicit data or execution contract."""


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def identifier(kind, value):
    return kind + ":" + digest(value)[:24]


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def lookup(text):
    return normalize(text).casefold()


def tokens(text):
    # CJK characters are separate lexical units; no claim of semantic tokenization.
    return re.findall(r"[\u3400-\u9fff]|[^\W_\u3400-\u9fff]+", lookup(text), flags=re.UNICODE)


def read_json(path):
    def reject_constant(value):
        raise ContractError("Non-finite JSON number: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ContractError(message)


def finite_unit(value, name, closed=True):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0 <= value and (value <= 1 if closed else value < 1),
            name + " is outside its allowed interval")
