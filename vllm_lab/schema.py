"""The contract between the model and the app: a support ticket as a structured record.

Everything here is plain standard library: the JSON Schema the server enforces, a small
validator that checks the same rules on the client (so free text and constrained output
are judged by one ruler), and a best-effort parser for half-written JSON from a stream.
"""

from __future__ import annotations

import json
import re

CATEGORIES = [
    "shipping", "damaged_product", "wrong_item", "payment", "account", "cancellation", "other",
]
URGENCY = ["low", "medium", "high"]
ACTIONS = ["refund", "exchange", "resend", "tracking", "cancellation", "information", "invoice"]
ORDER_ID_PATTERN = "^GT-[0-9]{6}$"
# For the regex mode of experiment 03: either a real order id or the literal NONE, so the
# model has a legal way to say "the ticket has no order id" instead of inventing one.
ORDER_ID_REGEX = "GT-[0-9]{6}|NONE"

TICKET_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "urgency": {"type": "string", "enum": URGENCY},
        "product": {"type": ["string", "null"]},
        "requested_action": {"type": "string", "enum": ACTIONS},
        "order_id": {"type": ["string", "null"], "pattern": ORDER_ID_PATTERN},
    },
    "required": ["category", "urgency", "product", "requested_action", "order_id"],
    "additionalProperties": False,
}

FIELDS = list(TICKET_SCHEMA["properties"])

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "ticket_record", "schema": TICKET_SCHEMA, "strict": True},
}

_TYPES = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Checks `value` against the subset of JSON Schema this repo uses.

    Supported: type (one or a list), enum, pattern, minLength, properties, required,
    additionalProperties false and items. Returns a list of errors; empty means valid.
    """
    errors: list[str] = []
    types = schema.get("type")
    if types is not None:
        allowed = [types] if isinstance(types, str) else types
        if not any(_TYPES[t](value) for t in allowed):
            return [f"{path}: expected {'|'.join(allowed)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errors.append(f"{path}: {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{path}: shorter than {schema['minLength']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing '{key}'")
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}: unexpected '{key}'")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], f"{path}[{i}]")
    return errors


def check_text(text: str, schema: dict = TICKET_SCHEMA) -> dict:
    """What the app does with a raw answer: json.loads it, then validate it."""
    try:
        value = json.loads(text)
    except ValueError as exc:
        return {"parsed": False, "valid": False, "error": f"json.loads: {exc.msg}"}
    errors = validate(value, schema)
    return {"parsed": True, "valid": not errors, "error": errors[0] if errors else None,
            "value": value}


def _close(prefix: str, close_strings: bool) -> str | None:
    """Turns a JSON prefix into a candidate document by closing what is still open."""
    stack: list[str] = []
    in_string = escape = False
    for ch in prefix:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
    text = prefix
    if in_string:
        if not close_strings:
            return None
        if escape:
            text = text[:-1]
        text += '"'
    text = text.rstrip()
    text = text.removesuffix(",")
    if text.endswith(":"):
        return None
    return text + "".join(reversed(stack))


def parse_partial(text: str, close_strings: bool = False) -> dict | None:
    """Best-effort parse of a JSON object that is still arriving.

    With close_strings=False only fields whose value is already complete come back (what a
    UI can trust). With close_strings=True the field being written also comes back with
    the characters received so far (what a UI can show as "typing").
    """
    start = text.find("{")
    if start < 0:
        return None
    s = text[start:]
    cuts = [len(s)] + [i for i in range(len(s) - 1, 0, -1) if s[i] == ","] + [1]
    for cut in cuts:
        candidate = _close(s[:cut], close_strings)
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None
