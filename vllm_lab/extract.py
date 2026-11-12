"""Runs the 20 tickets through the model once and judges every answer with one ruler."""

from __future__ import annotations

import statistics

from openai import OpenAI

from vllm_lab.client import complete
from vllm_lab.schema import FIELDS, check_text, unwrap
from vllm_lab.tickets import EXTRACT_SYSTEM, load_tickets, messages


def extract_all(client: OpenAI, **request_kwargs) -> dict:
    """Asks for a record per ticket; counts what parses, what validates and what matches.

    `request_kwargs` go straight to chat.completions.create (for example response_format).
    """
    rows = []
    for ticket in load_tickets():
        r = complete(client, messages(EXTRACT_SYSTEM, ticket["text"]), max_tokens=200,
                     **request_kwargs)
        verdict = check_text(r["text"])
        cleaned = check_text(unwrap(r["text"]))  # the same answer after dropping fences and chatter
        value = verdict.get("value") if verdict["valid"] else None
        matches = sum(value[f] == ticket["expected"][f] for f in FIELDS) if value else 0
        rows.append({
            "id": ticket["id"],
            "parsed": verdict["parsed"],
            "valid": verdict["valid"],
            "valid_after_unwrap": cleaned["valid"],
            "error": verdict["error"],
            "error_after_unwrap": cleaned["error"],
            "fields_matching_expected": matches,
            "completion_tokens": r["completion_tokens"],
            "wall_s": round(r["wall_s"], 4),
            "raw": r["text"][:160],
        })
    n = len(rows)
    parsed = sum(r["parsed"] for r in rows)
    valid = sum(r["valid"] for r in rows)
    valid_clean = sum(r["valid_after_unwrap"] for r in rows)
    return {
        "tickets": n,
        "parsed": parsed,
        "valid": valid,
        "parse_rate": round(parsed / n, 3),
        "valid_rate": round(valid / n, 3),
        "valid_after_unwrap": valid_clean,
        "valid_after_unwrap_rate": round(valid_clean / n, 3),
        # Agreement with the hand-written labels, only over valid records. Informative:
        # a schema guarantees the shape of the answer, not that the answer is right.
        "field_agreement": round(
            sum(r["fields_matching_expected"] for r in rows) / max(valid * len(FIELDS), 1), 3),
        "mean_latency_s": round(statistics.mean(r["wall_s"] for r in rows), 4),
        "mean_completion_tokens": round(statistics.mean(r["completion_tokens"] for r in rows), 1),
        "rows": rows,
    }
