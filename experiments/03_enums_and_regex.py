"""03. Smaller contracts: one of three words, or a string that matches a pattern.

Not every answer needs a whole schema. Here the app needs two single values per ticket:
the urgency (low, medium or high) and the order id (GT-123456, or NONE when the ticket
has none). Each one is asked twice: with the prompt only, and with vLLM's constrained
decoding (`structured_outputs` with `choice` or `regex` in extra_body). If the server
predates that field, the client falls back to `guided_choice` / `guided_regex`.

    python experiments/03_enums_and_regex.py [--mock]
"""

from __future__ import annotations

import re

from vllm_lab import Constrained, cli, complete, load_tickets, make_client, messages
from vllm_lab.schema import ORDER_ID_REGEX, URGENCY
from vllm_lab.tickets import ORDER_ID_SYSTEM, URGENCY_SYSTEM


def run(base_url: str) -> dict:
    client = make_client(base_url)
    constrained = Constrained(client)
    rows = []
    for t in load_tickets():
        u_msgs = messages(URGENCY_SYSTEM, t["text"])
        o_msgs = messages(ORDER_ID_SYSTEM, t["text"])
        rows.append({
            "id": t["id"],
            "urgency_free": complete(client, u_msgs, max_tokens=12)["text"].strip(),
            "urgency_choice": constrained(u_msgs, "choice", URGENCY)["text"].strip(),
            "order_free": complete(client, o_msgs, max_tokens=16)["text"].strip(),
            "order_regex": constrained(o_msgs, "regex", ORDER_ID_REGEX)["text"].strip(),
            "expected_urgency": t["expected"]["urgency"],
            "expected_order": t["expected"]["order_id"] or "NONE",
        })

    def rate(key: str, ok) -> float:
        return round(sum(ok(r[key]) for r in rows) / len(rows), 3)

    def in_enum(text: str) -> bool:
        return text in URGENCY

    def matches(text: str) -> bool:
        return re.fullmatch(ORDER_ID_REGEX, text) is not None

    return {
        "param_style": constrained.style,
        "choices": URGENCY,
        "regex": ORDER_ID_REGEX,
        "urgency_free_in_enum": rate("urgency_free", in_enum),
        "urgency_choice_in_enum": rate("urgency_choice", in_enum),
        "order_free_matches": rate("order_free", matches),
        "order_regex_matches": rate("order_regex", matches),
        # Shape is guaranteed; the value can still be wrong. Informative only.
        "urgency_choice_agrees": round(
            sum(r["urgency_choice"] == r["expected_urgency"] for r in rows) / len(rows), 3),
        "order_regex_agrees": round(
            sum(r["order_regex"] == r["expected_order"] for r in rows) / len(rows), 3),
        "free_misses": [
            {"id": r["id"], "urgency": r["urgency_free"], "order_id": r["order_free"]}
            for r in rows if not (in_enum(r["urgency_free"]) and matches(r["order_free"]))
        ],
        "rows": rows,
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    result.pop("rows")
    cli.show("Prompt only vs choice / regex", result)


if __name__ == "__main__":
    main()
