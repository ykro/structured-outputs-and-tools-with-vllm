"""02. The schema goes in the request: response_format with a JSON Schema.

Same 20 tickets, same prompt, plus response_format={"type": "json_schema", ...}. vLLM
turns the schema into a grammar and, at every decode step, masks the tokens that would
break it. Measures the parse and validation rate and what the mask costs in latency
against free text (the first request also pays for compiling the grammar).

    python experiments/02_json_schema_output.py [--mock]
"""

from __future__ import annotations

import time

from vllm_lab import RESPONSE_FORMAT, cli, complete, make_client, messages
from vllm_lab.extract import extract_all
from vllm_lab.tickets import EXTRACT_SYSTEM


def run(base_url: str) -> dict:
    client = make_client(base_url)
    warm = messages(EXTRACT_SYSTEM, "Hola, ¿a qué hora abren?")
    complete(client, warm, max_tokens=60)  # warmup: first request after a cold start
    t0 = time.perf_counter()
    complete(client, warm, max_tokens=60, response_format=RESPONSE_FORMAT)
    first_schema_call_s = time.perf_counter() - t0  # includes compiling the grammar once

    free = extract_all(client)
    schema = extract_all(client, response_format=RESPONSE_FORMAT)
    return {
        "response_format": RESPONSE_FORMAT,
        "first_schema_call_s": round(first_schema_call_s, 4),
        "free_text": {k: free[k] for k in ("parse_rate", "valid_rate", "mean_latency_s",
                                           "mean_completion_tokens")},
        "json_schema": {k: schema[k] for k in ("parse_rate", "valid_rate", "field_agreement",
                                               "mean_latency_s", "mean_completion_tokens")},
        "latency_x_vs_free": round(schema["mean_latency_s"] / max(free["mean_latency_s"], 1e-9),
                                   2),
        "invalid_ids": [r["id"] for r in schema["rows"] if not r["valid"]],
        "sample": schema["rows"][0]["raw"],
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    result.pop("response_format")
    cli.show("Free text vs response_format json_schema", result)


if __name__ == "__main__":
    main()
