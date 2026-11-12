"""01. Asking for JSON in the prompt is only a wish.

Sends the 20 support tickets with a system prompt that says "respond only with the JSON"
and nothing else. Every answer goes through what an app would do: json.loads, then the
schema validator. Counts how many survive and shows why the others did not.

    python experiments/01_free_text_breaks_parsers.py          # live server (tunnel open)
    python experiments/01_free_text_breaks_parsers.py --mock   # no GPU
"""

from __future__ import annotations

from vllm_lab import cli, make_client
from vllm_lab.extract import extract_all


def run(base_url: str) -> dict:
    client = make_client(base_url)
    result = extract_all(client)  # no response_format: the prompt is the only constraint
    result["failures"] = [
        {"id": r["id"], "error": r["error"], "raw": r["raw"]}
        for r in result["rows"] if not r["valid"]
    ]
    # What is still wrong after a careful app strips fences and chatter: values outside the
    # schema, the failure that passes the parser and breaks something later.
    result["still_invalid_after_unwrap"] = [
        {"id": r["id"], "error": r["error_after_unwrap"], "raw": r["raw"]}
        for r in result["rows"] if not r["valid_after_unwrap"]
    ]
    return result


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    cli.show("Free text, JSON asked for in the prompt", {
        k: result[k] for k in ("tickets", "parsed", "valid", "parse_rate", "valid_rate",
                               "valid_after_unwrap", "mean_latency_s", "mean_completion_tokens")})
    cli.show("What broke", result["failures"])
    cli.show("Still invalid after removing fences and chatter", result["still_invalid_after_unwrap"])


if __name__ == "__main__":
    main()
