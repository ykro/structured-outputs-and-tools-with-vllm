"""05. Streaming a structured answer: what the UI can show before the end.

Streams one ticket record with response_format json_schema. After every chunk it tries
to parse the text received so far: the fields whose value is already closed are safe to
render, and the field still being written can be shown as "typing". At the end the
whole text must parse and validate, like any non-streamed answer.

    python experiments/05_streaming_json_to_ui.py [--mock]
"""

from __future__ import annotations

from vllm_lab import (
    RESPONSE_FORMAT,
    check_text,
    cli,
    load_tickets,
    make_client,
    messages,
    parse_partial,
    stream_deltas,
)
from vllm_lab.schema import FIELDS
from vllm_lab.tickets import EXTRACT_SYSTEM

TICKET_ID = "T19"


def run(base_url: str) -> dict:
    client = make_client(base_url)
    ticket = next(t for t in load_tickets() if t["id"] == TICKET_ID)
    text = ""
    arrived: dict[str, float] = {}
    frames = []
    chunks = 0
    ttft_s = None
    for t, delta in stream_deltas(client, messages(EXTRACT_SYSTEM, ticket["text"]),
                                  max_tokens=200, response_format=RESPONSE_FORMAT):
        ttft_s = t if ttft_s is None else ttft_s
        chunks += 1
        text += delta
        done = parse_partial(text) or {}
        for field in FIELDS:
            if field in done and field not in arrived:
                arrived[field] = round(t, 4)
        typing = parse_partial(text, close_strings=True) or {}
        frames.append({"t_s": round(t, 4), "complete": sorted(done),
                       "typing": {k: v for k, v in typing.items() if k not in done}})
    total_s = frames[-1]["t_s"] if frames else 0.0
    verdict = check_text(text)
    # Keep a handful of frames for the screen: one each time a field starts being written
    # and one each time a field becomes complete.
    shown, last = [], None
    for f in frames:
        state = (len(f["complete"]), tuple(f["typing"]))
        if state != last:
            shown.append(f)
            last = state
    return {
        "ticket": ticket["text"],
        "chunks": chunks,
        "ttft_s": round(ttft_s or 0.0, 4),
        "total_s": total_s,
        "field_arrival_s": arrived,
        "first_field_s": min(arrived.values()) if arrived else None,
        "frames": shown,
        "final_text": text,
        "final_parsed": verdict["parsed"],
        "final_valid": verdict["valid"],
    }


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    cli.show("What the UI can render while the JSON streams", result["frames"])
    cli.show("Summary", {k: result[k] for k in ("chunks", "ttft_s", "first_field_s",
                                                 "total_s", "field_arrival_s",
                                                 "final_parsed", "final_valid")})


if __name__ == "__main__":
    main()
