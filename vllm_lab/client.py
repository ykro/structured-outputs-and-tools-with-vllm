"""OpenAI-compatible client pointed at your own vLLM server.

The only thing that changes from a hosted API is BASE_URL. With the Cloud Run tunnel
(`gcloud run services proxy vllm-gemma --region us-east4 --port 8000`) the server is
at http://localhost:8000/v1 and no API key is needed: the tunnel signs each request with
your gcloud identity.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator

import openai
from openai import OpenAI

BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
MODEL = os.environ.get("VLLM_MODEL_NAME", "gemma")

# vLLM 0.33 takes constrained decoding in one request field, `structured_outputs`, with
# exactly one of: json, regex, choice, grammar, json_object. Servers from before the
# rename took `guided_regex`, `guided_choice`, ... as top-level fields instead.
STYLES = ("structured_outputs", "guided")


def make_client(base_url: str | None = None) -> OpenAI:
    # vLLM ignores the key unless it was started with --api-key; the SDK just needs a string.
    return OpenAI(base_url=base_url or BASE_URL, api_key=os.environ.get("VLLM_API_KEY", "local"))


def complete(client: OpenAI, messages: list[dict], max_tokens: int = 200, **kwargs) -> dict:
    """One non-streaming request at temperature 0. Returns text, tool calls, usage, time."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0, seed=7, **kwargs
    )
    wall = time.perf_counter() - t0
    msg = resp.choices[0].message
    return {
        "text": msg.content or "",
        "tool_calls": [
            {"id": c.id, "name": c.function.name, "arguments": c.function.arguments}
            for c in (msg.tool_calls or [])
        ],
        "finish_reason": resp.choices[0].finish_reason,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else 0,
        "wall_s": wall,
    }


def constraint_body(kind: str, value, style: str) -> dict:
    """The extra_body for one constraint (kind is 'choice', 'regex' or 'json')."""
    if style == "structured_outputs":
        return {"structured_outputs": {kind: value}}
    return {f"guided_{kind}": value}


class Constrained:
    """Sends constrained requests, falling back to the older parameter names once if the
    server rejects `structured_outputs` with a 400."""

    def __init__(self, client: OpenAI):
        self.client = client
        self.style = STYLES[0]

    def __call__(self, messages: list[dict], kind: str, value, max_tokens: int = 32) -> dict:
        try:
            return complete(self.client, messages, max_tokens,
                            extra_body=constraint_body(kind, value, self.style))
        except openai.BadRequestError:
            if self.style == STYLES[-1]:
                raise
            self.style = STYLES[-1]
            return complete(self.client, messages, max_tokens,
                            extra_body=constraint_body(kind, value, self.style))


def stream_deltas(client: OpenAI, messages: list[dict], max_tokens: int = 200,
                  **kwargs) -> Iterator[tuple[float, str]]:
    """Streams one answer; yields (seconds since the request, text delta) per chunk."""
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, max_tokens=max_tokens, temperature=0, seed=7,
        stream=True, **kwargs,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield time.perf_counter() - t0, chunk.choices[0].delta.content
