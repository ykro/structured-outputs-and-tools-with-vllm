"""A tiny fake of vLLM's OpenAI-compatible server, for running everything without a GPU.

It speaks the same HTTP as `vllm serve` (/health, /v1/models, /v1/chat/completions with
and without streaming) and understands the four ways this repo asks for an answer:

- free text: it plays a model that was only asked nicely for JSON, so some answers come
  wrapped in Markdown fences, with a friendly sentence in front, or with a value outside
  the schema. Which ticket breaks how is fixed (it depends on the ticket id), never random.
- `response_format={"type": "json_schema", ...}`: always a record that fits the schema,
  like a real server that masks every token the grammar does not allow.
- `structured_outputs` with `choice` or `regex`: always one of the choices / a match.
- `tools`: decides between calling `lookup_order`, `create_return` or answering, from
  the conversation so far, the way a tool-calling model plays the loop.

It recognizes the prompts of this repo and the 20 tickets in data/tickets.json. Its times
are made up on purpose; only the live judge (`make verify-live`) measures a real GPU.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from vllm_lab.tickets import ORDER_ID_SYSTEM, URGENCY_SYSTEM, load_tickets

PREFILL_FIXED_S = 0.004
PREFILL_PER_TOKEN_S = 0.00002
DECODE_PER_TOKEN_S = 0.002
GRAMMAR_COMPILE_S = 0.040   # first request with a new schema/regex/choice list
GRAMMAR_MASK_OVERHEAD = 0.15  # each constrained decode step costs 15 % more
STREAM_CHARS_PER_TOKEN = 4

_compiled: set[str] = set()
_lock = threading.Lock()

_ORDER_RE = re.compile(r"GT-\d{6}")
_RETURN_WORDS = ("devolver", "devolución", "devolverla", "devolverlo", "reembolso")
_SPANISH_URGENCY = {"low": "Baja", "medium": "Media", "high": "Alta"}


def count_tokens(text: str) -> int:
    return max(1, len(text) // STREAM_CHARS_PER_TOKEN)


def _content(m: dict) -> str:
    c = m.get("content")
    return c if isinstance(c, str) else ""


def _find_ticket(prompt: str) -> tuple[int, dict]:
    for t in load_tickets():
        if t["text"] in prompt:
            return int(t["id"][1:]), dict(t["expected"])
    found = _ORDER_RE.search(prompt)
    return 0, {"category": "other", "urgency": "medium", "product": None,
               "requested_action": "information", "order_id": found.group() if found else None}


def _free_record(n: int, record: dict) -> str:
    """A model asked for JSON only in the prompt. Half the tickets come back clean."""
    variant = (n * 5) % 8
    clean = json.dumps(record, ensure_ascii=False)
    if variant == 2:
        return f"```json\n{json.dumps(record, ensure_ascii=False, indent=2)}\n```"
    if variant == 3:
        return f"Claro, aquí está el registro del ticket:\n{clean}"
    if variant == 5:
        return json.dumps({**record, "urgency": "urgente"}, ensure_ascii=False)
    if variant == 6:
        broken = {k: v for k, v in record.items() if k != "requested_action"}
        if record["order_id"]:
            broken["order_id"] = "#" + record["order_id"][3:]
        return json.dumps(broken, ensure_ascii=False)
    return clean


def _free_urgency(n: int, urgency: str) -> str:
    return [urgency, _SPANISH_URGENCY[urgency], f"La urgencia es {urgency}.", urgency][
        (n * 3) % 4]


def _free_order_id(n: int, order_id: str | None) -> str:
    if order_id is None:
        return ["NONE", "No trae número de pedido.", "NONE.", "NONE"][(n * 3) % 4]
    return [order_id, f"Pedido #{order_id[3:]}", f"{order_id}.", order_id][(n * 3) % 4]


def _constrained(spec: dict, record: dict) -> tuple[str, str]:
    """Returns (text, cache key) for one structured_outputs request."""
    if "choice" in spec:
        choices = list(spec["choice"])
        pick = record["urgency"] if record["urgency"] in choices else choices[0]
        return pick, "choice:" + json.dumps(choices)
    if "regex" in spec:
        pattern = spec["regex"]
        for candidate in (record["order_id"], "NONE"):
            if candidate and re.fullmatch(pattern, candidate):
                return candidate, "regex:" + pattern
        return "", "regex:" + pattern
    if "json" in spec:
        return json.dumps(record, ensure_ascii=False), "json:" + json.dumps(spec["json"])
    raise ValueError("structured_outputs needs one of: json, regex, choice")


def _tool_turn(messages: list[dict]) -> dict:
    """One assistant turn of the tool loop: either tool calls or the final answer."""
    user = next(_content(m) for m in messages if m["role"] == "user")
    found = _ORDER_RE.search(user)
    wants_return = any(w in user.lower() for w in _RETURN_WORDS)
    names = {c["id"]: c["function"]["name"]
             for m in messages if m["role"] == "assistant" for c in (m.get("tool_calls") or [])}
    results = {names.get(m.get("tool_call_id"), "?"): json.loads(_content(m) or "{}")
               for m in messages if m["role"] == "tool"}
    n_calls = len(names)

    def call(name: str, args: dict) -> dict:
        return {"content": None, "tool_calls": [{
            "id": f"call_{n_calls + 1}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}

    if found is None:
        return {"content": "Con gusto le ayudamos. Por ahora vendemos solo en línea y enviamos "
                           "a todo el país. Si tiene un pedido, compártanos el número GT-123456."}
    order_id = found.group()
    if "lookup_order" not in results:
        return call("lookup_order", {"order_id": order_id})
    order = results["lookup_order"]
    if wants_return and order.get("returnable") and "create_return" not in results:
        return call("create_return", {"order_id": order_id, "reason": "solicitud del cliente"})
    if "create_return" in results and "return_id" in results["create_return"]:
        text = (f"Listo, creamos la devolución {results['create_return']['return_id']} del "
                f"pedido {order_id}. Le escribiremos para coordinar la recolección.")
    elif "error" in order:
        text = f"No encontramos el pedido {order_id}. ¿Puede confirmar el número?"
    elif wants_return:
        text = (f"El pedido {order_id} se entregó hace {order['delivered_days_ago']} días y ya "
                "pasó el plazo de 30 días para devoluciones. Podemos revisar la garantía.")
    else:
        text = f"Su pedido {order_id} ({order['product']}) está en estado {order['status']}."
    return {"content": text}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the terminal clean
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {})
        elif self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": "gemma", "object": "model", "owned_by": "vllm", "max_model_len": 8192,
                 "root": "google/gemma-4-E4B-it"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        msgs = body["messages"]
        prompt = "".join(_content(m) for m in msgs)
        system = next((_content(m) for m in msgs if m["role"] == "system"), "")
        n, record = _find_ticket(prompt)
        grammar_key = None
        tool_calls = None
        try:
            if body.get("tools"):
                turn = _tool_turn(msgs)
                text, tool_calls = turn["content"] or "", turn.get("tool_calls")
            elif (body.get("response_format") or {}).get("type") == "json_schema":
                schema = body["response_format"]["json_schema"]["schema"]
                text, grammar_key = json.dumps(record, ensure_ascii=False), json.dumps(schema)
            elif body.get("structured_outputs"):
                text, grammar_key = _constrained(body["structured_outputs"], record)
            elif system == URGENCY_SYSTEM:
                text = _free_urgency(n, record["urgency"])
            elif system == ORDER_ID_SYSTEM:
                text = _free_order_id(n, record["order_id"])
            else:
                text = _free_record(n, record)
        except (KeyError, ValueError, TypeError) as exc:
            self._json(400, {"object": "error", "message": str(exc), "type": "BadRequestError",
                             "code": 400})
            return

        ptoks = count_tokens(prompt)
        ctoks = count_tokens(json.dumps(tool_calls) if tool_calls else text)
        usage = {"prompt_tokens": ptoks, "completion_tokens": ctoks,
                 "total_tokens": ptoks + ctoks}
        step = DECODE_PER_TOKEN_S * (1 + (GRAMMAR_MASK_OVERHEAD if grammar_key else 0))
        delay = PREFILL_FIXED_S + PREFILL_PER_TOKEN_S * ptoks
        if grammar_key is not None:
            with _lock:
                if grammar_key not in _compiled:
                    _compiled.add(grammar_key)
                    delay += GRAMMAR_COMPILE_S
        time.sleep(delay)
        finish = "tool_calls" if tool_calls else "stop"

        if not body.get("stream"):
            time.sleep(step * ctoks)
            message = {"role": "assistant", "content": text or None}
            if tool_calls:
                message["tool_calls"] = tool_calls
            self._json(200, {
                "id": "chatcmpl-mock", "object": "chat.completion", "model": "gemma",
                "choices": [{"index": 0, "finish_reason": finish, "message": message}],
                "usage": usage})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        pieces = [text[i:i + STREAM_CHARS_PER_TOKEN]
                  for i in range(0, len(text), STREAM_CHARS_PER_TOKEN)]
        for i, piece in enumerate(pieces):
            if i:
                time.sleep(step)
            self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk",
                         "model": "gemma", "choices": [{"index": 0, "delta": {"content": piece},
                                                        "finish_reason": None}]})
        self._chunk({"id": "chatcmpl-mock", "object": "chat.completion.chunk", "model": "gemma",
                     "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        self._raw(b"data: [DONE]\n\n")
        self._raw(b"")

    def _chunk(self, obj: dict) -> None:
        self._raw(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())

    def _raw(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()


class _QuietServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):  # clients closing keep-alive sockets
        pass


def start(port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Starts the mock in a background thread; returns the server and its /v1 base URL."""
    server = _QuietServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}/v1"


if __name__ == "__main__":
    srv, url = start(8000)
    print(f"Mock vLLM server at {url} (Ctrl+C to stop)")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        srv.shutdown()
