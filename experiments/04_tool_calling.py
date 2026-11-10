"""04. The model decides, your code acts: tool calling with the OpenAI `tools` API.

Two tools live in vllm_lab/store.py as plain Python with fake data: lookup_order and
create_return. The model only sees their names, descriptions and JSON parameters. The loop
sends the conversation, runs whatever tool calls come back, appends the results as `tool`
messages and repeats until the model answers in text. Four fixed customer messages check
that it calls the right tool, with the right order id, and refuses what the data forbids.

The server needs tool calling turned on at startup; for Gemma 4 in vLLM 0.33:
    --enable-auto-tool-choice --tool-call-parser gemma4

    python experiments/04_tool_calling.py [--mock]
"""

from __future__ import annotations

import json

from openai import OpenAI

from vllm_lab import cli, complete, make_client
from vllm_lab.store import TOOLS, call_tool

SYSTEM = (
    "Eres el asistente de soporte de una tienda en línea de Guatemala. Usa las herramientas. "
    "Antes de crear una devolución, consulta siempre el pedido con lookup_order y crea la "
    "devolución solo si returnable es true. Responde en español, en una o dos frases."
)
MAX_TURNS = 5

CASES = [
    {"id": "status",
     "message": "Hola, ¿por dónde viene mi pedido GT-104233? Lo pedí el lunes.",
     "must_call": ["lookup_order"], "must_not_call": ["create_return"]},
    {"id": "return_ok",
     "message": "Quiero devolver la licuadora del pedido GT-220871, vino con el vaso rajado.",
     "must_call": ["create_return"], "must_not_call": [], "order_id": "GT-220871"},
    {"id": "return_too_late",
     "message": "Quiero devolver la mochila del pedido GT-990417, ya no me gustó.",
     "must_call": ["lookup_order"], "must_not_call": ["create_return"]},
    {"id": "no_order",
     "message": "¿Tienen tienda física en Xela o solo venden en línea?",
     "must_call": [], "must_not_call": ["lookup_order", "create_return"]},
]


def tool_loop(client: OpenAI, user_message: str) -> dict:
    """Runs the conversation until the model answers without asking for a tool."""
    msgs: list[dict] = [{"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user_message}]
    calls: list[dict] = []
    for turn in range(1, MAX_TURNS + 1):
        # tool_choice "auto": the Gemma 4 parser in vLLM 0.33 does not support
        # "required" or a named tool, so the model decides on its own.
        r = complete(client, msgs, max_tokens=256, tools=TOOLS, tool_choice="auto")
        if not r["tool_calls"]:
            return {"calls": calls, "answer": r["text"].strip(), "turns": turn}
        msgs.append({"role": "assistant", "content": r["text"] or None, "tool_calls": [
            {"id": c["id"], "type": "function",
             "function": {"name": c["name"], "arguments": c["arguments"]}}
            for c in r["tool_calls"]]})
        for c in r["tool_calls"]:
            result = call_tool(c["name"], c["arguments"])
            calls.append({"name": c["name"], "arguments": c["arguments"], "result": result})
            msgs.append({"role": "tool", "tool_call_id": c["id"],
                         "content": json.dumps(result, ensure_ascii=False)})
    return {"calls": calls, "answer": None, "turns": MAX_TURNS}


def judge_case(case: dict, out: dict) -> bool:
    names = [c["name"] for c in out["calls"]]
    ok = out["answer"] is not None
    ok &= all(name in names for name in case["must_call"])
    ok &= not any(name in names for name in case["must_not_call"])
    if "order_id" in case:
        returns = [json.loads(c["arguments"] or "{}") for c in out["calls"]
                   if c["name"] == "create_return"]
        ok &= any(a.get("order_id") == case["order_id"] for a in returns)
    return bool(ok)


def run(base_url: str) -> dict:
    client = make_client(base_url)
    cases = []
    for case in CASES:
        out = tool_loop(client, case["message"])
        cases.append({"id": case["id"], "message": case["message"],
                      "tools_called": [c["name"] for c in out["calls"]],
                      "calls": out["calls"], "answer": out["answer"], "turns": out["turns"],
                      "ok": judge_case(case, out)})
    return {"cases": cases, "passed": sum(c["ok"] for c in cases), "total": len(cases)}


def main() -> None:
    args = cli.parse(__doc__.splitlines()[0])
    result = run(cli.resolve(args))
    for case in result["cases"]:
        cli.show(f"{case['id']}: {case['message']}", {
            k: case[k] for k in ("calls", "answer", "turns", "ok")})
    print(f"\n{result['passed']}/{result['total']} cases called the right tools")


if __name__ == "__main__":
    main()
