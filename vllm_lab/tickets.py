"""The app thread of this repo: support tickets from an online store in Guatemala.

`data/tickets.json` holds 20 fixed tickets and the record a person would write for each
one. The prompts live here so every experiment (and the mock server) uses the same words.
"""

from __future__ import annotations

import json
from functools import cache
from pathlib import Path

from vllm_lab.schema import ACTIONS, CATEGORIES, URGENCY

DATA = Path(__file__).resolve().parent.parent / "data" / "tickets.json"

EXTRACT_SYSTEM = (
    "Eres el sistema de soporte de una tienda en línea de Guatemala. Convierte el ticket en un "
    "registro JSON con exactamente estos campos: "
    f"category (uno de {', '.join(CATEGORIES)}), "
    f"urgency (uno de {', '.join(URGENCY)}), "
    "product (el producto mencionado, o null), "
    f"requested_action (uno de {', '.join(ACTIONS)}) y "
    "order_id (con formato GT-123456, o null si el ticket no trae número de pedido). "
    "Responde solo con el JSON."
)

URGENCY_SYSTEM = (
    "Clasifica la urgencia del ticket de soporte. "
    f"Responde solo con la urgencia, una palabra: {', '.join(URGENCY)}."
)

ORDER_ID_SYSTEM = (
    "Responde solo con el número de pedido del ticket, con formato GT-123456. "
    "Si el ticket no trae número de pedido, responde NONE."
)


@cache
def load_tickets() -> tuple[dict, ...]:
    return tuple(json.loads(DATA.read_text(encoding="utf-8")))


def messages(system: str, ticket_text: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Ticket:\n{ticket_text}"},
    ]
