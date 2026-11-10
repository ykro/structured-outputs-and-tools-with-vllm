"""The two tools of experiment 04, implemented locally in Python with fake data.

The model never touches this data. It only asks, through the OpenAI `tools` API, for a
function to be called with some arguments; this code decides whether that is allowed.
"""

from __future__ import annotations

import json

RETURN_WINDOW_DAYS = 30

ORDERS: dict[str, dict] = {
    "GT-104233": {"product": "audífonos inalámbricos", "status": "in_transit",
                  "city": "Mixco", "delivered_days_ago": None},
    "GT-220871": {"product": "licuadora", "status": "delivered",
                  "city": "Guatemala", "delivered_days_ago": 3},
    "GT-990417": {"product": "mochila", "status": "delivered",
                  "city": "Antigua Guatemala", "delivered_days_ago": 47},
    "GT-504826": {"product": "estufa", "status": "delivered",
                  "city": "Guatemala", "delivered_days_ago": 1},
}


def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id)
    if order is None:
        return {"order_id": order_id, "error": "order_not_found"}
    days = order["delivered_days_ago"]
    returnable = days is not None and days <= RETURN_WINDOW_DAYS
    return {"order_id": order_id, **order, "returnable": returnable}


def create_return(order_id: str, reason: str) -> dict:
    order = lookup_order(order_id)
    if "error" in order:
        return order
    if not order["returnable"]:
        return {"order_id": order_id, "error": "outside_return_window"}
    return {"order_id": order_id, "return_id": "DEV-" + order_id[3:], "reason": reason,
            "status": "created"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Consulta el estado de un pedido: producto, si ya se entregó y si "
                           "todavía se puede devolver (returnable).",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string",
                                            "description": "Número de pedido, GT-123456"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Crea una devolución. Solo para pedidos con returnable true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Número de pedido, GT-123456"},
                    "reason": {"type": "string", "description": "Motivo en pocas palabras"},
                },
                "required": ["order_id", "reason"],
            },
        },
    },
]

IMPLEMENTATIONS = {"lookup_order": lookup_order, "create_return": create_return}


def call_tool(name: str, arguments: str) -> dict:
    """Runs one tool call exactly as the model asked for it. Bad input becomes an error."""
    if name not in IMPLEMENTATIONS:
        return {"error": f"unknown_tool:{name}"}
    try:
        kwargs = json.loads(arguments or "{}")
        return IMPLEMENTATIONS[name](**kwargs)
    except (ValueError, TypeError) as exc:
        return {"error": f"bad_arguments:{exc}"}
