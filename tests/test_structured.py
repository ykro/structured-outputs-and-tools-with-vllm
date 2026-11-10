"""Runs the helpers and every experiment against the built-in mock server (no GPU needed)."""

import importlib.util
import json
from pathlib import Path

import pytest

from vllm_lab import TICKET_SCHEMA, check_text, load_tickets, mock_server, parse_partial, validate
from vllm_lab.store import call_tool, create_return, lookup_order

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def base_url():
    server, url = mock_server.start()
    yield url
    server.shutdown()


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_hand_label_fits_the_schema():
    tickets = load_tickets()
    assert len(tickets) == 20
    assert all(validate(t["expected"], TICKET_SCHEMA) == [] for t in tickets)


def test_validator_catches_enum_pattern_missing_and_extra():
    good = dict(load_tickets()[0]["expected"])
    assert validate({**good, "urgency": "urgente"}, TICKET_SCHEMA)
    assert validate({**good, "order_id": "#104233"}, TICKET_SCHEMA)
    assert validate({k: v for k, v in good.items() if k != "product"}, TICKET_SCHEMA)
    assert validate({**good, "notes": "x"}, TICKET_SCHEMA)
    assert validate({**good, "order_id": None}, TICKET_SCHEMA) == []


def test_check_text_separates_unparseable_from_invalid():
    assert not check_text("```json\n{}\n```")["parsed"]
    r = check_text('{"category": "other"}')
    assert r["parsed"] and not r["valid"]


def test_parse_partial_only_returns_closed_fields():
    text = '{"category": "payment", "urgency": "hi'
    assert parse_partial(text) == {"category": "payment"}
    assert parse_partial(text, close_strings=True) == {"category": "payment", "urgency": "hi"}
    assert parse_partial('{"a": "x, y", "b": nu') == {"a": "x, y"}
    assert parse_partial("Claro") is None


def test_tools_refuse_what_the_data_forbids():
    assert lookup_order("GT-220871")["returnable"]
    assert create_return("GT-990417", "x")["error"] == "outside_return_window"
    assert call_tool("drop_tables", "{}")["error"].startswith("unknown_tool")
    assert "bad_arguments" in call_tool("lookup_order", json.dumps({"id": 1}))["error"]


def test_01_free_text_breaks_some_parsers(base_url):
    r = load("01_free_text_breaks_parsers").run(base_url)
    assert r["tickets"] == 20 and r["valid"] < r["parsed"] < r["tickets"]
    assert r["failures"]


def test_02_schema_validates_everything(base_url):
    r = load("02_json_schema_output").run(base_url)
    assert r["json_schema"]["valid_rate"] == 1.0 > r["free_text"]["valid_rate"]


def test_03_choice_and_regex_always_hold(base_url):
    r = load("03_enums_and_regex").run(base_url)
    assert r["urgency_choice_in_enum"] == r["order_regex_matches"] == 1.0
    assert r["urgency_free_in_enum"] < 1.0 and r["param_style"] == "structured_outputs"


def test_04_tool_loop_calls_the_right_tools(base_url):
    r = load("04_tool_calling").run(base_url)
    assert r["passed"] == r["total"] == 4
    by_id = {c["id"]: c["tools_called"] for c in r["cases"]}
    assert by_id["return_ok"] == ["lookup_order", "create_return"]
    assert by_id["no_order"] == []


def test_05_stream_shows_fields_before_the_end(base_url):
    r = load("05_streaming_json_to_ui").run(base_url)
    assert r["final_valid"] and r["first_field_s"] < r["total_s"]
    assert len(r["field_arrival_s"]) == 5
