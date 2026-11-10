"""The Outside Judge for structured outputs and tools: runs the five experiments once and checks what must hold.

    python verify_structured.py          # live vLLM through the tunnel (real numbers)
    python verify_structured.py --mock   # fake server, checks the logic without a GPU

The checks are about behavior: with a schema every record validates (and never fewer than
with free text), enums and regex are always respected, the tool loop calls the right tool
for each fixed case, and a streamed record parses at the end. Rates for free text and
latencies are reported, not judged; they change with the model and the run.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from vllm_lab.client import BASE_URL

EXPERIMENTS = [
    "01_free_text_breaks_parsers",
    "02_json_schema_output",
    "03_enums_and_regex",
    "04_tool_calling",
    "05_streaming_json_to_ui",
]


SHORT = {"lookup_order": "lookup", "create_return": "return"}


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def server_version(base_url: str) -> str:
    try:
        root = base_url.rsplit("/v1", 1)[0]
        with urllib.request.urlopen(root + "/version", timeout=30) as r:
            return json.loads(r.read()).get("version", "?")
    except (OSError, ValueError):
        return "mock"


def pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--mock", action="store_true")
    p.add_argument("--base-url", default=BASE_URL)
    p.add_argument("--verbose", action="store_true", help="print each experiment's raw result")
    args = p.parse_args()

    from vllm_lab import cli, load_tickets

    base_url = cli.resolve(args)
    res = {name: load(name).run(base_url) for name in EXPERIMENTS}
    r1, r2, r3, r4, r5 = (res[n] for n in EXPERIMENTS)
    if args.verbose:
        for name, value in res.items():
            cli.show(name, value)

    js = r2["json_schema"]
    rows = [
        ("01. JSON asked in the prompt",
         f"parse {r1['parsed']}/{r1['tickets']}, valid {r1['valid']}/{r1['tickets']}",
         f"{pct(r1['valid_rate'])} valid", r1["tickets"] == len(load_tickets())),
        ("02. response_format json_schema",
         f"valid {pct(js['valid_rate'])} vs {pct(r1['valid_rate'])} free",
         f"latency {r2['latency_x_vs_free']:.2f}x free",
         js["valid_rate"] == 1.0 and js["valid_rate"] >= r1["valid_rate"]),
        ("03. choice and regex",
         (f"enum {pct(r3['urgency_choice_in_enum'])}, regex {pct(r3['order_regex_matches'])}"),
         (f"free {pct(r3['urgency_free_in_enum'])} / {pct(r3['order_free_matches'])}"),
         r3["urgency_choice_in_enum"] == 1.0 and r3["order_regex_matches"] == 1.0),
        ("04. Tool calling loop",
         ", ".join("+".join(SHORT.get(t, t) for t in c["tools_called"]) or "-"
                    for c in r4["cases"]),
         f"{r4['passed']}/{r4['total']} cases", r4["passed"] == r4["total"]),
        ("05. Streaming JSON to a UI",
         f"first field {r5['first_field_s'] or 0:.3f} s of {r5['total_s']:.3f} s",
         "parses at end" if r5["final_valid"] else "broken at end",
         r5["final_valid"] and r5["first_field_s"] is not None
         and r5["first_field_s"] < r5["total_s"]),
    ]

    mode = "mock (fake server, numbers are not real)" if args.mock else "live"
    version = server_version(base_url)
    lines = [
        "=" * 100,
        "THE OUTSIDE JUDGE — Structured Outputs and Tools with vLLM",
        "=" * 100,
        f"Mode    : {mode}",
        (f"Server  : vLLM {version} | constrained params via '{r3['param_style']}' | "
         f"first schema request {r2['first_schema_call_s']:.3f} s"),
        f"Client  : Python {platform.python_version()} | {platform.system()} {platform.machine()}",
        "-" * 100,
        f"{'Experiment':32} | {'Measured':34} | {'Result':20} | Check",
        "-" * 100,
    ]
    for name, measured, result, ok in rows:
        lines.append(f"{name:32} | {measured:34} | {result:20} | {'PASS' if ok else 'FAIL'}")
    lines += [
        "-" * 100,
        (f"Agreement with hand labels (informative): schema {pct(js['field_agreement'])} of "
         f"fields, urgency {pct(r3['urgency_choice_agrees'])}, order id "
         f"{pct(r3['order_regex_agrees'])}."),
        ("A schema guarantees the shape of the answer; whether the values are right is a "
         "separate check."),
    ]
    all_ok = all(ok for *_, ok in rows)
    lines.append("ALL CHECKS PASSED." if all_ok else "SOME CHECKS FAILED.")
    report = "\n".join(lines)
    print(report)

    if not args.mock:
        out = ROOT / "benchmarks"
        out.mkdir(exist_ok=True)
        payload = {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "vllm_version": version,
            "experiments": res,
            "checks": {name: ok for name, *_, ok in rows},
        }
        (out / "latest_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        (out / "terminal_output.txt").write_text(report + "\n")
        print("Saved benchmarks/latest_results.json and benchmarks/terminal_output.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
