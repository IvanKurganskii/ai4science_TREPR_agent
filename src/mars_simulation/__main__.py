"""CLI and newline-delimited JSON tool-call server (stdout contains only JSON)."""

import argparse
import contextlib
import json
import sys
from pathlib import Path

from .tools import MarsSimulationTools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schemas", action="store_true", help="Print the six function schemas")
    parser.add_argument("--serve", action="store_true", help="One JSON tool call per input/output line")
    parser.add_argument("--request", type=Path, help="JSON file containing name and arguments")
    parser.add_argument("--output-dir", type=Path,
                        help="Output folder (default: <working directory>/results/mars_simulations)")
    args = parser.parse_args()
    tools = MarsSimulationTools(args.output_dir)
    if args.schemas:
        print(json.dumps(tools.schemas, ensure_ascii=True, indent=2))
        return 0

    def execute(request):
        call_id = request.get("id") if isinstance(request, dict) else None
        if not isinstance(request, dict):
            return {"id": call_id, "result": {"ok": False, "error": {"message": "Request must be an object"}}}
        # Some MaRS backends print progress. Keep the transport channel clean.
        with contextlib.redirect_stdout(sys.stderr):
            result = tools.dispatch(request.get("name"), request.get("arguments", {}))
        return {"id": call_id, "result": result}

    if args.serve:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                reply = execute(json.loads(line))
            except (ValueError, TypeError) as exc:
                reply = {"id": None, "result": {"ok": False, "error": {"message": str(exc)}}}
            print(json.dumps(reply, ensure_ascii=True, allow_nan=False), flush=True)
        return 0
    if args.request:
        reply = execute(json.loads(args.request.read_text(encoding="utf-8")))
        print(json.dumps(reply, ensure_ascii=True, indent=2, allow_nan=False))
        return 0 if reply["result"]["ok"] else 1
    parser.error("Choose --schemas, --request FILE or --serve")


if __name__ == "__main__":
    raise SystemExit(main())
