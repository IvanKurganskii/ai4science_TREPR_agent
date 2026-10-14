"""CLI and newline-delimited JSON tool-call transport."""
import argparse
import contextlib
import json
import sys
from pathlib import Path
from .tools import MarsCleaningTools


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--schemas", action="store_true")
    mode.add_argument("--request", type=Path)
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--working-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, help="Default: <working-dir>/results/mars_simulations")
    args = parser.parse_args()
    tools = MarsCleaningTools(args.output_dir, working_dir=args.working_dir)
    if args.schemas:
        print(json.dumps(tools.schemas, indent=2, ensure_ascii=True))
        return 0

    def execute(request):
        if not isinstance(request, dict):
            return {"id": None, "result": {"ok": False, "error": {"message": "Request must be an object"}}}
        with contextlib.redirect_stdout(sys.stderr):
            result = tools.dispatch(request.get("name"), request.get("arguments", {}))
        return {"id": request.get("id"), "result": result}

    if args.request:
        result = execute(json.loads(args.request.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2, ensure_ascii=True, allow_nan=False))
        return 0 if result["result"]["ok"] else 1
    for line in sys.stdin:
        if line.strip():
            try:
                result = execute(json.loads(line))
            except (ValueError, TypeError) as exc:
                result = {"id": None, "result": {"ok": False, "error": {"message": str(exc)}}}
            print(json.dumps(result, ensure_ascii=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
