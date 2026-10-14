"""Run from the project root; replace the illustrative bounds for your data."""
import argparse
from src.mars_cleaning import MarsCleaningTools


def checked(result):
    if not result["ok"]:
        raise RuntimeError(result["error"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Notebook CSV or PKL input")
    parser.add_argument("--dataset")
    parser.add_argument("--time-unit", required=True, choices=["s", "ms", "us", "ns", "ps"])
    parser.add_argument("--field-bounds-T", nargs=2, type=float, required=True)
    parser.add_argument("--trigger-time-s", type=float, required=True)
    parser.add_argument("--rank", type=int, required=True)
    args = parser.parse_args()
    tools = MarsCleaningTools()
    source = {"path": args.path, "time_unit": args.time_unit}
    if args.dataset:
        source["dataset"] = args.dataset
    print(checked(tools.dispatch("inspect_epr_2d", {"source": source})))
    corrected = checked(tools.dispatch("correct_epr_2d_baseline", {
        "source": source,
        "baseline": {"field_bounds_T": args.field_bounds_T,
                     "time_bounds_s": [args.trigger_time_s, None]},
        "output": {"name": "baseline_corrected"},
    }))
    canonical = {"path": corrected["artifacts"]["npz"]}
    print(checked(tools.dispatch("inspect_epr_2d", {"source": canonical, "singular_values": 20})))
    filtered = checked(tools.dispatch("denoise_epr_2d", {
        "source": canonical, "denoising": {"method": "svd", "svd_rank": args.rank},
        "output": {"name": "denoised", "formats": ["npz", "pkl"]},
    }))
    print(filtered["artifacts"])


if __name__ == "__main__":
    main()
