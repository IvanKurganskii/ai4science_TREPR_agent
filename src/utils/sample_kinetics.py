"""Sample experimental and fitted 2D kinetics at declared fields, in common units."""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    from canonical_orientations import add_position_arguments, resolve_positions, validate_positions
    from fit_draw_io import cli_result, fingerprint, read_fit, sample_axis
else:
    from .canonical_orientations import add_position_arguments, resolve_positions, validate_positions
    from .fit_draw_io import cli_result, fingerprint, read_fit, sample_axis


def sample_kinetics(data, positions, *, field_half_width_T=0.):
    positions = validate_positions(positions)
    fields = [p["field_T"] for p in positions]
    exp, sampling = sample_axis(data["fields_T"], data["experimental"], fields,
                                sample_axis=1, half_width=field_half_width_T)
    fit, _ = sample_axis(data["fields_T"], data["fit"], fields,
                         sample_axis=1, half_width=field_half_width_T)
    return {"times_s": data["times_s"].copy(), "experimental": exp, "fit": fit,
            "positions": positions, "sampling": sampling}


def _run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    add_position_arguments(parser)
    parser.add_argument("--field-half-width-T", type=float, default=0.)
    parser.add_argument("--output", type=Path, required=True, help="New NPZ with sampled kinetics, no array output to terminal")
    args = parser.parse_args(argv)
    data = read_fit(args.fit, "2d")
    positions, info = resolve_positions(args)
    sampled = sample_kinetics(data, positions, field_half_width_T=args.field_half_width_T)
    meta = {"source": data["source"], "positions": positions, "position_source": info,
            "sampling": sampled["sampling"], "axis_order": ["time", "position"],
            "time_unit": "s", "field_unit": "T", "time_zero_s": data["time_zero_s"]}
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, times_s=sampled["times_s"], fields_T=np.array([p["field_T"] for p in positions]),
            experimental=sampled["experimental"], fitted=sampled["fit"], metadata_json=np.asarray(json.dumps(meta, allow_nan=False)))
    return {"status": "sampled", "output": fingerprint(args.output), "n_times": len(sampled["times_s"]),
            "n_positions": len(positions), "positions": positions, "sampling": sampled["sampling"]}


def main(argv=None):
    return cli_result(_run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
