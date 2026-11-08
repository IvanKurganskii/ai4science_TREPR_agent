"""One SVG: six field-selected kinetics and six time-selected spectral overlays."""
import argparse

if __package__ in (None, ""):
    from canonical_orientations import add_position_arguments, resolve_positions, validate_positions
    from draw_fit_2d_kinetics import draw_kinetics
    from draw_fit_2d_slices import draw_slices, sample_slices
    from fit_draw_io import add_draw_arguments, cli_result, output_path, pyplot, read_fit, save_svg
    from sample_kinetics import sample_kinetics
else:
    from .canonical_orientations import add_position_arguments, resolve_positions, validate_positions
    from .draw_fit_2d_kinetics import draw_kinetics
    from .draw_fit_2d_slices import draw_slices, sample_slices
    from .fit_draw_io import add_draw_arguments, cli_result, output_path, pyplot, read_fit, save_svg
    from .sample_kinetics import sample_kinetics


def draw_fit_2d(fit_path, output_dir, spectrum_id, fit_id, positions, times_s, *,
                field_half_width_T=0., time_half_width_s=0., position_source=None):
    data = read_fit(fit_path, "2d")
    path = output_path(output_dir, spectrum_id)
    positions = validate_positions(positions, 6)
    kinetics = sample_kinetics(data, positions, field_half_width_T=field_half_width_T)
    slices = sample_slices(data, times_s, time_half_width_s=time_half_width_s)
    plt = pyplot()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13), layout="constrained")
    fig.suptitle(f"{spectrum_id} | {fit_id}\nKinetics (top 6) and spectra (bottom 6)", fontsize=14)
    panels = draw_kinetics(axes.ravel()[:6], kinetics)+draw_slices(axes.ravel()[6:], slices)
    return save_svg(fig, path, {"schema": "ai4science.fitdraw.v1", "dimension": "2d",
        "spectrum_id": spectrum_id, "fit_id": fit_id, "fit_source": data["source"],
        "fit_key": data["fit_key"], "normalization": "none_saved_scale", "panels": panels,
        "position_source": position_source or {"method": "supplied_positions"},
        "time_zero_s": data["time_zero_s"], "comparison_window_s": data["comparison_window_s"]})


def _run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    add_draw_arguments(parser)
    add_position_arguments(parser)
    group = parser.add_mutually_exclusive_group(required=True)
    for unit in ("s", "us", "ns"):
        group.add_argument("--times-"+unit, type=float, nargs=6, help="Six times on the original experimental recording axis")
    parser.add_argument("--field-half-width-T", type=float, default=0.)
    parser.add_argument("--time-half-width-ns", type=float, default=0.)
    args = parser.parse_args(argv)
    positions, info = resolve_positions(args, 6)
    if args.times_s is not None:
        times = args.times_s
    elif args.times_us is not None:
        times = [t*1e-6 for t in args.times_us]
    else:
        times = [t*1e-9 for t in args.times_ns]
    return draw_fit_2d(args.fit, args.output_dir, args.spectrum_id, args.fit_id, positions, times,
                       field_half_width_T=args.field_half_width_T, time_half_width_s=args.time_half_width_ns*1e-9,
                       position_source=info)


def main(argv=None):
    return cli_result(_run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
