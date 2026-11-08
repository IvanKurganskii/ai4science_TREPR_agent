"""Overlay the saved, scaled 1D fit and its experiment in fitdraw_<spectrum_id>.svg."""
import argparse

if __package__ in (None, ""):
    from fit_draw_io import add_draw_arguments, cli_result, output_path, plot_pair, pyplot, read_fit, save_svg
else:
    from .fit_draw_io import add_draw_arguments, cli_result, output_path, plot_pair, pyplot, read_fit, save_svg


def draw_fit_1d(fit_path, output_dir, spectrum_id, fit_id):
    data = read_fit(fit_path, "1d")
    path = output_path(output_dir, spectrum_id)
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    panel = plot_pair(ax, data["fields_T"]*1e3, data["experimental"], data["fit"],
                      "fitdraw_1d_slice_0", spectrum_id, "Magnetic field (mT)")
    return save_svg(fig, path, {"schema": "ai4science.fitdraw.v1", "dimension": "1d",
        "spectrum_id": spectrum_id, "fit_id": fit_id, "fit_source": data["source"],
        "fit_key": data["fit_key"], "normalization": "none_saved_scale", "panels": [panel]})


def _run(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    add_draw_arguments(parser)
    args = parser.parse_args(argv)
    return draw_fit_1d(args.fit, args.output_dir, args.spectrum_id, args.fit_id)


def main(argv=None):
    return cli_result(_run, argv)


if __name__ == "__main__":
    raise SystemExit(main())
