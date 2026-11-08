"""Six kinetic overlay panels; used by draw_fit_2d.py on one shared figure."""
if __package__ in (None, ""):
    from fit_draw_io import plot_pair
    from sample_kinetics import sample_kinetics
else:
    from .fit_draw_io import plot_pair
    from .sample_kinetics import sample_kinetics


def draw_kinetics(axes, sampled):
    if len(axes) != 6 or len(sampled["positions"]) != 6:
        raise ValueError("Kinetic drawing needs six axes and six field positions")
    panels = []
    for i, (ax, position) in enumerate(zip(axes, sampled["positions"])):
        panel = plot_pair(ax, sampled["times_s"]*1e6, sampled["experimental"][:, i], sampled["fit"][:, i],
            f"fitdraw_2d_kinetic_{i}", f"{position['label']}: B = {position['field_T']*1e3:.4g} mT", "Recording time (us)")
        panel.update(position=position, sampling=sampled["sampling"][i])
        panels.append(panel)
    return panels
