"""Six time-selected spectral overlays; used by draw_fit_2d.py."""
if __package__ in (None, ""):
    from fit_draw_io import finite_array, plot_pair, sample_axis
else:
    from .fit_draw_io import finite_array, plot_pair, sample_axis


def sample_slices(data, times_s, *, time_half_width_s=0.):
    times = finite_array(times_s, "slice times", 1)
    if len(times) != 6 or len(set(times)) != 6:
        raise ValueError("Specify six distinct recording times for spectral slices")
    exp, sampling = sample_axis(data["times_s"], data["experimental"], times, half_width=time_half_width_s)
    fit, _ = sample_axis(data["times_s"], data["fit"], times, half_width=time_half_width_s)
    return {"times_s": times, "experimental": exp, "fit": fit, "sampling": sampling, "fields_T": data["fields_T"]}


def draw_slices(axes, sampled):
    if len(axes) != 6:
        raise ValueError("Spectral drawing needs six axes")
    panels = []
    for i, ax in enumerate(axes):
        time = float(sampled["times_s"][i])
        panel = plot_pair(ax, sampled["fields_T"]*1e3, sampled["experimental"][:, i], sampled["fit"][:, i],
            f"fitdraw_2d_slice_{i}", f"t = {time*1e6:.5g} us", "Magnetic field (mT)")
        panel.update(recording_time_s=time, sampling=sampled["sampling"][i])
        panels.append(panel)
    return panels
