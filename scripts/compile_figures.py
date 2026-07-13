import pathlib

import numpy as np
import svgutils.transform as svgt


def compile_figure(
    save_path,
    panel_paths,
    template_path=None,
    sz=None,
    tiling=None,
    panel_sz=(500, 450),
    panel_offset=(0, -40),
    panel_positions=None,
    panel_scalings=None,
    label_strings=None,
    label_size=20,
    label_offset=(5, 20),
    label_positions=None,
):
    """
    Create a figure from a set of panels.

    Parameters
    ----------
    save_path : pathlib.Path
        Path to save the figure.
    panel_paths : list of pathlib.Path
        Paths to the panels to include in the figure.
    template_path : pathlib.Path, optional
        Path to an SVG template file onto which the panels will be placed. If None, no
        template is used.
    sz : tuple of int, optional
        Size of the figure in pixels (width, height). If None, the size is calculated
        based on the number of panels and the panel size.
    tiling : tuple of int, optional
        Number of rows and columns to tile the panels in the figure. If None, the
        number of rows and columns is calculated automatically.
    panel_sz : tuple of int, optional
        Size of the panels in pixels (width, height) (default is (500, 450))
    panel_offset : tuple of int, optional
        Offset of the panels from the top-left corner of the figure in pixels (x, y)
        (default is (0, -40)).
    panel_positions : list of tuple of int, optional
        Positions of the panels in the figure in pixels (x, y). Overrides other settings
        determining panel positions if provided.
    panel_scalings : list of float, optional
        Scaling factors for the panels. If not provided, the panels are not scaled.
    label_strings : list of str, optional
        Strings to use as labels for the panels. If not provided, labels are generated
        automatically as A., B., C., etc.
    label_size : int, optional
        Font size of the labels (default is 20).
    label_offset : tuple of int, optional
        Offset of the labels from the top-left corner of the figure in pixels (x, y)
        (default is (5, 20)).
    label_positions : list of tuple of int, optional
        Positions of the labels in the figure in pixels (x, y). Overrides other settings
        determining label positions if provided.

    Returns
    -------
    None
    """
    no_panels = len(panel_paths)
    if tiling is None:
        rows = 1 + (no_panels - 1) // 3
        cols = int(np.ceil(no_panels / rows))
    else:
        cols, rows = tiling
    if sz is None:
        sz = (panel_sz[0] * cols, panel_sz[1] * rows)
    if panel_positions is None:
        panel_positions = [
            (
                panel_offset[0] + panel_sz[0] * (i % cols),
                panel_offset[1] + panel_sz[1] * (i // cols),
            )
            for i in range(no_panels)
        ]
    if panel_scalings is None:
        panel_scalings = [1] * no_panels
    if label_strings is None:
        label_strings = [chr(65 + i) + "." for i in range(no_panels)]
    if label_positions is None:
        label_positions = [
            (
                label_offset[0] + panel_sz[0] * (i % cols),
                label_offset[1] + panel_sz[1] * (i // cols),
            )
            for i in range(no_panels)
        ]
    # create new SVG figure
    if template_path is not None:
        fig = svgt.fromfile(template_path)
    else:
        fig = svgt.SVGFigure()
        fig.set_size((str(sz[0]) + "px", str(sz[1]) + "px"))
    # load matpotlib-generated figures
    panels = []
    for path in panel_paths:
        if path is not None:
            panel = svgt.fromfile(path).getroot()
        else:
            panel = svgt.TextElement(0, 0, "")
        panels.append(panel)
    for panel, position, scaling in zip(panels, panel_positions, panel_scalings):
        panel.moveto(position[0], position[1], scale_x=scaling)
    # add text labels
    labels = [
        svgt.TextElement(
            position[0], position[1], string, size=label_size, font="Arial"
        )
        for string, position in zip(label_strings, label_positions)
    ]
    # append plots and labels to figure
    fig.append(panels + labels)
    # save generated SVG files
    save_path.parent.mkdir(exist_ok=True, parents=True)
    fig.save(save_path)


def compile_paper_figures():
    figure_dir = pathlib.Path(__file__).parents[1] / "figures"
    # Panels with twin axes or two-line ylabels overrun the default slot and overlap
    # their neighbours; give those figures a wider panel slot.
    wide_panel_sz = (520, 450)
    # Figure 2
    compile_figure(
        save_path=figure_dir / "figure_2.svg",
        panel_paths=[
            figure_dir / "sim_study" / f"{x}.svg"
            for x in [
                "rep_no",
                "example_outbreak_prob",
                "many_outbreak_example",
                "many_outbreak_decision",
            ]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure 3
    compile_figure(
        save_path=figure_dir / "figure_3.svg",
        panel_paths=[
            figure_dir / f"{x}.svg"
            for x in [
                "weather_suitability_data/temperature",
                "weather_suitability_data/suitability_model",
                "weather_suitability_data/suitability",
                "lazio_outbreak/serial_interval_dist",
            ]
        ],
    )
    # Figure 4
    compile_figure(
        save_path=figure_dir / "figure_4.svg",
        panel_paths=[
            figure_dir / "lazio_outbreak" / f"{x}.svg"
            for x in ["suitability", "rep_no", "additional_case_prob", "decision"]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure 5 (retrospective under-reporting): latent true cases, additional-case probability,
    # and decision delay, with full-reporting benchmarks on the latter two panels.
    compile_figure(
        save_path=figure_dir / "figure_5.svg",
        panel_paths=[
            figure_dir / "lazio_underreporting_retro" / f"{x}.svg"
            for x in ["cases", "additional_case_prob", "decision"]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S1
    compile_figure(
        save_path=figure_dir / "figure_S1.svg",
        panel_paths=[
            figure_dir / "sim_sensitivity" / f"{x}.svg"
            for x in [
                "rep_no_factor_curves",
                "rep_no_factor_low",
                "rep_no_factor_high",
                "decay_speed_curves",
                "decay_speed_low",
                "decay_speed_high",
            ]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S2
    compile_figure(
        save_path=figure_dir / "figure_S2.svg",
        panel_paths=[
            figure_dir / "lazio_outbreak" / "scaling_factor.svg",
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S3 (Lazio R_t-method comparison): frozen autoregressive (A-C) vs EpiEstim
    # (D-F), each showing R_t, additional-case probability and decision delay.
    compile_figure(
        save_path=figure_dir / "figure_S3.svg",
        panel_paths=[
            figure_dir / "lazio_frozen" / f"{x}.svg"
            for x in ["rep_no", "additional_case_prob", "decision"]
        ]
        + [
            figure_dir / "lazio_epiestim" / f"{x}.svg"
            for x in ["rep_no", "additional_case_prob", "decision"]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S4 (full-reporting quasi-real-time analysis)
    compile_figure(
        save_path=figure_dir / "figure_S4.svg",
        panel_paths=[
            figure_dir / "lazio_outbreak_qrt" / f"{x}.svg"
            for x in [
                "suitability",
                "scaling_factor",
                "rep_no",
                "additional_case_prob",
                "decision",
            ]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S5 (retrospective under-reporting — inference diagnostics): suitability / R_t-factor /
    # R_t posteriors from the suitability fit and the R_t posterior from the autoregressive fit,
    # each overlaid with the full-reporting (no under-reporting) estimate.
    compile_figure(
        save_path=figure_dir / "figure_S5.svg",
        panel_paths=[
            figure_dir / "lazio_underreporting_retro" / f"{x}.svg"
            for x in ["suitability", "scaling_factor", "rep_no", "rep_no_ar"]
        ],
        panel_sz=wide_panel_sz,
    )
    # Figure S6 (inference test)
    compile_figure(
        save_path=figure_dir / "figure_S6.svg",
        panel_paths=[
            figure_dir / "inference_test" / f"{x}.svg"
            for x in [
                "suitability",
                "scaling_factor",
                "rep_no",
                "additional_case_prob",
                "decision",
            ]
        ],
        panel_sz=wide_panel_sz,
    )


if __name__ == "__main__":
    compile_paper_figures()
