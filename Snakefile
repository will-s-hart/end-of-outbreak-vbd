import glob

lazio_incidence_path = "data/lazio_chik_2017.csv"
suitability_grid_path = "data/tegar_suitability_grid.csv"

results_paths_schematic = ["results/schematic/outbreak.csv"]
results_paths_weather_suitability_data = expand(
    "results/weather_suitability_data/{name}.csv", name=["all", "2017"]
)
results_paths_sim_study = expand(
    "results/sim_study/{name}.csv",
    name=[
        "example_outbreak_prob",
        "example_outbreak_decision",
        "many_outbreak_example",
        "many_outbreak_decision",
    ],
)


def get_results_paths_lazio_outbreak(qrt):
    return expand(
        "results/lazio_outbreak{qrt}/{name}.csv",
        qrt=qrt,
        name=[
            "autoregressive",
            "autoregressive_diagnostics",
            "suitability",
            "suitability_diagnostics",
        ],
    )


results_paths_lazio_outbreak = get_results_paths_lazio_outbreak(qrt=["", "_qrt"])


def get_results_paths_inference_test(qrt):
    return expand(
        "results/inference_test{qrt}/{name}.csv",
        qrt=qrt,
        name=[
            "outbreak_data",
            "autoregressive",
            "autoregressive_diagnostics",
            "suitability",
            "suitability_diagnostics",
        ],
    )


results_paths_inference_test = get_results_paths_inference_test(qrt=[""])

results_paths_sim_sensitivity = expand(
    "results/sim_sensitivity/{name}.csv",
    name=[
        "rep_no_factor_low",
        "rep_no_factor_high",
        "decay_speed_low",
        "decay_speed_high",
    ],
)
results_paths_lazio_frozen = ["results/lazio_frozen/autoregressive_frozen.csv"]
results_paths_lazio_epiestim = ["results/lazio_epiestim/epiestim.csv"]
results_paths_lazio_underreporting_retro = expand(
    "results/lazio_underreporting_retro/{name}.csv",
    name=[
        "suitability_p60",
        "suitability_p60_diagnostics",
        "autoregressive_p60",
        "autoregressive_p60_diagnostics",
    ],
)
results_paths = (
    results_paths_schematic
    + results_paths_weather_suitability_data
    + results_paths_sim_study
    + results_paths_lazio_outbreak
    + results_paths_inference_test
    + results_paths_sim_sensitivity
    + results_paths_lazio_frozen
    + results_paths_lazio_epiestim
    + results_paths_lazio_underreporting_retro
)

figure_paths_weather_suitability_data = expand(
    "figures/weather_suitability_data/{name}.svg",
    name=["temperature", "suitability_model", "suitability"],
)
figure_paths_sim_study = expand(
    "figures/sim_study/{name}.svg",
    name=[
        "rep_no",
        "example_outbreak_prob",
        "example_outbreak_decision",
        "many_outbreak_example",
        "many_outbreak_decision",
    ],
)


def get_figure_paths_lazio_outbreak(qrt):
    return expand(
        "figures/lazio_outbreak{qrt}/{name}.svg",
        qrt=qrt,
        name=[
            "serial_interval_dist",
            "rep_no",
            "additional_case_prob",
            "decision",
            "suitability",
            "scaling_factor",
        ],
    )


figure_paths_lazio_outbreak = get_figure_paths_lazio_outbreak(qrt=["", "_qrt"])


def get_figure_paths_inference_test(qrt):
    return expand(
        "figures/inference_test{qrt}/{name}.svg",
        qrt=qrt,
        name=[
            "rep_no",
            "additional_case_prob",
            "decision",
            "suitability",
            "scaling_factor",
        ],
    )


figure_paths_inference_test = get_figure_paths_inference_test(qrt=[""])

figure_paths_sim_sensitivity = expand(
    "figures/sim_sensitivity/{name}.svg",
    name=[
        "rep_no_factor_curves",
        "rep_no_factor_low",
        "rep_no_factor_high",
        "decay_speed_curves",
        "decay_speed_low",
        "decay_speed_high",
    ],
)
figure_paths_lazio_frozen = expand(
    "figures/lazio_frozen/{name}.svg",
    name=["rep_no", "additional_case_prob", "decision"],
)
figure_paths_lazio_epiestim = expand(
    "figures/lazio_epiestim/{name}.svg",
    name=["rep_no", "additional_case_prob", "decision"],
)
figure_paths_lazio_underreporting_retro = expand(
    "figures/lazio_underreporting_retro/{name}.svg",
    name=[
        "cases",
        "additional_case_prob",
        "decision",
        "suitability",
        "scaling_factor",
        "rep_no",
        "rep_no_ar",
    ],
)
schematic_figure_path = "figures/figure_1.svg"
compiled_paper_figure_paths = expand(
    "figures/figure_{number}.svg",
    number=["2", "3", "4", "5", "S1", "S2", "S3", "S4", "S5", "S6"],
)
paper_figure_paths = [schematic_figure_path] + compiled_paper_figure_paths
paper_figure_png_paths = [x.replace(".svg", ".png") for x in paper_figure_paths]


figure_paths = (
    figure_paths_weather_suitability_data
    + figure_paths_sim_study
    + figure_paths_lazio_outbreak
    + figure_paths_inference_test
    + figure_paths_sim_sensitivity
    + figure_paths_lazio_frozen
    + figure_paths_lazio_epiestim
    + figure_paths_lazio_underreporting_retro
)

# Globbed rather than listed by hand: every package module is an input to every analysis, and
# an omission here silently stops results being rebuilt when that module changes.
package_paths = sorted(glob.glob("endoutbreakvbd/*.py"))
# Every analysis/plot script also imports parameter assembly from scripts/inputs.py.
shared_input_paths = package_paths + ["scripts/inputs.py"]


wildcard_constraints:
    # allow qrt to be "" (non-QRT) or "_qrt"
    qrt="(_qrt)?",


rule all:
    input:
        paper_figure_paths,


rule paper_figures_png:
    input:
        paper_figure_paths,
    output:
        paper_figure_png_paths,
    shell:
        r"""
        for svg in {input}; do
            png="${{svg%.svg}}.png"
            echo "Converting $svg -> $png"
            inkscape "$svg" --export-type=png --export-filename="$png"
        done
        """


rule results:
    input:
        results_paths,


rule paper_figures:
    input:
        figure_paths,
        "scripts/compile_figures.py",
    output:
        compiled_paper_figure_paths,
    shell:
        """
        pixi run python scripts/compile_figures.py
        """


rule schematic_results:
    input:
        shared_input_paths,
        "scripts/schematic.py",
    output:
        results_paths_schematic,
    shell:
        """
        pixi run python scripts/schematic.py -r
        """


rule schematic_plots:
    input:
        shared_input_paths,
        "scripts/schematic_plots.py",
        results_paths_schematic,
        "figures/schematic/intervention_graphic.png",
        "figures/schematic/safe_graphic.png",
    output:
        schematic_figure_path,
    shell:
        """
        pixi run python scripts/schematic_plots.py
        """


rule weather_suitability_data_results:
    input:
        shared_input_paths,
        suitability_grid_path,
        "scripts/weather_suitability_data.py",
    output:
        results_paths_weather_suitability_data,
    shell:
        """
        pixi run python scripts/weather_suitability_data.py -r
        """


rule weather_suitability_data_plots:
    input:
        shared_input_paths,
        "scripts/weather_suitability_data_plots.py",
        results_paths_weather_suitability_data,
    output:
        figure_paths_weather_suitability_data,
    shell:
        """
        pixi run python scripts/weather_suitability_data_plots.py
        """


rule sim_study_results:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/sim_study.py",
    output:
        results_paths_sim_study,
    shell:
        """
        pixi run python scripts/sim_study.py -r
        """


rule sim_study_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/sim_study_plots.py",
        results_paths_sim_study,
    output:
        figure_paths_sim_study,
    shell:
        """
        pixi run python scripts/sim_study_plots.py
        """


rule lazio_outbreak_results:
    input:
        shared_input_paths,
        lazio_incidence_path,
        results_paths_weather_suitability_data,
        "scripts/lazio_outbreak.py",
    output:
        get_results_paths_lazio_outbreak(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/lazio_outbreak.py -r {params.qrt_flag}
        """


rule lazio_outbreak_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/lazio_outbreak_plots.py",
        get_results_paths_lazio_outbreak(qrt="{qrt}"),
    output:
        get_figure_paths_lazio_outbreak(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/lazio_outbreak_plots.py {params.qrt_flag}
        """


rule inference_test_results:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/inference_test.py",
        "scripts/lazio_outbreak.py",
    output:
        get_results_paths_inference_test(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/inference_test.py -r {params.qrt_flag}
        """


rule inference_test_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/inference_test_plots.py",
        "scripts/lazio_outbreak_plots.py",
        get_results_paths_inference_test(qrt="{qrt}"),
    output:
        get_figure_paths_inference_test(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/inference_test_plots.py {params.qrt_flag}
        """


rule sim_sensitivity_results:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/sim_sensitivity.py",
        "scripts/sim_study.py",
    output:
        results_paths_sim_sensitivity,
    shell:
        """
        pixi run python scripts/sim_sensitivity.py -r
        """


rule sim_sensitivity_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        "scripts/sim_sensitivity_plots.py",
        "scripts/sim_study_plots.py",
        results_paths_sim_sensitivity,
    output:
        figure_paths_sim_sensitivity,
    shell:
        """
        pixi run python scripts/sim_sensitivity_plots.py
        """


rule lazio_frozen_results:
    input:
        shared_input_paths,
        lazio_incidence_path,
        results_paths_weather_suitability_data,
        "scripts/lazio_frozen.py",
        "scripts/lazio_outbreak.py",
    output:
        results_paths_lazio_frozen,
    shell:
        """
        pixi run python scripts/lazio_frozen.py -r
        """


rule lazio_frozen_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        get_results_paths_lazio_outbreak(qrt=""),
        results_paths_lazio_frozen,
        "scripts/lazio_frozen_plots.py",
        "scripts/lazio_outbreak_plots.py",
    output:
        figure_paths_lazio_frozen,
    shell:
        """
        pixi run python scripts/lazio_frozen_plots.py
        """


rule lazio_epiestim_results:
    input:
        shared_input_paths,
        lazio_incidence_path,
        results_paths_weather_suitability_data,
        "scripts/lazio_epiestim.py",
    output:
        results_paths_lazio_epiestim,
    shell:
        """
        pixi run python scripts/lazio_epiestim.py -r
        """


rule lazio_epiestim_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        get_results_paths_lazio_outbreak(qrt=""),
        results_paths_lazio_epiestim,
        "scripts/lazio_epiestim_plots.py",
        "scripts/lazio_outbreak_plots.py",
    output:
        figure_paths_lazio_epiestim,
    shell:
        """
        pixi run python scripts/lazio_epiestim_plots.py
        """


rule lazio_underreporting_retro_results:
    input:
        shared_input_paths,
        lazio_incidence_path,
        results_paths_weather_suitability_data,
        "scripts/lazio_underreporting_retro.py",
    output:
        results_paths_lazio_underreporting_retro,
    shell:
        """
        pixi run python scripts/lazio_underreporting_retro.py -r
        """


rule lazio_underreporting_retro_plots:
    input:
        shared_input_paths,
        results_paths_weather_suitability_data,
        results_paths_lazio_underreporting_retro,
        get_results_paths_lazio_outbreak(qrt=""),
        "scripts/lazio_underreporting_retro_plots.py",
    output:
        figure_paths_lazio_underreporting_retro,
    shell:
        """
        pixi run python scripts/lazio_underreporting_retro_plots.py
        """
