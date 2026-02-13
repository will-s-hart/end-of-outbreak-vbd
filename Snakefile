results_files_weather_suitability_data = expand(
    "results/weather_suitability_data/{name}.csv", name=["all", "2017"]
)
results_files_sim_study = expand(
    "results/sim_study/{name}.csv",
    name=[
        "example_outbreak_risk",
        "example_outbreak_declaration",
        "many_outbreak",
    ],
)


def get_results_files_lazio_outbreak(qrt):
    return expand(
        "results/lazio_outbreak{qrt}/{name}.csv",
        qrt=qrt,
        name=["autoregressive", "suitability"],
    )


results_files_lazio_outbreak = get_results_files_lazio_outbreak(qrt=["", "_qrt"])


def get_results_files_inference_test(qrt):
    return expand(
        "results/inference_test{qrt}/{name}.csv",
        qrt=qrt,
        name=[
            "outbreak_data",
            "autoregressive",
            "suitability",
        ],
    )


results_files_inference_test = get_results_files_inference_test(qrt=["", "_qrt"])

results_files = (
    results_files_weather_suitability_data
    + results_files_sim_study
    + results_files_lazio_outbreak
    + results_files_inference_test
)

plot_files_weather_suitability_data = expand(
    "figures/weather_suitability_data/{name}.svg", name=["temperature", "suitability"]
)
plot_files_sim_study = expand(
    "figures/sim_study/{name}.svg",
    name=[
        "rep_no",
        "example_outbreak_risk",
        "example_outbreak_declaration",
        "many_outbreak",
    ],
)


def get_plot_files_lazio_outbreak(qrt):
    return expand(
        "figures/lazio_outbreak{qrt}/{name}.svg",
        qrt=qrt,
        name=[
            "gen_time_dist",
            "rep_no",
            "risk",
            "declaration",
            "suitability",
            "scaling_factor",
        ],
    )


plot_files_lazio_outbreak = get_plot_files_lazio_outbreak(qrt=["", "_qrt"])


def get_plot_files_inference_test(qrt):
    return expand(
        "figures/inference_test{qrt}/{name}.svg",
        qrt=qrt,
        name=[
            "rep_no",
            "risk",
            "declaration",
            "suitability",
            "scaling_factor",
        ],
    )


plot_files_inference_test = get_plot_files_inference_test(qrt=["", "_qrt"])

paper_figure_files = expand(
    "figures/figure_{number}.svg", number=["1", "2", "3", "4", "S1", "S2"]
)
paper_figure_files_png = [x.replace(".svg", ".png") for x in paper_figure_files]


plot_files = (
    plot_files_weather_suitability_data
    + plot_files_sim_study
    + plot_files_lazio_outbreak
    + plot_files_inference_test
)

package_files = [
    f"endoutbreakvbd/{name}.py"
    for name in [
        "__init__",
        "further_case_risk",
        "inference",
        "inputs",
        "model",
        "utils",
    ]
]


wildcard_constraints:
    # allow qrt to be "" (non-QRT) or "_qrt"
    qrt="(_qrt)?",


rule all:
    input:
        paper_figure_files,


rule paper_figures_png:
    input:
        paper_figure_files,
    output:
        paper_figure_files_png,
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
        results_files,


rule paper_figures:
    input:
        plot_files,
    output:
        paper_figure_files,
    shell:
        """
        pixi run python scripts/compile_figures.py
        """


rule weather_suitability_data_results:
    input:
        package_files,
        "scripts/weather_suitability_data.py",
    output:
        results_files_weather_suitability_data,
    shell:
        """
        pixi run python scripts/weather_suitability_data.py -r
        """


rule weather_suitability_data_plots:
    input:
        package_files,
        "scripts/weather_suitability_data_plots.py",
        results_files_weather_suitability_data,
    output:
        plot_files_weather_suitability_data,
    shell:
        """
        pixi run python scripts/weather_suitability_data_plots.py
        """


rule sim_study_results:
    input:
        package_files,
        results_files_weather_suitability_data,
        "scripts/sim_study.py",
    output:
        results_files_sim_study,
    shell:
        """
        pixi run python scripts/sim_study.py -r
        """


rule sim_study_plots:
    input:
        package_files,
        results_files_weather_suitability_data,
        "scripts/sim_study_plots.py",
        results_files_sim_study,
    output:
        plot_files_sim_study,
    shell:
        """
        pixi run python scripts/sim_study_plots.py
        """


rule lazio_outbreak_results:
    input:
        package_files,
        results_files_weather_suitability_data,
        "scripts/lazio_outbreak.py",
    output:
        get_results_files_lazio_outbreak(qrt="{qrt}"),
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
        package_files,
        results_files_weather_suitability_data,
        "scripts/lazio_outbreak_plots.py",
        get_results_files_lazio_outbreak(qrt="{qrt}"),
    output:
        get_plot_files_lazio_outbreak(qrt="{qrt}"),
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
        package_files,
        results_files_weather_suitability_data,
        "scripts/inference_test.py",
        "scripts/lazio_outbreak.py",
    output:
        get_results_files_inference_test(qrt="{qrt}"),
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
        package_files,
        results_files_weather_suitability_data,
        "scripts/inference_test_plots.py",
        "scripts/lazio_outbreak_plots.py",
        get_results_files_inference_test(qrt="{qrt}"),
    output:
        get_plot_files_inference_test(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/inference_test_plots.py {params.qrt_flag}
        """
