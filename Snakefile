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

figure_files_weather_suitability_data = expand(
    "figures/weather_suitability_data/{name}.svg", name=["temperature", "suitability"]
)
figure_files_sim_study = expand(
    "figures/sim_study/{name}.svg",
    name=[
        "rep_no",
        "example_outbreak_risk",
        "example_outbreak_declaration",
        "many_outbreak",
    ],
)


def get_figure_files_lazio_outbreak(qrt):
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


figure_files_lazio_outbreak = get_figure_files_lazio_outbreak(qrt=["", "_qrt"])


def get_figure_files_inference_test(qrt):
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


figure_files_inference_test = get_figure_files_inference_test(qrt=["", "_qrt"])


figure_files = (
    figure_files_weather_suitability_data
    + figure_files_sim_study
    + figure_files_lazio_outbreak
    + figure_files_inference_test
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
        results_files + figure_files,


rule figures:
    input:
        figure_files,


rule results:
    input:
        results_files,


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
        figure_files_weather_suitability_data,
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
        figure_files_sim_study,
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
        get_figure_files_lazio_outbreak(qrt="{qrt}"),
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
        get_figure_files_inference_test(qrt="{qrt}"),
    params:
        qrt_flag=lambda wildcards: (
            "--quasi-real-time" if wildcards.qrt == "_qrt" else ""
        ),
    shell:
        """
        pixi run python scripts/inference_test_plots.py {params.qrt_flag}
        """
