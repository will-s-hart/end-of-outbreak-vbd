RESULTS_FILES_WEATHER_SUITABILITY_DATA = [
    f"results/weather_suitability_data/{name}.csv" for name in ["all", "2017"]
]
RESULTS_FILES_SIM_STUDY = [
    f"results/sim_study/{name}.csv"
    for name in [
        "example_outbreak_risk",
        "example_outbreak_declaration",
        "many_outbreak",
    ]
]
RESULTS_FILES_LAZIO_OUTBREAK = [
    f"results/lazio_outbreak/{name}.csv" for name in ["autoregressive", "suitability"]
]

RESULTS_FILES = (
    RESULTS_FILES_WEATHER_SUITABILITY_DATA
    + RESULTS_FILES_SIM_STUDY
    + RESULTS_FILES_LAZIO_OUTBREAK
)

FIGURE_FILES_WEATHER_SUITABILITY_DATA = [
    f"figures/weather_suitability_data/{name}.svg"
    for name in ["temperature", "suitability"]
]
FIGURE_FILES_SIM_STUDY = [
    f"figures/sim_study/{name}.svg"
    for name in [
        "rep_no",
        "example_outbreak_risk",
        "example_outbreak_declaration",
        "many_outbreak",
    ]
]
FIGURE_FILES_LAZIO_OUTBREAK = [
    f"figures/lazio_outbreak/{name}.svg"
    for name in [
        "gen_time_dist",
        "rep_no",
        "risk",
        "declaration",
        "suitability",
        "scaling_factor",
    ]
]

FIGURE_FILES = (
    FIGURE_FILES_WEATHER_SUITABILITY_DATA
    + FIGURE_FILES_SIM_STUDY
    + FIGURE_FILES_LAZIO_OUTBREAK
)

PACKAGE_FILES = [
    f"endoutbreakvbd/{name}.py"
    for name in [
        "__init__",
        "chikungunya",
        "further_case_risk",
        "inference",
        "model",
        "utils",
    ]
]


rule all:
    input:
        RESULTS_FILES + FIGURE_FILES,


rule figures:
    input:
        FIGURE_FILES,


rule results:
    input:
        RESULTS_FILES,


rule weather_suitability_data_results:
    input:
        PACKAGE_FILES,
        "scripts/weather_suitability_data.py",
    output:
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
    shell:
        """
        pixi run python scripts/weather_suitability_data.py -r
        """


rule weather_suitability_data_figures:
    input:
        PACKAGE_FILES,
        "scripts/weather_suitability_data.py",
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
    output:
        FIGURE_FILES_WEATHER_SUITABILITY_DATA,
    shell:
        """
        pixi run python scripts/weather_suitability_data.py -p
        """


rule sim_study_results:
    input:
        PACKAGE_FILES,
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
        "scripts/sim_study.py",
    output:
        RESULTS_FILES_SIM_STUDY,
    shell:
        """
        pixi run python scripts/sim_study.py -r
        """


rule sim_study_figures:
    input:
        PACKAGE_FILES,
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
        "scripts/sim_study.py",
        RESULTS_FILES_SIM_STUDY,
    output:
        FIGURE_FILES_SIM_STUDY,
    shell:
        """
        pixi run python scripts/sim_study.py -p
        """


rule lazio_outbreak_results:
    input:
        PACKAGE_FILES,
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
        "scripts/lazio_outbreak.py",
    output:
        RESULTS_FILES_LAZIO_OUTBREAK,
    shell:
        """
        pixi run python scripts/lazio_outbreak.py -r
        """


rule lazio_outbreak_figures:
    input:
        PACKAGE_FILES,
        RESULTS_FILES_WEATHER_SUITABILITY_DATA,
        "scripts/lazio_outbreak.py",
        RESULTS_FILES_LAZIO_OUTBREAK,
    output:
        FIGURE_FILES_LAZIO_OUTBREAK,
    shell:
        """
        pixi run python scripts/lazio_outbreak.py -p
        """
