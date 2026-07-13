import pandas as pd

from scripts.inputs import (
    get_inputs_lazio_outbreak,
    get_inputs_lazio_underreporting_retro,
)

full_reporting_inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
underreporting_inputs = get_inputs_lazio_underreporting_retro()

existing_decisions = full_reporting_inputs["existing_decisions"]
blood_measures_lifted_date = existing_decisions["blood_resumed_anzio"]["date"]
blood_measures_lifted_outbreak_day = existing_decisions["blood_resumed_anzio"][
    "outbreak_day"
]
final_case_to_blood_measures_lifted = existing_decisions["blood_resumed_anzio"][
    "days_from_final_case"
]

start_date = full_reporting_inputs["start_date"]
time_final_case = full_reporting_inputs["time_final_case"]

example_prob_level = 0.01

reporting_scenarios = [
    ("full reporting", full_reporting_inputs["results_paths"]),
    (
        "under-reporting",
        {
            "autoregressive": underreporting_inputs["results_paths"][
                "autoregressive_p60"
            ],
            "suitability": underreporting_inputs["results_paths"]["suitability_p60"],
        },
    ),
]

for reporting_label, results_paths in reporting_scenarios:
    for model in ["autoregressive", "suitability"]:
        df = pd.read_csv(results_paths[model])
        time_prob_below_example_level = df[
            df["additional_case_prob"] < example_prob_level
        ]["day_of_outbreak"].iloc[0]
        final_case_to_prob_below_example_level = (
            time_prob_below_example_level - time_final_case
        )
        date_prob_below_example_level = start_date + pd.Timedelta(
            days=time_prob_below_example_level
        )
        prob_when_blood_measures_lifted = df[
            df["day_of_outbreak"] == blood_measures_lifted_outbreak_day
        ]["additional_case_prob"].iloc[0]
        print(
            f"\n{model.capitalize()} model ({reporting_label}):\n"
            f"Risk first below {100 * example_prob_level:.2f}% on "
            f"{date_prob_below_example_level:%d-%m-%Y} "
            f"({final_case_to_prob_below_example_level} days after final case)\n"
            f"Risk when blood measures lifted ({blood_measures_lifted_date:%d-%m-%Y}, "
            f"{final_case_to_blood_measures_lifted} days from final case): "
            f"{(100 * prob_when_blood_measures_lifted):.2f}%"
        )
