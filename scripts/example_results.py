import pandas as pd

from scripts.inputs import (
    get_inputs_lazio_outbreak,
    get_inputs_lazio_underreporting_retro,
)

full_reporting_inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
underreporting_inputs = get_inputs_lazio_underreporting_retro()

existing_decisions = full_reporting_inputs["existing_decisions"]
blood_measures_lifted_date = existing_decisions["blood_resumed_anzio"]["decision_date"]
t_blood_measures_lifted = existing_decisions["blood_resumed_anzio"]["t"]
days_after_final_case_when_blood_measures_lifted = existing_decisions[
    "blood_resumed_anzio"
]["days_after_final_case"]

outbreak_start_date = full_reporting_inputs["outbreak_start_date"]
t_final_case = full_reporting_inputs["t_final_case"]

example_additional_case_prob = 0.01

reporting_scenario_specs = [
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

for reporting_label, results_paths in reporting_scenario_specs:
    for model in ["autoregressive", "suitability"]:
        results_df = pd.read_csv(results_paths[model])
        t_prob_below_example_level = results_df[
            results_df["additional_case_prob"] < example_additional_case_prob
        ]["day_of_outbreak"].iloc[0]
        days_after_final_case_when_prob_below_example_level = (
            t_prob_below_example_level - t_final_case
        )
        prob_below_example_level_date = outbreak_start_date + pd.Timedelta(
            days=t_prob_below_example_level
        )
        prob_when_blood_measures_lifted = results_df[
            results_df["day_of_outbreak"] == t_blood_measures_lifted
        ]["additional_case_prob"].iloc[0]
        print(
            f"\n{model.capitalize()} model ({reporting_label}):\n"
            f"Risk first below {100 * example_additional_case_prob:.2f}% on "
            f"{prob_below_example_level_date:%d-%m-%Y} "
            f"({days_after_final_case_when_prob_below_example_level} days after final case)\n"
            f"Risk when blood measures lifted ({blood_measures_lifted_date:%d-%m-%Y}, "
            f"{days_after_final_case_when_blood_measures_lifted} days from final case): "
            f"{(100 * prob_when_blood_measures_lifted):.2f}%"
        )
