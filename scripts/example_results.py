import pandas as pd

from endoutbreakvbd.inputs import get_inputs_lazio_outbreak

inputs = get_inputs_lazio_outbreak(quasi_real_time=False)

existing_declarations = inputs["existing_declarations"]
blood_measures_lifted_date = existing_declarations["blood_resumed_anzio"]["date"]
blood_measures_lifted_outbreak_day = existing_declarations["blood_resumed_anzio"][
    "outbreak_day"
]
last_case_to_blood_measures_lifted = existing_declarations["blood_resumed_anzio"][
    "days_from_last_case"
]

start_date = inputs["start_date"]
time_last_case = inputs["time_last_case"]

example_risk_level = 0.05

for model in ["autoregressive", "suitability"]:
    results_path = inputs["results_paths"][model]
    df = pd.read_csv(results_path)
    time_risk_below_example_level = df[df["further_case_risk"] < example_risk_level][
        "day_of_outbreak"
    ].iloc[0]
    last_case_to_risk_below_example_level = (
        time_risk_below_example_level - time_last_case
    )
    date_risk_below_example_level = start_date + pd.Timedelta(
        days=time_risk_below_example_level
    )
    risk_when_blood_measures_lifted = df[
        df["day_of_outbreak"] == blood_measures_lifted_outbreak_day
    ]["further_case_risk"].iloc[0]
    print(
        f"\n{model.capitalize()} model:\n"
        f"Risk first below {100 * example_risk_level:.2f}% on "
        f"{date_risk_below_example_level:%d-%m-%Y} "
        f"({last_case_to_risk_below_example_level} days after last case)\n"
        f"Risk when blood measures lifted ({blood_measures_lifted_date:%d-%m-%Y}, "
        f"{last_case_to_blood_measures_lifted} days from last case): "
        f"{(100 * risk_when_blood_measures_lifted):.2f}%"
    )
