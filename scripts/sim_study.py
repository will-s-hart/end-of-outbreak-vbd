import argparse
import functools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from endoutbreakvbd import (
    calc_decision_delay,
    calc_additional_case_prob_analytical,
    calc_additional_case_prob_simulation,
)
from endoutbreakvbd.model import run_renewal_model
from scripts.inputs import get_inputs_sim_study
from scripts.sim_study_plots import make_plots


def run_analyses(track_premature_decisions=False):
    inputs = get_inputs_sim_study()
    rng = np.random.default_rng(2)
    _run_example_outbreak_prob_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        doy_start_vals=inputs["example_outbreak_doy_start_vals"],
        n_sims=inputs["example_outbreak_n_sims"],
        rng=rng,
        save_path=inputs["results_paths"]["example_outbreak_prob"],
    )
    _run_example_outbreak_decision_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        perc_risk_threshold_vals=inputs["example_outbreak_perc_risk_threshold_vals"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        save_path=inputs["results_paths"]["example_outbreak_decision"],
    )
    _run_many_outbreak_analysis(
        n_sims=inputs["many_outbreak_n_sims"],
        outbreak_size_threshold=inputs["many_outbreak_outbreak_size_threshold"],
        perc_risk_threshold_vals=inputs["many_outbreak_perc_risk_threshold_vals"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        example_outbreak_idx=inputs["many_outbreak_example_outbreak_idx"],
        track_premature_decisions=track_premature_decisions,
        rng=rng,
        save_path=inputs["results_paths"]["many_outbreak_decision"],
        save_path_example=inputs["results_paths"]["many_outbreak_example"],
    )


def _run_example_outbreak_prob_analysis(
    *,
    incidence_vec,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    doy_start_vals,
    n_sims,
    rng,
    save_path,
):
    prob_days = np.arange(1, len(incidence_vec) + len(serial_interval_dist_vec) + 1)
    prob_days_sim = np.arange(1, len(incidence_vec) + len(serial_interval_dist_vec) + 1)
    prob_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [doy_start_vals, prob_days],
            names=["initial_case_day_of_year", "day_of_outbreak"],
        )
    )

    for doy_start in doy_start_vals:
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)

        prob_vals = calc_additional_case_prob_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=prob_days,
        )
        prob_vals_sim = calc_additional_case_prob_simulation(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=prob_days_sim,
            n_sims=n_sims,
            rng=rng,
            parallel=True,
        )
        prob_df.loc[(doy_start, prob_days), "analytical"] = prob_vals
        prob_df.loc[(doy_start, prob_days_sim), "simulation"] = prob_vals_sim
    prob_df.to_csv(save_path)


def _run_example_outbreak_decision_analysis(
    *,
    incidence_vec,
    perc_risk_threshold_vals,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    save_path,
):
    time_final_case = np.nonzero(incidence_vec)[0][-1]
    prob_days = np.arange(
        time_final_case + 1, time_final_case + len(serial_interval_dist_vec) + 2
    )

    doy_final_case_vec = np.arange(1, 366)
    decision_delay_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [perc_risk_threshold_vals, doy_final_case_vec],
            names=["perc_risk_threshold", "final_case_day_of_year"],
        ),
        columns=["delay_to_decision"],
    )

    for doy_final_case in doy_final_case_vec:
        doy_start = doy_final_case - len(incidence_vec) + 1
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        prob_vals = calc_additional_case_prob_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=prob_days,
        )
        decision_delay_vals = calc_decision_delay(
            prob_vec=prob_vals,
            perc_risk_threshold=perc_risk_threshold_vals,
            delay_of_first_prob=1,
        )
        decision_delay_df.loc[
            (list(perc_risk_threshold_vals), doy_final_case), "delay_to_decision"
        ] = decision_delay_vals
    decision_delay_df.to_csv(save_path)


def _run_many_outbreak_analysis(
    *,
    n_sims,
    outbreak_size_threshold,
    perc_risk_threshold_vals,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    track_premature_decisions,
    rng,
    save_path,
    example_outbreak_idx=None,
    save_path_example=None,
):
    tasks = []
    full_output_flags = [
        True if i == example_outbreak_idx else False for i in range(n_sims)
    ]
    child_rngs = rng.spawn(n_sims)
    for child_rng, full_output in zip(child_rngs, full_output_flags):
        tasks.append(
            (
                outbreak_size_threshold,
                perc_risk_threshold_vals,
                rep_no_from_doy_start,
                serial_interval_dist_vec,
                track_premature_decisions,
                full_output,
                child_rng,
            )
        )
    results = list(
        tqdm(
            Parallel(
                n_jobs=-1,
                prefer="processes",
                return_as="generator",
                batch_size="auto",
            )(delayed(_many_outbreak_analysis_one_sim)(task) for task in tasks),
            total=n_sims,
            desc="Simulating outbreaks",
        )
    )
    (
        doy_start_vals,
        no_cases_vals,
        doy_final_case_vals,
        decision_delay_vals,
        output_vals,
        premature_decisions_vals,
    ) = zip(*results)
    df = pd.DataFrame(
        {
            "initial_case_day_of_year": doy_start_vals,
            "number_of_cases": no_cases_vals,
            "final_case_day_of_year": doy_final_case_vals,
        },
        index=range(n_sims),
    )
    # One decision-delay column per risk threshold (delays stacked rows x thresholds).
    decision_delays = np.array(decision_delay_vals)
    for j, perc_risk_threshold in enumerate(perc_risk_threshold_vals):
        df[f"delay_to_decision_{perc_risk_threshold}"] = decision_delays[:, j]
    df.to_csv(save_path)
    if example_outbreak_idx is not None:
        example_output = output_vals[example_outbreak_idx]
        example_output.to_csv(save_path_example)
    if track_premature_decisions:
        n_premature_decision_outbreaks = np.sum(np.array(premature_decisions_vals) > 0)
        print(
            f"{n_premature_decision_outbreaks} "
            "outbreaks had a premature decision "
            f"({100 * n_premature_decision_outbreaks / n_sims:.1f}%)"
        )
        n_premature_decisions = np.sum(premature_decisions_vals)
        print(
            f"{n_premature_decisions} premature decisions across all outbreaks "
            f"({n_premature_decisions / n_sims:.1f} per outbreak)"
        )


def _many_outbreak_analysis_one_sim(args):
    (
        outbreak_size_threshold,
        perc_risk_threshold_vals,
        rep_no_from_doy_start,
        serial_interval_dist_vec,
        track_premature_decisions,
        full_output,
        rng,
    ) = args
    outbreak_found = False
    while not outbreak_found:
        doy_start = rng.integers(1, 366)
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            t_stop=1000,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rng=rng,
            incidence_init=1,
        )
        if np.sum(incidence_vec) >= outbreak_size_threshold:
            outbreak_found = True
    time_final_case = np.nonzero(incidence_vec)[0][-1]
    doy_final_case = doy_start + time_final_case
    prob_days = np.arange(
        time_final_case + 1, time_final_case + len(serial_interval_dist_vec) + 2
    )
    prob_vals = calc_additional_case_prob_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=prob_days,
    )
    decision_delay = np.atleast_1d(
        calc_decision_delay(
            prob_vec=prob_vals,
            perc_risk_threshold=perc_risk_threshold_vals,
            delay_of_first_prob=1,
        )
    )
    if 150 < doy_final_case < 250 and np.any(decision_delay == 0):
        print("Possible error - zero days to decision for outbreak ending mid-year")
    no_cases = np.sum(incidence_vec)
    output = None
    premature_decisions = None
    if full_output or track_premature_decisions:
        prob_vals_to_final_case = calc_additional_case_prob_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=np.arange(time_final_case + 1),
        )
    if full_output:
        prob_vals_all = np.concatenate([prob_vals_to_final_case, prob_vals])
        output = pd.DataFrame(
            {
                "day_of_year": np.arange(doy_start, doy_start + len(incidence_vec)),
                "cases": incidence_vec,
                "additional_case_prob": prob_vals_all[: len(incidence_vec)],
            }
        )
    if track_premature_decisions:
        # Use the most lenient threshold, which triggers a decision earliest.
        premature_decisions = np.sum(
            prob_vals_to_final_case < (max(perc_risk_threshold_vals) / 100)
        )
    return (
        doy_start,
        no_cases,
        doy_final_case,
        decision_delay,
        output,
        premature_decisions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument(
        "-t",
        "--track-premature-decisions",
        action="store_true",
        help="Track whether any outbreaks had a premature decision",
    )
    args_in = parser.parse_args()
    run_analyses(track_premature_decisions=args_in.track_premature_decisions)
    if not args_in.results_only:
        make_plots()
