import argparse
import functools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from endoutbreakvbd import (
    calc_additional_case_prob_analytical,
    calc_additional_case_prob_simulation,
    calc_decision_delay,
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
        results_path=inputs["results_paths"]["example_outbreak_prob"],
    )
    _run_example_outbreak_decision_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        risk_threshold_pct_vals=inputs["example_outbreak_risk_threshold_pct_vals"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        results_path=inputs["results_paths"]["example_outbreak_decision_delay"],
    )
    _run_many_outbreak_analysis(
        n_sims=inputs["many_outbreak_n_sims"],
        outbreak_min_size=inputs["many_outbreak_min_size"],
        risk_threshold_pct_vals=inputs["many_outbreak_risk_threshold_pct_vals"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        example_outbreak_idx=inputs["many_outbreak_example_idx"],
        track_premature_decisions=track_premature_decisions,
        rng=rng,
        results_path=inputs["results_paths"]["many_outbreak_decision_delay"],
        example_outbreak_results_path=inputs["results_paths"]["many_outbreak_example"],
    )


def _run_example_outbreak_prob_analysis(
    *,
    incidence_vec,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    doy_start_vals,
    n_sims,
    rng,
    results_path,
):
    t_calc_vec = np.arange(1, len(incidence_vec) + len(serial_interval_dist_vec) + 1)
    prob_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [doy_start_vals, t_calc_vec],
            names=["initial_case_day_of_year", "day_of_outbreak"],
        )
    )

    for doy_start in doy_start_vals:
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)

        prob_vec = calc_additional_case_prob_analytical(
            incidence=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc_vec,
        )
        prob_sim_vec = calc_additional_case_prob_simulation(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc_vec,
            n_sims=n_sims,
            rng=rng,
            parallel=True,
        )
        prob_df.loc[(doy_start, t_calc_vec), "analytical"] = prob_vec
        prob_df.loc[(doy_start, t_calc_vec), "simulation"] = prob_sim_vec
    prob_df.to_csv(results_path)


def _run_example_outbreak_decision_analysis(
    *,
    incidence_vec,
    risk_threshold_pct_vals,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    results_path,
):
    t_final_case = np.nonzero(incidence_vec)[0][-1]
    t_calc_vec = np.arange(
        t_final_case + 1, t_final_case + len(serial_interval_dist_vec) + 2
    )

    doy_final_case_vec = np.arange(1, 366)
    decision_delay_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [risk_threshold_pct_vals, doy_final_case_vec],
            names=["risk_threshold_pct", "final_case_day_of_year"],
        ),
        columns=["delay_to_decision"],
    )

    for doy_final_case in doy_final_case_vec:
        doy_start = doy_final_case - len(incidence_vec) + 1
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        prob_vec = calc_additional_case_prob_analytical(
            incidence=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc_vec,
        )
        decision_delay_vec = calc_decision_delay(
            prob_vec=prob_vec,
            t_vec=t_calc_vec,
            risk_threshold_pct=risk_threshold_pct_vals,
            t_final_case=t_final_case,
        )
        decision_delay_df.loc[
            (list(risk_threshold_pct_vals), doy_final_case), "delay_to_decision"
        ] = decision_delay_vec
    decision_delay_df.to_csv(results_path)


def _run_many_outbreak_analysis(
    *,
    n_sims,
    outbreak_min_size,
    risk_threshold_pct_vals,
    rep_no_from_doy_start,
    serial_interval_dist_vec,
    track_premature_decisions,
    rng,
    results_path,
    example_outbreak_idx=None,
    example_outbreak_results_path=None,
):
    simulation_tasks = []
    return_outbreak_df_flags = [
        simulation_idx == example_outbreak_idx for simulation_idx in range(n_sims)
    ]
    child_rngs = rng.spawn(n_sims)
    for child_rng, return_outbreak_df in zip(child_rngs, return_outbreak_df_flags):
        simulation_tasks.append(
            (
                outbreak_min_size,
                risk_threshold_pct_vals,
                rep_no_from_doy_start,
                serial_interval_dist_vec,
                track_premature_decisions,
                return_outbreak_df,
                child_rng,
            )
        )
    simulation_results = list(
        tqdm(
            Parallel(
                n_jobs=-1,
                prefer="processes",
                return_as="generator",
                batch_size="auto",
            )(
                delayed(_many_outbreak_analysis_one_sim)(task)
                for task in simulation_tasks
            ),
            total=n_sims,
            desc="Simulating outbreaks",
        )
    )
    (
        doy_start_results,
        n_cases_results,
        doy_final_case_results,
        decision_delay_results,
        outbreak_dfs,
        n_premature_decisions_results,
    ) = zip(*simulation_results)
    doy_start_vec = np.asarray(doy_start_results)
    n_cases_vec = np.asarray(n_cases_results)
    doy_final_case_vec = np.asarray(doy_final_case_results)
    decision_delay_mat = np.stack(decision_delay_results)
    n_premature_decisions_vec = np.asarray(n_premature_decisions_results)
    results_df = pd.DataFrame(
        {
            "initial_case_day_of_year": doy_start_vec,
            "number_of_cases": n_cases_vec,
            "final_case_day_of_year": doy_final_case_vec,
        },
        index=range(n_sims),
    )
    # One decision-delay column per risk threshold (delays stacked rows x thresholds).
    for j, risk_threshold_pct in enumerate(risk_threshold_pct_vals):
        results_df[f"delay_to_decision_{risk_threshold_pct}"] = decision_delay_mat[:, j]
    results_df.to_csv(results_path)
    if example_outbreak_idx is not None:
        example_outbreak_df = outbreak_dfs[example_outbreak_idx]
        example_outbreak_df.to_csv(example_outbreak_results_path)
    if track_premature_decisions:
        n_outbreaks_with_premature_decision = np.sum(n_premature_decisions_vec > 0)
        print(
            f"{n_outbreaks_with_premature_decision} "
            "outbreaks had a premature decision "
            f"({100 * n_outbreaks_with_premature_decision / n_sims:.1f}%)"
        )
        n_premature_decisions = np.sum(n_premature_decisions_vec)
        print(
            f"{n_premature_decisions} premature decisions across all outbreaks "
            f"({n_premature_decisions / n_sims:.1f} per outbreak)"
        )


def _many_outbreak_analysis_one_sim(args):
    (
        outbreak_min_size,
        risk_threshold_pct_vals,
        rep_no_from_doy_start,
        serial_interval_dist_vec,
        track_premature_decisions,
        return_outbreak_df,
        rng,
    ) = args
    is_outbreak_found = False
    while not is_outbreak_found:
        doy_start = rng.integers(1, 366)
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            t_stop=1000,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rng=rng,
            incidence_init=1,
        )
        if np.sum(incidence_vec) >= outbreak_min_size:
            is_outbreak_found = True
    t_final_case = np.nonzero(incidence_vec)[0][-1]
    doy_final_case = doy_start + t_final_case
    t_calc_vec = np.arange(
        t_final_case + 1, t_final_case + len(serial_interval_dist_vec) + 2
    )
    prob_vec = calc_additional_case_prob_analytical(
        incidence=incidence_vec,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc_vec,
    )
    decision_delay_vec = calc_decision_delay(
        prob_vec=prob_vec,
        t_vec=t_calc_vec,
        risk_threshold_pct=risk_threshold_pct_vals,
        t_final_case=t_final_case,
    )
    if 150 < doy_final_case < 250 and np.any(decision_delay_vec == 0):
        print("Possible error - zero days to decision for outbreak ending mid-year")
    n_cases = np.sum(incidence_vec)
    outbreak_df = None
    n_premature_decisions = None
    if return_outbreak_df or track_premature_decisions:
        prob_through_final_case_vec = calc_additional_case_prob_analytical(
            incidence=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=np.arange(t_final_case + 1),
        )
    if return_outbreak_df:
        prob_full_vec = np.concatenate([prob_through_final_case_vec, prob_vec])
        outbreak_df = pd.DataFrame(
            {
                "day_of_year": np.arange(doy_start, doy_start + len(incidence_vec)),
                "incidence": incidence_vec,
                "additional_case_prob": prob_full_vec[: len(incidence_vec)],
            }
        )
    if track_premature_decisions:
        # Use the most lenient threshold, which triggers a decision earliest.
        n_premature_decisions = np.sum(
            prob_through_final_case_vec < (max(risk_threshold_pct_vals) / 100)
        )
    return (
        doy_start,
        n_cases,
        doy_final_case,
        decision_delay_vec,
        outbreak_df,
        n_premature_decisions,
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
    args = parser.parse_args()
    run_analyses(track_premature_decisions=args.track_premature_decisions)
    if not args.results_only:
        make_plots()
