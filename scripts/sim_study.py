import argparse
import functools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from endoutbreakvbd import (
    calc_declaration_delay,
    calc_further_case_risk_analytical,
    calc_further_case_risk_simulation,
    run_renewal_model,
)
from endoutbreakvbd.inputs import get_inputs_sim_study
from scripts.sim_study_plots import make_plots


def run_analyses():
    inputs = get_inputs_sim_study()
    rng = np.random.default_rng(2)
    _run_example_outbreak_risk_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        doy_start_vals=inputs["example_outbreak_doy_start_vals"],
        rng=rng,
        save_path=inputs["results_paths"]["example_outbreak_risk"],
    )
    _run_example_outbreak_declaration_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        perc_risk_threshold_vals=inputs["example_outbreak_perc_risk_threshold_vals"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        save_path=inputs["results_paths"]["example_outbreak_declaration"],
    )
    _run_many_outbreak_analysis(
        n_sims=inputs["many_outbreak_n_sims"],
        outbreak_size_threshold=inputs["many_outbreak_outbreak_size_threshold"],
        perc_risk_threshold=inputs["many_outbreak_perc_risk_threshold"],
        rep_no_from_doy_start=inputs["rep_no_from_doy_start"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        rng=rng,
        save_path=inputs["results_paths"]["many_outbreak"],
    )


def _run_example_outbreak_risk_analysis(
    *,
    incidence_vec,
    rep_no_from_doy_start,
    gen_time_dist_vec,
    doy_start_vals,
    rng,
    save_path,
):
    risk_days = np.arange(1, len(incidence_vec) + len(gen_time_dist_vec) + 1, dtype=int)
    risk_days_sim = np.arange(
        1, len(incidence_vec) + len(gen_time_dist_vec) + 1, dtype=int
    )
    risk_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [doy_start_vals, risk_days], names=["start_day_of_year", "day_of_outbreak"]
        )
    )

    for doy_start in doy_start_vals:
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)

        risk_vals = calc_further_case_risk_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days,
        )
        risk_vals_sim = calc_further_case_risk_simulation(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days_sim,
            n_sims=10000,
            rng=rng,
            parallel=True,
        )
        risk_df.loc[(doy_start, risk_days), "analytical"] = risk_vals
        risk_df.loc[(doy_start, risk_days_sim), "simulation"] = risk_vals_sim
    risk_df.to_csv(save_path)


def _run_example_outbreak_declaration_analysis(
    *,
    incidence_vec,
    perc_risk_threshold_vals,
    rep_no_from_doy_start,
    gen_time_dist_vec,
    save_path,
):
    time_last_case = np.nonzero(incidence_vec)[0][-1]
    risk_days = np.arange(
        time_last_case + 1, time_last_case + len(gen_time_dist_vec) + 2, dtype=int
    )

    doy_last_case_vec = np.arange(1, 366, dtype=int)
    declaration_delay_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [perc_risk_threshold_vals, doy_last_case_vec],
            names=["perc_risk_threshold", "final_case_day_of_year"],
        ),
        columns=["delay_to_declaration"],
    )

    for doy_last_case in doy_last_case_vec:
        doy_start = doy_last_case - len(incidence_vec) + 1
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        risk_vals = calc_further_case_risk_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days,
        )
        declaration_delay_vals = calc_declaration_delay(
            risk_vec=risk_vals,
            perc_risk_threshold=perc_risk_threshold_vals,
            delay_of_first_risk=1,
        )
        declaration_delay_df.loc[
            (list(perc_risk_threshold_vals), doy_last_case), "delay_to_declaration"
        ] = declaration_delay_vals
    declaration_delay_df.to_csv(save_path)


def _run_many_outbreak_analysis(
    *,
    n_sims,
    outbreak_size_threshold,
    perc_risk_threshold,
    rep_no_from_doy_start,
    gen_time_dist_vec,
    rng,
    save_path,
):
    tasks = []
    child_rngs = rng.spawn(n_sims)
    for child_rng in child_rngs:
        tasks.append(
            (
                outbreak_size_threshold,
                perc_risk_threshold,
                rep_no_from_doy_start,
                gen_time_dist_vec,
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
    doy_start_vals, doy_last_case_vals, declaration_delay_vals = zip(*results)
    df = pd.DataFrame(
        {
            "first_case_day_of_year": doy_start_vals,
            "final_case_day_of_year": doy_last_case_vals,
            "delay_to_declaration": declaration_delay_vals,
        },
        index=range(n_sims),
    )
    df.to_csv(save_path)


def _many_outbreak_analysis_one_sim(args):
    (
        outbreak_size_threshold,
        perc_risk_threshold,
        rep_no_from_doy_start,
        gen_time_dist_vec,
        rng,
    ) = args
    outbreak_found = False
    while not outbreak_found:
        doy_start = rng.integers(1, 366)
        rep_no_func = functools.partial(rep_no_from_doy_start, doy_start=doy_start)
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            t_stop=1000,
            gen_time_dist_vec=gen_time_dist_vec,
            rng=rng,
            incidence_init=1,
        )
        if np.sum(incidence_vec) >= outbreak_size_threshold:
            outbreak_found = True
    time_last_case = np.nonzero(incidence_vec)[0][-1]
    doy_last_case = doy_start + time_last_case
    risk_days = np.arange(
        time_last_case + 1, time_last_case + len(gen_time_dist_vec) + 2, dtype=int
    )
    risk_vals = calc_further_case_risk_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_func,
        gen_time_dist_vec=gen_time_dist_vec,
        t_calc=risk_days,
    )
    declaration_delay = calc_declaration_delay(
        risk_vec=risk_vals,
        perc_risk_threshold=perc_risk_threshold,
        delay_of_first_risk=1,
    )
    if 150 < doy_last_case < 250 and declaration_delay == 0:
        print("Possible error - zero days to declaration for outbreak ending mid-year")
    return doy_start, doy_last_case, declaration_delay


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args_in = parser.parse_args()
    run_analyses()
    if not args_in.results_only:
        make_plots()
