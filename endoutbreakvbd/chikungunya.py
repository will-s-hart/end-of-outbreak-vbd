import functools

import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import scipy.stats


def get_parameters():
    gen_time_dist_vec = _get_gen_time_dist()
    return {
        "gen_time_dist_vec": gen_time_dist_vec,
    }


def get_data():
    df = pd.read_csv(
        "data/lazio_chik_2017.csv", index_col="onset_date", parse_dates=True
    )
    df["doy"] = df.index.day_of_year
    return df


def get_suitability_data():
    df = pd.read_csv(
        "results/rome_weather_suitability_2017.csv",
        index_col="date",
        parse_dates=True,
    )
    df["doy"] = df.index.day_of_year
    return df


def _get_gen_time_dist():
    gen_time_max = 40
    gen_time_dist_cont = scipy.stats.gamma(a=8.53, scale=1.46)
    gen_time_dist_vec = _discretise_cori(gen_time_dist_cont, max_val=gen_time_max)
    return gen_time_dist_vec


def _discretise_cori(dist_cont, max_val=None, allow_zero=False):
    """
    Function for discretising a continuous distribution using the method
    described in https://doi.org/10.1093/aje/kwt133 (web appendix 11).
    """

    def _integrand_fun(x, y):
        # To get probability mass function at time x, need to integrate this expression
        # with respect to y between y=x-1 and and y=x+1
        return (1 - abs(x - y)) * dist_cont.pdf(y)

    # Set up vector of x values and pre-allocate vector of probabilities
    x_vec = np.arange(0, max_val + 1, dtype=int)
    p_vec = np.zeros(len(x_vec))
    # Calculate probability mass function at each x value
    for i in range(len(x_vec)):  # pylint: disable=consider-using-enumerate
        x = x_vec[i]
        integrand = functools.partial(_integrand_fun, x)
        p_vec[i] = scipy.integrate.quad(
            integrand,
            x - 1 if x > 0 else 1e-12,
            x + 1,
        )[0]
    if not allow_zero:
        # Assign mass from 0 to 1
        x_vec = x_vec[1:]
        p_vec[1] = p_vec[1] + p_vec[0]
        p_vec = p_vec[1:]
    # Assign residual mass to x_max
    p_vec[-1] = p_vec[-1] + 1 - np.sum(p_vec)
    return p_vec
