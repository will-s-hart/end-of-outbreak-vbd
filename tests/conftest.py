import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg", force=True)


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


@pytest.fixture
def posterior_factory():
    def _factory(var_names=("rep_no",), n_times=3, n_chains=1, n_draws=4, fill=1.0):
        coords = {
            "chain": np.arange(n_chains),
            "draw": np.arange(n_draws),
            "time": np.arange(n_times),
        }
        data_vars = {}
        for i, var_name in enumerate(var_names):
            posterior_samples = np.full(
                (n_chains, n_draws, n_times), fill + i, dtype=float
            )
            data_vars[var_name] = (("chain", "draw", "time"), posterior_samples)
        return xr.Dataset(data_vars, coords=coords)

    return _factory
