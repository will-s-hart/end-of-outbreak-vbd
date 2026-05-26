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
    def _factory(var_names=("rep_no",), time=3, chain=1, draw=4, fill=1.0):
        coords = {
            "chain": np.arange(chain),
            "draw": np.arange(draw),
            "time": np.arange(time),
        }
        data_vars = {}
        for i, var_name in enumerate(var_names):
            values = np.full((chain, draw, time), fill + i, dtype=float)
            data_vars[var_name] = (("chain", "draw", "time"), values)
        return xr.Dataset(data_vars, coords=coords)

    return _factory
