# Note that AI tools were used to generate tests

from typing import Any, cast

import numpy as np
import pytest

import endoutbreakvbd.additional_case_prob as acp
from endoutbreakvbd._types import FloatArray


def test_calc_additional_case_prob_analytical_t_calc_zero_is_one():
    prob = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        t_calc=0,
    )
    assert prob == 1


def test_calc_additional_case_prob_analytical_after_t_max_is_zero():
    prob = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=lambda t: 1.0,
        serial_interval_dist_vec=[1.0],
        t_calc=2,
    )
    assert prob == 0


def test_calc_additional_case_prob_analytical_vectorized_t_calc_shape():
    prob = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=lambda t: np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0, 1, 2], dtype=int),
    )
    assert isinstance(prob, np.ndarray)
    assert prob.shape == (3,)
    np.testing.assert_allclose(prob, np.array([1.0, 0.0, 0.0]))


def test_calc_additional_case_prob_analytical_scalar_closed_form_case():
    prob = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=lambda t: 0.5,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    assert prob == pytest.approx(1 - np.exp(-0.5))


def test_calc_additional_case_prob_analytical_handles_posterior_matrix_rep_no():
    def rep_no_func(t):
        t_vec = np.atleast_1d(t)
        return np.column_stack(
            [
                np.full(t_vec.size, 0.2, dtype=float),
                np.full(t_vec.size, 0.6, dtype=float),
            ]
        )

    prob = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    expected = 1 - np.mean(np.exp(-np.array([0.2, 0.6])))
    assert prob == pytest.approx(expected)


def test_calc_additional_case_prob_analytical_matrix_incidence_equals_per_sample_mean():
    # A (time, sample) incidence matrix with a scalar rep_no must equal the mean over the
    # per-sample 1-D probabilities.
    rng = np.random.default_rng(0)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1  # keep an index case in every sample
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func(t):
        return 0.5

    for t_calc in (1, 2, 3):
        prob_average = acp.calc_additional_case_prob_analytical(
            incidence=incidence_mat,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc,
        )
        per_sample_prob_vals = [
            acp.calc_additional_case_prob_analytical(
                incidence=incidence_mat[:, s],
                rep_no_func=rep_no_func,
                serial_interval_dist_vec=serial_interval_dist_vec,
                t_calc=t_calc,
            )
            for s in range(incidence_mat.shape[1])
        ]
        assert prob_average == pytest.approx(float(np.mean(per_sample_prob_vals)))


def test_calc_additional_case_prob_analytical_matrix_incidence_and_rep_align():
    # When both incidence and rep_no carry a sample dimension, draw s of the cases must be
    # paired with draw s of the reproduction number (aligned, not an outer product).
    rng = np.random.default_rng(1)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1
    rep_no_mat = rng.uniform(0.1, 0.9, size=(20, 4))
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func_mat(t):
        return rep_no_mat[np.asarray(t)]

    for t_calc in (1, 2, 3):
        prob_average = acp.calc_additional_case_prob_analytical(
            incidence=incidence_mat,
            rep_no_func=rep_no_func_mat,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc,
        )
        per_sample_prob_vals = [
            acp.calc_additional_case_prob_analytical(
                incidence=incidence_mat[:, s],
                rep_no_func=lambda t, s=s: rep_no_mat[np.asarray(t), s],
                serial_interval_dist_vec=serial_interval_dist_vec,
                t_calc=t_calc,
            )
            for s in range(incidence_mat.shape[1])
        ]
        assert prob_average == pytest.approx(float(np.mean(per_sample_prob_vals)))


def test_calc_additional_case_prob_analytical_singleton_sample_matches_1d():
    # A (time, 1) incidence column must give the same probability as the 1-D series.
    incidence_vec = np.array([1, 2, 0, 1, 0])
    serial_interval_dist_vec = [0.5, 0.3, 0.2]
    prob_from_vec = acp.calc_additional_case_prob_analytical(
        incidence=incidence_vec,
        rep_no_func=lambda t: 0.4,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=2,
    )
    prob_from_col = acp.calc_additional_case_prob_analytical(
        incidence=incidence_vec.reshape(-1, 1),
        rep_no_func=lambda t: 0.4,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=2,
    )
    assert prob_from_col == pytest.approx(prob_from_vec)


def test_calc_additional_case_prob_analytical_broadcast_keeps_samples():
    # additional_dims="broadcast" returns per-sample probabilities; averaging them must
    # reproduce the default "average" result.
    rng = np.random.default_rng(2)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1
    rep_no_mat = rng.uniform(0.1, 0.9, size=(20, 4))
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func(t):
        return rep_no_mat[np.asarray(t)]

    t_calc_vec = np.array([1, 2, 3], dtype=int)
    per_sample_prob_mat = acp.calc_additional_case_prob_analytical(
        incidence=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc_vec,
        additional_dims="broadcast",
    )
    averaged = acp.calc_additional_case_prob_analytical(
        incidence=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc_vec,
    )
    assert per_sample_prob_mat.shape == (3, 4)
    np.testing.assert_allclose(per_sample_prob_mat.mean(axis=1), averaged)


def test_calc_additional_case_prob_analytical_scalar_broadcast_keeps_samples_once():
    calls = []

    def rep_no_func(t):
        t_vec = np.atleast_1d(t)
        calls.append(t_vec.copy())
        return np.column_stack([np.full(t_vec.size, 0.2), np.full(t_vec.size, 0.6)])

    per_sample_prob_vec: FloatArray = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
        additional_dims="broadcast",
    )

    np.testing.assert_allclose(per_sample_prob_vec, 1 - np.exp(-np.array([0.2, 0.6])))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], np.array([1]))


def test_calc_additional_case_prob_analytical_scalar_broadcast_probes_early_shape():
    calls = []

    def rep_no_func(t):
        t_vec = np.atleast_1d(t)
        calls.append(t_vec.copy())
        return np.full((t_vec.size, 3), 0.5)

    per_sample_prob_vec: FloatArray = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=0,
        additional_dims="broadcast",
    )

    assert per_sample_prob_vec.shape == (3,)
    np.testing.assert_allclose(per_sample_prob_vec, 1.0)
    # The shape probe passes an *empty* time vector: it recovers the trailing sample dimensions
    # without evaluating R_t at any particular outbreak time.
    assert len(calls) == 1
    assert calls[0].size == 0


def test_calc_additional_case_prob_analytical_all_early_broadcast_probes_once():
    calls = []

    def rep_no_func(t):
        t_vec = np.atleast_1d(t)
        calls.append(t_vec.copy())
        return np.full((t_vec.size, 3), 0.5)

    per_sample_prob_mat = acp.calc_additional_case_prob_analytical(
        incidence=[1],
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0, 2], dtype=int),
        additional_dims="broadcast",
    )

    assert per_sample_prob_mat.shape == (2, 3)
    np.testing.assert_allclose(per_sample_prob_mat, np.array([[1.0] * 3, [0.0] * 3]))
    # Probed once, with an empty time vector (no particular outbreak time is evaluated).
    assert len(calls) == 1
    assert calls[0].size == 0


def test_calc_additional_case_prob_analytical_broadcast_broadcasts_early_returns():
    # An early-return t_calc (t_calc=0 -> prob 1) is broadcast to the sample shape so it
    # stacks with the per-sample results.
    incidence_mat = np.ones((4, 3), dtype=int)

    def rep_no_func(t):
        t_vec = np.atleast_1d(t)
        return np.full((t_vec.size, 3), 0.5)

    per_sample_prob_mat = acp.calc_additional_case_prob_analytical(
        incidence=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0, 1], dtype=int),
        additional_dims="broadcast",
    )
    assert per_sample_prob_mat.shape == (2, 3)
    np.testing.assert_allclose(per_sample_prob_mat[0], 1.0)


def test_calc_additional_case_prob_analytical_invalid_additional_dims():
    with pytest.raises(ValueError, match="additional_dims"):
        acp.calc_additional_case_prob_analytical(
            incidence=[1],
            rep_no_func=lambda t: 0.5,
            serial_interval_dist_vec=[1.0],
            t_calc=1,
            additional_dims=cast(Any, "nope"),
        )


def test_calc_additional_case_prob_analytical_all_zero_incidence_returns_zero():
    prob = acp.calc_additional_case_prob_analytical(
        incidence=[0, 0, 0],
        rep_no_func=lambda t: 1.0,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    assert prob == 0


def test_calc_additional_case_prob_sampled_histories_share_later_index_time():
    incidence_mat = np.array([[0, 0], [1, 2], [0, 0]])

    prob_vec: FloatArray = acp.calc_additional_case_prob_analytical(
        incidence=incidence_mat,
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        t_calc=0,
        additional_dims="broadcast",
    )

    np.testing.assert_array_equal(prob_vec, np.ones(2))


def test_calc_additional_case_prob_sampled_histories_all_zero():
    prob_vec: FloatArray = acp.calc_additional_case_prob_analytical(
        incidence=np.zeros((3, 2), dtype=int),
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        t_calc=0,
        additional_dims="broadcast",
    )

    np.testing.assert_array_equal(prob_vec, np.zeros(2))


def test_calc_additional_case_prob_rejects_different_sampled_index_times():
    incidence_mat = np.array([[1, 0], [0, 1], [0, 0]])

    with pytest.raises(ValueError, match="same index-case time"):
        acp.calc_additional_case_prob_analytical(
            incidence=incidence_mat,
            rep_no_func=lambda t: 0.0,
            serial_interval_dist_vec=[1.0],
            t_calc=0,
        )


def test_calc_additional_case_prob_rejects_mixed_empty_sampled_histories():
    incidence_mat = np.array([[1, 0], [0, 0], [0, 0]])

    with pytest.raises(ValueError, match="either all be zero or all contain"):
        acp.calc_additional_case_prob_analytical(
            incidence=incidence_mat,
            rep_no_func=lambda t: 0.0,
            serial_interval_dist_vec=[1.0],
            t_calc=0,
        )


def test_calc_additional_case_prob_simulation_parallel_false_deterministic_zero_prob(
    rng,
):
    prob = acp.calc_additional_case_prob_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([1, 2], dtype=int),
        n_sims=20,
        rng=rng,
        parallel=False,
    )
    assert isinstance(prob, np.ndarray)
    assert prob.shape == (2,)
    np.testing.assert_allclose(prob, np.array([0.0, 0.0]))


def test_calc_additional_case_prob_simulation_scalar_returns_float(rng):
    prob = acp.calc_additional_case_prob_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
        n_sims=5,
        rng=rng,
        parallel=False,
    )
    assert isinstance(prob, float)
    assert prob == 0.0


def test_has_additional_case_one_sim_uses_run_renewal_model(monkeypatch):
    called = {}

    def fake_run_renewal_model(**kwargs):
        called.update(kwargs)
        return np.array([1, 0, 1])

    monkeypatch.setattr(acp, "run_renewal_model", fake_run_renewal_model)

    out = acp._has_additional_case_one_sim(
        (
            np.array([1]),
            lambda t: 1.0,
            np.array([1.0, 0.0]),
            1,
            np.random.default_rng(0),
            0,
            0,
        )
    )

    assert called["t_stop"] == 3
    assert called["_break_on_case"] is True
    assert out == (0, 0, True)


def test_has_additional_case_one_sim_raises_when_t_calc_does_not_match_incidence_length():
    with pytest.raises(ValueError, match="does not match length of incidence_vec"):
        acp._has_additional_case_one_sim(
            (
                np.array([1, 0]),
                lambda t: 1.0,
                np.array([1.0, 0.0]),
                1,
                np.random.default_rng(0),
                0,
                0,
            )
        )


def test_calc_decision_delay_contiguous_times_and_nan_when_never_below():
    # Contiguous outbreak times measured from the final case (the retrospective use):
    # probabilities at t=1..3
    # after a final case on day 0. 5% first crossed on day 2 -> delay 2; 2% on day 3 -> delay 3;
    # 0.5% never crossed -> NaN (not an error).
    decision_delay_vec = acp.calc_decision_delay(
        prob_vec=np.array([0.2, 0.04, 0.01]),
        t_vec=np.array([1, 2, 3]),
        risk_threshold_pct=np.array([5, 2, 0.5]),
        t_final_case=0,
    )
    np.testing.assert_array_equal(decision_delay_vec[:2], np.array([2.0, 3.0]))
    assert np.isnan(decision_delay_vec[2])


def test_calc_decision_delay_maps_non_contiguous_times():
    # Probability measured at non-contiguous outbreak times; delay is
    # (crossing time - final-case time), and NaN
    # for a threshold the risk never falls below.
    t_vec = np.array([2, 4, 6, 8])
    prob_vec = np.array([0.9, 0.5, 0.2, 0.02])
    decision_delay_vec = acp.calc_decision_delay(
        prob_vec=prob_vec,
        t_vec=t_vec,
        risk_threshold_pct=np.array([60, 30, 1]),
        t_final_case=3,
    )
    # 60% first crossed on day 4 -> delay 1; 30% on day 6 -> delay 3; 1% never -> NaN.
    np.testing.assert_array_equal(decision_delay_vec[:2], np.array([1.0, 3.0]))
    assert np.isnan(decision_delay_vec[2])


def test_calc_decision_delay_ignores_days_before_final_case():
    # A sub-threshold day before the final case does not count.
    t_vec = np.array([0, 1, 2, 3, 4])
    prob_vec = np.array([0.0, 0.9, 0.9, 0.9, 0.1])
    decision_delay_vec = acp.calc_decision_delay(
        prob_vec=prob_vec,
        t_vec=t_vec,
        risk_threshold_pct=np.array([50]),
        t_final_case=2,
    )
    np.testing.assert_array_equal(decision_delay_vec, np.array([2.0]))


def test_simulation_prob_t_calc_zero_should_not_crash(rng):
    prob = acp.calc_additional_case_prob_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0], dtype=int),
        n_sims=5,
        rng=rng,
        parallel=False,
    )
    assert isinstance(prob, np.ndarray)
    np.testing.assert_allclose(prob, np.array([1.0]))
