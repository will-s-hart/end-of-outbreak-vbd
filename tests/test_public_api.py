# Note that AI tools were used to generate tests

import endoutbreakvbd


def test_public_api_exports_expected_symbols():
    expected = {
        "calc_decision_delay",
        "calc_additional_case_prob_analytical",
        "calc_additional_case_prob_simulation",
    }
    assert set(endoutbreakvbd.__all__) == expected


def test_public_api_exports_are_callables():
    for name in endoutbreakvbd.__all__:
        assert hasattr(endoutbreakvbd, name)
        assert callable(getattr(endoutbreakvbd, name))
