import endoutbreakvbd


def test_public_api_exports_expected_symbols():
    expected = {
        "calc_declaration_delay",
        "calc_further_case_risk_analytical",
        "calc_further_case_risk_simulation",
        "run_renewal_model",
        "rep_no_from_grid",
    }
    assert set(endoutbreakvbd.__all__) == expected


def test_public_api_exports_are_callables():
    for name in endoutbreakvbd.__all__:
        assert hasattr(endoutbreakvbd, name)
        assert callable(getattr(endoutbreakvbd, name))
