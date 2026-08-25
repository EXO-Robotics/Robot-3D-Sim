from exo_bench.qualification import ChassisQualifier


def test_cq_007_requires_upright_unforced_recovery() -> None:
    gate = ChassisQualifier().perturbation_recovery()
    assert gate["status"] == "PASS", gate
    assert gate["evidence"]["fallen"] is False
    assert gate["evidence"]["external_force_cleared"] is True
    assert gate["evidence"]["final_height_m"] >= gate["evidence"]["minimum_final_height_m"]
    assert gate["evidence"]["final_tilt_rad"] <= gate["evidence"]["maximum_final_tilt_rad"]
    assert (
        gate["evidence"]["final_window_average_speed_mps"]
        <= gate["evidence"]["maximum_final_window_average_speed_mps"]
    )
