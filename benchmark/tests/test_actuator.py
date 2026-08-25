import numpy as np

from exo_bench.actuator import MotorCommand, TrustedActuatorBoundary


def test_trusted_actuator_boundary_clamps_gains_feedforward_and_output() -> None:
    boundary = TrustedActuatorBoundary(
        {
            "mode": "pd_ff",
            "transmission": "direct_joint",
            "torque_limits": [10, 20],
            "kp_range": [0, 100],
            "kd_range": [0, 10],
        },
        joint_count=2,
    )
    command = MotorCommand(
        q_des=np.asarray([10.0, -10.0]),
        dq_des=np.zeros(2),
        kp=np.asarray([1_000_000.0, 1_000_000.0]),
        kd=np.asarray([1_000_000.0, 1_000_000.0]),
        tau_ff=np.asarray([1_000.0, -1_000.0]),
    )
    torque = boundary.resolve(command, np.zeros(2), np.zeros(2))
    np.testing.assert_array_equal(torque, np.asarray([10.0, -20.0]))


def test_trusted_actuator_boundary_rejects_non_finite_commands() -> None:
    boundary = TrustedActuatorBoundary(
        {
            "mode": "pd_ff",
            "transmission": "direct_joint",
            "torque_limits": [10],
            "kp_range": [0, 100],
            "kd_range": [0, 10],
        },
        joint_count=1,
    )
    command = MotorCommand(
        q_des=np.asarray([np.nan]),
        dq_des=np.zeros(1),
        kp=np.ones(1),
        kd=np.ones(1),
        tau_ff=np.zeros(1),
    )
    try:
        boundary.resolve(command, np.zeros(1), np.zeros(1))
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite motor command was accepted")

