import pytest
import numpy as np
import math
from roverlad_control.roverlad_helper import RoverladPIDDrive

def test_PIDCalc_zeroError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(0)

    assert linear == pytest.approx(0.225)
    assert angular == pytest.approx(0.0)


def test_PIDCalc_positiveError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(10)

    # kp * error + kd * derivative
    # 0.02 * 10 + 0.01 * 10 = 0.3
    assert linear == pytest.approx(0.15)
    assert angular == pytest.approx(0.3)


def test_PIDCalc_negativeError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(-10)

    # 0.02 * -10 + 0.01 * -10 = -0.3
    assert linear == pytest.approx(0.15)
    assert angular == pytest.approx(-0.3)


def test_PIDCalc_largePositiveError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(100)

    # Linear speed should be reduced for large error
    assert linear == pytest.approx(0.045)

    # Angular velocity should be clamped to maxAngular
    assert angular == pytest.approx(1.5)


def test_PIDCalc_largeNegativeError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(-100)

    assert linear == pytest.approx(0.045)

    # Angular velocity should be clamped
    assert angular == pytest.approx(-1.5)


def test_PIDCalc_smallError() -> None:
    pidDrive = RoverladPIDDrive()

    linear, angular = pidDrive.PIDCalc(3)

    # abs(error) < 5
    assert linear == pytest.approx(0.225)

    # 0.02 * 3 + 0.01 * 3 = 0.09
    assert angular == pytest.approx(0.09)


def test_PIDCalc_integralUpdates() -> None:
    pidDrive = RoverladPIDDrive()

    pidDrive.PIDCalc(10)

    assert pidDrive.integral == pytest.approx(10)


def test_PIDCalc_prevErrorUpdates() -> None:
    pidDrive = RoverladPIDDrive()

    pidDrive.PIDCalc(10)

    assert pidDrive.prevError == pytest.approx(10)


def test_PIDCalc_derivativeUsesPreviousError() -> None:
    pidDrive = RoverladPIDDrive()

    # First call:
    # error = 10
    # previous error = 0
    # derivative = 10
    pidDrive.PIDCalc(10)

    # Second call:
    # error = 20
    # previous error = 10
    # derivative = 10
    linear, angular = pidDrive.PIDCalc(20)

    # integral = 10 + 20 = 30
    #
    # angular =
    # kp * error
    # + ki * integral
    # + kd * derivative
    #
    # = 0.02 * 20 + 0 + 0.01 * 10
    # = 0.5

    assert angular == pytest.approx(0.5)
    assert pidDrive.prevError == pytest.approx(20)
    assert pidDrive.integral == pytest.approx(30)


def test_PIDCalc_integralClamped() -> None:
    pidDrive = RoverladPIDDrive()

    pidDrive.PIDCalc(2000)

    assert pidDrive.integral == pytest.approx(1000)


def test_PIDCalc_angularClampedPositive() -> None:
    pidDrive = RoverladPIDDrive()

    _, angular = pidDrive.PIDCalc(1000)

    assert angular == pytest.approx(1.5)


def test_PIDCalc_angularClampedNegative() -> None:
    pidDrive = RoverladPIDDrive()

    _, angular = pidDrive.PIDCalc(-1000)

    assert angular == pytest.approx(-1.5)
