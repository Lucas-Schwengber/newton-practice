
#1/x, x*ln(x)-x, flipped(x^2), sin(1/x)
import numpy as np
import newton
import pytest
import warnings

def test_cos_near_0():
    assert np.abs(newton.optimize(np.cos,0.1))<1e-6

def test_correct_input_type():
    with pytest.raises(TypeError):
        newton.optimize(1,np.cos)

def test_zero_second_derivative():
    with pytest.warns(UserWarning, match="Hessian norm below tolerance, stopping early."):
        f = lambda x: x**4
        newton.optimize(f,0.1)

def test_max():
    with pytest.warns(UserWarning, match="Probably counverged to a max"):
        f = lambda x: -x**2
        newton.optimize(f,0.1)

def test_f_exists():
    with pytest.raises(ValueError):
        def der_abs(x):
            if x == 0:
                raise ValueError
            else:
                return np.sign(x)

        newton.optimize(der_abs,0)

def test_conv_inf():
    with pytest.raises(RuntimeError):
        def f(x):
            if x == 0:
                raise ValueError("Function cannot be evaluated at point")
            else:
                return 0.5/x

        newton.optimize(f, 0.2)
