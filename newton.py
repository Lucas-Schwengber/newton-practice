import numpy as np
from typing import Callable
import warnings

def derivative(f, eps=1e-7):
    """ 
    Evaluates the derivative of a function (as a function) using finite differences.

    Arguments:
    -f: the real function to be differenciated
    -eps: the finite difference approximation step size

    Output:
    f_prime: a real function which evaluates the finite difference approximation of the derivative of f at a given point
    """

    def f_prime(x):
        return (f(x + eps) - f(x)) / eps

    return f_prime


def second_derivative(f, eps=1e-7):
    """
    Evaluates the second derivative of a given function f (as a function) using finite differences.

    Arguments:
    -f: the real function to be twice differenciated
    -eps: the finite difference approximation step size

    Output:
    f_double_prime: a real function which evaluates the finite difference approximation of the second derivative of f at a given point
    """
    f_double_prime = derivative(derivative(f, eps), eps)

    return f_double_prime


def optimize(f:Callable, x0:float=0, max_iter:int=100, eps:float=1e-6):
    """
    Runs newton's method to find local minima (or maxima) of a given real function f.

    Arguments:
    -f: the real function to be optimized
    -x0: starting value
    -max_iter: maximum number of iterations to run the algorithm
    -eps: finite difference approximation step size
    """
    tol = 1e-8

    f_prime = derivative(f, eps)
    f_double_prime = second_derivative(f, eps)
    x = x0
    step = 0
    new_step = 0

    for i in range(0, max_iter):
        if np.abs(f_double_prime(x)) < tol:
            warnings.warn("Hessian norm below tolerance, stopping early.", UserWarning)
            break

        step = f_prime(x) / f_double_prime(x)

        if np.abs(step) < tol:
            print("Stopping criteria reached.")
            break
        else:
            x = x - step
            new_step = f_prime(x) / f_double_prime(x)

        if i == max_iter - 1:
            warnings.warn("Maximum number of iterations reached.", UserWarning)
        elif new_step > 10*step:
            warnings.warn("Method seems to be diverging, stopping early.", UserWarning)
            break        
        print(f"Iteration {i} approximation: {x}")

    if f_double_prime(x) < 0:
        warnings.warn("Probably counverged to a max.")

    return x


if __name__ == "__main__":
    x = optimize(lambda x: x**3/3 - 2*x + 1, x0=1)
    print(x-np.sqrt(2))
