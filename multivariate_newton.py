import numpy as np
from typing import Callable
import warnings
import numdifftools as nd

def optimize(f:Callable, x0:float=0, max_iter:int=100, eps:float=1e-6):
    """
    Runs multivariate newton's method to find local minima (or maxima) of a given scalar function f.

    Arguments:
    -f: the scalar function to be optimized
    -x0: starting value
    -max_iter: maximum number of iterations to run the algorithm
    """
    grad = nd.Gradient(f)
    hessian = nd.Hessian(f)
    x = x0

    for i in range(0, max_iter):
        x = x - np.linalg.inv(hessian(x)) @ grad(x)

        if i == max_iter - 1:
            warnings.warn("Maximum number of iterations reached.", UserWarning)
        print(f"Iteration {i} approximation: {x}")

        return x


if __name__ == "__main__":
    f = lambda x: np.sum(x**2)
    optimize(f, x0 = np.array([1,1,1]))
