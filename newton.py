import numpy as np


def derivative(f, eps=1e-6):
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


def second_derivative(f, eps=1e-6):
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


def optimize(f, x0=0, max_iter=100, eps=1e-6):
    """
    Runs newton's method to find local minima (or maxima) of a given real function f.

    Arguments:
    -f: the real function to be optimized
    -x0: starting value
    -max_iter: maximum number of iterations to run the algorithm
    -eps: finite difference approximation step size
    """
    tol = 1e-6

    f_prime = derivative(f, eps)
    f_double_prime = second_derivative(f, eps)
    x = x0

    for i in range(0, max_iter):
        if np.abs(f_double_prime(x)) < tol:
            print("Hessian norm below tolerance, stopping early.")
            break

        if np.abs(f_prime(x) / f_double_prime(x)) < tol:
            print("Stopping criteria reached.")
            break

        else:
            x = x - f_prime(x) / f_double_prime(x)
            if i == max_iter - 1:
                print("Maximum number of iterations reached")

        print(f"Iteration {i} approximation: {x}")

    return x


if __name__ == "__main__":
    x = optimize(np.sin, x0=1)
    print(x)
