import numpy as np


def derivative(f, eps=1e-6):
    def f_prime(x):
        return (f(x+eps)-f(x))/eps

    return f_prime


def second_derivative(f, eps=1e-6):
    return derivative(derivative(f))


def optimize(f, x0=0, n_iter=100, tol=1e-6):
    f_prime = derivative(f)
    f_double_prime = second_derivative(f)
    x = x0

    for i in range(0,n_iter):
        if np.abs(f_prime(x)) < tol:
            print("Gradient norm below tolerance, stopping early.")
            break
        
        if  np.abs(f_prime(x)/f_double_prime(x)) < tol:
            print("Stopping criteria reached.")
            break
        
        else:
            x = x - f_prime(x)/f_double_prime(x)
            if i == n_iter-1:
               print("Maximum number of iterations reached") 
        
        print(f"Iteration {i} approximation: {x}") 
    
    return x



if __name__ == "__main__":
    f = lambda x: np.sin(x)

    x = optimize(f, x0=1)
    print(x)
