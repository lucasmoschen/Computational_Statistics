#! usr/bin/env/python

import numpy as np

def pi_dist(theta, k):
    """
    The model distributions.
    """
    if k == 1:
        return np.exp(-0.5*theta**2)
    elif k == 2:
        return np.exp(-0.5*(theta[0]**2 + theta[1]**2))
    else:
        raise Exception("Error - There is only two models. ")

class ReversibleJump:
    """
    (Simulation question - Paper sheet 4 (Reversible jump MCMC))
    Consider two models. For model 1 the toy target distribution is given
    pi(theta | k = 1) = exp(-theta^2/2) and
    pi(theta | k = 1) = exp(-theta_1^2/2 - theta_2^2/2)

    We want to sample from (k, theta).
    """

    def __init__(self) -> None:
        pass

    def kernel_model1(self, sigma, iterations):
        """
        Metropolis-Hastings kernel for model 1.
        """
        x_t = np.random.normal(loc=0, scale=sigma)
        x_values = np.zeros(iterations)
        for index in range(iterations):
            x_new = np.random.normal(loc=0, scale=sigma)
            alpha = min(1, pi_dist(x_new, k=1)/pi_dist(x_t, k=1))
            uniform = np.random.uniform()
            if uniform <= alpha:
                x_t = x_new
            x_values[index] = x_t
        return x_values

    def kernel_model2(self, sigma, iterations):
        """
        Metropolis-Hastings kernel for model 2.
        """
