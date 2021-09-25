#!usr/bin/env python 

import numpy as np 
import matplotlib.pyplot as plt 
from time import time
from numpy.core.fromnumeric import mean
from numpy.lib.index_tricks import nd_grid
from scipy.stats import norm, t

def simple_simulation(R, n_samples): 
    """
    We sample x,y uniformly in the disc centered at 
    the origin with radius R. In order to sample from then, we consider the
    following transformation: 
    x = sqrt{r_1})(cos(theta_1), sin(theta_1)),
    y = sqrt{r_2})(cos(theta_2), sin(theta_2)).
    The density of (r_1, theta_1) is uniform times the determinant of the
    jacobian 
    | (1/2 sqrt{r1})cos(theta_1)   -sqrt{r1}sin(theta_1) |
    | (1/2 sqrt{r1})sin(theta_1)    sqrt{r1}cos(theta_1) |
    that is 1/2. Then 
    g(r_1, theta_1) = (2pi R^2)^{-1} in the rectangle [0,R^2] x [0,2pi], which
    is the uniform distribution. 
    """
    r1, r2, theta1, theta2 = np.random.uniform(size = (4, n_samples))
    dist = np.sqrt(r1 + r2 - 2*np.sqrt(r1*r2)*np.cos(2*np.pi*(theta1 - theta2)))
    return R*dist.mean(), dist

if __name__ == '__main__': 

    R = 5
    V = (1 - (128/(45 * np.pi))**2) * R**2
    alpha = 0.05
    n_samples = 10000

    print('INFO - Starting simulation')
    t0 = time()
    E_D, D_array = simple_simulation(R, n_samples)
    t1 = time()
    print('INFO - Simulation finished') 
    print('INFO - Duration: {}'.format(t1-t0))
    print('INFO - E_D = {}'.format(E_D))
    print('INFO - True value = {}'.format(128*R/(45 * np.pi)))

    order = np.linspace(2,n_samples,n_samples-1)
    mean_cum = R*D_array.cumsum()[1:]/order

    z = norm().ppf(1 - alpha/2)*np.sqrt(V/order)

    T = t.ppf(1-alpha/2, df = range(1, n_samples))

    sigma_prime = (R**2 * (D_array**2).cumsum()[1:] - order * mean_cum**2)/(order - 1)
    sigma_prime = T*np.sqrt(sigma_prime/order)

    print('INFO - Interval Estimate (known V): ({}, {})'.format(E_D - z[-1], E_D + z[-1]))
    print('INFO - Interval Estimate (unknown V): ({}, {})'.format(E_D - sigma_prime[-1], E_D + sigma_prime[-1]))

    fig, ax = plt.subplots(1,2,figsize = (15,5))

    ax[0].plot(range(1,n_samples), mean_cum, color = 'black', label = 'Point estimate')
    ax[0].fill_between(x = range(1,n_samples), 
                       y1 = mean_cum - z, y2 = mean_cum + z, 
                       alpha = 0.3, color = 'blue', 
                       label = '95-confidence interval (known V)')
    ax[0].fill_between(x = range(1,n_samples), 
                       y1 = mean_cum - sigma_prime, 
                       y2 = mean_cum + sigma_prime, 
                       alpha = 0.3, color = 'green', 
                       label = '95-confidence interval')
    ax[0].axhline(128*R/(45 * np.pi), linestyle = '--', color = 'red', 
                  alpha = 0.5, label = 'True value')
    ax[0].set_title('Convergence of simulation')
    ax[0].set_xlabel('Number of simulations')
    ax[0].set_ylabel('Cumulative mean')
    ax[0].set_xscale('log')
    ax[0].legend()

    ax[1].plot(abs(mean_cum - 128*R/(45 * np.pi)), color = 'grey')
    ax[1].set_title('Absolute error of simulations')
    ax[1].set_xlabel('Number of simulations')
    ax[1].set_ylabel('Absolute error')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')

    folder = '/home/lucasmoschen/Documents/GitHub/computational-statistics/assignments/warmup_assignment/'
    plt.savefig(folder + 'figure_simulation.png', bbox_inches = 'tight', dpi = 300)
    plt.show()