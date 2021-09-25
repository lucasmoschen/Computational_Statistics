#!usr/bin/env python 

import numpy as np 

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
    r1, r2, theta1, theta2 = np.random.uniform(low = [0,0,0,0], 
                                               high = [R**2, R**2, 2*np.pi, 2*np.pi], 
                                               size = (n_samples, 4)).transpose()
    dist = np.sqrt(r1 + r2 - 2*np.sqrt(r1*r2)*np.cos(theta1 - theta2))
    return dist.mean()

if __name__ == '__main__': 
    R = 5
    D = simple_simulation(R, 10000000)
    print(D)
    print(128*R/(45 * np.pi))
