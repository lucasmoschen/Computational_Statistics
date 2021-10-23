#!usr/bin/env python

import pickle
import numpy as np
import pystan as ps
from scipy.special import comb
from tqdm import tqdm

def initiate_model():
    """
    It gets the Stan model.
    """
    folder = '/home/lucasmoschen/Documents/GitHub/computational-statistics/exercises/exercise_5_ps2/'
    try:
        stan_model = pickle.load(open(folder + 'exercise_5_ps2.pkl', 'rb'))
    except FileNotFoundError:
        stan_model = ps.StanModel(file=folder + "exercise_5_ps2.stan")
        with open(folder + "exercise_5_ps2.pkl", 'wb') as file:
            pickle.dump(stan_model, file)
    print("INFO - Model done!")
    return stan_model 

def binomial_product(y, z, m, n, theta1, theta2):
    pdf = np.log(comb(n, y)) +  y*np.log(theta2) + (n-y)*np.log1p(-theta2) 
    pdf += np.log(comb(m, z-y)) +  (z-y)*np.log(theta1) + (m-z+y)*np.log1p(-theta1)
    return np.exp(pdf)

def probabilities_binomial_product(z, m, n, theta1, theta2):
    p = np.zeros(min(n, z) - max(0, z-m)+1)
    i=-1
    for y in range(max(0, z-m), min(n, z)+1): 
        i += 1
        p[i] = binomial_product(y, z, m, n, theta1, theta2)
    return p/sum(p)

def binomial_product_rng(z, m, n, theta1, theta2):
    u = np.random.uniform()
    p = probabilities_binomial_product(z, m, n, theta1, theta2)
    index = np.argmax(u < np.cumsum(p))
    return index + max(0,z-m)

def gibbs_sampler(z, m, n, T, warmup, iterations):
    x = np.zeros(T)
    y = np.zeros(T)

    theta1 = np.random.uniform()
    theta2 = np.random.uniform()
    for _ in tqdm(range(warmup)): 
        for i in range(T): 
            y[i] = binomial_product_rng(z[i], m[i], n[i], theta1, theta2)
            x[i] = z[i] - y[i]
        theta1 = np.random.beta(1 + sum(x), 1 + sum(m-x))
        theta2 = np.random.beta(1 + sum(y), 1 + sum(n-y))

    samples = np.zeros((iterations, 2))
    for ite in tqdm(range(iterations)):
        for i in range(T): 
            y[i] = binomial_product_rng(z[i], m[i], n[i], theta1, theta2)
            x[i] = z[i] - y[i]
        theta1 = np.random.beta(1 + sum(x), 1 + sum(m-x))
        theta2 = np.random.beta(1 + sum(y), 1 + sum(n-y))   
        samples[ite, 0] = theta1
        samples[ite, 1] = theta2
    return samples

def main():
    """
    Main function.
    """
    theta1 = 0.3
    theta2 = 0.5
    m = np.random.randint(0, 20, size=30)
    n = np.random.randint(0, 20, size=30)
    X = np.random.binomial(n=m, p=theta1)
    Y = np.random.binomial(n=n, p=theta2)
    Z = X + Y

    stan_model = initiate_model()
    data = {"T": len(m), "m": m, "n": n, "z": Z, "Z_max": max(m + n)}
    fitting = stan_model.sampling(data=data)
    print(fitting)

    sampling = gibbs_sampler(Z, m, n, len(m), warmup=2000, iterations=1000)

    print(sampling[:, 0].mean())
    print(fitting.extract()['theta1'].mean())
    print(sampling[:, 1].mean())
    print(fitting.extract()['theta2'].mean())

if __name__ == '__main__':

    main()
