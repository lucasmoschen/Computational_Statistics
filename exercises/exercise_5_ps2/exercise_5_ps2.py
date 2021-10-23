#!usr/bin/env python

import pickle
import numpy as np
import pystan as ps

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

def main():
    """
    Main function.
    """
    theta1 = 0.3
    theta2 = 0.5
    m = np.random.randint(0, 20, size=50)
    n = np.random.randint(0, 20, size=50)
    X = np.random.binomial(n=m, p=theta1)
    Y = np.random.binomial(n=n, p=theta2)
    Z = X + Y

    stan_model = initiate_model()
    data = {"T": len(m), "m": m, "n": n, "z": Z, "Z_max": max(m + n)}
    fitting = stan_model.sampling(data=data)
    print(fitting)

if __name__ == '__main__':

    main()
