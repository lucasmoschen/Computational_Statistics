#!usr/bin/env/python
"""
Final assignment - Computational Statistics
Lucas Moschen
Bootstrap
"""
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def random_sample(data_points, format):
    """
    Imitates the function numpy.choice with replacement.
    """
    size = format[0] * format[1]
    n = len(data_points)
    unif = np.random.random(size=size)
    choices = np.zeros(size)
    for i in range(size):
        pos = 0
        while pos/n < unif[i]:
            pos += 1
        choices[i] = data_points[pos - 1]
    choices = np.reshape(choices, format)
    return choices

def bootstrap_nonparametric(data_points, B):
    """
    Denotes the non parametric version of bootstrap
    """
    n = len(data_points)
    bootstrap_samples = random_sample(data_points, format=(n, B))
    return bootstrap_samples

def bootstrap_parametric(data_points, B):
    """
    Denotes the parametric version of bootstrap
    """
    n = len(data_points)
    bootstrap_samples = np.random.normal(loc=np.mean(data_points),
                                         scale=np.std(data_points, ddof=1),
                                         size=(n, B))
    return bootstrap_samples

def bootstrap_bayesian(data_points, B):
    """
    Bayesian bootstrap.
    """
    n = len(data_points)
    unif = np.random.random(size=(n-1, B))
    unif_sort = np.zeros((n+1, B))
    unif_sort[1:-1, :] = np.sort(unif, axis=0)
    unif_sort[-1, :] = 1
    g = np.diff(unif_sort, axis=0)
    return g

def jackknife_method(data_points):
    """
    Denotes the jackknife method for resampling
    """
    n = len(data_points)
    jack_samples = np.zeros((n-1, n))
    for i in range(n):
        jack_samples[:i, i] = data_points[:i]
        jack_samples[i:, i] = data_points[i+1:]
    return jack_samples

def trimmed_mean_bayesian(data_points, alpha, B):
    """
    Calcultate samples from the trimmed mean posterior
    """
    n = len(data_points)
    g = int(n * alpha)
    r = n * alpha - g
    dp_sort = np.sort(data_points)
    bootstrap_samples = bootstrap_bayesian(data_points, B)
    trimmed_mean = (1-r) * (bootstrap_samples[g] * dp_sort[g] + bootstrap_samples[n-g-1] * dp_sort[n-g-1])
    trimmed_mean = trimmed_mean + (bootstrap_samples[g+1:n-g-1] * dp_sort[g+1:n-g-1].reshape(-1, 1)).sum(axis=0)
    trimmed_mean /= ((1-r) * (bootstrap_samples[g] + bootstrap_samples[n-g-1]) + bootstrap_samples[g+1:n-g-1].sum(axis=0))
    return trimmed_mean

def trimmed_mean(data_points, alpha, B, method='bootstrap'):
    """
    Calculate samples from the trimmed mean estimator.
    """
    n = len(data_points)
    g = int(n * alpha)
    r = n * alpha - g
    if method == 'bootstrap':
        bootstrap = bootstrap_nonparametric(data_points, B)
    elif method == 'bootstrap-parametric':
        bootstrap = bootstrap_parametric(data_points, B)
    elif method == 'bootstrap-bayesian':
        return trimmed_mean_bayesian(data_points, alpha, B)
    else:
        bootstrap = jackknife_method(data_points)
    dp_sort = np.sort(bootstrap, axis=0)
    trimmed_mean = (1-r) * (dp_sort[g] + dp_sort[n-g-1]) + dp_sort[g+1:n-g-1].sum(axis=0)
    trimmed_mean = trimmed_mean/(n * (1 - 2*alpha))
    return trimmed_mean

def true_standard_error(F_distribution, n=15, alpha=0.25):
    """
    Calculate an approximation for the true standard error.
    """
    if F_distribution == 'normal':
        samples = np.random.normal(size=(n, 1000000))
    elif F_distribution == 't':
        samples = np.random.standard_t(df=5, size=(n, 1000000))
    else:
        samples = np.random.exponential(scale=1, size=(n, 1000000))
    samples = np.sort(samples, axis=0)
    g = int(n * alpha)
    r = n * alpha - g
    cov_matrix = np.cov(samples, rowvar=True)
    var = cov_matrix[g+1:n-g-1, g+1:n-g-1].sum()
    var += 2*(1-r) * cov_matrix[g, g+1:n-g-1].sum()
    var += 2*(1-r) * cov_matrix[n-g-1, g+1:n-g-1].sum()
    var += (1-r)**2 * (cov_matrix[g, g] + 2 * cov_matrix[g, n-g-1] + cov_matrix[n-g-1, n-g-1])
    var /= (n*(1-2*alpha))**2
    standard_error = np.sqrt(var)
    return standard_error

def monte_carlo_estimates(n=15, alpha=0.25, B=200, B2=10000): 

    m = 1000
    table = np.zeros((m, 8))
    for k in tqdm(range(m)):
        y_norm = np.random.normal(size=n)
        y_exp = np.random.exponential(scale=1, size=n)
        tm_samples_norm_boots = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap')
        tm_samples_exp_boots = trimmed_mean(y_exp, alpha=alpha, B=B, method='bootstrap')
        tm_samples_norm_boots_more = trimmed_mean(y_norm, alpha=alpha, B=B2, method='bootstrap')
        tm_samples_exp_boots_more = trimmed_mean(y_exp, alpha=alpha, B=B2, method='bootstrap')
        tm_samples_norm_jack = trimmed_mean(y_norm, alpha=alpha, B=B, method='jack')
        tm_samples_exp_jack = trimmed_mean(y_exp, alpha=alpha, B=B, method='jack')
        tm_samples_norm_boots_bayes = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap-bayesian')
        tm_samples_exp_boots_bayes = trimmed_mean(y_exp, alpha=alpha, B=B, method='bootstrap-bayesian')

        table[k, 0] = tm_samples_norm_boots.std(ddof=1)
        table[k, 1] = tm_samples_exp_boots.std(ddof=1)
        table[k, 2] = tm_samples_norm_boots_more.std(ddof=1)
        table[k, 3] = tm_samples_exp_boots_more.std(ddof=1)
        table[k, 4] = (n-1)/np.sqrt(n) * tm_samples_norm_jack.std(ddof=1)
        table[k, 5] = (n-1)/np.sqrt(n) * tm_samples_exp_jack.std(ddof=1)
        table[k, 6] = tm_samples_norm_boots_bayes.std(ddof=1)
        table[k, 7] = tm_samples_exp_boots_bayes.std(ddof=1)

    print(table.mean(axis=0))
    print(table.std(axis=0, ddof=1))
    return table


if __name__ == '__main__':

    #np.random.seed(17)
    #table = monte_carlo_estimates()
    # print("True standard error exponential trimmed mean:")
    # _ = true_standard_error(F_distribution='exp')
    # print('--------------------------------------')
    # print("True standard error normal trimmed mean")
    # _ = true_standard_error(F_distribution='normal')

    # np.random.seed(671)
    # n = 15
    # alpha = 0.25
    # y_norm = np.random.normal(size=n)
    # y_exp = np.random.exponential(scale=1, size=n)

    # B_values = np.linspace(10, 100000, 200)
    # table = np.zeros((2, 200))
    # for k, B in tqdm(enumerate(B_values)):
    #     tm_samples_norm_boots = trimmed_mean(y_norm, alpha=alpha, B=int(B), method='bootstrap')
    #     tm_samples_exp_boots = trimmed_mean(y_exp, alpha=alpha, B=int(B), method='bootstrap')
    #     table[0, k] = np.std(tm_samples_norm_boots, ddof=1)
    #     table[1, k] = np.std(tm_samples_exp_boots, ddof=1)

    # fig, ax = plt.subplots(2, 1)
    # ax[0].plot(B_values, table[0])
    # ax[1].plot(B_values, table[1])
    # ax[0].set_title('Standard normal distribution F', fontsize=20)
    # ax[1].set_title('Exponential distribution F', fontsize=20)
    # fig.tight_layout(pad=3)
    # plt.show()

    # np.random.seed(728190)
    # m = 1000
    # n = 15
    # alpha = 0.25
    # B = 200
    # table = np.zeros((4, m))
    # for k in tqdm(range(m)):
    #     y_norm = np.random.normal(size=n)
    #     y_stud = np.random.standard_t(df=5, size=n)
    #     tm_samples_norm_boots = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap-parametric')
    #     tm_samples_stud_boots = trimmed_mean(y_stud, alpha=alpha, B=B, method='bootstrap-parametric')
    #     table[0, k] = np.std(tm_samples_norm_boots, ddof=1)
    #     table[1, k] = np.std(tm_samples_stud_boots, ddof=1)
    #     tm_samples_norm_boots = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap')
    #     tm_samples_stud_boots = trimmed_mean(y_stud, alpha=alpha, B=B, method='bootstrap')
    #     table[2, k] = np.std(tm_samples_norm_boots, ddof=1)
    #     table[3, k] = np.std(tm_samples_stud_boots, ddof=1)

    # fig, ax = plt.subplots(2, 2, sharex=True)
    # ax[0, 0].hist(table[0], bins=25)
    # ax[0, 1].hist(table[1], bins=25)
    # ax[1, 0].hist(table[2], bins=25)
    # ax[1, 1].hist(table[3], bins=25)
    # ax[0, 0].axvline(true_standard_error(F_distribution='normal'), linestyle='--', color='red')
    # ax[0, 1].axvline(true_standard_error(F_distribution='t'), linestyle='--', color='red')
    # ax[1, 0].axvline(true_standard_error(F_distribution='normal'), linestyle='--', color='red')
    # ax[1, 1].axvline(true_standard_error(F_distribution='t'), linestyle='--', color='red')
    # ax[0, 0].set_title('Normal(0,1) data', fontsize=10)
    # ax[0, 1].set_title('t-Student(df=5) data', fontsize=10)
    # ax[0, 0].set_ylabel('Parametric with normal dist.', fontsize=10)
    # ax[1, 0].set_ylabel('Non parametric', fontsize=10)
    # # fig.tight_layout(pad=3)
    # plt.show()

    np.random.seed(6321)
    # alpha=0.25
    # n = 15
    # B=200
    # y_norm = np.random.normal(size=n)
    # tm_samples_norm_boots = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap')
    # tm_samples_norm_boots_bayes = trimmed_mean(y_norm, alpha=alpha, B=B, method='bootstrap-bayesian')
    # plt.hist(tm_samples_norm_boots, label='Regular bootstrap')
    # plt.hist(tm_samples_norm_boots_bayes, label='Bayesian bootstrap')
    # plt.legend()
    # plt.title('Comparing regular and bayesian bootstraps')
    # plt.show()
