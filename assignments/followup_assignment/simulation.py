#!usr/bin/env python
"""
Followup assignment simulation: Average distance

Rejection sampling
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
PI = np.pi
FOLDER = '/home/lucasmoschen/Documents/GitHub/computational-statistics/assignments/followup_assignment/images/'
import json

def calculate_triangle_points(radius, n):
    """
    Return an array of points (x,y) to determine the triangles to sample.
    """
    points = np.zeros((n, 2))
    points_m = np.zeros((n, 2))
    points[:, 0] = radius * np.array([np.cos(i * 2*PI/n) for i in range(n)])
    points[:, 1] = radius * np.array([np.sin(i * 2*PI/n) for i in range(n)])
    points_m[:, 0] = radius * np.array([np.cos((i+0.5) * 2*PI/n) for i in range(n)])
    points_m[:, 1] = radius * np.array([np.sin((i+0.5) * 2*PI/n) for i in range(n)])
    x = (points_m[:, 0]**2/points_m[:, 1] + points_m[:, 1])
    x /= (points_m[:, 0]/points_m[:, 1] + points[:, 1]/points[:, 0])
    y = points[:, 1]/points[:, 0] * x
    return points, points_m, x, y

def reflextion_paralelogram(m, c, x1, y1):
    d = (x1 + (y1 - c) * m)/(1 + m**2)
    x_refl = 2 * d - x1
    y_refl = 2 * d * m - y1 + 2 * c
    return x_refl, y_refl

def rejection_sampling(n, m, R, x, y):
        
    samples = []
    n_samples = 0
    while len(samples) < m:
        index = np.random.randint(-1, n-1)
        x_value_1 = x[index]
        y_value_1 = y[index]
        x_value_2 = x[index+1]
        y_value_2 = y[index+1]

        u1, u2 = np.random.random(size=2)
        if u1 + u2 > 1:
            u1 = 1 - u1
            u2 = 1 - u2
        sample_x = x_value_1 * u1 + x_value_2 * u2
        sample_y = y_value_1 * u1 + y_value_2 * u2
        if sample_x * sample_x + sample_y * sample_y <= R * R:
            samples.append([sample_x, sample_y])
        n_samples += 1
    samples = np.array(samples)
    return samples, n_samples

def I1_hat(m, R):
    r1, r2 = np.random.uniform(0, R, size=(2, m))
    theta1, theta2 = np.random.uniform(0, 2*PI, size=(2,m))
    estimates = r1**2 + r2**2 - 2*r1*r2*np.cos(theta2 - theta1)
    estimates = np.sqrt(estimates) * r1 * r2
    mean_estimate = np.sum(estimates) * 4 / (m * R**2)
    var_estimate = np.sum((4 / R**2 * estimates - mean_estimate)**2)/(m-1)
    return mean_estimate, var_estimate, estimates

def I2_hat(m, R):
    r1, r2, theta1, theta2 = np.random.random(size=(4, m))
    estimates = r1**2 + r2**2 - 2*r1*r2*np.cos(2*PI*(theta2 - theta1))
    estimates = np.sqrt(estimates) * r1 * r2
    mean_estimate = np.sum(estimates) * 4 * R / m 
    var_estimate = np.sum((4 * R * estimates - mean_estimate)**2)/(m-1)
    return mean_estimate, var_estimate, estimates

def I3_hat(n, m, R, x, y):
    samples1, n_samples1 = rejection_sampling(n, m, R, x, y)
    samples2, n_samples2 = rejection_sampling(n, m, R, x, y)
    estimates = np.sqrt(np.sum((samples1 - samples2)**2, axis=1))
    mean_estimate = estimates.sum()/m
    var_estimate = np.sum((estimates - mean_estimate)**2)/(m-1)
    return mean_estimate, var_estimate, n_samples1, n_samples2, estimates

if __name__ == '__main__':

    M = 500
    m = 2000
    data = {}
    for R in tqdm([0.01, 0.1, 1, 10, 100, 1000, 10000]):
        points = {}
        for k in [4, 8, 20]:
            _, _, x, y = calculate_triangle_points(radius=R, n=k)
            points[k] = [x, y]
        for i in range(M):
            mean1, variance1, est1 = I1_hat(m=m, R=R)
            mean2, variance2, est2 = I2_hat(m=m, R=R)
            data[R] = {}
            data[R]['I1'] = [mean1, variance1, list(est1)]
            data[R]['I2'] = [mean2, variance2, list(est2)]
            for k in [4, 8, 20]:
                mean, variance, n1, n2, est3 = I3_hat(n=k, m=m, R=R, x=points[k][0], y=points[k][1])
                data[R]['IR_{}'.format(k)] = [mean, variance, n1, n2, list(est3)]
    with open(FOLDER + 'file.json', 'w') as f:
        json.dump(data, f)
            
    if input('Do you want to show the figure? [y/n]') == 'y':
        # Figure 1
        n = 6
        R = 1
        points, points_m, x, y = calculate_triangle_points(R, n)
        t = np.linspace(0, 2*PI, 1000)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(R*np.cos(t), R*np.sin(t), color='black')
        ax.scatter(points[:, 0], points[:, 1], color='red')
        for p in range(n):
            ax.plot([0, points[p, 0]], [0, points[p, 1]],
                    linestyle='--', color='darkgrey', alpha=0.7)
        ax.scatter([0], [0], color='black')
        ax.set_title('Circle divided in n = 6 sections.')
        plt.savefig(FOLDER+'figure1.pdf', bbopoints_mx_inches='tight')
        plt.show()

        # Figure 2
        n = 6
        R = 1
        points, points_m, x, y = calculate_triangle_points(R, n)
        t = np.linspace(0, 2*PI, 1000)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(R*np.cos(t), R*np.sin(t), color='black')
        ax.scatter(points[:, 0], points[:, 1], color='red')
        for p in range(n):
            ax.plot([0, points[p, 0]], [0, points[p, 1]],
                    linestyle='--', color='darkgrey', alpha=0.7)
        ax.scatter([0], [0], color='black')
        ax.scatter(points_m[:, 0], points_m[:, 1], color='blue')
        ax.set_title('Circle divided in n = 6 sections with midpoints.')
        plt.savefig(FOLDER+'figure2.pdf', bbox_inches='tight')
        plt.show()

        # Figure 3
        n = 6
        R = 1
        points, points_m, x, y = calculate_triangle_points(R, n)
        t = np.linspace(0, 2*PI, 1000)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(R*np.cos(t), R*np.sin(t), color='black', alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], color='red', alpha=0.3)
        ax.scatter([0], [0], color='black')
        ax.scatter(points_m[:, 0], points_m[:, 1], color='blue', alpha=0.3)
        ax.set_title('Circle circumscribed by the hexagon')
        ax.scatter(x, y, color='green')
        for i in range(n):
            if i == n-1:
                i = -1
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='green', linewidth=3)
            ax.plot([0, x[i+1]], [0, y[i+1]],
                    linestyle='--', color='darkgrey', alpha=0.7)
        plt.savefig(FOLDER+'figure3.pdf', bbox_inches='tight')
        plt.show()

        # Figure 4
        n_values = [3, 4, 6, 7, 10, 12]
        R = 1
        t = np.linspace(0, 2*PI, 1000)      
        u1, u2 = np.random.random(size=(2, 10000))
        verify = u1 + u2 > 1
        u1[verify] = 1 - u1[verify]
        u2[verify] = 1 - u2[verify]
        index = np.random.randint(n, size=10000)
        x_array_1 = [x[i] for i in index]
        y_array_1 = [y[i] for i in index]
        x_array_2 = [x[i+1] if i < n-1 else x[0] for i in index]
        y_array_2 = [y[i+1] if i < n-1 else y[0] for i in index]
        plt.scatter(x_array_1 * u1 + x_array_2 * u2, y_array_1 * u1 + y_array_2 * u2, s=1)
        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        fig.suptitle('Different approximations of the circle by polygons', 
                    fontsize=20)
        for k, n in enumerate(n_values):
            i = k // 3
            j = k % 3
            points, points_m, x, y = calculate_triangle_points(R, n)
            ax[i, j].plot(R*np.cos(t), R*np.sin(t), color='black', alpha=0.5)
            ax[i, j].set_title('n = {}'.format(n))
            ax[i, j].scatter(x, y, color='green')
            for p in range(n):
                if p == n-1:
                    p = -1
                ax[i, j].plot([x[p], x[p+1]], [y[p], y[p+1]], color='green', linewidth=3)
        plt.savefig(FOLDER+'figure4.pdf', bbox_inches='tight')
        plt.show()

        # Figure 5
        n = 6
        R = 1
        points, points_m, x, y = calculate_triangle_points(R, n)
        t = np.linspace(0, 2*PI, 1000)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(R*np.cos(t), R*np.sin(t), color='black', alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], color='red', alpha=0.3)
        ax.scatter([0], [0], color='black')
        ax.scatter(points_m[:, 0], points_m[:, 1], color='blue', alpha=0.3)
        ax.set_title('Sampling from the parallelogram given by the first triangle')
        ax.scatter(x, y, color='green')
        for i in range(n):
            if i == n-1:
                i = -1
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='green', linewidth=3)
            ax.plot([0, x[i+1]], [0, y[i+1]],
                    linestyle='--', color='darkgrey', alpha=0.7)
        u1, u2 = np.random.random(size=(2, 10000))
        plt.scatter(x[0] * u1 + x[1] * u2, y[0] * u1 + y[1] * u2, s=2)
        plt.savefig(FOLDER+'figure5.pdf', bbox_inches='tight')
        plt.show()

        # Figure 6
        n = 6
        R = 1
        points, points_m, x, y = calculate_triangle_points(R, n)
        t = np.linspace(0, 2*PI, 1000)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot(R*np.cos(t), R*np.sin(t), color='black', alpha=0.5)
        ax.scatter(points[:, 0], points[:, 1], color='red', alpha=0.3)
        ax.scatter([0], [0], color='black')
        ax.scatter(points_m[:, 0], points_m[:, 1], color='blue', alpha=0.3)
        ax.set_title('Sampling from the parallelogram given by the first triangle')
        ax.scatter(x, y, color='green')
        for i in range(n):
            if i == n-1:
                i = -1
            ax.plot([x[i], x[i+1]], [y[i], y[i+1]], color='green', linewidth=3)
            ax.plot([0, x[i+1]], [0, y[i+1]],
                    linestyle='--', color='darkgrey', alpha=0.7)
        u1, u2 = np.random.random(size=(2, 10000))
        verify = u1 + u2 > 1
        u1[verify] = 1 - u1[verify]
        u2[verify] = 1 - u2[verify]
        index = np.random.randint(n, size = 10000)
        x_array_1 = [x[i] for i in index]
        y_array_1 = [y[i] for i in index]
        x_array_2 = [x[i+1] if i < n-1 else x[0] for i in index]
        y_array_2 = [y[i+1] if i < n-1 else y[0] for i in index]
        plt.scatter(x_array_1 * u1 + x_array_2 * u2, y_array_1 * u1 + y_array_2 * u2, s=1)
        plt.savefig(FOLDER+'figure6.pdf', bbox_inches='tight')
        plt.show()

        # Figure 7
        n_values = range(3, 50)
        M_n = [(n * np.tan(PI/n) / PI)**2 for n in n_values]
        plt.plot(n_values, M_n, color='black', linewidth=3)
        plt.axhline(1, color='red', linestyle='--')
        plt.xlabel('n')
        plt.title('M constants for different values of n')
        plt.savefig(FOLDER+'figure7.pdf', bbox_inches='tight')
        plt.show()

    
