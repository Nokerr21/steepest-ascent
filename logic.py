"""
Author: Wojciech Kondracki 310941
Date: 08.03.2024
"""

import numpy as np
import matplotlib.pyplot as plt
from autograd import grad


def steepest_ascent(x0, b, q, eps, max_iter, limit, plot=False, info=False):
    """
    Perform the steepest ascent optimization algorithm.

    Parameters:
        x0 (np.array): Initial point.
        b (float): Beta - step size.
        q (function): Function to be minimized.
        eps (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
        limit (int or float): Boundary limit for the solution.
        plot (bool, optional): If True, plot the path of the optimization.
        info (bool, optional): If True, print final q(x) and total iterations.

    Returns:
        np.array: The point that approximates the optimum.
    """

    q_grad = grad(q)
    x = x0
    iteration = 0

    stop_flag = False

    if plot:
        plt.plot(x[0], x[1], 'ro', markersize=4, label="Start point")

    while not stop_flag:
        iteration += 1
        d = q_grad(x)
        prev_x = x
        x = x - b * d

        for i in range(len(x)):
            if x[i] > limit:
                x[i] = limit

            elif x[i] < -limit:
                x[i] = -limit

        if plot:
            if np.linalg.norm(x[0] - prev_x[0]) < 2 and np.linalg.norm(x[1] - prev_x[1]) < 2:
                plt.arrow(prev_x[0], prev_x[1], x[0] - prev_x[0], x[1] - prev_x[1],
                          head_width=1, head_length=2, fc='k', ec='k')
            else:
                plt.arrow(prev_x[0], prev_x[1], x[0] - prev_x[0], x[1] - prev_x[1],
                          head_width=2.5, head_length=5, fc='k', ec='k')

        if np.linalg.norm(d) < eps or np.abs(q(prev_x) - q(x)) < eps or iteration == max_iter:
            if plot:
                plt.plot(x[0], x[1], 'go', markersize=4, label="End point")

            if info:
                print(f"q(x) = {q(x):.6f}, total iterations = {iteration}")

            stop_flag = True

    return x
