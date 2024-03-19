"""
Author: Wojciech Kondracki 310941
Date: 08.03.2024
"""

import matplotlib.pyplot as plt
import cec2017
import numpy as np
from cec2017.functions import f1, f2, f3
from logic import steepest_ascent
from booth import booth


if __name__ == '__main__':
    MAX_X = 100
    PLOT_STEP = 0.1
    # DIMENSIONALITY must be equal to 10 for the f1, f2 and f3 functions
    DIMENSIONALITY = 10

    x_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    y_arr = np.arange(-MAX_X, MAX_X, PLOT_STEP)
    x = np.random.uniform(-MAX_X, MAX_X, DIMENSIONALITY)

    X, Y = np.meshgrid(x_arr, y_arr)
    Z = np.empty(X.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # Z[i, j] = booth(np.array([X[i, j], Y[i, j]]))
            # Alternative functions commented out
            Z[i, j] = f1(np.array([X[i, j], Y[i, j]]))
            # Z[i, j] = f2(np.array([X[i, j], Y[i, j]]))
            # Z[i, j] = f3(np.array([X[i, j], Y[i, j]]))

    plt.contour(X, Y, Z, 300, zorder=0)

    # x = steepest_ascent(x, 0.01, booth, 0.0001, 30000, 100, True, True)
    # Alternate optimization function calls commented out
    x = steepest_ascent(x, 0.00000001, f1, 0.0001, 30000, 100, True, True)
    # x = steepest_ascent(x, 0.000000000000000001, f2, 0.001, 30000, 100, True, True)
    # x = steepest_ascent(x, 0.0000000000001, f3, 0.001, 30000, 100, True, True)

    print(x)
    plt.legend()
    plt.show()
