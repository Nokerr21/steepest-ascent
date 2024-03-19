"""
Author: Wojciech Kondracki 310941
Date: 08.03.2024
"""


def booth(x):
    """
    Calculate the Booth function.

    Parameters:
        x (list or np.array): A two-dimensional vector.

    Returns:
        float: The value of the Booth function at the given point.
    """

    result = (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2
    return result
