import numpy as np
from scipy import constants as cst

def potential_linecharge(x, y, k, x0, y0):
    """
    Electric potential line charge

    Inputs:
        (float) k  : line charge
        (float) x  : x coordinate potential
        (float) y  : y coordinate potential
        (float) x0 : x coordinate wire
        (float) y0 : y coordinate wire
    """
    dr = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    return (1 / ((4 * np.pi * cst.epsilon_0)) * k * np.log(dr))