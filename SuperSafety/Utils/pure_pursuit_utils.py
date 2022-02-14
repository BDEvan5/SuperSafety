"""
Code source: https://github.com/f1tenth/f1tenth_gym
Example waypoint_follow.py from f1tenth_gym

Minor adjustments have been made

"""

import numpy as np
from numpy import genfromtxt
import numpy.linalg as LA
import scipy.interpolate as interpolate
import json, time, collections

from numba import njit


EPSILON = 0.00000000001

@njit(fastmath=False, cache=True)
def get_rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


    # print min_dist_segment, dists[min_dist_segment], projections[min_dist_segment]
