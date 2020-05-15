#!/usr/bin/env python
"""
Description:        
Date Created:       
Last Modified:      
"""

import numpy as np
from scipy.spatial.distance import cdist

__author__ = "Matthew Woolley"

def calc_dist2points(curr_point, points, metric='euclidean'):
    """

    :param curr_point:
    :param points:
    :param metric:
    :return:
    """

    # Check curr_point is 1D
    if len(curr_point.shape) > 1:
        raise ValueError(f"curr_point must be 1D array: curr_point is of shape {curr_point.shape}")

    # Check if points is 2D
    if len(points.shape) > 2:
        raise ValueError(f"points must be 2D array: points is of shape {points.shape}")

    # Check for boolean inputs

    return cdist([curr_point], points, metric=metric)

x = np.array([1, 3, 5])
y = np.array([[2, 5, 1], [1, 1, 9], [4, 2, 1], [5, 9, 1]])
print(calc_dist2points(x, y))