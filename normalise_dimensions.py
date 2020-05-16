#!/usr/bin/env python
"""
Description:
Need to implement sklearn.preprocessing normalisers
Date Created:       
Last Modified:      
"""

import numpy as np

__author__ = "Matthew Woolley"

def normalise_dimensions(points: np.array, method: str='min-max') -> np.array:

    valid_methods = {'min-max', 'mean', 'z-score', 'unit-length'}
    if method not in valid_methods:
        raise ValueError(f"{method} is not a valid method; valid methods are {valid_methods}")

    if method == 'min-max':
        return normalise_using_min_max(points)
    elif method == 'mean':
        return normalise_using_mean(points)
    elif method == 'z-score':
        return normalise_using_z_score(points)
    else:
        return normalise_using_unit_length(points)

def normalise_using_min_max(points: np.array) -> np.array:

    min_vals = points.min(axis=0)
    return (points - min_vals) / (points.max(axis=0) - min_vals)

def normalise_using_mean(points: np.array) -> np.array:

    return (points - points.mean(axis=0)) / (points.max(axis=0) - points.min(axis=0))

def normalise_using_z_score(points: np.array) -> np.array:

    return (points - points.mean(axis=0)) / points.std(axis=0)

def normalise_using_unit_length(points: np.array) -> np.array:

    return points / np.linalg.norm(points, axis=0)




#normalise_dimensions(np.array([1, 2]), 'max')

x = np.array([[1, 2], [3, 1], [5, 3]])
print(normalise_dimensions(x, 'unit-length'))


