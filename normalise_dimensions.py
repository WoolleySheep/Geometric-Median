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

    if type(points) is not np.ndarray:
        raise TypeError("points must a a numpy array")
    if len(points.shape) != 2:
        raise TypeError(f"points must be a 2D array; currently of shape {points.shape}")
    if np.isrealobj(points) is False:
        raise TypeError("The data type of points must be real numeric")

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    zero_diff_indexes = np.isclose(min_vals, max_vals)
    # This needs work
    #return (points - min_vals) / (points.max(axis=0) - min_vals)
def normalise_using_mean(points: np.array) -> np.array:

    return (points - points.mean(axis=0)) / (points.max(axis=0) - points.min(axis=0))

def normalise_using_z_score(points: np.array) -> np.array:

    return (points - points.mean(axis=0)) / points.std(axis=0)

def normalise_using_unit_length(points: np.array) -> np.array:

    return points / np.linalg.norm(points, axis=0)
