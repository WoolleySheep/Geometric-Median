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
    if type(points) is not np.ndarray:
        raise TypeError("points must a a numpy array")
    if len(points.shape) != 2:
        raise TypeError(f"points must be a 2D array; currently of shape {points.shape}")
    if np.isrealobj(points) is False:
        raise TypeError("The data type of points must be real numeric")

    if method == 'min-max':
        return normalise_using_min_max(points)
    elif method == 'mean':
        return normalise_using_mean(points)
    elif method == 'z-score':
        return normalise_using_z_score(points)
    else:
        return normalise_using_unit_length(points)

def normalise_using_min_max(points: np.array) -> np.array:
    """
    Normalises the points between 0 and 1

    :param points:
    :return: The normalised points, shape(npoint, ndim) array of real numbers
    """

    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    # Identify the dimensions where all points have the same value
    zero_diff_indexes = np.isclose(min_vals, max_vals)
    result = np.empty(points.shape)
    result[:, zero_diff_indexes] = 1 / points.shape[0]
    result[:, ~zero_diff_indexes] = (points[:, ~zero_diff_indexes] - min_vals[~zero_diff_indexes]) / (max_vals[~zero_diff_indexes] - min_vals[~zero_diff_indexes])

    return result, min_vals, max_vals

def normalise_using_mean(points: np.array) -> np.array:
    """
    Normalises the points around the mean

    :param points: shape(npoint, ndim) array of real numbers
    :return: The normalised points, shape(npoint, ndim) array of real numbers
    """

    mean_vals = points.mean(axis=0)
    min_vals = points.min(axis=0)
    max_vals = points.max(axis=0)
    # Identify the dimensions where all points have the same value
    zero_diff_indexes = np.isclose(min_vals, max_vals)
    result = np.empty(points.shape)
    result[:, zero_diff_indexes] = 1 / points.shape[0]
    result[:, ~zero_diff_indexes] = (points[:, ~zero_diff_indexes] - mean_vals[~zero_diff_indexes]) / (max_vals[~zero_diff_indexes] - min_vals[~zero_diff_indexes])

    return result, mean_vals, min_vals, max_vals


def normalise_using_z_score(points: np.array) -> np.array:
    """
    Normalises

    :param points:
    :return:
    """

    mean_vals = points.mean(axis=0)
    std_vals = points.std(axis=0)
    # Identify the dimensions where all points have the same value (zero std)
    zero_std_indexes = np.isclose(std_vals, 0)
    result = np.empty(points.shape)
    result[:, zero_std_indexes] = 0
    result[:, ~zero_std_indexes] = (points[~zero_std_indexes] - mean_vals[~zero_std_indexes]) / std_vals[~zero_std_indexes]

    return result, mean_vals, std_vals

def normalise_using_unit_length(points: np.array) -> np.array:

    unit_lengths = np.linalg.norm(points, axis=0)
    zero_length_indexes = np.isclose(unit_lengths, 0)
    result = np.empty(points.shape)
    result[:, zero_length_indexes] = 0
    result[:, ~zero_length_indexes] = points[~zero_length_indexes] / unit_lengths[~zero_length_indexes]

    return result, zero_length_indexes

data = np.array([[1, 2], [1, 3]])
normalise_using_min_max(data)