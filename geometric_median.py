#!/usr/bin/env python
"""
Description:        
Date Created:       
Last Modified:      
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

__author__ = "Matthew Woolley"

def geometric_median(points: np.array, weights: np.array=None, method: str='weiszfeld', convergence_threshold: float=1e-5, iteration_limit: int=1000, dist_measure: str='euclidean', solver_method=None):

    # Raise errors for improper inputs

    # points errors
    if type(points) is not np.ndarray:
        raise TypeError(f"Type of points must be a numpy array; current type is {type(points)}")
    if (np.issubdtype(points.dtype, np.integer) or np.issubdtype(points.dtype, np.floating)) is False:
        raise TypeError(f"Datatype of points must be int or float; current datatype is {points.dtype}")
    if len(points.shape) != 2:
        raise ValueError(f"points must be a 2D array; currently shape is {points.shape}")

    npoint, ndim = points.shape

    if ndim < 1:
        raise ValueError(f"value of ndim must be >= 1; currently value is {ndim}")

    # weights errors
    if weights is None:
        weights = np.ones(npoint)
    else:
        if type(weights) is not np.ndarray:
            raise TypeError(f"Type of weights must be a numpy array; current type is {type(points)}")
        if (np.issubdtype(weights.dtype, np.integer) or np.issubdtype(weights.dtype, np.floating)) is False:
            raise TypeError(f"Datatype of points must be real numbers; current datatype is {points.dtype}")
        if len(weights.shape) != 1:
            raise ValueError(f"weights must be a 1D array; currently shape is {weights.shape}")

    # method errors
    valid_methods = {'weiszfeld', 'minimize'}
    if method not in valid_methods:
        raise ValueError(f"Invalid method given: {method} not in {valid_methods}")

    # convergence_threshold errors
    if type(convergence_threshold) not in (int, float):
        raise TypeError(f"Type of convergence_threshold must be int or float; currently of type {type(convergence_threshold)}")
    if convergence_threshold <= 0:
        raise ValueError(f"Value of convergence_threshold must be > 0; current value is {convergence_threshold}")

    # iteration_limit errors
    if type(iteration_limit) is not int:
        raise TypeError(f"Type of iteration_limit must be int; current type is {type(iteration_limit)}")
    if iteration_limit <= 0:
        raise ValueError(f"Value of iteration_limit must be > 0; current value is {iteration_limit}")

    # distance_measure errors
    # Caught by scipy.spatial.distance.cdist

    # solver_method errors
    # Caught by scipy.optimize.minimize

    if method == 'weiszfeld':
        return weiszfeld_algorithm(points, weights, convergence_threshold, iteration_limit, dist_measure)
    elif method == 'minimize':
        return minimize_algorithm(points, weights, convergence_threshold, iteration_limit, dist_measure, solver_method)


def weiszfeld_algorithm(points: np.ndarray, weights: np.array, convergence_threshold: float, iteration_limit: int, dist_measure: str) -> np.array:


    # Find the weighted centroid and set as the initial center
    curr_center = (weights[:, None] * points).sum(axis=0) / weights.sum()

    move_dist = float('inf')
    for _ in range(iteration_limit):
        if move_dist < convergence_threshold:
            return curr_center
        # If curr_center is the same as one of the points, move the current center slightly away from that point
        # The weiszfeld algorithm will fail to converge if it gets stuck on one of the points
        if np.isclose(points, curr_center).all(axis=1).any():
            _, ndim = points.shape
            dim = np.random.choice(range(ndim))
            dir = np.random.choice([1, -1])
            curr_center[dim] += dir * convergence_threshold / 2
        prev_center = curr_center
        # Calculate the weighted distances from the current center to all points
        weighted_distances = cdist(np.array([prev_center]), points, metric=dist_measure) / weights
        # Get new center prediction
        curr_center = (points / weighted_distances.T).sum(axis=0) / (1.0 / weighted_distances).sum()
        # Calculate the distance between the current center and the previous center
        move_dist = cdist(np.array([curr_center]), np.array([prev_center]), metric=dist_measure)[0]

    raise ValueError(f"Weiszfelds algorithm not able to converge within {iteration_limit} iterations")


def minimize_algorithm(points: np.ndarray, weights: np.array, convergence_threshold: float, iteration_limit: int, dist_measure: str, solver_method: str) -> np.array:


    def calc_weighted_distance(curr_center):

        return (weights * cdist(np.array([curr_center]), points, metric=dist_measure)).sum()

    # Find the weighted centroid and set as the initial center
    curr_center = (weights[:, None] * points).sum(axis=0) / weights.sum()

    optimize_result = minimize(calc_weighted_distance, curr_center, method=solver_method, tol=convergence_threshold, options={'maxiter': iteration_limit})

    return optimize_result.x


def predict_optiomal_method(npoint, ndim, convergence_threshold):

    # TODO: Need to find time complexity estimates of methods
    methods = ["Weizfeld", "Vardi-Zhang"]

    time = []
    # Weizfeld
    # Vardi-Zhang

    return

