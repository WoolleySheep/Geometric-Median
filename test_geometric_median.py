#!/usr/bin/env python
"""
Description: Tests for geometric_median.py
Date Created: 16-MAY-2020
Last Modified: 17-MAY-2020
"""
__author__ = "Matthew Woolley"


import numpy as np
import unittest
from geometric_median import geometric_median


class TestGeometricMedian(unittest.TestCase):

    def test_point_type(self):
        # Non numpy array raises Type Error
        self.assertRaises(TypeError, geometric_median, "Drax")
        self.assertRaises(TypeError, geometric_median, 1)
        self.assertRaises(TypeError, geometric_median, 1.0)
        self.assertRaises(TypeError, geometric_median, True)
        self.assertRaises(TypeError, geometric_median, [])
        self.assertRaises(TypeError, geometric_median, {})
        self.assertRaises(TypeError, geometric_median, dict())
        self.assertRaises(TypeError, geometric_median, (1,))
        # Non int or float dtype raises Type Error
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=complex))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=bool))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=str))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=dict))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=list))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=set))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=tuple))
        self.assertRaises(TypeError, geometric_median, np.array([], dtype=object))

    def test_point_value(self):
        # Non 2D array raises Value Error
        self.assertRaises(ValueError, geometric_median, np.array([1, 2]))
        self.assertRaises(ValueError, geometric_median, np.array([[[]]]))
        # Not having >= one point raises Value Error
        self.assertRaises(ValueError, geometric_median, np.array([]))
        # Not having >= one dimension raises Value Error
        self.assertRaises(ValueError, geometric_median, np.array([[], []]))

    def test_weight_type(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        self.assertRaises(TypeError, geometric_median, points, weights="Drax")
        self.assertRaises(TypeError, geometric_median, points, weights=1)
        self.assertRaises(TypeError, geometric_median, points, weights=1.0)
        self.assertRaises(TypeError, geometric_median, points, weights=True)
        self.assertRaises(TypeError, geometric_median, points, weights=[])
        self.assertRaises(TypeError, geometric_median, points, weights={})
        self.assertRaises(TypeError, geometric_median, points, weights=dict())
        self.assertRaises(TypeError, geometric_median, points, weights=(1,))
        # Non int or float dtype raises Type Error
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=complex))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=bool))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=str))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=dict))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=list))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=set))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=tuple))
        self.assertRaises(TypeError, geometric_median, points, weights=np.array([], dtype=object))

    def test_weights_value(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Non 1D array raises Value Error
        self.assertRaises(ValueError, geometric_median, points, weights=np.array([[1], [2]]))
        # Not having >= one point raises Value Error
        self.assertRaises(ValueError, geometric_median, points, weights=np.array([]))

    def test_method_value(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Object not in valid_methods list
        self.assertRaises(ValueError, geometric_median, points, method='Drax')

    def test_convergence_threshold_type(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Non int or float type raises Type Error
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold="Drax")
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold=True)
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold=[])
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold={})
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold=dict())
        self.assertRaises(TypeError, geometric_median, points, convergence_threshold=(1,))

    def test_convergence_threshold_value(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Value <= 0 raises Value Error
        self.assertRaises(ValueError, geometric_median, points, convergence_threshold=0)
        self.assertRaises(ValueError, geometric_median, points, convergence_threshold=-1)
        self.assertRaises(ValueError, geometric_median, points, convergence_threshold=-1.0)
        self.assertRaises(ValueError, geometric_median, points, convergence_threshold=-0.00001)

    def test_iteration_limit_type(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Non int or float type raises Type Error
        self.assertRaises(TypeError, geometric_median, points, iteration_limit="Drax")
        self.assertRaises(TypeError, geometric_median, points, iteration_limit=1.0)
        self.assertRaises(TypeError, geometric_median, points, iteration_limit=True)
        self.assertRaises(TypeError, geometric_median, points, iteration_limit=[])
        self.assertRaises(TypeError, geometric_median, points, iteration_limit={})
        self.assertRaises(TypeError, geometric_median, points, iteration_limit=dict())
        self.assertRaises(TypeError, geometric_median, points, iteration_limit=(1,))

    def test_iteration_limit_value(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # Value <= 0 raises Value Error
        self.assertRaises(ValueError, geometric_median, points, iteration_limit=0)
        self.assertRaises(ValueError, geometric_median, points, iteration_limit=-1)

    def test_dist_measure_type(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # String not in valid list (scipy.spatial.distance.cdist)
        self.assertRaises(TypeError, geometric_median, points, solver='weiszfeld', dist_measure='Drax')

    def test_solver_method_value(self):
        points = np.array([[1, 2], [2, 1], [2, 8]])
        # String not in valid list (scipy.optimize.minimize)
        self.assertRaises(TypeError, geometric_median, points, solver='minimize', solver_method="Drax")




