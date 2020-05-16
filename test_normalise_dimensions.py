#!/usr/bin/env python
"""
Description: Unit tests for normalise_dimensions.py
Date Created: 16-MAY-2020
Last Modified: 16-MAY-2020
"""
__author__ = "Matthew Woolley"

import numpy as np
import unittest
from normalise_dimensions import normalise_using_min_max


class TestNormaliseUsingMinMax(unittest.TestCase):


    def test_shape(self):
        # Test for no points
        self.assertRaises(TypeError, normalise_using_min_max, np.array([]))
        # Test for no dimensions
        self.assertRaises(TypeError, normalise_using_min_max, np.array([[]]))


    def test_point_type(self):
        # Test for non-numpy array inputs
        self.assertRaises(TypeError, normalise_using_min_max, None)
        self.assertRaises(TypeError, normalise_using_min_max, 'Drax')
        self.assertRaises(TypeError, normalise_using_min_max, 1)
        self.assertRaises(TypeError, normalise_using_min_max, 1.0)
        self.assertRaises(TypeError, normalise_using_min_max, [1, 2])
        self.assertRaises(TypeError, normalise_using_min_max, {1: 2, 3: 4})
        self.assertRaises(TypeError, normalise_using_min_max, {1, 2})


    def test_point_datatype(self):
        # Test for non real number datatypes
        # Test boolean
        self.assertRaises(TypeError, normalise_using_min_max, np.array([True, False]))
        # Test strings
        self.assertRaises(TypeError, normalise_using_min_max, np.array(["1", "2"]))
        # Test complex
        self.assertRaises(TypeError, normalise_using_min_max, np.array([1+1j, 2+2j]))


    def test_value(self):
        # Test standard input
        data = np.array([[1, 2], [3, 4], [2, 1]])
        result = np.array([[0, 1 / 3], [1, 1], [0.5, 0]])
        np.testing.assert_array_almost_equal(normalise_using_min_max(data), result)
        # Test for one point
        data = np.array([1, 2, 3, 4])
        result = np.array()

        # Test for one point (or repeated same points)
        # Test for one dimension
        # Test for nan
        # Test for repeated points
        # Test for same number of points as dimensions
        # Test for all values are zero



