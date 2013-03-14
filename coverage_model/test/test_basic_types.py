#!/usr/bin/env python

"""
@package coverage_model.test.test_basic_types
@file coverage_model/test/test_basic_types.py
@author Christopher Mueller
@brief Unit tests for basic_types module
"""

from nose.plugins.attrib import attr
from coverage_model import CoverageModelUnitTestCase

from coverage_model.basic_types import Span, NdSpan
import numpy as np

@attr('UNIT', group='now')
class TestBasicTypesUnit(CoverageModelUnitTestCase):

    def test_span(self):
        min_int = np.iinfo('int64').min
        max_int = np.iinfo('int64').max

        # Unlimited bounds
        s = Span(value=10)
        self.assertIn(min_int, s)
        self.assertIn(0, s)
        self.assertIn(max_int, s)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.value, 10)


        # Unlimited lower bound
        s = Span(None, 5, value='bob')
        self.assertIn(min_int, s)
        self.assertIn(-18, s)
        self.assertIn(0, s)
        self.assertNotIn(5, s)
        self.assertNotIn(6, s)
        self.assertNotIn(19, s)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.value, 'bob')

        # Unlimited upper bound
        s = Span(5, None, value=9.38)
        self.assertIn(max_int, s)
        self.assertIn(18, s)
        self.assertIn(6, s)
        self.assertIn(5, s)
        self.assertNotIn(4, s)
        self.assertNotIn(-12, s)
        self.assertEqual(len(s), 0)
        self.assertEqual(s.value, 9.38)

        # Fully bounded
        s = Span(10, 68, value=4)
        self.assertIn(34, s)
        self.assertNotIn(8, s)
        self.assertNotIn(90, s)
        self.assertEqual(len(s), 58)
        self.assertEqual(s.value, 4)

    def test_ndspan(self):
        min_int = np.iinfo('int64').min
        max_int = np.iinfo('int64').max
        
        s = NdSpan([Span(None,10), Span(3,30), Span(20,None)])
        self.assertIn((8, 5, 23), s)
        self.assertIn((min_int, 29, max_int), s)
        self.assertNotIn((12, 5, 28), s)
        self.assertNotIn([-29, 30, 21], s)
        self.assertNotIn([2, 23, 19], s)
        self.assertEqual(s.shape, (0, 27, 0))
        self.assertEqual(len(s), 0)

        s = NdSpan([Span(0, 20), Span(4,8), Span(9, 18)])
        self.assertIn((7, 4, 15), s)
        self.assertIn((0, 7, 10), s)
        self.assertNotIn((20, 3, 18), s)
        self.assertEqual(s.shape, (20, 4, 9))
        self.assertEqual(len(s), 20 * 4 * 9)

    