#!/usr/bin/env python

"""
@package 
@file test_numexpr_utils
@author Christopher Mueller
@brief 
"""

from coverage_model import *
from coverage_model.numexpr_utils import nest_wheres, denest_wheres, is_nested_where, is_well_formed_nested, is_well_formed_where
from nose.plugins.attrib import attr
from unittest import TestCase

@attr('UNIT',group='cov')
class TestNumexprUtilsUnit(TestCase):

    def test_make_range_expr(self):

        s='c*10'
        expr = make_range_expr(10)
        self.assertEqual(s, expr)

        s = 'where(x > 99, 8, -999)'
        expr = make_range_expr(8, min=99, else_val=-999)
        self.assertEqual(s, expr)

        s = 'where(x >= 99, 8, -999)'
        expr = make_range_expr(8, min=99, min_incl=True, else_val=-999)
        self.assertEqual(s, expr)

        s = 'where(x < 10, 100, -999)'
        expr = make_range_expr(100, max=10, max_incl=False, else_val=-999)
        self.assertEqual(s, expr)

        s = 'where(x <= 10, 100, -999)'
        expr = make_range_expr(100, max=10, else_val=-999)
        self.assertEqual(s, expr)

        s = 'where((x > 0) & (x <= 100), 55, 100)'
        expr = make_range_expr(55, min=0, max=100, else_val=100)
        self.assertEqual(s, expr)

        s = 'where((x >= 0) & (x <= 100), 55, 100)'
        expr = make_range_expr(55, min=0, max=100, min_incl=True, else_val=100)
        self.assertEqual(s, expr)

        s = 'where((x >= 0) & (x < 100), 55, 100)'
        expr = make_range_expr(55, min=0, max=100, min_incl=True, max_incl=False, else_val=100)
        self.assertEqual(s, expr)

        s = 'where((x > 0) & (x < 100), 55, 100)'
        expr = make_range_expr(55, min=0, max=100, min_incl=False, max_incl=False, else_val=100)
        self.assertEqual(s, expr)

        self.assertTrue(is_well_formed_where(expr))
        self.assertFalse(is_nested_where(expr))
        self.assertFalse(is_well_formed_nested(expr))

    def test_nest_wheres(self):
        expr1 = make_range_expr(val=-888, max=0, max_incl=False)
        expr2 = make_range_expr(val=111, min=0, min_incl=True, max=10, max_incl=False)
        expr3 = make_range_expr(val=222, max=20, max_incl=False, else_val=-999)

        # Tests that expressions are nested properly
        expr = nest_wheres(expr1, expr2, expr3)
        s = 'where(x < 0, -888, where((x >= 0) & (x < 10), 111, where(x < 20, 222, -999)))'
        self.assertEqual(s, expr)
        self.assertTrue(is_nested_where(expr))
        self.assertTrue(is_well_formed_nested(expr))

        # Tests that invalid expressions are thrown out
        expr = nest_wheres('should not be included', expr1, expr2, 'where(also not included)', expr3)
        self.assertEqual(s, expr)
        self.assertTrue(is_nested_where(expr))
        self.assertTrue(is_well_formed_where(expr))
        self.assertTrue(is_well_formed_nested(expr))

        s = 'where(x < 0, -888, where((x >= 0) & (x < 10), 111, where(x < 20, 222, -999)])'
        self.assertFalse(is_well_formed_nested(s))
        s = 'where(x < 0, -888, where((x >= 0) & (x < 10), 111, where(bad)))'
        self.assertFalse(is_well_formed_nested(s))

        self.assertRaises(IndexError, nest_wheres, 'bob')

    def test_denest_wheres(self):
        expr1 = make_range_expr(val=-888, max=0, max_incl=False)
        expr2 = make_range_expr(val=111, min=0, min_incl=True, max=10, max_incl=False)
        expr3 = make_range_expr(val=222, max=20, max_incl=False, else_val=-9999)

        expr = nest_wheres(expr1, expr2, expr3)

        exprs = denest_wheres(expr)
        self.assertIsInstance(exprs, list)

        e1,e2,e3 = exprs[:]
        self.assertEqual(expr1, e1)
        self.assertEqual(expr2, e2)
        self.assertEqual(expr3, e3)

        self.assertRaises(ValueError, denest_wheres, 'bob')



