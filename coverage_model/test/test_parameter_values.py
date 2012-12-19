#!/usr/bin/env python

"""
@package
@file test_parameter_values.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
import coverage_model.parameter_types as ptypes
import coverage_model.basic_types as btypes
import coverage_model as cm
import numpy as np
import random
from unittest import TestCase

@attr('UNIT',group='cov')
class TestParameterValuesUnit(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_values_outside_coverage(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        # QuantityType example
        qtype = ptypes.QuantityType(value_encoding=np.dtype('float32'))
        qval = cm.get_value_class(qtype, domain_set=dom)

        # ArrayType example
        atype = ptypes.ArrayType()
        aval = cm.get_value_class(atype, domain_set=dom)

        # RecordType example
        rtype = ptypes.RecordType()
        rval = cm.get_value_class(rtype, domain_set=dom)

        # ConstantType example
        ctype = ptypes.ConstantType(ptypes.QuantityType(value_encoding=np.dtype('int32')))
        cval = cm.get_value_class(ctype, domain_set=dom)

        # FunctionType example
        ftype = ptypes.FunctionType(ptypes.QuantityType(value_encoding=np.dtype('float32')))
        fval = cm.get_value_class(ftype, domain_set=dom)

        # CategoryType example
        cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
        cattype = ptypes.CategoryType(categories=cat)
        catval = cm.get_value_class(cattype, domain_set=dom)

        # Add data to the values
        qval[:] = np.random.random_sample(num_rec)*(50-20)+20 # array of 10 random values between 20 & 50

        letts='abcdefghij'
        catkeys = cat.keys()
        for x in xrange(num_rec):
            aval[x] = np.random.bytes(np.random.randint(1,20)) # One value (which is a byte string) for each member of the domain
            rval[x] = {letts[x]: letts[x:]} # One value (which is a dict) for each member of the domain
            catval[x] = [random.choice(catkeys)]

        cval[0] = 200 # Doesn't matter what index (or indices) you assign this to - it's used everywhere!!

        fval[:] = cm.make_range_expr(100, min=0, max=4, min_incl=True, max_incl=False, else_val=-9999)
        fval[:] = cm.make_range_expr(200, min=4, max=6, min_incl=True, else_val=-9999)
        fval[:] = cm.make_range_expr(300, min=6, else_val=-9999)

        self.assertTrue(aval.shape == rval.shape == cval.shape)

        qval_val = qval[:]
        aval_val = aval[:]
        rval_val = rval[:]
        cval_val = cval[:]
        fval_val = fval[:]