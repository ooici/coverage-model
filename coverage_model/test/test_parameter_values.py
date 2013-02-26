#!/usr/bin/env python

"""
@package
@file test_parameter_values.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
import coverage_model.parameter_types as ptypes
import coverage_model as cm
import numpy as np
import random

# TODO: Revisit this test class and expand/elaborate the tests

@attr('UNIT',group='cov')
class TestParameterValuesUnit(cm.CoverageModelUnitTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # QuantityType
    def test_quantity_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        qtype = ptypes.QuantityType(value_encoding=np.dtype('float32'))
        qval = cm.get_value_class(qtype, domain_set=dom)

        data = np.arange(10)
        qval[:] = data
        self.assertTrue(np.array_equal(data, qval[:]))

    # BooleanType
    def test_boolean_values(self):
        num_rec = 10

        dom = cm.SimpleDomainSet((num_rec,))

        btype = ptypes.BooleanType()
        bval = cm.get_value_class(btype, domain_set=dom)

        data = [True, False, True, False, False, True, False, True, True, True]
        bval[:] = data
        self.assertTrue(np.array_equal(data, bval[:]))

    # ArrayType
    def test_array_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        atype = ptypes.ArrayType()
        aval = cm.get_value_class(atype, domain_set=dom)

        for x in xrange(num_rec):
            aval[x] = np.random.bytes(np.random.randint(1,20)) # One value (which is a byte string) for each member of the domain

        self.assertIsInstance(aval[0], np.ndarray)
        self.assertIsInstance(aval[0][0], basestring)
        self.assertTrue(1 <= len(aval[0][0]) <= 20)

    # RecordType
    def test_record_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        rtype = ptypes.RecordType()
        rval = cm.get_value_class(rtype, domain_set=dom)

        letts='abcdefghij'
        for x in xrange(num_rec):
            rval[x] = {letts[x]: letts[x:]} # One value (which is a dict) for each member of the domain

        self.assertIsInstance(rval[0], dict)

    # ConstantType
    def test_constant_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        ctype = ptypes.ConstantType(ptypes.QuantityType(value_encoding=np.dtype('int32')))
        cval = cm.get_value_class(ctype, domain_set=dom)
        cval[0] = 200 # Doesn't matter what index (or indices) you assign this to - it's used everywhere!!
        self.assertEqual(cval[0], 200)
        self.assertEqual(cval[7], 200)
        self.assertEqual(cval[2,9], 200)
        self.assertTrue(np.array_equal(cval[[2,7],], np.array([200,200], dtype='int32')))

    # ConstantRangeType
    def test_constant_range_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        crtype = ptypes.ConstantRangeType(ptypes.QuantityType(value_encoding=np.dtype('int16')))
        crval = cm.get_value_class(crtype, domain_set=dom)
        crval[:] = (-10, 10)
        self.assertEqual(crval[0], (-10, 10))
        self.assertEqual(crval[6], (-10, 10))
        comp=np.empty(2,dtype='object')
        comp.fill((-10,10))

        self.assertTrue(np.array_equal(crval[[2,7],], comp))

    # CategoryType
    def test_category_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        # CategoryType example
        cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
        cattype = ptypes.CategoryType(categories=cat)
        catval = cm.get_value_class(cattype, domain_set=dom)

        catkeys = cat.keys()
        for x in xrange(num_rec):
            catval[x] = random.choice(catkeys)

        with self.assertRaises(IndexError):
            catval[20]

        self.assertTrue(catval[0] in cat.values())

    # FunctionType
    def test_function_values(self):
        num_rec = 10
        dom = cm.SimpleDomainSet((num_rec,))

        ftype = ptypes.FunctionType(ptypes.QuantityType(value_encoding=np.dtype('float32')))
        fval = cm.get_value_class(ftype, domain_set=dom)

        fval[:] = cm.make_range_expr(100, min=0, max=4, min_incl=True, max_incl=False, else_val=-9999)
        fval[:] = cm.make_range_expr(200, min=4, max=6, min_incl=True, else_val=-9999)
        fval[:] = cm.make_range_expr(300, min=6, else_val=-9999)

        self.assertEqual(fval[0], 100)
        self.assertEqual(fval[5], 200)
        self.assertEqual(fval[9], 300)
