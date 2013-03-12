#!/usr/bin/env python

"""
@package
@file test_parameter_values.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
from coverage_model import *
import numpy as np
import random

# TODO: Revisit this test class and expand/elaborate the tests


@attr('UNIT',group='cov')
class TestParameterValuesUnit(CoverageModelUnitTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # QuantityType
    def test_quantity_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        qtype = QuantityType(value_encoding=np.dtype('float32'))
        qval = get_value_class(qtype, domain_set=dom)

        data = np.arange(10)
        qval[:] = data
        self.assertTrue(np.array_equal(data, qval[:]))

    # BooleanType
    def test_boolean_values(self):
        num_rec = 10

        dom = SimpleDomainSet((num_rec,))

        btype = BooleanType()
        bval = get_value_class(btype, domain_set=dom)

        data = [True, False, True, False, False, True, False, True, True, True]
        bval[:] = data
        self.assertTrue(np.array_equal(data, bval[:]))

    # ArrayType
    def test_array_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        atype = ArrayType()
        aval = get_value_class(atype, domain_set=dom)

        for x in xrange(num_rec):
            aval[x] = np.random.bytes(np.random.randint(1,20)) # One value (which is a byte string) for each member of the domain

        self.assertIsInstance(aval[0], np.ndarray)
        self.assertIsInstance(aval[0][0], basestring)
        self.assertTrue(1 <= len(aval[0][0]) <= 20)

    # RecordType
    def test_record_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        rtype = RecordType()
        rval = get_value_class(rtype, domain_set=dom)

        letts='abcdefghij'
        for x in xrange(num_rec):
            rval[x] = {letts[x]: letts[x:]} # One value (which is a dict) for each member of the domain

        self.assertIsInstance(rval[0], dict)

    # ConstantType
    def test_constant_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        ctype = ConstantType(QuantityType(value_encoding=np.dtype('int32')))
        cval = get_value_class(ctype, domain_set=dom)
        cval[0] = 200 # Doesn't matter what index (or indices) you assign this to - it's used everywhere!!
        self.assertEqual(cval[0], 200)
        self.assertEqual(cval[7], 200)
        self.assertEqual(cval[2,9], 200)
        self.assertTrue(np.array_equal(cval[[2,7],], np.array([200,200], dtype='int32')))

    # ConstantRangeType
    def test_constant_range_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        crtype = ConstantRangeType(QuantityType(value_encoding=np.dtype('int16')))
        crval = get_value_class(crtype, domain_set=dom)
        crval[:] = (-10, 10)
        self.assertEqual(crval[0], (-10, 10))
        self.assertEqual(crval[6], (-10, 10))
        comp=np.empty(2,dtype='object')
        comp.fill((-10,10))

        self.assertTrue(np.array_equal(crval[[2,7],], comp))

    # CategoryType
    def test_category_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        # CategoryType example
        cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
        cattype = CategoryType(categories=cat)
        catval = get_value_class(cattype, domain_set=dom)

        catkeys = cat.keys()
        for x in xrange(num_rec):
            catval[x] = random.choice(catkeys)

        with self.assertRaises(IndexError):
            catval[20]

        self.assertTrue(catval[0] in cat.values())

    # FunctionType
    def test_function_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        ftype = FunctionType(QuantityType(value_encoding=np.dtype('float32')))
        fval = get_value_class(ftype, domain_set=dom)

        fval[:] = make_range_expr(100, min=0, max=4, min_incl=True, max_incl=False, else_val=-9999)
        fval[:] = make_range_expr(200, min=4, max=6, min_incl=True, else_val=-9999)
        fval[:] = make_range_expr(300, min=6, else_val=-9999)

        self.assertEqual(fval[0], 100)
        self.assertEqual(fval[5], 200)
        self.assertEqual(fval[9], 300)


@attr('INT',group='cov')
class TestParameterValuesInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_value_caching(self):
        import collections

        cov = self._make_empty_oneparamcov()

        # Insert some timesteps (automatically expands other arrays)
        nt = 2000
        cov.insert_timesteps(nt)

        vals = np.arange(nt, dtype=cov._range_dictionary.get_context('time').param_type.value_encoding)
        cov.set_time_values(vals)

        # Make sure the _value_cache is an instance of OrderedDict and that it's empty
        self.assertIsInstance(cov._value_cache, collections.OrderedDict)
        self.assertEqual(len(cov._value_cache), 0)

        # Get the time values and make sure they match what we assigned
        got = cov.get_time_values()
        self.assertTrue(np.array_equal(vals, got))

        # Now check that there is 1 entry in the _value_cache and that it's a match for vals
        self.assertEqual(len(cov._value_cache), 1)
        self.assertTrue(np.array_equal(cov._value_cache[cov._value_cache.keys()[0]], vals))

        # Now retrieve a slice and make sure it matches the same slice of vals
        sl = slice(20, 1000, 3)
        got = cov.get_time_values(sl)
        self.assertTrue(np.array_equal(vals[sl], got))

        # Now check that there are 2 entries and that the second is a match for vals
        self.assertEqual(len(cov._value_cache), 2)
        self.assertTrue(np.array_equal(cov._value_cache[cov._value_cache.keys()[1]], vals[sl]))

        # Call get 40 times - check that the _value_cache stops growing @ 30
        for x in xrange(40):
            cov.get_time_values(x)
        self.assertEqual(len(cov._value_cache), 30)


    def _make_empty_oneparamcov(self):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)

        return scov
