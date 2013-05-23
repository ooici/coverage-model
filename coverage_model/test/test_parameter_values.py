#!/usr/bin/env python

"""
@package
@file test_parameter_values.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
import unittest
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

        self.assertIsInstance(aval[0], basestring)
        self.assertTrue(1 <= len(aval[0]) <= 20)

        vals = [[1, 2, 3]] * num_rec
        val_arr = np.empty(num_rec, dtype=object)
        val_arr[:] = vals

        aval[:] = vals
        self.assertTrue(np.array_equal(aval[:], val_arr))
        self.assertIsInstance(aval[0], list)
        self.assertEqual(aval[0], [1, 2, 3])

        aval[:] = val_arr
        self.assertTrue(np.array_equal(aval[:], val_arr))
        self.assertIsInstance(aval[0], list)
        self.assertEqual(aval[0], [1, 2, 3])


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

    def test_parameter_function_values(self):
        pass

    def test_sparse_constant_value(self):
        num_rec = 0
        dom = SimpleDomainSet((num_rec,))
        sctype = SparseConstantType(fill_value=-999)
        scval = get_value_class(sctype, dom)

        scval[:] = 10
        self.assertTrue(np.array_equal(scval[:], np.empty(0, dtype=sctype.value_encoding)))

        dom.shape = (10,)
        self.assertTrue(np.array_equal(scval[:], np.array([10] * 10, dtype=sctype.value_encoding)))

        scval[:] = 20
        dom.shape = (20,)
        out = np.empty(20, dtype=sctype.value_encoding)
        out[:10] = 10
        out[10:] = 20
        self.assertTrue(np.array_equal(scval[:], out))
        self.assertTrue(np.array_equal(scval[2:19], out[2:19]))
        self.assertTrue(np.array_equal(scval[8::3], out[8::3]))

        scval[:] = 30
        dom.shape = (30,)
        out = np.empty(30, dtype=sctype.value_encoding)
        out[:10] = 10
        out[10:20] = 20
        out[20:] = 30
        self.assertTrue(np.array_equal(scval[:], out))
        self.assertTrue(np.array_equal(scval[2:29], out[2:29]))
        self.assertTrue(np.array_equal(scval[12:25], out[12:25]))
        self.assertTrue(np.array_equal(scval[18::3], out[18::3]))

    def test_sparse_referred_value(self):
        num_rec = 0
        scdom = SimpleDomainSet((num_rec,))
        sctype = SparseConstantType(fill_value=-999)
        scval = get_value_class(sctype, scdom)

        qv = get_value_class(QuantityType(), SimpleDomainSet((10,)))
        qv[:] = np.arange(10)
        qv2 = get_value_class(QuantityType(), SimpleDomainSet((14,)))
        qv2[:] = np.arange(100, 114)

        scval[:] = qv
        scdom.shape = (10,)
        self.assertTrue(np.array_equal(scval[:], np.arange(10)))

        scval[:] = qv2
        scdom.shape = (24,)
        self.assertTrue(np.array_equal(scval[:10], np.arange(10)))
        self.assertTrue(np.array_equal(scval[10:], np.arange(100,114)))
        self.assertTrue(np.array_equal(scval[:], np.append(np.arange(10), np.arange(100,114))))


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

    def test_value_caching_with_domain_expansion(self):
        cov = self._make_empty_oneparamcov()

        # Insert some timesteps (automatically expands other arrays)
        nt = 100
        cov.insert_timesteps(nt)

        vals = np.arange(nt, dtype=cov._range_dictionary.get_context('time').param_type.value_encoding)
        cov.set_time_values(vals)

        # Prime the value_cache
        got = cov.get_time_values()

        # Expand the domain
        cov.insert_timesteps(nt)

        # Value cache should still hold 1 and the value should be equal to values retrieved prior to expansion ('got')
        self.assertEqual(len(cov._value_cache), 1)
        self.assertTrue(np.array_equal(cov._value_cache[cov._value_cache.keys()[0]], got))

        # Perform another get, just to make sure the following removes all entries for the parameter
        got = cov.get_time_values(slice(0, 10))

        # Set time values
        cov.set_time_values(range(cov.num_timesteps))

        # Value cache should now be empty because all values cached for 'time' should be removed
        self.assertEqual(len(cov._value_cache), 0)

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

@attr('INT',group='cov')
class TestParameterValuesInteropInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _interop_assertions(self, cov, pname, val_cls, assn_vals=None):
        if assn_vals is not None:
            val_cls[:] = assn_vals
            cov.set_parameter_values(pname, assn_vals)

        self.assertTrue(np.array_equal(cov.get_parameter_values(pname), val_cls[:]))
        self.assertTrue(np.array_equal(cov.get_parameter_values(pname, slice(-1, None)), val_cls[-1:]))
        self.assertTrue(np.array_equal(cov.get_parameter_values(pname, slice(None, None, 3)), val_cls[::3]))
        if isinstance(val_cls.parameter_type, ArrayType) or \
                (hasattr(val_cls.parameter_type, 'base_type') and isinstance(val_cls.parameter_type.base_type, ArrayType)):
            self.assertTrue(np.array_equal(cov.get_parameter_values(pname, 0), val_cls[0]))
            self.assertTrue(np.array_equal(cov.get_parameter_values(pname, -1), val_cls[-1]))
        else:
            self.assertEqual(cov.get_parameter_values(pname, 0), val_cls[0])
            self.assertEqual(cov.get_parameter_values(pname, -1), val_cls[-1])

    ## Must use a specialized set of assertions because np.array_equal doesn't work on arrays of type Sn!!
    def _interop_assertions_str(self, cov, pname, val_cls, assn_vals=None):
        if assn_vals is not None:
            val_cls[:] = assn_vals
            cov.set_parameter_values(pname, assn_vals)

        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname) == val_cls[:]).all())
        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname, slice(-1, None)) == val_cls[-1:]).all())
        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname, slice(None, None, 3)) == val_cls[::3]).all())
        self.assertEqual(cov.get_parameter_values(pname, 0), val_cls[0])
        self.assertEqual(cov.get_parameter_values(pname, -1), val_cls[-1])

    def _setup_cov(self, ntimes, names, types):
        pdict = ParameterDictionary()
        pdict.add_context(ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL), is_temporal=True)
        for i, n in enumerate(names):
            pdict.add_context(ParameterContext(n, param_type=types[i], variability=VariabilityEnum.TEMPORAL))
        tdom = GridDomain(GridShape('temporal', [0]), CRS([AxisTypeEnum.TIME]), MutabilityEnum.EXTENSIBLE)
        cov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom)
        if ntimes != 0:
            cov.insert_timesteps(ntimes)
            cov.set_time_values(range(ntimes))

        return cov

    def test_numeric_value_interop(self):
        # Setup the types
        i8_type = QuantityType(value_encoding='int8')
        i16_type = QuantityType(value_encoding='int16')
        i32_type = QuantityType(value_encoding='int32')
        i64_type = QuantityType(value_encoding='int64')
        f32_type = QuantityType(value_encoding='float32')
        f64_type = QuantityType(value_encoding='float64')

        # Setup the values
        ntimes = 20
        valsi8 = range(ntimes)
        valsi8_arr = np.arange(ntimes, dtype='int8')
        valsi16 = range(ntimes)
        valsi16_arr = np.arange(ntimes, dtype='int16')
        valsi32 = range(ntimes)
        valsi32_arr = np.arange(ntimes, dtype='int32')
        valsi64 = range(ntimes)
        valsi64_arr = np.arange(ntimes, dtype='int64')
        valsf32 = range(ntimes)
        valsf32_arr = np.arange(ntimes, dtype='float32')
        valsf64 = range(ntimes)
        valsf64_arr = np.arange(ntimes, dtype='float64')

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        i8_val = get_value_class(i8_type, dom)
        i16_val = get_value_class(i16_type, dom)
        i32_val = get_value_class(i32_type, dom)
        i64_val = get_value_class(i64_type, dom)
        f32_val = get_value_class(f32_type, dom)
        f64_val = get_value_class(f64_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['i8', 'i16', 'i32', 'i64', 'f32', 'f64'], [i8_type, i16_type, i32_type, i64_type, f32_type, f64_type])

        # Perform the assertions

        # List Assignment
        self._interop_assertions(cov, 'i8', i8_val, valsi8)

        self._interop_assertions(cov, 'i16', i16_val, valsi16)

        self._interop_assertions(cov, 'i32', i32_val, valsi32)

        self._interop_assertions(cov, 'i64', i64_val, valsi64)

        self._interop_assertions(cov, 'f32', f32_val, valsf32)

        self._interop_assertions(cov, 'f64', f64_val, valsf64)

        # Array Assignment
        self._interop_assertions(cov, 'i8', i8_val, valsi8_arr)

        self._interop_assertions(cov, 'i16', i16_val, valsi16_arr)

        self._interop_assertions(cov, 'i32', i32_val, valsi32_arr)

        self._interop_assertions(cov, 'i64', i64_val, valsi64_arr)

        self._interop_assertions(cov, 'f32', f32_val, valsf32_arr)

        self._interop_assertions(cov, 'f64', f64_val, valsf64_arr)

    def test_constant_value_interop(self):
        # Setup the type
        const_type_n = ConstantType()  # QuantityType of float32
        const_type_s = ConstantType(value_encoding='S9')

        # Setup the values
        ntimes = 20
        val = 20
        val_arr = np.array([val])
        sval = 'const str'
        sval_arr = np.array([sval])

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        cn_val = get_value_class(const_type_n, dom)
        cs_val = get_value_class(const_type_s, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['const_num', 'const_str'], [const_type_n, const_type_s])

        # Perform the assertions

        # Single value assignment, numeric
        self._interop_assertions(cov, 'const_num', cn_val, val)

        # Single value assignment, string
        self._interop_assertions_str(cov, 'const_str', cs_val, sval)

        # Array value assignment, numeric
        self._interop_assertions(cov, 'const_num', cn_val, val_arr)

        # Array value assignment, string
        self._interop_assertions_str(cov, 'const_str', cs_val, sval_arr)

    def test_constant_range_value_interop(self):
        # Setup the type
        cr_n_type = ConstantRangeType(value_encoding='float32')
        cr_s_type = ConstantRangeType(value_encoding='S5')

        # Setup the values
        ntimes = 20
        val = (20, 40)
        val_arr = np.empty(1, dtype=object)
        val_arr[0] = val
        val_arr2 = np.array([val])
        sval = ('low', 'high')
        sval_arr = np.empty(1, dtype=object)
        sval_arr[0] = sval
        sval_arr2 = np.array([sval])

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        crn_val = get_value_class(cr_n_type, dom)
        crs_val = get_value_class(cr_s_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['const_rng_num', 'const_rng_str'], [cr_n_type, cr_s_type])

        # Perform the assertions

        # Single value assignment, numeric
        self._interop_assertions(cov, 'const_rng_num', crn_val, val)

        # Single value assignment, string
        self._interop_assertions_str(cov, 'const_rng_str', crs_val, sval)

        # Object array assignment, numeric
        self._interop_assertions(cov, 'const_rng_num', crn_val, val_arr)

        # Object array assignment, string
        self._interop_assertions_str(cov, 'const_rng_str', crs_val, sval_arr)

        # Nd array assignment, numeric
        self._interop_assertions(cov, 'const_rng_num', crn_val, val_arr2)

        # Nd array assignment, string
        self._interop_assertions_str(cov, 'const_rng_str', crs_val, sval_arr2)

    def test_boolean_value_interop(self):
        # Setup the type
        bool_type = BooleanType()

        # Setup the values
        from random import choice
        ntimes = 20
        bvals = [choice([True, False]) for r in range(ntimes)]
        ivals = [choice([-1, 0, 1, 2]) for r in range(ntimes)]
        bvals_arr = np.array(bvals, dtype='bool')
        ivals_arr = np.array(ivals, dtype='int8')

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        bool_val = get_value_class(bool_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['bool'], [bool_type])

        # Perform the assertions

        # List assignment, boolean
        self._interop_assertions(cov, 'bool', bool_val, bvals)

        # List assignment, integer
        self._interop_assertions(cov, 'bool', bool_val, ivals)

        # Array assignment, boolean
        self._interop_assertions(cov, 'bool', bool_val, bvals_arr)

        # Array assignment, integer
        self._interop_assertions(cov, 'bool', bool_val, ivals_arr)

    def test_record_value_interop(self):
        # Setup the type
        rec_type = RecordType()

        # Setup the values
        ntimes = 20
        letts='abcdefghijklmnopqrstuvwxyz'
        rvals = [{letts[x]: letts[x:]} for x in range(ntimes)]
        rvals_arr = np.empty(ntimes, dtype=object)
        rvals_arr[:] = rvals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        rec_val = get_value_class(rec_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['rec'], [rec_type])

        # Perform the assertions

        # List assignment
        self._interop_assertions(cov, 'rec', rec_val, rvals)

        # Array assignment
        self._interop_assertions(cov, 'rec', rec_val, rvals_arr)

    def test_parameter_function_value_interop(self):
        # Setup the type
        numexpr_type = ParameterFunctionType(NumexprFunction('test_func', 'a*2', ['a'], param_map={'a': 'time'}), value_encoding='int32')
        pyfunc_type = ParameterFunctionType(PythonFunction('test_func', 'coverage_model.test.test_parameter_functions', 'pyfunc', ['a','b'], param_map={'a': 'time', 'b': 2}))


        # Setup the values
        ntimes = 20

        def get_vals(name, slice_):
            if name == 'time':
                return np.atleast_1d(range(ntimes))[slice_]
        numexpr_type._pval_callback = get_vals
        pyfunc_type._pval_callback = get_vals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        numexpr_val = get_value_class(numexpr_type, dom)
        pyfunc_val = get_value_class(pyfunc_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['numexpr', 'pyfunc'], [numexpr_type, pyfunc_type])

        # Perform the assertions

        # Make sure the value_encoding is enforced
        self.assertEqual(numexpr_val[:].dtype, np.dtype('int32'))
        self.assertEqual(pyfunc_val[:].dtype, np.dtype('float32'))
        self.assertEqual(cov.get_parameter_values('numexpr').dtype, np.dtype('int32'))
        self.assertEqual(cov.get_parameter_values('pyfunc').dtype, np.dtype('float32'))

        # NumexprFunction
        self._interop_assertions(cov, 'numexpr', numexpr_val)

        # PythonFunction
        self._interop_assertions(cov, 'pyfunc', pyfunc_val)

    @unittest.skip('VectorType not fully implemented')
    def test_vector_value_interop(self):
        # Setup the type

        # Setup the values

        # Setup the in-memory value

        # Setup the coverage

        # Perform the assertions
        pass

    def test_array_value_interop(self):
        # Setup the type
        arr_type = ArrayType()
        arr_type_ie = ArrayType(inner_encoding=np.dtype('int32'))

        # Setup the values
        ntimes = 20
        vals = [[1, 2, 3]] * ntimes
        vals_ie = [[1.2,2.3,3.4]] * ntimes
        vals_arr = np.empty(ntimes, dtype=object)
        vals_arr_ie = np.empty(ntimes, dtype=object)
        vals_arr[:] = vals
        vals_arr_ie[:] = vals_ie
        svals = []
        for x in xrange(ntimes):
            svals.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
        svals_arr = np.empty(ntimes, dtype=object)
        svals_arr[:] = svals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        arr_val = get_value_class(arr_type, dom)
        arr_val_ie = get_value_class(arr_type_ie, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['array', 'array_ie'], [arr_type, arr_type_ie])

        # Perform the assertions

        # Nested List Assignment
        self._interop_assertions(cov, 'array', arr_val, vals)
        self._interop_assertions(cov, 'array_ie', arr_val_ie, vals_ie)

        # Array Assignment
        self._interop_assertions(cov, 'array', arr_val, vals_arr)
        self._interop_assertions(cov, 'array_ie', arr_val_ie, vals_arr_ie)

        # String Assignment via list
        self._interop_assertions_str(cov, 'array', arr_val, svals)

        # String Assignment via array
        self._interop_assertions_str(cov, 'array', arr_val, svals_arr)

    def test_category_value_interop(self):
        # Setup the type
        cats = {0: 'turkey', 1: 'duck', 2: 'chicken', 3: 'empty'}
        cat_type = CategoryType(categories=cats)
        cat_type.fill_value = 3

        # Setup the values
        ntimes = 10
        key_vals = [1, 2, 0, 3, 2, 0, 1, 2, 1, 1]
        cat_vals = [cats[k] for k in key_vals]
        key_vals_arr = np.array(key_vals)
        cat_vals_arr = np.empty(ntimes, dtype=object)
        cat_vals_arr[:] = cat_vals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        cat_val = get_value_class(cat_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['category'], [cat_type])

        # Perform the assertions

        # Assign with a list of keys
        self._interop_assertions_str(cov, 'category', cat_val, key_vals)

        # Assign with a list of categories
        self._interop_assertions_str(cov, 'category', cat_val, cat_vals)

        # Assign with an array of keys
        self._interop_assertions_str(cov, 'category', cat_val, key_vals_arr)

        # Assign with an array of categories
        self._interop_assertions_str(cov, 'category', cat_val, cat_vals_arr)

    def test_sparse_constant_value_interop(self):
         # Setup the type
        scv_type = SparseConstantType(fill_value=-998, value_encoding='int32')
        ifv = 827.38
        scv_arr_type = SparseConstantType(base_type=ArrayType(inner_encoding='float32', inner_fill_value=ifv))

        # Setup the values
        ntimes = 10
        val = 20
        val_list = [20, 40]
        val_arr = np.array(val_list, dtype='int32')
        want = np.array([val] * ntimes, dtype='int32')

        aval = [[20, 39, 58]]
        aval_arr = np.empty(1, dtype=object)
        aval_arr[0] = aval[0]
        awant = np.array(aval * ntimes, dtype='float32')

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        scv_val = get_value_class(scv_type, dom)
        scv_arr_val = get_value_class(scv_arr_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['scv', 'scv_arr'], [scv_type, scv_arr_type])

        # Perform the assertions

        # Assign with val
        scv_val[:] = val
        cov.set_parameter_values('scv', val)
        self._interop_assertions(cov, 'scv', scv_val)
        self.assertTrue(np.array_equal(scv_val[:], want))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv'), want))

        # Assign with aval
        scv_arr_val[:] = aval
        cov.set_parameter_values('scv_arr', aval)
        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        self.assertTrue(np.array_equal(scv_arr_val[:], awant))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv_arr'), awant))

        # Backfill assignment

        # Assign with list
        scv_val[-1] = val_list
        cov.set_parameter_values('scv', val_list, -1)
        self._interop_assertions(cov, 'scv', scv_val)
        self.assertTrue(np.array_equal(scv_val[:], want))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv'), want))

        # Asign with array
        scv_val[-1] = val_arr
        cov.set_parameter_values('scv', val_arr, -1)
        self._interop_assertions(cov, 'scv', scv_val)
        self.assertTrue(np.array_equal(scv_val[:], want))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv'), want))

        scv_arr_val[-1] = aval_arr
        cov.set_parameter_values('scv_arr', aval_arr, -1)
        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        self.assertTrue(np.array_equal(scv_arr_val[:], awant))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv_arr'), awant))

        # Add a new value and expand the domain

        # Change the values
        val = 40
        val_list = [40, 80]
        val_arr = np.array(val_list, dtype='int32')
        want = np.append(want, np.array([val] * ntimes, dtype='int32'))

        aval = [[39, 2, 394, 55]]
        aval_arr = np.empty(1, dtype=object)
        aval_arr[0] = aval
        awant = np.hstack((awant, np.array([[827.38]] * ntimes)))
        awant = np.vstack((awant, np.array(aval * ntimes, dtype='float32')))

        # Assign with val
        scv_val[:] = val
        cov.set_parameter_values('scv', val)

        # Assign with aval
        scv_arr_val[:] = aval
        cov.set_parameter_values('scv_arr', aval)

        # Expand the domain
        dom.shape = (dom.shape[0] + ntimes,)
        cov.insert_timesteps(ntimes)

        # Validate values
        self._interop_assertions(cov, 'scv', scv_val)
        self.assertTrue(np.array_equal(scv_val[:], want))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv'), want))

        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        self.assertTrue(np.allclose(scv_arr_val[:], awant))
        self.assertTrue(np.allclose(cov.get_parameter_values('scv_arr'), awant))

        # Reassign last segment
        val = 44
        want[ntimes:] = val

        aval = [[39, 2, 3, 44]]
        awant[ntimes:] = aval

        # Assign with val
        scv_val[-1] = val
        cov.set_parameter_values('scv', val, -1)

        # Assign with aval
        scv_arr_val[-1] = aval
        cov.set_parameter_values('scv_arr', aval, -1)

        # Validate values
        self._interop_assertions(cov, 'scv', scv_val)
        self.assertTrue(np.array_equal(scv_val[:], want))
        self.assertTrue(np.array_equal(cov.get_parameter_values('scv'), want))

        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        self.assertTrue(np.allclose(scv_arr_val[:], awant))
        self.assertTrue(np.allclose(cov.get_parameter_values('scv_arr'), awant))
