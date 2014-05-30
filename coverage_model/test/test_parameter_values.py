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
        np.testing.assert_array_equal(data, qval[:])

    # BooleanType
    def test_boolean_values(self):
        num_rec = 10

        dom = SimpleDomainSet((num_rec,))

        btype = BooleanType()
        bval = get_value_class(btype, domain_set=dom)

        data = [True, False, True, False, False, True, False, True, True, True]
        bval[:] = data
        np.testing.assert_array_equal(data, bval[:])

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
        np.testing.assert_array_equal(aval[:], val_arr)
        self.assertIsInstance(aval[0], list)
        self.assertEqual(aval[0], [1, 2, 3])

        aval[:] = val_arr
        np.testing.assert_array_equal(aval[:], val_arr)
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
        np.testing.assert_array_equal(cval[[2,7],], np.array([200,200], dtype='int32'))

    # ConstantRangeType
    def test_constant_range_values(self):
        num_rec = 10
        dom = SimpleDomainSet((num_rec,))

        crtype = ConstantRangeType(QuantityType(value_encoding=np.dtype('int16')), fill_value=(0,0))
        crval = get_value_class(crtype, domain_set=dom)
        crval[:] = (-10, 10)
        self.assertEqual(tuple(crval[0]), (-10, 10))
        self.assertEqual(tuple(crval[6]), (-10, 10))
        comp=np.empty(2,dtype='object')
        comp.fill((-10,10))

        np.testing.assert_array_equal(crval[[2,7],], comp)

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
        np.testing.assert_array_equal(scval[:], np.empty(0, dtype=sctype.value_encoding))

        dom.shape = (10,)
        np.testing.assert_array_equal(scval[:], np.array([10] * 10, dtype=sctype.value_encoding))

        scval[:] = 20
        dom.shape = (20,)
        out = np.empty(20, dtype=sctype.value_encoding)
        out[:10] = 10
        out[10:] = 20
        np.testing.assert_array_equal(scval[:], out)
        np.testing.assert_array_equal(scval[2:19], out[2:19])
        np.testing.assert_array_equal(scval[8::3], out[8::3])

        scval[:] = 30
        dom.shape = (30,)
        out = np.empty(30, dtype=sctype.value_encoding)
        out[:10] = 10
        out[10:20] = 20
        out[20:] = 30
        np.testing.assert_array_equal(scval[:], out)
        np.testing.assert_array_equal(scval[2:29], out[2:29])
        np.testing.assert_array_equal(scval[12:25], out[12:25])
        np.testing.assert_array_equal(scval[18::3], out[18::3])

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
        np.testing.assert_array_equal(scval[:], np.arange(10))

        scval[:] = qv2
        scdom.shape = (24,)
        np.testing.assert_array_equal(scval[:10], np.arange(10))
        np.testing.assert_array_equal(scval[10:], np.arange(100,114))
        np.testing.assert_array_equal(scval[:], np.append(np.arange(10), np.arange(100,114)))


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

        vals = {cov.temporal_parameter_name: np.arange(nt, dtype=cov._range_dictionary.get_context('time').param_type.value_encoding)}
        cov.set_parameter_values(make_parameter_data_dict(vals))

        # Make sure the _value_cache is an instance of OrderedDict and that it's empty
        self.assertIsInstance(cov._value_cache, collections.deque)
        self.assertEqual(len(cov._value_cache), 0)

        # Get the time values and make sure they match what we assigned
        got = cov.get_time_values(time_segement=(0,20000))
        np.testing.assert_array_equal(vals['time'].get_data(), got)

        # Now check that there is 1 entry in the _value_cache and that it's a match for vals
        self.assertEqual(len(cov._value_cache), 1)
        cached_vals = None
        for t in cov._value_cache:
            cached_vals = t[2].get_data()
        np.testing.assert_array_equal(cached_vals['time'], got)

        # Now retrieve a slice and make sure it matches the same slice of vals
        sl = np.arange(18,1000, 3)
        got = cov.get_time_values(time_segement=(18, 1000), stride_length=3)
        np.testing.assert_array_equal(sl, got)

        # Now check that there are 2 entries and that the second is a match for vals
        self.assertEqual(len(cov._value_cache), 2)
        cached_vals = None
        for t in cov._value_cache:
            cached_vals = t[2].get_data()
        np.testing.assert_array_equal(cached_vals['time'], sl)

        # Call get 40 times with the same request - check that the _value_cache doesn't grow
        expected_length = len(cov._value_cache) + 1
        for x in xrange(40):
            cov.get_time_values(time_segement=(0,10))
        self.assertEqual(len(cov._value_cache), expected_length)

        # Call get 40 times - check that the _value_cache stops growing @ 30
        for x in xrange(40):
            cov.get_time_values(time_segement=(0,x))
        self.assertEqual(len(cov._value_cache), 5)

    def test_value_caching_with_domain_expansion(self):
        cov = self._make_empty_oneparamcov()

        # Insert some timesteps (automatically expands other arrays)
        nt = 100

        vals = {cov.temporal_parameter_name: np.arange(nt, dtype=cov._range_dictionary.get_context('time').param_type.value_encoding)}
        cov.set_parameter_values(make_parameter_data_dict(vals))

        # Prime the value_cache
        got = cov.get_time_values(time_segement=(0,2000))

        # Expand the domain
        vals = {cov.temporal_parameter_name: np.arange(2*nt, 3*nt, dtype=cov._range_dictionary.get_context('time').param_type.value_encoding)}
        cov.set_parameter_values(make_parameter_data_dict(vals))

        # Value cache should still hold 1 and the value should be equal to values retrieved prior to expansion ('got')
        self.assertEqual(len(cov._value_cache), 1)
        np.testing.assert_array_equal(cov._value_cache[0][2].get_data()['time'], got)

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

    def _interop_assertions(self, cov, pname, val_cls, assn_vals=None, ts=None):
        start = None
        end = None
        if ts is not None:
            start = ts[0]
            end = ts[1]
        vals = None
        if assn_vals is not None:
            vals = cov.parameter_dictionary[pname].param_type.create_filled_array(assn_vals[cov.temporal_parameter_name].size)
            if isinstance(assn_vals[pname], ConstantOverTime):
                vals[:] = assn_vals[pname].get_data()
            else:
                vals = cov.parameter_dictionary[pname].param_type.create_data_array(assn_vals[pname], assn_vals[cov.temporal_parameter_name].size)
            start = assn_vals[cov.temporal_parameter_name][0]
            end = assn_vals[cov.temporal_parameter_name][-1]
            cov.set_parameter_values(assn_vals)
        if vals is None:
            vals = val_cls.__getitem__((start,end))
            start = 0
            end = len(vals)-1

        print cov.get_parameter_values(pname, time_segment=(start, end), fill_empty_params=True).get_data()[pname].dtype
        print vals[:].dtype
        np.testing.assert_array_equal(cov.get_parameter_values(pname, time_segment=(start, end), fill_empty_params=True).get_data()[pname], vals[:])
        np.testing.assert_array_equal(cov.get_parameter_values(pname, time_segment=(end, None)).get_data()[pname], vals[-1:])
        np.testing.assert_array_equal(cov.get_parameter_values(pname, time_segment=(start, end), stride_length=3).get_data()[pname], vals[0::3])
        if isinstance(val_cls.parameter_type, ArrayType) or \
                (hasattr(val_cls.parameter_type, 'base_type') and isinstance(val_cls.parameter_type.base_type, ArrayType)):
            np.testing.assert_array_equal(cov.get_parameter_values(pname, time=start).get_data()[pname], vals[0])
            np.testing.assert_array_equal(cov.get_parameter_values(pname, time=end).get_data()[pname], vals[-1])
            pass
        else:
            self.assertEqual(cov.get_parameter_values(pname, time=start).get_data()[pname], vals[0])
            self.assertEqual(cov.get_parameter_values(pname, time=end).get_data()[pname], vals[-1:])

    ## Must use a specialized set of assertions because np.array_equal doesn't work on arrays of type Sn!!
    def _interop_assertions_str(self, cov, pname, val_cls, assn_vals=None):
        start = None
        end = None
        vals = val_cls
        if assn_vals is not None:
            vals = cov.parameter_dictionary[pname].param_type.create_filled_array(assn_vals[cov.temporal_parameter_name].size)
            if isinstance(assn_vals[pname], ConstantOverTime):
                vals[:] = assn_vals[pname].get_data()
            else:
                vals = cov.parameter_dictionary[pname].param_type.create_data_array(assn_vals[pname], assn_vals[cov.temporal_parameter_name].size)
            start = assn_vals[cov.temporal_parameter_name][0]
            end = assn_vals[cov.temporal_parameter_name][-1]
            cov.set_parameter_values(assn_vals)

        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname, time_segment=(start,end), as_record_array=False).get_data()[pname] == vals).all())
        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname, time_segment=(end,None), as_record_array=False).get_data()[pname] == vals[-1:]).all())
        self.assertTrue(np.atleast_1d(cov.get_parameter_values(pname, time_segment=(start,end), as_record_array=False, stride_length=3).get_data()[pname] == vals[0::3]).all())
        self.assertEqual(cov.get_parameter_values(pname, time_segment=(start,start), as_record_array=False).get_data()[pname], vals[0])
        self.assertEqual(cov.get_parameter_values(pname, time_segment=(end,end), as_record_array=False).get_data()[pname], vals[-1])
        self.assertEqual(cov.get_parameter_values(pname, time=start, as_record_array=False).get_data()[pname], vals[0])
        self.assertEqual(cov.get_parameter_values(pname, time=end, as_record_array=False).get_data()[pname], vals[-1])

    def _setup_cov(self, ntimes, names, types):
        pdict = ParameterDictionary()
        pdict.add_context(ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL), is_temporal=True)
        for i, n in enumerate(names):
            pdict.add_context(ParameterContext(n, param_type=types[i], variability=VariabilityEnum.TEMPORAL))
        tdom = GridDomain(GridShape('temporal', [0]), CRS([AxisTypeEnum.TIME]), MutabilityEnum.EXTENSIBLE)
        cov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom)
        if ntimes != 0:
            cov.set_parameter_values(make_parameter_data_dict({cov.temporal_parameter_name: np.arange(ntimes)}))

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
        cur_time = 10000
        cur_time+=ntimes
        valsi8_arr = { 'i8': np.arange(cur_time, cur_time+ntimes, dtype='int8'), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        valsi16_arr = {'i16': np.arange(cur_time, cur_time+ntimes, dtype='int16'), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        valsi32_arr = {'i32': np.arange(cur_time, cur_time+ntimes, dtype='int32'), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        valsi64_arr = {'i64': np.arange(cur_time, cur_time+ntimes, dtype='int64'), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        valsf32_arr = {'f32': np.arange(cur_time, cur_time+ntimes, dtype='float32'), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        valsf64_arr = {'f64': np.arange(cur_time, cur_time+ntimes, dtype='float64'), 'time': np.arange(cur_time, cur_time+ntimes) }

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
        cur_time = 100
        val_arr = {'time': np.arange(cur_time, cur_time+ntimes), 'const_num': ConstantOverTime('const_num', val)}
        cur_time += ntimes
        sval = 'const_str'
        sval_arr = {'time': np.arange(cur_time, cur_time+ntimes), 'const_str': ConstantOverTime('const_str', sval)}

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        cn_val = get_value_class(const_type_n, dom)
        cs_val = get_value_class(const_type_s, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['const_num', 'const_str'], [const_type_n, const_type_s])

        # Perform the assertions

        # Array value assignment, numeric
        self._interop_assertions(cov, 'const_num', cn_val, val_arr)

        # Array value assignment, string
        self._interop_assertions_str(cov, 'const_str', cs_val, sval_arr)

    def test_constant_range_value_interop(self):
        # Setup the type
        cr_n_type = ConstantRangeType(value_encoding='float32', fill_value=(0.,0.))
        cr_s_type = ConstantRangeType(value_encoding='S5', fill_value=("", ""))

        # Setup the values
        ntimes = 20
        val = (20, 40)
        cur_time = 100
        val_arr = {'time': np.arange(cur_time, cur_time+ntimes), 'const_rng_num': ConstantOverTime('const_rng_num', val)}
        cur_time += ntimes
        sval = ('low', 'high')
        sval_arr = {'time': np.arange(cur_time, cur_time+ntimes), 'const_rng_str': ConstantOverTime('const_rng_str', sval)}

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        crn_val = get_value_class(cr_n_type, dom)
        crs_val = get_value_class(cr_s_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['const_rng_num', 'const_rng_str'], [cr_n_type, cr_s_type])

        # Perform the assertions

        # Object array assignment, numeric
        self._interop_assertions(cov, 'const_rng_num', crn_val, val_arr)

        # Object array assignment, string
        self._interop_assertions_str(cov, 'const_rng_str', crs_val, sval_arr)

    def test_boolean_value_interop(self):
        # Setup the type
        bool_type = BooleanType()

        # Setup the values
        from random import choice
        ntimes = 20
        bvals = [choice([True, False]) for r in range(ntimes)]
        ivals = [choice([-1, 0, 1, 2]) for r in range(ntimes)]
        cur_time = 1000
        bvals_arr = {'bool': bool_type.create_data_array(bvals), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        ivals_arr = {'bool': bool_type.create_data_array(ivals), 'time': np.arange(cur_time, cur_time+ntimes) }
        cur_time+=ntimes
        ivals_arr2 = {'bool': bool_type.create_data_array(size=len(ivals)), 'time': np.arange(cur_time, cur_time+ntimes) }
        ivals_arr2['bool'][:] = ivals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        bool_val = get_value_class(bool_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['bool'], [bool_type])

        # Perform the assertions

        # Array assignment, boolean
        self._interop_assertions(cov, 'bool', bool_val, bvals_arr)

        # Array assignment, integer
        self._interop_assertions(cov, 'bool', bool_val, ivals_arr)

        # Array assignment, integer
        self._interop_assertions(cov, 'bool', bool_val, ivals_arr2)

    def test_record_value_interop(self):
        # Setup the type
        rec_type = RecordType()

        # Setup the values
        ntimes = 20
        letts='abcdefghijklmnopqrstuvwxyz'
        rvals = [{letts[x]: letts[x:]} for x in range(ntimes)]
        rvals_arr = np.empty(ntimes, dtype=object)
        rvals_arr[:] = rvals
        rvals_dict = { 'time': np.arange(10000, 10000+ntimes), 'rec': rvals_arr }

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        rec_val = get_value_class(rec_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['rec'], [rec_type])

        # Perform the assertions

        # List assignment
        # self._interop_assertions(cov, 'rec', rec_val, rvals)

        # Array assignment
        self._interop_assertions(cov, 'rec', rec_val, rvals_dict)

    def test_parameter_function_value_interop(self):
        # Setup the type
        numexpr_type = ParameterFunctionType(NumexprFunction('test_func', 'a*2', ['a'], param_map={'a': 'time'}), value_encoding='int32')
        pyfunc_type = ParameterFunctionType(PythonFunction('test_func', 'coverage_model.test.test_parameter_functions', 'pyfunc', ['a','b'], param_map={'a': 'time', 'b': 2}))


        # Setup the values
        ntimes = 20

        def get_vals(name, time_segment=None, stride=None):
            if name == 'time':
                arr = np.atleast_1d(range(ntimes))
                if time_segment is not None:
                    start = time_segment[0]
                    end = time_segment[1]
                    if time_segment[0] is None and time_segment[1] is None:
                        arr =  arr
                    elif time_segment[0] is None and time_segment[1] is not None:
                        arr =  arr[:time_segment[1]+1]
                    elif time_segment[0] is not None and time_segment[1] is None:
                        arr =  arr[time_segment[1]:]
                    else:
                        arr = arr[time_segment[0]:time_segment[1]+1]

                return NumpyDictParameterData({name: arr})
        numexpr_type.callback = get_vals
        pyfunc_type.callback = get_vals

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        numexpr_val = get_value_class(numexpr_type, dom)
        pyfunc_val = get_value_class(pyfunc_type, dom)

       # Setup the coverage
        cov = self._setup_cov(ntimes, ['numexpr', 'pyfunc'], [numexpr_type, pyfunc_type])

        # cov2 = AbstractCoverage.load(cov.persistence_dir, cov.persistence_guid)

        # Perform the assertions

        # Make sure the value_encoding is enforced
        self.assertEqual(numexpr_val.__getitem__().dtype, np.dtype('int32'))
        self.assertEqual(pyfunc_val.__getitem__().dtype, np.dtype('float32'))
        # self.assertEqual(cov.get_parameter_values('numexpr').get_data()['numexpr'].dtype, np.dtype('int32'))
        # self.assertEqual(cov.get_parameter_values('pyfunc').get_data()['pyfunc'].dtype, np.dtype('float32'))

        cov.set_parameter_function('numexpr', get_vals)
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
        arr_type = ArrayType('int32', inner_length=3)
        arr_type_ie = ArrayType(inner_encoding=np.dtype('float32'), inner_length=3)

        # Setup the values
        ntimes = 20
        vals = [(1, 2, 3)] * ntimes
        vals_ie = [(1.2,2.3,3.4)] * ntimes
        vals_arr = np.array(vals, dtype=np.dtype(', '.join([np.dtype(np.int).name for x in range(3)])))
        vals_arr_ie = np.empty(ntimes, dtype=np.dtype(', '.join([np.dtype(np.float64).name for x in range(3)])))
        vals_arr_ie[:] = vals_ie
        cur_time=100
        vals_arr = {'array_': vals_arr, 'time': np.arange(cur_time, cur_time+ntimes)}
        cur_time+=ntimes
        vals_arr_ie = {'array_ie': vals_arr_ie, 'time': np.arange(cur_time, cur_time+ntimes)}
        svals = []
        for x in xrange(ntimes):
            svals.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
        svals_arr = np.empty(ntimes, dtype=object)
        svals_arr[:] = svals
        cur_time+=ntimes
        expected_svals_arr = svals_arr
        svals_arr = {'array_': svals_arr, 'time': np.arange(cur_time, cur_time+ntimes)}

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        arr_val = get_value_class(arr_type, dom)
        arr_val_ie = get_value_class(arr_type_ie, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['array_', 'array_ie'], [arr_type, arr_type_ie])

        # Perform the assertions

        # Array Assignment
        self._interop_assertions(cov, 'array_', arr_val, vals_arr)
        self._interop_assertions(cov, 'array_ie', arr_val_ie, vals_arr_ie)

        # String Assignment via array
        self._interop_assertions_str(cov, 'array_', arr_val, svals_arr)

    def test_category_value_interop(self):
        # Setup the type
        cats = {0: 'turkey', 1: 'duck', 2: 'chicken', 3: 'empty'}
        cat_type = CategoryType(categories=cats)
        cat_type.fill_value = 3

        # Setup the values
        ntimes = 10
        key_vals = [1, 2, 0, 3, 2, 0, 1, 2, 1, 1]
        cat_vals = [cats[k] for k in key_vals]
        cur_time = 100
        key_vals_arr = {'category':np.array(key_vals), 'time': np.arange(cur_time, cur_time+ntimes)}
        cat_vals_arr = np.empty(ntimes, dtype=object)
        cat_vals_arr[:] = cat_vals
        cur_time+=ntimes
        cat_vals_arr = {'category': cat_vals_arr, 'time': np.arange(cur_time, cur_time+ntimes)}

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        cat_val = get_value_class(cat_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['category'], [cat_type])

        # Perform the assertions

        # Assign with an array of keys
        self._interop_assertions_str(cov, 'category', cat_vals, key_vals_arr)

        val_arr = cov.get_parameter_values('category', time_segment=(100,109)).get_data()['category']
        return_vals = np.array([cats[k] for k in val_arr])
        np.testing.assert_array_equal(return_vals, cat_vals_arr['category'])

    @unittest.skip('Sparse values replaced in R3')
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
        cur_time = 10000
        cov.set_parameter_values({'scv': ConstantOverTime('scv', val), cov.temporal_parameter_name: np.arange(cur_time, cur_time+ntimes)})
        self._interop_assertions(cov, 'scv', scv_val, ts=(cur_time, cur_time+ntimes-1))
        np.testing.assert_array_equal(scv_val[:], want)
        np.testing.assert_array_equal(cov.get_parameter_values('scv', time_segment=(cur_time, cur_time+ntimes-1)).get_data()['scv'], want)

        # Assign with aval
        scv_arr_val[:] = aval
        cur_time+=ntimes
        cov.set_parameter_values({'scv_arr': ConstantOverTime('scv_arr', aval_arr), cov.temporal_parameter_name: np.arange(cur_time, cur_time+ntimes)})
        self._interop_assertions(cov, 'scv_arr', scv_arr_val, ts=(cur_time, cur_time+ntimes-1))
        np.testing.assert_array_equal(scv_arr_val[:], awant)
        np.testing.assert_array_equal(cov.get_parameter_values('scv_arr', time_segment=(cur_time, cur_time+ntimes-1)).get_data()['scv_arr'], awant)

        # Backfill assignment

        # Assign with list
        scv_val[-1] = val_list
        cov.set_parameter_values('scv', val_list, -1)
        self._interop_assertions(cov, 'scv', scv_val)
        np.testing.assert_array_equal(scv_val[:], want)
        np.testing.assert_array_equal(cov.get_parameter_values('scv'), want)

        # Asign with array
        scv_val[-1] = val_arr
        cov.set_parameter_values('scv', val_arr, -1)
        self._interop_assertions(cov, 'scv', scv_val)
        np.testing.assert_array_equal(scv_val[:], want)
        np.testing.assert_array_equal(cov.get_parameter_values('scv'), want)

        scv_arr_val[-1] = aval_arr
        cov.set_parameter_values('scv_arr', aval_arr, -1)
        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        np.testing.assert_array_equal(scv_arr_val[:], awant)
        np.testing.assert_array_equal(cov.get_parameter_values('scv_arr'), awant)

        # Add a new value and expand the domain

        # Change the values
        val = 40
        val_list = [40, 80]
        val_arr = np.array(val_list, dtype='int32')
        want = np.append(want, np.array([val] * ntimes, dtype='int32'))

        aval = [[39, 2, 394, 55]]
        aval_arr = np.empty(1, dtype=object)
        aval_arr[0] = aval
        awant = np.hstack((awant, np.array([[ifv]] * ntimes)))
        awant = np.vstack((awant, np.array(aval * ntimes, dtype='float32')))

        # Assign with val
        scv_val[:] = val
        cov.set_parameter_values('scv', val)

        # Assign with aval
        scv_arr_val[:] = aval
        cov.set_parameter_values('scv_arr', aval)

        # Expand the domain
        dom.shape = (dom.shape[0] + ntimes,)

        # Validate values
        self._interop_assertions(cov, 'scv', scv_val)
        np.testing.assert_array_equal(scv_val[:], want)
        np.testing.assert_array_equal(cov.get_parameter_values('scv'), want)

        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        np.testing.assert_allclose(scv_arr_val[:], awant)
        np.testing.assert_allclose(cov.get_parameter_values('scv_arr'), awant)

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
        np.testing.assert_array_equal(scv_val[:], want)
        np.testing.assert_array_equal(cov.get_parameter_values('scv'), want)

        self._interop_assertions(cov, 'scv_arr', scv_arr_val)
        np.testing.assert_allclose(scv_arr_val[:], awant)
        np.testing.assert_allclose(cov.get_parameter_values('scv_arr'), awant)

    @unittest.skip('Sparse arrays replaced in R3')
    def test_sparse_constant_value_ndarray_interop(self):
        ifv = 827.38
        scv_arr_type = SparseConstantType(base_type=ArrayType(inner_encoding='float32', inner_fill_value=ifv))

        # Setup the values
        ntimes = 10
        ndaval = [[12, 32, 33], [3, 44, 52], [16, 2, 76, 1]]
        cur_time = 100
        from coverage_model.parameter_data import RepeatOverTime
        ndaval_arr = {'time': np.arange(cur_time, cur_time+ntimes), 'scv_ndarr': RepeatOverTime('scv_ndarr', ndaval)}
        # ndaval_arr = np.array(ndaval, dtype=np.object)
        ndawant = np.array(ndaval * ntimes, dtype=np.object)

        # Setup the in-memory value
        dom = SimpleDomainSet((ntimes,))
        scv_ndarr_val = get_value_class(scv_arr_type, dom)

        # Setup the coverage
        cov = self._setup_cov(ntimes, ['scv_ndarr'], [scv_arr_type])

        # Perform the assertions

        # Assign with ndaval
        scv_ndarr_val[:] = ndaval
        # cov.set_parameter_values( ndaval_arr )
        self._interop_assertions(cov, 'scv_ndarr', scv_ndarr_val, ndaval_arr)
        np.testing.assert_array_equal(scv_ndarr_val[:], ndawant)
        np.testing.assert_array_equal(cov.get_parameter_values('scv_ndarr').get_data()['scv_ndarr'], ndawant)

        # Backfill assignment

        # Assign with array
        scv_ndarr_val[-1] = ndaval_arr
        cov.set_parameter_values('scv_ndarr', ndaval_arr, -1)
        self._interop_assertions(cov, 'scv_ndarr', scv_ndarr_val)
        np.testing.assert_array_equal(scv_ndarr_val[:], ndawant)
        np.testing.assert_array_equal(cov.get_parameter_values('scv_ndarr'), ndawant)

        # Change the values
        ndaval = [[[88, 323], [4, 34]]]
        ndaval_arr = np.array(ndaval, dtype='float32')
        w = np.array([[[88, 323, ifv], [4, 34, ifv], [ifv, ifv, ifv]]]*ntimes, dtype='float32')
        o_ndawant = ndawant.copy()
        ndawant = np.vstack((ndawant, w))

        # Assign with ndaval_arr
        scv_ndarr_val[:] = ndaval_arr
        cov.set_parameter_values('scv_ndarr', ndaval_arr)

        # Expand the domain
        dom.shape = (dom.shape[0] + ntimes,)

        # Validate the values
        self._interop_assertions(cov, 'scv_ndarr', scv_ndarr_val)
        np.testing.assert_allclose(scv_ndarr_val[:], ndawant)
        np.testing.assert_allclose(cov.get_parameter_values('scv_ndarr'), ndawant)

        # Reassign last segment
        ndaval = [[[44, 581, 52], [33, 90, ifv]]]
        w = np.array([[[44, 581, 52], [33, 90, ifv], [ifv, ifv, ifv]]]*ntimes, dtype='float32')
        ndawant = np.vstack((o_ndawant, w))

         # Assign with ndaval
        scv_ndarr_val[-1] = ndaval
        cov.set_parameter_values('scv_ndarr', ndaval, -1)

        self._interop_assertions(cov, 'scv_ndarr', scv_ndarr_val)
        np.testing.assert_allclose(scv_ndarr_val[:], ndawant)
        np.testing.assert_allclose(cov.get_parameter_values('scv_ndarr'), ndawant)
