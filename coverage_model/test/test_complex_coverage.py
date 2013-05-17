#!/usr/bin/env python

"""
@package coverage_model.test.test_complex_coverage
@file coverage_model/test/test_complex_coverage.py
@author Christopher Mueller
@brief Unit & Integration tests for ComplexCoverage
"""

from ooi.logging import log
import os
import numpy as np
import random
from coverage_model import *
from nose.plugins.attrib import attr
import mock
import unittest
from copy import deepcopy

from coverage_test_base import CoverageIntTestBase, get_props


def _make_cov(root_dir, params, nt=10, data_dict=None, make_temporal=True):
    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    if isinstance(params, ParameterDictionary):
        pdict = params
    else:
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        if make_temporal:
            # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
            t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('float32')))
            t_ctxt.uom = 'seconds since 01-01-1970'
            pdict.add_context(t_ctxt, is_temporal=True)

        for p in params:
            if isinstance(p, ParameterContext):
                pdict.add_context(p)
            elif isinstance(params, tuple):
                pdict.add_context(ParameterContext(p[0], param_type=QuantityType(value_encoding=np.dtype(p[1]))))
            else:
                pdict.add_context(ParameterContext(p, param_type=QuantityType(value_encoding=np.dtype('float32'))))

    scov = SimplexCoverage(root_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)

    scov.insert_timesteps(nt)
    for p in scov.list_parameters():
        if data_dict is not None and p in data_dict:
            dat = data_dict[p]
        else:
            dat = range(nt)

        try:
            scov.set_parameter_values(p, dat)
        except Exception as ex:
            import sys
            raise Exception('Error setting values for {0}: {1}'.format(p, data_dict[p])), None, sys.exc_traceback

    scov.close()

    return os.path.realpath(scov.persistence_dir)


@attr('INT',group='cov')
class TestComplexCoverageInt(CoverageModelIntTestCase, CoverageIntTestBase):

    # Make a deep copy of the base TESTING_PROPERTIES dict and then modify for this class
    TESTING_PROPERTIES = deepcopy(CoverageIntTestBase.TESTING_PROPERTIES)
    TESTING_PROPERTIES['test_props_decorator'] = {'test_props': 10}
    TESTING_PROPERTIES['test_get_time_data_metrics'] = {'time_data_size': 0.01907348}

    @get_props()
    def test_props_decorator(self):
        props = self.test_props_decorator.props
        self.assertIsInstance(props, dict)
        expected = {'time_steps': 30, 'test_props': 10, 'brick_size': 1000}
        self.assertEqual(props, expected)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def get_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=30, auto_flush_values=True):
        # Many tests assume nt is the 'total' number of timesteps, must divide between the 3 coverages
        sz1 = sz2 = sz3 = int(nt) / 3
        sz3 += nt - sum([sz1, sz2, sz3])

        first_times = np.arange(0, sz1, dtype='float32')
        first_data = np.arange(0, sz1, dtype='float32')

        second_times = np.arange(sz1, sz1+sz2, dtype='float32')
        second_data = np.arange(sz1, sz1+sz2, dtype='float32')

        third_times = np.arange(sz1+sz2, nt, dtype='float32')
        third_data = np.arange(sz1+sz2, nt, dtype='float32')

        cova_pth = _make_cov(cls.working_dir, ['data_all', 'data_a'], nt=sz1,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(cls.working_dir, ['data_all', 'data_b'], nt=sz2,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(cls.working_dir, ['data_all', 'data_c'], nt=sz3,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(cls.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        return comp_cov, 'TestComplexCoverageInt'

    ######################
    # Overridden base tests
    ######################

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        # Cannot set values against a ComplexCoverage - just return True
        return True

    def test_append_parameter(self):
        nt = 60
        ccov, cov_name = self.get_cov(inline_data_writes=True, nt=nt)

        parameter_name = 'a*b'
        func = NumexprFunction('a*b', 'a*b', ['a', 'b'], {'a': 'data_a', 'b': 'data_b'})
        pc_in = ParameterContext(parameter_name, param_type=ParameterFunctionType(function=func, value_encoding=np.dtype('float32')))

        ccov.append_parameter(pc_in)

        sample_values = ccov.get_parameter_values('data_a') * ccov.get_parameter_values('data_b')

        self.assertTrue(np.array_equal(sample_values, ccov.get_parameter_values(parameter_name)))

        ccov.insert_timesteps(100)
        self.assertEqual(len(ccov.get_parameter_values(parameter_name)), nt + 100)

        nvals = np.arange(nt, nt + 100, dtype='f')
        ccov.set_parameter_values('data_a', value=nvals, tdoa=slice(nt, None))
        ccov.set_parameter_values('data_b', value=nvals, tdoa=slice(nt, None))

        sample_values = ccov.get_parameter_values('data_a') * ccov.get_parameter_values('data_b')

        self.assertTrue(np.array_equal(sample_values, ccov.get_parameter_values(parameter_name)))

        with self.assertRaises(ValueError):
            ccov.append_parameter(pc_in)

    @unittest.skip('Functionality verified in \'test_temporal_aggregation\'')
    def test_refresh(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_create_multi_bricks(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_coverage_pickle_and_in_memory(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_coverage_mode_expand_domain(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_coverage_mode_set_value(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_pickle_problems_in_memory(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_allparams_five_bricks(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_allparams_one_brick(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_time_five_bricks(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_time_five_bricks_strided(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_time_one_brick(self):
        pass

    @unittest.skip('Does not apply to ComplexCoverage')
    def test_set_time_one_brick_strided(self):
        pass

    ######################
    # Additional tests specific to Complex Coverage
    ######################

    def test_parametric_strict(self):
        num_times = 10

        first_data = np.arange(num_times, dtype='float32') * 0.2
        second_data = np.random.random_sample(num_times) * (50 - 10) + 10
        apple_data = np.arange(num_times, dtype='float32')
        orange_data = np.arange(num_times, dtype='float32') * 2

        cova_pth = _make_cov(self.working_dir, ['first_param'], data_dict={'first_param': first_data})
        covb_pth = _make_cov(self.working_dir, ['second_param'], data_dict={'second_param': second_data})
        covc_pth = _make_cov(self.working_dir, ['apples', 'oranges'], data_dict={'apples': apple_data, 'oranges': orange_data})

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        ab_func = NumexprFunction('aXb', 'a*b', ['a', 'b'], {'a': 'first_param', 'b': 'second_param'})
        ab_ctxt = ParameterContext('aXb', param_type=ParameterFunctionType(function=ab_func, value_encoding=np.dtype('float32')))
        pdict.add_context(ab_ctxt)

        aplorng_func = NumexprFunction('apples_to_oranges', 'a*cos(sin(b))+c', ['a', 'b', 'c'], {'a': 'apples', 'b': 'oranges', 'c': 'first_param'})
        aplorng_ctxt = ParameterContext('apples_to_oranges', param_type=ParameterFunctionType(function=aplorng_func, value_encoding=np.dtype('float32')))
        pdict.add_context(aplorng_ctxt)

        # Instantiate the ComplexCoverage
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                               reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                               parameter_dictionary=pdict,
                               complex_type=ComplexCoverageType.PARAMETRIC_STRICT)

        self.assertEqual(ccov.list_parameters(),
                         ['aXb', 'apples', 'apples_to_oranges', 'first_param', 'oranges', 'second_param', 'time'])

        self.assertEqual(ccov.temporal_parameter_name, 'time')
        self.assertEqual(ccov.num_timesteps, num_times)

        self.assertTrue(np.array_equal(ccov.get_parameter_values('first_param'), first_data))
        self.assertTrue(np.allclose(ccov.get_parameter_values('second_param'), second_data))
        self.assertTrue(np.array_equal(ccov.get_parameter_values('apples'), apple_data))
        self.assertTrue(np.array_equal(ccov.get_parameter_values('oranges'), orange_data))

        aXb_want = first_data * second_data
        self.assertTrue(np.allclose(ccov.get_parameter_values('aXb'), aXb_want))
        aplorng_want = apple_data * np.cos(np.sin(orange_data)) + first_data
        self.assertTrue(np.allclose(ccov.get_parameter_values('apples_to_oranges'), aplorng_want))

    def test_parametric_strict_warnings(self):
        num_times = 10

        first_data = np.arange(num_times, dtype='float32') * 0.2
        second_data = np.random.random_sample(num_times) * (50 - 10) + 10
        apple_data = np.arange(num_times, dtype='float32')
        orange_data = np.arange(num_times, dtype='float32') * 2

        cova_pth = _make_cov(self.working_dir, ['first_param'], data_dict={'first_param': first_data})
        covb_pth = _make_cov(self.working_dir, ['second_param'], data_dict={'second_param': second_data, 'time': np.arange(123, 133, dtype='int64')})
        covc_pth = _make_cov(self.working_dir, ['apples', 'oranges'], data_dict={'apples': apple_data, 'oranges': orange_data})

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        ab_func = NumexprFunction('aXb', 'a*b', ['a', 'b'], {'a': 'first_param', 'b': 'second_param'})
        ab_ctxt = ParameterContext('aXb', param_type=ParameterFunctionType(function=ab_func, value_encoding=np.dtype('float32')))
        pdict.add_context(ab_ctxt)

        aplorng_func = NumexprFunction('apples_to_oranges', 'a*cos(sin(b))+c', ['a', 'b', 'c'], {'a': 'apples', 'b': 'oranges', 'c': 'first_param'})
        aplorng_ctxt = ParameterContext('apples_to_oranges', param_type=ParameterFunctionType(function=aplorng_func, value_encoding=np.dtype('float32')))
        pdict.add_context(aplorng_ctxt)

        with mock.patch('coverage_model.coverage.log') as log_mock:
            ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                   parameter_dictionary=pdict,
                                   complex_type=ComplexCoverageType.PARAMETRIC_STRICT)

            self.assertEquals(log_mock.warn.call_args_list[0],
                              mock.call('Coverage timestamps do not match; cannot include: %s', covb_pth))
            self.assertEquals(log_mock.info.call_args_list[0],
                              mock.call("Parameter '%s' from coverage '%s' already present, skipping...", 'time', covc_pth))

        with mock.patch('coverage_model.coverage.log') as log_mock:
            ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                                   reference_coverage_locs=[cova_pth, cova_pth],
                                   parameter_dictionary=pdict,
                                   complex_type=ComplexCoverageType.PARAMETRIC_STRICT)

            self.assertEquals(log_mock.info.call_args_list[0],
                              mock.call("Coverage '%s' already present; ignoring", cova_pth))

        with mock.patch('coverage_model.coverage.log') as log_mock:
            covb_pth = _make_cov(self.working_dir, ['second_param'], data_dict={'second_param': second_data}, make_temporal=False)
            ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   parameter_dictionary=pdict,
                                   complex_type=ComplexCoverageType.PARAMETRIC_STRICT)

            self.assertEquals(log_mock.warn.call_args_list[0],
                              mock.call("Coverage '%s' does not have a temporal_parameter; ignoring", covb_pth))

        with mock.patch('coverage_model.coverage.log') as log_mock:
            pdict.add_context(ParameterContext('discard_me'))
            ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                                   reference_coverage_locs=[cova_pth, cova_pth],
                                   parameter_dictionary=pdict,
                                   complex_type=ComplexCoverageType.PARAMETRIC_STRICT)
            self.assertEqual(log_mock.warn.call_args_list[0],
                             mock.call("Parameters stored in a ComplexCoverage must be ParameterFunctionType parameters: discarding '%s'", 'discard_me'))

    def test_temporal_aggregation(self):
        size = 100000
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        third_times = np.arange(size*2, size*3, dtype='float32')
        third_data = np.arange(size*7, size*8, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(self.working_dir, ['data_all', 'data_c'], nt=size,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        self.assertEqual(comp_cov.num_timesteps, 3*size)
        tvals = comp_cov.get_time_values()
        self.assertTrue(np.array_equal(tvals, np.arange(3*size, dtype='float32')))
        self.assertEqual(tvals.dtype, np.dtype('float32'))  # np.array_equal does NOT check dtype!!

        all_data = np.empty(0, dtype='float32')
        all_data = np.append(all_data, first_data)
        all_data = np.append(all_data, second_data)
        all_data = np.append(all_data, third_data)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_all'), all_data))
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_all', slice(0,size)), first_data))

        fill_arr = np.empty(size, dtype='float32')
        fill_arr[:] = -9999.0
        a_data = np.empty(0, dtype='float32')
        a_data = np.append(a_data, first_data)
        a_data = np.append(a_data, fill_arr)
        a_data = np.append(a_data, fill_arr)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_a'), a_data))

        b_data = np.empty(0, dtype='float32')
        b_data = np.append(b_data, fill_arr)
        b_data = np.append(b_data, second_data)
        b_data = np.append(b_data, fill_arr)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_b'), b_data))

        c_data = np.empty(0, dtype='float32')
        c_data = np.append(c_data, fill_arr)
        c_data = np.append(c_data, fill_arr)
        c_data = np.append(c_data, third_data)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_c'), c_data))

        # Check that the head_coverage_path is correct
        self.assertEqual(comp_cov.head_coverage_path, covc_pth)

        # Add some data to the last coverage (covc) and make sure it comes in
        cov_c = AbstractCoverage.load(covc_pth, mode='a')
        cov_c.insert_timesteps(size)
        cov_c.set_time_values(range(size*3, size*4), slice(size, None))
        addnl_c_data = np.arange(size*8, size*9, dtype='float32')
        cov_c.set_parameter_values('data_all', addnl_c_data, slice(size, None))
        cov_c.set_parameter_values('data_c', addnl_c_data, slice(size, None))
        cov_c.close()

        # Refresh the complex coverage
        comp_cov.refresh()

        all_data = np.append(all_data, addnl_c_data)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_all'), all_data))

        c_data = np.append(c_data, addnl_c_data)
        self.assertTrue(np.array_equal(comp_cov.get_parameter_values('data_c'), c_data))

        # Check that the head_coverage_path is still correct
        self.assertEqual(comp_cov.head_coverage_path, covc_pth)

    def test_temporal_aggregation_warnings(self):
        size = 100000
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        third_times = np.arange(size*2, size*3, dtype='float32')
        third_data = np.arange(size*7, size*8, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': first_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(self.working_dir, ['data_all', 'data_c'], nt=size,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        with mock.patch('coverage_model.coverage.log') as log_mock:
            comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                       reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                       complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

            self.assertEquals(log_mock.warn.call_args_list[3],
                              mock.call("Coverage with time bounds '%s' already present; ignoring", (first_times.min(), first_times.max(), 0)))

            self.assertEquals(log_mock.info.call_args_list[0],
                              mock.call("Parameter '%s' from coverage '%s' already present, skipping...", 'data_all', covc_pth))

            self.assertEquals(log_mock.info.call_args_list[1],
                              mock.call("Parameter '%s' from coverage '%s' already present, skipping...", 'time', covc_pth))

    def _setup_allparams(self, size=10, num_covs=2, sequential_covs=True):
        # Setup types
        types = []
        types.append(('qtype', QuantityType()))
        types.append(('atype_n', ArrayType()))
        types.append(('atype_s', ArrayType()))
        letts='abcdefghijklmnopqrstuvwxyz'
        while len(letts) < size:
            letts += letts
        types.append(('rtype', RecordType()))
        types.append(('btype', BooleanType()))
        types.append(('ctype_n', ConstantType(QuantityType(value_encoding=np.dtype('int32')))))
        types.append(('ctype_s', ConstantType(QuantityType(value_encoding=np.dtype('S21')))))
        types.append(('crtype', ConstantRangeType(QuantityType(value_encoding=np.dtype('int16')))))
        types.append(('pftype', ParameterFunctionType(NumexprFunction('v*10', 'v*10', ['v'], {'v': 'ctype_n'}))))
        cat = {99:'empty',0:'turkey',1:'duck',2:'chicken'}
        catkeys = cat.keys()
        types.append(('cattype', CategoryType(categories=cat)))
        types.append(('sctype', SparseConstantType(fill_value=-998, value_encoding='int32')))

        # Make coverages
        covs = []
        cov_data = []
        for i in xrange(num_covs):
            ii = i + 1
            # Make parameters
            pdict = ParameterDictionary()
            tpc = ParameterContext('time', param_type=QuantityType(value_encoding='float32'))
            tpc.axis = AxisTypeEnum.TIME
            pdict.add_context(tpc)
            for t in types:
                pdict.add_context(ParameterContext(t[0], param_type=t[1], variability=VariabilityEnum.TEMPORAL))

            # Make the data
            data_dict = {}
            if sequential_covs:
                tmax = ii * size
                tmin = i * size
                tdata = np.random.random_sample(size) * (tmax - tmin) + tmin
                tdata.sort()
            else:
                tdata = np.random.random_sample(size) * (200 - 0) + 0
                tdata.sort()
            data_dict['time'] = tdata
            data_dict['atype_n'] = [[ii for a in xrange(random.choice(range(1,size)))] for r in xrange(size)]
            data_dict['atype_s'] = [np.random.bytes(np.random.randint(1,20)) for r in xrange(size)]
            data_dict['qtype'] = np.random.random_sample(size) * (50 - 10) + 10
            data_dict['rtype'] = [{letts[r]: letts[r:]} for r in xrange(size)]
            data_dict['btype'] = [random.choice([True, False]) for r in xrange(size)]
            data_dict['ctype_n'] = [ii*20] * size
            data_dict['ctype_s'] = ['const_str_{0}'.format(i)] * size
            crarr = np.empty(size, dtype=object)
            crarr[:] = [(ii*10, ii*20)]
            data_dict['crtype'] = crarr
            #    data_dict['pftype'] # Calculated on demand, nothing assigned!!
            data_dict['cattype'] = [random.choice(catkeys) for r in xrange(size)]
            data_dict['sctype'] = [ii*30] * size

            # Create the coverage
            covs.append(_make_cov(self.working_dir, pdict, nt=size, data_dict=data_dict))

            # Now add values for pftype, for later comparison
            data_dict['pftype'] = [x*10 for x in data_dict['ctype_n']]
            # And update the values for cattype
            data_dict['cattype'] = [cat[k] for k in data_dict['cattype']]

            # Add the data_dict to the cov_data list
            cov_data.append(data_dict)

        return covs, cov_data

    def test_temporal_aggregation_all_param_types(self):
        size = 10

        covs, cov_data = self._setup_allparams(size=size)

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=covs,
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        for p in comp_cov.list_parameters():
            for i in xrange(len(covs)):
                ddict = cov_data[i]
                if p in ['qtype', 'time']:
                    self.assertTrue(np.allclose(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)), ddict[p]))
                elif p == 'ctype_s':
                    self.assertTrue(np.atleast_1d(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)) == ddict[p]).all())
                else:
                    self.assertTrue(np.array_equal(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)), ddict[p]))

    def test_temporal_interleaved(self):
        num_times = 200
        tpc = num_times / 2

        first_times = np.random.random_sample(tpc) * (20 - 0) + 0
        # first_times =  np.array([0,1,2,5,6,10,11,13,14,16], dtype='float32')
        first_times.sort()
        first_data = np.arange(tpc, dtype='float32') * 0.2
        first_full = np.random.random_sample(tpc) * (80 - 60) + 60

        second_times = np.random.random_sample(tpc) * (20 - 0) + 0
        # second_times = np.array([3,4,7,8,9,12,15,17,18,19], dtype='float32')
        second_times.sort()
        second_data = np.random.random_sample(tpc) * (50 - 10) + 10
        second_full = np.random.random_sample(tpc) * (80 - 60) + 60

        log.debug('\nCov A info:\n%s\n%s\n%s\n---------', first_times, first_data, first_full)
        log.debug('\nCov B info:\n%s\n%s\n%s\n---------', second_times, second_data, second_full)

        cova_pth = _make_cov(self.working_dir,
                             ['first_param', 'full_param'], nt=tpc,
                             data_dict={'time': first_times, 'first_param': first_data, 'full_param': first_full})
        covb_pth = _make_cov(self.working_dir,
                             ['second_param', 'full_param'], nt=tpc,
                             data_dict={'time': second_times, 'second_param': second_data, 'full_param': second_full})

        # Instantiate the ComplexCoverage
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                               reference_coverage_locs=[cova_pth, covb_pth],
                               complex_type=ComplexCoverageType.TEMPORAL_INTERLEAVED)

        self.assertEqual(ccov.list_parameters(), ['first_param', 'full_param', 'second_param', 'time'])

        self.assertEqual(ccov.temporal_parameter_name, 'time')
        self.assertEqual(ccov.num_timesteps, num_times)

        time_interleave = np.append(first_times, second_times)
        sort_i = np.argsort(time_interleave)
        self.assertTrue(np.allclose(ccov.get_time_values(), time_interleave[sort_i]))

        full_interleave = np.append(first_full, second_full)
        self.assertTrue(np.allclose(ccov.get_parameter_values('full_param'), full_interleave[sort_i]))

        first_interleave = np.empty((num_times,))
        first_interleave.fill(ccov.get_parameter_context('first_param').fill_value)
        first_interleave[:tpc] = first_data
        self.assertTrue(np.allclose(ccov.get_parameter_values('first_param'), first_interleave[sort_i]))

        second_interleave = np.empty((num_times,))
        second_interleave.fill(ccov.get_parameter_context('second_param').fill_value)
        second_interleave[tpc:] = second_data
        self.assertTrue(np.allclose(ccov.get_parameter_values('second_param'), second_interleave[sort_i]))

    def test_temporal_interleaved_all_param_types(self):
        size = 10
        covs, cov_data = self._setup_allparams(size=size, num_covs=5, sequential_covs=False)

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=covs,
                                   complex_type=ComplexCoverageType.TEMPORAL_INTERLEAVED)

        time_interleave = np.empty(0)
        for c in cov_data:
            time_interleave = np.append(time_interleave, c['time'])
        sort_i = np.argsort(time_interleave)

        self.assertTrue(np.allclose(comp_cov.get_time_values(), time_interleave[sort_i]))

    def test_append_reference_coverage(self):
        size = 100000
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        third_times = np.arange(size*2, size*3, dtype='float32')
        third_data = np.arange(size*7, size*8, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(self.working_dir, ['data_all', 'data_c'], nt=size,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        # Verify stuff worked normally...
        self.assertEqual(comp_cov.num_timesteps, 2*size)
        tvals = comp_cov.get_time_values()
        self.assertTrue(np.array_equal(tvals, np.arange(2*size, dtype='float32')))

        # Append the new coverage
        comp_cov.append_reference_coverage(covc_pth)

        # Now make sure the new data is there!
        self.assertEqual(comp_cov.num_timesteps, 3*size)
        tvals = comp_cov.get_time_values()
        self.assertTrue(np.array_equal(tvals, np.arange(3*size, dtype='float32')))

    def test_head_coverage_path(self):
        size = 10
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})

        # Ensure the correct path is returned from ComplexCoverage.head_coverage_path in CC --> SC & SC scenario
        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)
        self.assertEqual(comp_cov.head_coverage_path, covb_pth)

        # Ensure the correct path is returned from ComplexCoverage.head_coverage_path in CC --> SC & VC scenario
        vcov = ViewCoverage(self.working_dir, create_guid(), 'test', covb_pth)
        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, vcov.persistence_dir],
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)
        self.assertEqual(comp_cov.head_coverage_path, covb_pth)
        self.assertEqual(comp_cov.head_coverage_path, vcov.head_coverage_path)

