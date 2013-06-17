#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_simplex_coverage.py
@author James Case
@author Christopher Mueller
@brief Tests for the SimplexCoverage class.
"""

from coverage_model import *
from nose.plugins.attrib import attr
import unittest
import numpy as np
from pyon.public import log
import random
from copy import deepcopy
import os

from coverage_test_base import CoverageIntTestBase, get_props, get_parameter_dict, EXEMPLAR_CATEGORIES

@attr('INT', group='cov')
class TestSampleCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    # Make a deep copy of the base TESTING_PROPERTIES dict and then modify for this class
    TESTING_PROPERTIES = deepcopy(CoverageIntTestBase.TESTING_PROPERTIES)
    TESTING_PROPERTIES['test_props_decorator'] = {'test_props': 10}

    @get_props()
    def test_props_decorator(self):
        props = self.test_props_decorator.props
        self.assertIsInstance(props, dict)
        expected = {'time_steps': 30, 'test_props': 10, 'brick_size': 1000}
        self.assertEqual(props, expected)

    def setUp(self):
        pass

    @classmethod
    def get_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        # Instantiate a ParameterDictionary
        pname_filter = ['time',
                        'lat',
                        'lon',
                        'temp',
                        'conductivity']
        if only_time:
            pname_filter = ['time']

        pdict = get_parameter_dict(parameter_list=pname_filter)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestSampleCovInt'
        else:
            scov.insert_timesteps(nt)

            # Add data for each parameter
            scov.set_parameter_values('time', value=np.arange(nt))
            if not only_time:
                scov.set_parameter_values('lat', value=45)
                scov.set_parameter_values('lon', value=-71)
                # make a random sample of 10 values between 23 and 26
                # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
                # --> To sample  multiply the output of random_sample by (b-a) and add a
                tvals = utils.get_random_sample(nt, 23, 26)
                scov.set_parameter_values('temp', value=tvals)
                scov.set_parameter_values('conductivity', value=utils.get_random_sample(nt, 90, 110))

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        return scov, 'TestSampleCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        # Function to test variable occurances of getting and setting values across parameter(s)
        data = data[_slice]
        ret = []

        scov.insert_timesteps(timesteps)
        param_list = []
        if param == 'all':
            param_list = scov.list_parameters()
        else:
            param_list.append(param)

        for param in param_list:
            scov.set_parameter_values(param, data, _slice)
            scov.get_dirty_values_async_result().get(timeout=60)
            # TODO: Is the res = assignment below correct?
            ret = scov.get_parameter_values(param, _slice)
        return (ret == data).all()

    def test_list_parameters_coords_only(self):
        cov, cov_name = self.get_cov()
        coords_params = cov.list_parameters(coords_only=True)
        self.assertEqual(coords_params, ['lat', 'lon', 'time'])
        data_params = cov.list_parameters(data_only=True)
        self.assertEqual(data_params, ['conductivity', 'temp'])

@attr('INT', group='cov')
class TestOneParamCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    @classmethod
    def get_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        """
        Construct coverage
        """
        pname_filter = ['time']
        pdict = get_parameter_dict(parameter_list=pname_filter)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestOneParamCovInt'
        else:
            scov.insert_timesteps(nt)

            # Add data for the parameter
            scov.set_parameter_values('time', value=np.arange(nt))

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        return scov, 'TestOneParamCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        # Function to test variable occurances of getting and setting values across parameter(s)
        data = data[_slice]
        ret = []

        scov.insert_timesteps(timesteps)
        param_list = []
        if param == 'all':
            param_list = scov.list_parameters()
        else:
            param_list.append(param)

        for param in param_list:
            scov.set_parameter_values(param, data, _slice)
            scov.get_dirty_values_async_result().get(timeout=60)
            # TODO: Is the res = assignment below correct?
            ret = scov.get_parameter_values(param, _slice)
        return (ret == data).all()

    @unittest.skip('Does not apply to empty coverage.')
    def test_get_all_data_metrics(self):
        pass

@attr('INT', group='cov')
class TestEmptySampleCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    # Make a deep copy of the base TESTING_PROPERTIES dict and then modify for this class
    TESTING_PROPERTIES = deepcopy(CoverageIntTestBase.TESTING_PROPERTIES)
    TESTING_PROPERTIES['defaults'] = {'time_steps': 0}

    @get_props()
    def test_props_decorator(self):
        props = self.test_props_decorator.props
        self.assertIsInstance(props, dict)
        expected = {'time_steps': 0, 'test_props': 'base_test_props', 'brick_size': 1000}
        self.assertEqual(props, expected)

    def setUp(self):
        pass

    @classmethod
    def get_cov(cls, only_time=False, param_filter=None, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        # Instantiate a ParameterDictionary
        pname_filter = ['time',
                            'lat',
                            'lon',
                            'temp',
                            'conductivity']
        if only_time:
            pname_filter = ['time']
        pdict = get_parameter_dict(parameter_list=pname_filter)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        return scov, 'TestEmptySampleCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        return True

    def test_get_by_slice(self):
        cov = self.get_cov()[0]

        expected = np.empty(0, dtype=cov.get_parameter_context('time').param_type.value_encoding)

        slices = [slice(None), slice(0, None), slice(None, 10), slice(2, 8), slice(3, 19, 8)]

        for s in slices:
            ret = cov.get_parameter_values('time', s)
            self.assertTrue(np.array_equal(ret, expected))

    def test_get_by_int(self):
        cov = self.get_cov()[0]

        expected = np.empty(0, dtype=cov.get_parameter_context('time').param_type.value_encoding)

        ints = [0, 2, 5, 109]

        for i in ints:
            ret = cov.get_parameter_values('time', i)
            self.assertTrue(np.array_equal(ret, expected))

    def test_get_by_list(self):
        cov = self.get_cov()[0]

        expected = np.empty(0, dtype=cov.get_parameter_context('time').param_type.value_encoding)

        lists = [[[1,2,3]], [[3,5,19]]]

        for l in lists:
            ret = cov.get_parameter_values('time', l)
            self.assertTrue(np.array_equal(ret, expected))

    def test_slice_raises_index_error_out_out(self):
        # Tests that an array defined totally outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        _slice = slice(5010, 5020, None)
        self.assertTrue(np.array_equal(
            scov.get_parameter_values('time', _slice),
            np.empty(0, dtype=scov.get_parameter_context('time').param_type.value_encoding)))

    def test_int_raises_index_error(self):
        # Tests that an integer defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        self.assertTrue(np.array_equal(
            scov.get_parameter_values('time', 9000),
            np.empty(0, dtype=scov.get_parameter_context('time').param_type.value_encoding)))

    def test_array_raises_index_error(self):
        # Tests that an array defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        self.assertTrue(np.array_equal(
            scov.get_parameter_values('time', [[5,9000]]),
            np.empty(0, dtype=scov.get_parameter_context('time').param_type.value_encoding)))

    # @unittest.skip('Does not apply to empty coverage.')
    # def test_refresh(self):
    #     pass

    def test_get_time_data_metrics(self):
        cov = self.get_cov(only_time=True)[0]

        with self.assertRaises(ValueError):
            cov.get_data_bounds('time')

        with self.assertRaises(ValueError):
            cov.get_data_bounds_by_axis(axis=AxisTypeEnum.TIME)

        self.assertEqual(cov.get_data_extents('time'), (0,))
        self.assertEqual(cov.get_data_extents_by_axis(AxisTypeEnum.TIME), (0,))

    def test_get_all_data_metrics(self):
        cov = self.get_cov()[0]

        with self.assertRaises(ValueError):
            cov.get_data_bounds()

        self.assertEqual(cov.get_data_extents(),
                         {'conductivity': (0,), 'lat': (0,), 'lon': (0,), 'temp': (0,), 'time': (0,)})

    @unittest.skip('Does not apply to empty coverage.')
    def test_get_data_after_load(self):
        pass

    @unittest.skip('Does not apply to empty coverage.')
    def test_append_parameter(self):
        pass

    @unittest.skip('Does not apply to empty coverage.')
    def test_create_multi_bricks(self):
        pass

    @unittest.skip('Does not apply to empty coverage.')
    def test_pickle_problems_in_memory(self):
        pass

    @unittest.skip('Does not apply to empty coverage.')
    def test_slice_stop_greater_than_size(self):
        pass

    @unittest.skip('Does not apply to empty coverage.')
    def test_slice_stop_greater_than_size_with_step(self):
        pass


@attr('INT', group='cov')
class TestPtypesCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    @classmethod
    def get_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        """
        Construct coverage
        """
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        pname_filter = ['time',
                            'boolean',
                            'const_float',
                            'const_int',
                            'const_str',
                            'const_rng_flt',
                            'const_rng_int',
                            'numexpr_func',
                            'category',
                            'quantity',
                            'array',
                            'record',
                            'fixed_str',
                            'sparse']

        if only_time:
            pname_filter = ['time']

        pdict = get_parameter_dict(parameter_list=pname_filter)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestPtypesCovInt'
        else:
            # Add data for each parameter
            print nt
            if only_time:
                scov.insert_timesteps(nt)
                scov.set_parameter_values('time', value=np.arange(nt))
            else:
                scov.set_parameter_values('sparse', [[[2, 4, 6], [8, 10, 12]]])
                scov.insert_timesteps(nt/2)

                scov.set_parameter_values('sparse', [[[4, 8], [16, 20]]])
                scov.insert_timesteps(nt/2)

                scov.set_parameter_values('time', value=np.arange(nt))
                scov.set_parameter_values('boolean', value=[True, True, True], tdoa=[[2,4,14]])
                scov.set_parameter_values('const_float', value=-71.11) # Set a constant with correct data type
                scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
                scov.set_parameter_values('const_str', value='constant string value') # Set with a string
                scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
                scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list

                scov.set_parameter_values('quantity', value=np.random.random_sample(nt)*(26-23)+23)

                arrval = []
                recval = []
                catval = []
                fstrval = []
                catkeys = EXEMPLAR_CATEGORIES.keys()
                letts='abcdefghijklmnopqrstuvwxyz'
                while len(letts) < nt:
                    letts += 'abcdefghijklmnopqrstuvwxyz'
                for x in xrange(nt):
                    arrval.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
                    d = {letts[x]: letts[x:]}
                    recval.append(d) # One value (which is a dict) for each member of the domain
                    catval.append(random.choice(catkeys))
                    fstrval.append(''.join([random.choice(letts) for x in xrange(8)])) # A random string of length 8
                scov.set_parameter_values('array', value=arrval)
                scov.set_parameter_values('record', value=recval)
                scov.set_parameter_values('category', value=catval)
                scov.set_parameter_values('fixed_str', value=fstrval)

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        return scov, 'TestPtypesCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        # TODO: Only tests time parameter so far.
        param = 'time'
        data = data[_slice]
        scov.insert_timesteps(timesteps)
        scov.set_parameter_values(param, data, _slice)
        scov.get_dirty_values_async_result().get(timeout=60)
        ret = scov.get_parameter_values(param, _slice)
        return (ret == data).all()

    def test_ptypescov_get_values(self):
        # Tests retrieval of values from QuantityType, ConstantType,
        # TODO: Implement getting values from FunctionType, ArrayType and RecordType
        results = []
        ptypes_cov, cov_name = self.get_cov(nt=2000)
        self.assertIsInstance(ptypes_cov, AbstractCoverage)

        # QuantityType
        results.append((ptypes_cov._range_value.time[:] == np.arange(2000)).any())
        # ConstantType
        results.append(ptypes_cov._range_value.const_int[0] == 45)
        # ConstantRangeType
        results.append(ptypes_cov._range_value.const_rng_int[0] == (-10, 10))
        ptypes_cov.close()
        self.assertTrue(False not in results)

    @unittest.skip('Does not apply to empty coverage.')
    def test_get_all_data_metrics(self):
        pass
