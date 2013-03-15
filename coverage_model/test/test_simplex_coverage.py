#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_simplex_coverage.py
@author James Case
@brief Tests for the SimplexCoverage class.
"""

from coverage_model import *
from nose.plugins.attrib import attr
import numpy as np
from pyon.public import log
import random

from coverage_test_base import *


@attr('INT', group='cov')
class TestSampleCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    @classmethod
    def get_cov(self, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=False):
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
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

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
                tvals=np.random.random_sample(nt)*(26-23)+23
                scov.set_parameter_values('temp', value=tvals)
                scov.set_parameter_values('conductivity', value=np.random.random_sample(nt)*(110-90)+90)

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'sample.cov'))

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

@attr('INT', group='cov')
class TestOneParamCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=False):
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
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestOneParamCovInt'
        else:
            scov.insert_timesteps(nt)

            # Add data for the parameter
            scov.set_parameter_values('time', value=np.arange(nt))

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'sample.cov'))

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

@attr('INT', group='cov')
class TestEmptySampleCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, only_time=False, param_filter=None, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=False):
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
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'sample.cov'))

        return scov, 'TestEmptySampleCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        return True

    def test_get_all_data_metadata(self):
        pass

    def test_get_data_after_load(self):
        pass

    def test_append_parameter(self):
        pass


@attr('INT', group='cov')
class TestPtypesCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=False):
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
                            'fixed_str']

        if only_time:
            pname_filter = ['time']

        pdict = get_parameter_dict(parameter_list=pname_filter)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestPtypesCovInt'
        else:
            scov.insert_timesteps(nt)

            # Add data for each parameter
            scov.set_parameter_values('time', value=np.arange(nt))
            if not only_time:
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
            SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'sample.cov'))

        return scov, 'TestPtypesCovInt'

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        return True
        # # Function to test variable occurances of getting and setting values across parameter(s)
        # data = data[_slice]
        # ret = []
        #
        # scov.insert_timesteps(timesteps)
        #
        # scov.set_parameter_values('time', value=data, slice=_slice)
        # if param == 'all':
        #     scov.set_parameter_values('boolean', value=[True, True, True], tdoa=[[2,4,14]])
        #     scov.set_parameter_values('const_float', value=-71.11) # Set a constant with correct data type
        #     scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
        #     scov.set_parameter_values('const_str', value='constant string value') # Set with a string
        #     scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
        #     scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list
        #
        #     scov.set_parameter_values('quantity', value=np.random.random_sample(nt)*(26-23)+23, slice=_slice)
        #
        #     arrval = []
        #     recval = []
        #     catval = []
        #     fstrval = []
        #     catkeys = EXEMPLAR_CATEGORIES.keys()
        #     letts='abcdefghijklmnopqrstuvwxyz'
        #     while len(letts) < nt:
        #         letts += 'abcdefghijklmnopqrstuvwxyz'
        #     for x in xrange(nt):
        #         arrval.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
        #         d = {letts[x]: letts[x:]}
        #         recval.append(d) # One value (which is a dict) for each member of the domain
        #         catval.append(random.choice(catkeys))
        #         fstrval.append(''.join([random.choice(letts) for x in xrange(8)])) # A random string of length 8
        #     scov.set_parameter_values('array', value=arrval)
        #     scov.set_parameter_values('record', value=recval)
        #     scov.set_parameter_values('category', value=catval)
        #     scov.set_parameter_values('fixed_str', value=fstrval)
        #
        # #TODO: Loop over all parameters?????
        # ret = scov.get_parameter_values('time', slice=_slice)
        #
        # return (ret == data).all()

    def test_ptypescov_get_values(self):
        # Tests retrieval of values from QuantityType, ConstantType,
        # TODO: Implement getting values from FunctionType, ArrayType and RecordType
        results = []
        ptypes_cov, cov_name = self.get_cov(nt=2000)
        self.assertIsInstance(ptypes_cov, SimplexCoverage)

        # QuantityType
        results.append((ptypes_cov._range_value.time[:] == np.arange(2000)).any())
        # ConstantType
        results.append(ptypes_cov._range_value.const_int[0] == 45)
        # ConstantRangeType
        results.append(ptypes_cov._range_value.const_rng_int[0] == (-10, 10))
        ptypes_cov.close()
        self.assertTrue(False not in results)

    # def test_slice_and_dice(self):
    #     pass