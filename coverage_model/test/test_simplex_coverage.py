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

@attr('INT', group='jdc')
class TestOneParamCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, save_coverage=False, in_memory=False, inline_data_writes=True, make_empty=False):
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

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        bricking_scheme = {'brick_size':1000,'chunk_size':500}
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

        # Insert some timesteps (automatically expands other arrays)
        nt = 1000
        scov.insert_timesteps(nt)

        # Add data for the parameter
        scov.set_parameter_values('time', value=np.arange(nt))

        return scov


@attr('INT', group='jdc')
class TestEmptySampleCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, save_coverage=False, in_memory=False, inline_data_writes=True, make_empty=False, brick_size=None):
        # Instantiate a ParameterDictionary
        pname_filter = ['time',
                        'lat',
                        'lon',
                        'temp',
                        'conductivity']
        pdict = get_parameter_dict(parameter_list=pname_filter)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':10}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage('test_data', create_guid(), 'empty sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

        return scov

@attr('INT', group='jdc')
class TestPtypesCovInt(CoverageModelIntTestCase, CoverageIntTestBase):

    def setUp(self):
        pass

    def get_cov(self, save_coverage=False, in_memory=False, inline_data_writes=True, make_empty=False):
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

        pdict = get_parameter_dict(parameter_list=pname_filter)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

        # Insert some timesteps (automatically expands other arrays)
        if make_empty:
            return scov

        nt = 20
        scov.insert_timesteps(nt)

        # Add data for each parameter
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

        return scov
