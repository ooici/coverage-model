#!/usr/bin/env python

"""
@package coverage_model.test.test_complex_coverage
@file coverage_model/test/test_complex_coverage.py
@author Christopher Mueller
@brief Unit & Integration tests for ComplexCoverage
"""

import os
import numpy as np
from coverage_model import *
from nose.plugins.attrib import attr


def _make_cov(root_dir, params, nt=10, data_dict=None):
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

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
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
            scov.set_parameter_values(p, data_dict[p])
        else:
            scov.set_parameter_values(p, range(nt))

    scov.close()

    return os.path.realpath(scov.persistence_dir)


@attr('INT',group='cm')
class TestComplexCoverageInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_parametric(self):
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

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        ccov = ComplexCoverage(self.working_dir, 'test_complex', 'sample complex coverage',
                               reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                               parameter_dictionary=pdict,
                               complex_type=ComplexCoverageType.PARAMETRIC)

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



