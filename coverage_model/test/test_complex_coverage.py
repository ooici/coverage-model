#!/usr/bin/env python

"""
@package coverage_model.test.test_complex_coverage
@file coverage_model/test/test_complex_coverage.py
@author Christopher Mueller
@brief Unit & Integration tests for ComplexCoverage
"""

import numpy as np
from coverage_model import *
from nose.plugins.attrib import attr

@attr('INT',group='cm')
class TestComplexCoverageInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def _make_cov(self, params, nt=10):
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

        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)

        scov.insert_timesteps(nt)
        # scov.set_time_values('time', range(nt))
        for p in scov.list_parameters(data_only=True):
            scov.set_parameter_values(p, range(nt))

        scov.close()

        return scov.persistence_dir

    def test_parametric(self):
        cova_pth = self._make_cov(['first_param'])
        covb_pth = self._make_cov(['second_param'])

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        func = NumexprFunction('aXb', 'a*b', ['a', 'b'], {'a': 'first_param', 'b': 'second_param'})
        val_ctxt = ParameterContext('aXb', param_type=ParameterFunctionType(function=func, value_encoding=np.dtype('float32')))
        pdict.add_context(val_ctxt)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        ccov = ComplexCoverage('test_data', 'test_complex', 'sample complex coverage', reference_coverages=[cova_pth, covb_pth], parameter_dictionary=pdict, complex_type=ComplexCoverageType.PARAMETRIC)

        want = np.arange(10, dtype='float32')
        self.assertTrue(np.array_equal(ccov.get_parameter_values('first_param'), want))
        self.assertTrue(np.array_equal(ccov.get_parameter_values('second_param'), want))

        aXb_want = want * want
        self.assertTrue(np.array_equal(ccov.get_parameter_values('aXb'), aXb_want))

