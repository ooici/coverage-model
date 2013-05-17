#!/usr/bin/env python

"""
@package coverage_model.test.test_recovery
@file coverage_model/test/test_recovery.py
@author Christopher Mueller
@brief Tests for coverage_model.recovery module
"""

from coverage_model import *
from coverage_model.recovery import CoverageDoctor

from nose.plugins.attrib import attr
import unittest
import numpy as np
from coverage_model.base_test_cases import CoverageModelIntTestCase
from interface.objects import Dataset

from subprocess import call
import re

not_have_h5stat = call('which h5stat'.split(), stdout=open('/dev/null','w'))
if not not_have_h5stat:
    from subprocess import check_output
    from distutils.version import StrictVersion
    output = check_output('h5stat -V'.split())
    version_str = re.match(r'.*(\d+\.\d+\.\d+).*', output).groups()[0]
    h5stat_correct_version = StrictVersion(version_str) >= StrictVersion('1.8.9')

@attr('INT', group='rec')
class TestRecoveryInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @unittest.skipIf(not_have_h5stat, 'h5stat is not accessible in current PATH')
    @unittest.skipIf(not not_have_h5stat and not h5stat_correct_version, 'HDF is the incorrect version: %s' % version_str)
    def test_coverage_recovery(self):
        # Create the coverage
        cov, dset = self.get_cov()
        cov_pth = cov.persistence_dir

        # Analyze the valid coverage
        dr = CoverageDoctor(cov_pth, 'dprod', dset)

        dr_result = dr.analyze()

        # TODO: Turn these into meaningful Asserts
        self.assertEqual(len(dr_result.get_brick_corruptions()), 0)
        self.assertEqual(len(dr_result.get_brick_size_ratios()), 6)
        self.assertEqual(len(dr_result.get_corruptions()), 0)
        self.assertEqual(len(dr_result.get_master_corruption()), 0)
        self.assertEqual(len(dr_result.get_param_corruptions()), 0)
        self.assertEqual(len(dr_result.get_param_size_ratios()), 3)
        self.assertEqual(len(dr_result.get_master_size_ratio()), 1)
        self.assertEqual(len(dr_result.get_size_ratios()), 10)
        self.assertEqual(dr_result.master_status[1], 'NORMAL')

        self.assertFalse(dr_result.is_corrupt)
        self.assertEqual(dr_result.param_file_count, 3)
        self.assertEqual(dr_result.brick_file_count, 6)
        self.assertEqual(dr_result.total_file_count, 10)

        # Get original values (mock)
        orig_cov = AbstractCoverage.load(cov_pth)
        time_vals_orig = orig_cov.get_time_values()

        # Corrupt the Master File
        fo = open(cov._persistence_layer.master_manager.file_path, "wb")
        fo.write('Junk')
        fo.close()
        # corrupt_res = dr.analyze(reanalyze=True)
        # self.assertTrue(corrupt_res.is_corrupt)

        # TODO: Destroy the metadata files

        # TODO: RE-analyze coverage

        # TODO: Should be corrupt, take action to repair if so

        # Repair the metadata files
        dr.repair_metadata()

        # TODO: Re-analyze fixed coverage

        fixed_cov = AbstractCoverage.load(cov_pth)
        self.assertIsInstance(fixed_cov, AbstractCoverage)

        time_vals_fixed = fixed_cov.get_time_values()
        self.assertTrue(np.array_equiv(time_vals_orig, time_vals_fixed))

    def get_cov(self):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt, is_temporal=True)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.axis = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.axis = AxisTypeEnum.LON
        lon_ctxt.uom = 'degree_east'
        pdict.add_context(lon_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=True, in_memory_storage=False)

        # Insert some timesteps (automatically expands other arrays)
        nt = 200000
        scov.insert_timesteps(nt)

        # Add data for each parameter
        scov.set_parameter_values('time', value=np.arange(nt))
        scov.set_parameter_values('lat', value=45)
        scov.set_parameter_values('lon', value=-71)

        dset = Dataset()
        dset.parameter_dictionary = pdict.dump()
        dset.spatial_domain = sdom.dump()
        dset.temporal_domain = tdom.dump()


        return scov, dset