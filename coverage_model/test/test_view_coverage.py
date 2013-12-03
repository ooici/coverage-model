#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_view_coverage.py
@author James Case
@author Christopher Mueller
@brief Tests for the ViewCoverage class.
"""

from coverage_model import *
from nose.plugins.attrib import attr
import unittest
from pyon.public import log

from test_simplex_coverage import TestSampleCovInt as sc
from coverage_test_base import *


@attr('INT', group='cov')
class TestSampleCovViewInt(CoverageModelIntTestCase, CoverageIntTestBase):

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
        ref_cov, ref_cov_name = sc.get_cov(only_time=only_time, save_coverage=save_coverage, in_memory=in_memory, inline_data_writes=inline_data_writes, brick_size=brick_size, make_empty=make_empty, nt=nt, auto_flush_values=auto_flush_values)
        view_pdict = get_parameter_dict(parameter_list=['time','lat'])
        cov = ViewCoverage(cls.working_dir,
                           create_guid(),
                           name='sample coverage_model',
                           reference_coverage_location=ref_cov.persistence_dir,
                           parameter_dictionary=view_pdict)

        return cov, 'TestSampleCovViewInt'

    ######################
    # Additional tests specific to View Coverage
    ######################

    def test_replace_reference_coverage(self):
        cov1, _ = sc.get_cov(nt=10)

        cov2, _ = sc.get_cov(only_time=True, nt=10)
        cov2.set_time_values([i*2 for i in range(10)])

        vcov = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=cov1.persistence_dir)
        self.assertTrue(np.array_equal(vcov.get_time_values(), cov1.get_time_values()))
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])

        vcov.replace_reference_coverage(cov2.persistence_dir)
        self.assertTrue(np.array_equal(vcov.get_time_values(), cov2.get_time_values()))
        self.assertEqual(vcov.list_parameters(), ['time'])

        vcov.close()
        vcov = AbstractCoverage.load(vcov.persistence_dir)
        self.assertTrue(np.array_equal(vcov.get_time_values(), cov2.get_time_values()))
        self.assertEqual(vcov.list_parameters(), ['time'])

        vcov.close()
        vcov = AbstractCoverage.load(vcov.persistence_dir, mode='a')
        vcov.replace_reference_coverage(cov1.persistence_dir)

        self.assertTrue(np.array_equal(vcov.get_time_values(), cov1.get_time_values()))
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])

    def test_replace_reference_coverage_change_pdict(self):
        cov1, _ = sc.get_cov(nt=10)

        cov2, _ = sc.get_cov(only_time=True, nt=10)
        cov2.set_time_values([i*2 for i in range(10)])

        vcov = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=cov1.persistence_dir)
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])
        np.testing.assert_array_equal(vcov.get_time_values(), cov1.get_time_values())

        vcov.replace_reference_coverage(use_current_param_dict=False, parameter_dictionary=['time', 'temp'])
        self.assertEqual(vcov.list_parameters(), ['temp', 'time'])
        np.testing.assert_array_equal(vcov.get_time_values(), cov1.get_time_values())

        vcov.replace_reference_coverage(use_current_param_dict=False, parameter_dictionary='temp')
        self.assertEqual(vcov.list_parameters(), ['temp'])
        np.testing.assert_array_equal(vcov.get_parameter_values('temp'), cov1.get_parameter_values('temp'))

        vcov.replace_reference_coverage(use_current_param_dict=False, parameter_dictionary=None)
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])
        np.testing.assert_array_equal(vcov.get_time_values(), cov1.get_time_values())

    def test_replace_simplex_with_complex(self):
        cov, _ = sc.get_cov(nt=10)
        cov_pth = cov.persistence_dir
        cov.close()
        del cov

        vcov = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=cov_pth)
        self.assertEqual(vcov.head_coverage_path, cov_pth)
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])

        # Grab the view coverage path, then close it and delete the reference (for cleanliness)
        vcov_pth = vcov.persistence_dir
        vcov.close()
        del vcov

        # Load the view coverage WITH WRITE PERMISSIONS
        vcov = AbstractCoverage.load(vcov_pth, mode='a')

        # Create a complex coverage that uses the simplex coverage reference by the view coverage
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                               reference_coverage_locs=[vcov.head_coverage_path,],
                               parameter_dictionary=ParameterDictionary(),
                               complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)
        ccov_pth = ccov.persistence_dir
        self.assertEqual(ccov.head_coverage_path, cov_pth)
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])
        ccov.close()
        del ccov

        # Then replace the path in the view coverage
        vcov.replace_reference_coverage(ccov_pth)

        # Refresh the ViewCoverage - not actually be required, but...
        vcov.refresh()

        # Open the view,
        vcov = AbstractCoverage.load(vcov_pth)
        # and ensure that the .reference_coverage is the complex,
        self.assertEqual(vcov.reference_coverage.persistence_dir, ccov_pth)
        # but the .head_coverage_path is the simplex
        self.assertEqual(vcov.head_coverage_path, cov_pth)
        self.assertEqual(vcov.list_parameters(), ['conductivity', 'lat', 'lon', 'temp', 'time'])

    def test_head_coverage_path(self):
        cov1, _ = sc.get_cov(only_time=True, nt=10)
        if cov1._persistence_layer.master_manager.storage_type() != 'hdf':
            # TODO: Check for something Cassandra related
            self.assertTrue(True)
        else:
            # Ensure that for a first-order (VC --> SC) ViewCoverage.head_coverage_path reveals the underlying SimplexCoverage
            vcov1 = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=cov1.persistence_dir)
            self.assertEqual(vcov1.head_coverage_path, cov1.persistence_dir)

            # Ensure that for a second-order (VC --> VC --> SC) ViewCoverage.head_coverage_path reveals the underlying SimplexCoverage
            vcov2 = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=vcov1.persistence_dir)
            self.assertEqual(vcov2.head_coverage_path, cov1.persistence_dir)

    ######################
    # Overridden base tests
    ######################

    def test_refresh(self):
        brick_size = 1000
        time_steps = 5000

        from coverage_model.test.test_simplex_coverage import TestSampleCovInt as sc
        # Get a writable coverage
        write_cov, cov_name = sc.get_cov(only_time=True, brick_size=brick_size, nt=time_steps)

        # Get a ViewCoverage of that coverage
        read_cov = ViewCoverage(self.working_dir, create_guid(), name='sample view cov', reference_coverage_location=write_cov.persistence_dir)

        # Add some data to the writable copy & ensure a flush
        write_cov.insert_timesteps(100)
        tdat = range(write_cov.num_timesteps - 100, write_cov.num_timesteps)
        write_cov.set_time_values(tdat, slice(-100, None))

        # Refresh the read coverage
        read_cov.refresh()

        self.assertTrue(np.array_equal(write_cov.get_time_values(), read_cov.get_time_values()))

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_time_one_brick(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_allparams_five_bricks(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_allparams_one_brick(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_time_five_bricks(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_time_five_bricks_strided(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_set_time_one_brick_strided(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_persistence_variation1(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_persistence_variation2(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_persistence_variation3(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_persistence_variation4(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_pickle_problems_in_memory(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_coverage_pickle_and_in_memory(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_load_options_pd_pg(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_dot_load_fails_bad_guid(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_append_parameter(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_close_coverage_before_done_using_it(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_error_set_invalid_parameter(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_dot_load_options_pd(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_dot_load_options_pd_pg(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_create_multi_bricks(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_get_data_after_load(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_coverage_mode_expand_domain(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_coverage_mode_set_value(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_get_value_dict_tslice(self):
        pass

