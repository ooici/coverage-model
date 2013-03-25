#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_view_coverage.py
@author James Case
@brief Tests for the ViewCoverage class.
"""

from coverage_model import *
from nose.plugins.attrib import attr
import unittest
from pyon.public import log

from test_simplex_coverage import TestSampleCovInt as sc

from coverage_test_base import *


@attr('INT', group='view_cov')
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
    def get_cov(self, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=False):
        ref_cov, ref_cov_name = sc.get_cov(only_time=only_time, save_coverage=save_coverage, in_memory=in_memory, inline_data_writes=inline_data_writes, brick_size=brick_size, make_empty=make_empty, nt=nt, auto_flush_values=auto_flush_values)
        view_pdict = get_parameter_dict(parameter_list=['time'])
        cov = ViewCoverage(self.working_dir,
                           create_guid(),
                           reference_coverage_location=ref_cov.persistence_dir,
                           name='sample coverage_model',
                           parameter_dictionary=view_pdict)

        return cov, 'TestSampleCovViewInt'

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_load_init_succeeds(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_load_succeeds_with_options(self):
        pass

    @unittest.skip('Does not apply to ViewCoverage.')
    def test_dot_load_succeeds(self):
        pass

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
    def test_get_time_metadata(self):
        pass