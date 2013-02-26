#!/usr/bin/env python

"""
@package coverage_model.base_test_classes
@file coverage_model/base_test_classes.py
@author Christopher Mueller
@brief Base classes for Unit and Int testing within the coverage model
"""

from unittest import TestCase
import os, shutil, tempfile


class CoverageModelUnitTestCase(TestCase):

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None


class CoverageModelIntTestCase(TestCase):

    working_dir = os.path.join(tempfile.gettempdir(), 'cov_mdl_tests')

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None

    @classmethod
    def setUpClass(cls):
        os.mkdir(cls.working_dir)

    @classmethod
    def tearDownClass(cls):
        # Removes temporary files
        # Comment this out if you need to inspect the HDF5 files.
        shutil.rmtree(cls.working_dir)