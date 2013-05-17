#!/usr/bin/env python

"""
@package coverage_model.base_test_classes
@file coverage_model/base_test_classes.py
@author Christopher Mueller
@brief Base classes for Unit and Int testing within the coverage model
"""

from unittest import TestCase
import os, shutil, tempfile

from pyon.core import log as logutil
if not logutil.is_logging_configured():
    logutil.configure_logging(logutil.DEFAULT_LOGGING_PATHS)

class CoverageModelUnitTestCase(TestCase):

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None

    # override __str__ and __repr__ behavior to show a copy-pastable nosetest name for ion tests
    #  ion.module:TestClassName.test_function_name
    def __repr__(self):
        name = self.id()
        name = name.split('.')
        if name[0] not in ["coverage_model"]:
            return "%s (%s)" % (name[-1], '.'.join(name[:-1]))
        else:
            return "%s ( %s )" % (name[-1], '.'.join(name[:-2]) + ":" + '.'.join(name[-2:]))
    __str__ = __repr__


class CoverageModelIntTestCase(TestCase):

    working_dir = os.path.join(tempfile.gettempdir(), 'cov_mdl_tests')

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.working_dir):
            shutil.rmtree(cls.working_dir)

        os.mkdir(cls.working_dir)

    @classmethod
    def tearDownClass(cls):
        # Removes temporary files
        # Comment this out if you need to inspect the HDF5 files.
        shutil.rmtree(cls.working_dir)

    # override __str__ and __repr__ behavior to show a copy-pastable nosetest name for ion tests
    #  ion.module:TestClassName.test_function_name
    def __repr__(self):
        name = self.id()
        name = name.split('.')
        if name[0] not in ["coverage_model"]:
            return "%s (%s)" % (name[-1], '.'.join(name[:-1]))
        else:
            return "%s ( %s )" % (name[-1], '.'.join(name[:-2]) + ":" + '.'.join(name[-2:]))
    __str__ = __repr__