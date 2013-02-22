#!/usr/bin/env python

"""
@package coverage_model.base_test_classes
@file coverage_model/base_test_classes.py
@author Christopher Mueller
@brief Base classes for Unit and Int testing within the coverage model
"""

from unittest import TestCase


class CoverageModelUnitTestCase(TestCase):

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None


class CoverageModelIntTestCase(TestCase):

    # Prevent test docstring from printing - uses test name instead
    # @see
    # http://www.saltycrane.com/blog/2012/07/how-prevent-nose-unittest-using-docstring-when-verbosity-2/
    def shortDescription(self):
        return None