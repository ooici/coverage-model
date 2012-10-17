#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_coverage.py
@author Christopher Mueller
@brief Test cases for the coverage_model module
"""

from pyon.public import log
from pyon.util.int_test import IonIntegrationTestCase
from nose.plugins.attrib import attr
from mock import patch, Mock

import unittest

@attr('INT', group='cov')
class TestCoverageModelBasicsInt(IonIntegrationTestCase):

    def setUp(self):
        pass

