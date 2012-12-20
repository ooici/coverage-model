#!/usr/bin/env python

"""
@package
@file test_parameter.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
from coverage_model import *
import numpy as np
from unittest import TestCase

@attr('UNIT',group='cov')
class TestParameterUnit(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dir_parameter_context(self):
        pc = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        d = dir(pc)
        self.assertEqual(d, dir(pc))

    def test_parameter_context_name(self):
        pc = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        self.assertEqual(pc.name, 'time')

    def test_parameter_context_is_coordinate(self):
        time_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        self.assertFalse(time_ctxt.is_coordinate)
        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.reference_frame = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        lat_ctxt.axis = AxisTypeEnum.GEO_Y
        self.assertTrue(lat_ctxt.is_coordinate)

    def test_parameter_context_ne(self):
        time_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        self.assertNotEqual(time_ctxt, lat_ctxt)

    def test_pass_pc_list_into_pd(self):
        time_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_list = [time_ctxt, lat_ctxt]
        pd = ParameterDictionary(pc_list)
        self.assertIsInstance(pd, ParameterDictionary)

    def test_get_context_variants(self):
        time_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_list = [time_ctxt, lat_ctxt]
        pd = ParameterDictionary(pc_list)
        self.assertEqual(time_ctxt, pd.get_context_by_ord(1))
        with self.assertRaises(KeyError):
            pd.get_context('unknown')

    def test_size(self):
        time_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_list = [time_ctxt, lat_ctxt]
        pd = ParameterDictionary(pc_list)
        len(pd)
        pd.size()