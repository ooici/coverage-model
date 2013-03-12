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

@attr('UNIT',group='cov')
class TestParameterUnit(CoverageModelUnitTestCase):

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
        lat_ctxt.axis = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
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

        self.assertEqual(len(pd), pd.size())

    def test_pc_mod_deps(self):
        owner = 'gsw'
        sal_func = 'SP_from_C'
        sal_arglist = ['C', 't', 'p']
        sal_pmap = {'C': NumexprFunction('CONDWAT_L1*10', 'C*10', ['C'], param_map={'C': 'CONDWAT_L1'}), 't': 'TEMPWAT_L1', 'p': 'PRESWAT_L1'}
        sal_kwargmap = None
        expr = PythonFunction('PRACSAL', owner, sal_func, sal_arglist, sal_kwargmap, sal_pmap)
        sal_ctxt = ParameterContext('PRACSAL', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)

        self.assertEqual(('numexpr', 'gsw'), sal_ctxt.get_module_dependencies())

