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

    def test_parameter_dictionary_comparisons(self):
        # Tests ParameterDictionary and ParameterContext creation
        # Instantiate a ParameterDictionary
        pdict_1 = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        pdict_1.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
        pdict_1.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
        pdict_1.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
        pdict_1.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))

        # Instantiate a ParameterDictionary
        pdict_2 = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        pdict_2.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
        pdict_2.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
        pdict_2.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
        pdict_2.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))

        # Instantiate a ParameterDictionary
        pdict_3 = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        pdict_3.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
        pdict_3.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
        pdict_3.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
        pdict_3.add_context(ParameterContext('temp2', param_type=QuantityType(uom='degree_Celsius')))

        # Instantiate a ParameterDictionary
        pdict_4 = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        pdict_4.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
        pdict_4.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
        pdict_4.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))

        temp_ctxt = ParameterContext('temp', param_type=QuantityType(uom = 'degree_Celsius'))
        pdict_4.add_context(temp_ctxt)

        temp2_ctxt = ParameterContext(name=temp_ctxt, new_name='temp2')
        pdict_4.add_context(temp2_ctxt)

        with self.assertRaises(SystemError):
            ParameterContext([10,20,30], param_type=QuantityType(uom = 'bad name'))

        with self.assertRaises(SystemError):
            ParameterContext(None,None)

        with self.assertRaises(SystemError):
            ParameterContext(None)

        with self.assertRaises(TypeError):
            ParameterContext()

        with self.assertRaises(SystemError):
            ParameterContext(None, param_type=QuantityType(uom = 'bad name'))

        # Should be equal and compare one-to-one with nothing in the None list
        self.assertEquals(pdict_1, pdict_2)
        self.assertEquals(pdict_1.compare(pdict_2), {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp'], 'time': ['time']})

        # Should be unequal and compare with an empty list for 'temp' and 'temp2' in the None list
        self.assertNotEquals(pdict_1, pdict_3)
        self.assertEquals(pdict_1.compare(pdict_3), {'lat': ['lat'], 'lon': ['lon'], None: ['temp2'], 'temp': [], 'time': ['time']})

        # Should be unequal and compare with both 'temp' and 'temp2' in 'temp' and nothing in the None list
        self.assertNotEquals(pdict_1,  pdict_4)
        self.assertEquals(pdict_1.compare(pdict_4), {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp', 'temp2'], 'time': ['time']})

    def test_bad_pc_from_dict(self):
        # Tests improper load of a ParameterContext
        pc1 = ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius'))
        with self.assertRaises(TypeError):
            pc1._fromdict('junk', pc1.dump())
        pc2 = pc1._fromdict(pc1.dump())
        self.assertEquals(pc1, pc2)

    def test_dump_and_load_from_dict(self):
        # Tests improper load of a ParameterContext
        pc1 = ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius'))
        pc2 = pc1._fromdict(pc1.dump())
        self.assertEquals(pc1, pc2)

    def test_param_dict_from_dict(self):
        pdict_1 = ParameterDictionary()
        pdict_1.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
        pdict_1.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
        pdict_1.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
        pdict_1.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))
        new_pdict = ParameterDictionary._fromdict(pdict_1._todict())
        self.assertTrue(pdict_1 == new_pdict)