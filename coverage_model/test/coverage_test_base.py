#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/
@author James Case
@brief Base tests for all coverages
"""

from pyon.public import log
from nose.plugins.attrib import attr
import numpy as np
from coverage_model import *

class CoverageIntTestBase(object):
# class CoverageIntTestBase(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def get_cov(self):
        raise NotImplementedError()

    # ############################
    # CONSTRUCTION

    def test_create_cov(self):
        cov = self.get_cov()
        self.assertIsInstance(cov, SimplexCoverage)


    # ############################
    # LOADING

    def test_load_succeeds(self):
        # Creates a valid coverage, inserts data and loads coverage back up from the HDF5 files.
        scov = self.get_cov()
        # self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage(base_path, guid)
        self.assertIsInstance(lcov, SimplexCoverage)
        lcov.close()

    # ############################
    # MODES

    # ############################
    # GET

    # ############################
    # SET

    # ############################
    # INLINE & OUT OF BAND R/W

    # ############################
    # CACHING

    # ############################
    # ERRORS



def get_parameter_dict_info():
    pdict_info = {}

    for p in MASTER_PDICT:
        if hasattr(p, 'description') and hasattr(p.param_type, 'value_encoding') and hasattr(p, 'name'):
            pdict_info[p.name] = [p.description, p.param_type.value_encoding]
        else:
            raise ValueError('Parameter {0} does not contain appropriate attributes.'.format(p))

    return pdict_info

def get_parameter_dict(parameter_list=None):
    from copy import deepcopy

    pdict_ret = ParameterDictionary()

    if parameter_list is None:
        return MASTER_PDICT
    else:
        for pname in parameter_list:
            if pname in MASTER_PDICT:
                pdict_ret.add_context(deepcopy(MASTER_PDICT.get_context(pname)))

    return pdict_ret

EXEMPLAR_CATEGORIES = {0:'turkey',1:'duck',2:'chicken',99:'None'}

def _make_master_parameter_dict():
    # Construct ParameterDictionary of all supported types
    pdict = ParameterDictionary()

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
    temp_ctxt.description = 'example of a parameter type QuantityType, base_type float32'
    temp_ctxt.uom = 'degree_Celsius'
    pdict.add_context(temp_ctxt)

    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    cond_ctxt.description = ''
    cond_ctxt.uom = 'unknown'
    pdict.add_context(cond_ctxt)

    bool_ctxt = ParameterContext('boolean', param_type=BooleanType(), variability=VariabilityEnum.TEMPORAL)
    bool_ctxt.description = ''
    pdict.add_context(bool_ctxt)

    cnst_flt_ctxt = ParameterContext('const_float', param_type=ConstantType(), variability=VariabilityEnum.NONE)
    cnst_flt_ctxt.description = 'example of a parameter of type ConstantType, base_type float (default)'
    cnst_flt_ctxt.long_name = 'example of a parameter of type ConstantType, base_type float (default)'
    cnst_flt_ctxt.axis = AxisTypeEnum.LON
    cnst_flt_ctxt.uom = 'degree_east'
    pdict.add_context(cnst_flt_ctxt)

    cnst_int_ctxt = ParameterContext('const_int', param_type=ConstantType(QuantityType(value_encoding=np.dtype('int32'))), variability=VariabilityEnum.NONE)
    cnst_int_ctxt.description = 'example of a parameter of type ConstantType, base_type int32'
    cnst_int_ctxt.axis = AxisTypeEnum.LAT
    cnst_int_ctxt.uom = 'degree_north'
    pdict.add_context(cnst_int_ctxt)

    cnst_str_ctxt = ParameterContext('const_str', param_type=ConstantType(QuantityType(value_encoding=np.dtype('S21'))), fill_value='', variability=VariabilityEnum.NONE)
    cnst_str_ctxt.description = 'example of a parameter of type ConstantType, base_type fixed-len string'
    pdict.add_context(cnst_str_ctxt)

    cnst_rng_flt_ctxt = ParameterContext('const_rng_flt', param_type=ConstantRangeType(), variability=VariabilityEnum.NONE)
    cnst_rng_flt_ctxt.description = 'example of a parameter of type ConstantRangeType, base_type float (default)'
    pdict.add_context(cnst_rng_flt_ctxt)

    cnst_rng_int_ctxt = ParameterContext('const_rng_int', param_type=ConstantRangeType(QuantityType(value_encoding='int16')), variability=VariabilityEnum.NONE)
    cnst_rng_int_ctxt.long_name = 'example of a parameter of type ConstantRangeType, base_type int16'
    pdict.add_context(cnst_rng_int_ctxt)

    func = NumexprFunction('numexpr_func', expression='q*10', arg_list=['q'], param_map={'q':'quantity'})
    pfunc_ctxt = ParameterContext('parameter_function', param_type=ParameterFunctionType(function=func), variability=VariabilityEnum.TEMPORAL)
    pfunc_ctxt.description = 'example of a parameter of type ParameterFunctionType'
    pdict.add_context(pfunc_ctxt)

    cat_ctxt = ParameterContext('category', param_type=CategoryType(categories=EXEMPLAR_CATEGORIES), variability=VariabilityEnum.TEMPORAL)
    cat_ctxt.description = ''
    pdict.add_context(cat_ctxt)

    quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    quant_ctxt.description = 'example of a parameter of type QuantityType'
    quant_ctxt.uom = 'degree_Celsius'
    pdict.add_context(quant_ctxt)

    arr_ctxt = ParameterContext('array', param_type=ArrayType())
    arr_ctxt.description = 'example of a parameter of type ArrayType, will be filled with variable-length \'byte-string\' data'
    pdict.add_context(arr_ctxt)

    rec_ctxt = ParameterContext('record', param_type=RecordType())
    rec_ctxt.description = 'example of a parameter of type RecordType, will be filled with dictionaries'
    pdict.add_context(rec_ctxt)

    fstr_ctxt = ParameterContext('fixed_str', param_type=QuantityType(value_encoding=np.dtype('S8')), fill_value='')
    fstr_ctxt.description = 'example of a fixed-length string parameter'
    pdict.add_context(fstr_ctxt)

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    from coverage_model.test.test_parameter_functions import _create_all_params
    contexts = _create_all_params()
    pdict.add_context(contexts.pop('time'), is_temporal=True)  # Add time
    map(pdict.add_context, contexts.values())  # Add others

    return pdict

MASTER_PDICT = _make_master_parameter_dict()

# @attr('INT', group='jdc')
# def test_pdict_helper():
#     pdict = get_parameter_dict()
#     self.assertEqual(MASTER_PDICT.keys(), pdict.keys())
#
#     pname_filter = ['TIME','CONDUCTIVITY','TEMPWAT_L0']
#     pdict = get_parameter_dict(parameter_list=pname_filter)
#     self.assertIsInstance(pdict, ParameterDictionary)
#     self.assertEqual(pname_filter, pdict.keys())