#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/
@author James Case
@brief Base tests for all coverages
"""

from coverage_model import *
from nose.plugins.attrib import attr
import numpy as np
import os
from pyon.public import log


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

    def test_create_guid_valid(self):
        # Tests that the create_guid() function outputs a properly formed GUID
        self.assertTrue(len(create_guid()) == 36)

    def test_create_name_invalid(self):
        # Tests condition where the coverage name is an invalid type
        pdict = get_parameter_dict()
        tcrs = _make_tcrs()
        tdom = _make_tdom(tcrs)
        scrs = _make_scrs()
        sdom = _make_sdom(scrs)
        in_memory = False
        name = np.arange(10) # Numpy array is not a valid coverage name
        with self.assertRaises(TypeError):
            SimplexCoverage(
                root_dir=self.working_dir,
                persistence_guid=create_guid(),
                name=name,
                parameter_dictionary=pdict,
                temporal_domain=tdom,
                spatial_domain=sdom,
                in_memory_storage=in_memory)




    # ############################
    # LOADING

    def test_load_succeeds(self):
        # Creates a valid coverage, inserts data and loads coverage back up from the HDF5 files.
        scov = self.get_cov()
        _insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage(base_path, guid)
        self.assertIsInstance(lcov, SimplexCoverage)

        lcov = SimplexCoverage.load(scov.persistence_dir)
        self.assertIsInstance(lcov, SimplexCoverage)
        lcov.close()

    def test_dot_load_succeeds(self):
        # Creates a valid coverage, inserts data and .load coverage back up from the HDF5 files.
        scov = self.get_cov()
        _insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage.load(base_path, guid)
        lcov.close()
        self.assertIsInstance(lcov, SimplexCoverage)

    def test_get_data_after_load(self):
        # Creates a valid coverage, inserts data and .load coverage back up from the HDF5 files.
        results =[]
        scov = self.get_cov()
        _insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage.load(base_path, guid)
        ret_data = lcov.get_parameter_values('time', slice(0,50))
        results.append(np.arange(50).any() == ret_data.any())
        self.assertTrue(False not in results)
        lcov.close()
        self.assertIsInstance(lcov, SimplexCoverage)

    def test_load_fails_bad_guid(self):
        # Tests load fails if coverage exists and path is correct but GUID is incorrect
        scov = self.get_cov()
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertTrue(os.path.exists(scov.persistence_dir))
        guid = 'some_incorrect_guid'
        base_path = scov.persistence_dir
        scov.close()
        with self.assertRaises(SystemError) as se:
            SimplexCoverage(base_path, guid)
            self.assertEquals(se.message, 'Cannot find specified coverage: {0}'.format(os.path.join(base_path, guid)))

    def test_dot_load_fails_bad_guid(self):
        # Tests load fails if coverage exists and path is correct but GUID is incorrect
        scov = self.get_cov()
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertTrue(os.path.exists(scov.persistence_dir))
        guid = 'some_incorrect_guid'
        base_path = scov.persistence_dir
        scov.close()
        with self.assertRaises(SystemError) as se:
            SimplexCoverage.load(base_path, guid)
            self.assertEquals(se.message, 'Cannot find specified coverage: {0}'.format(os.path.join(base_path, guid)))

    def test_load_only_pd_raises_error(self):
        scov = self.get_cov()
        scov.close()
        with self.assertRaises(TypeError):
            SimplexCoverage(scov.persistence_dir)

    def test_load_options_pd_pg(self):
        scov = self.get_cov()
        scov.close()
        cov = SimplexCoverage(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd(self):
        scov = self.get_cov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd_pg(self):
        scov = self.get_cov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_load_succeeds_with_options(self):
        # Tests loading a SimplexCoverage using init parameters
        scov = self.get_cov()
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        scov.close()
        base_path = root_path.replace(guid,'')
        name = 'coverage_name'
        pdict = ParameterDictionary()
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        lcov = SimplexCoverage(base_path, guid, name, pdict, tdom, sdom)
        lcov.close()
        self.assertIsInstance(lcov, SimplexCoverage)

    # ############################
    # MODES

    def test_coverage_mode_expand_domain(self):
        scov = self.get_cov()
        self.assertEqual(scov.mode, 'a')
        scov.close()
        rcov = SimplexCoverage.load(scov.persistence_dir, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov.insert_timesteps(10)

    def test_coverage_mode_set_value(self):
        scov = self.get_cov()
        self.assertEqual(scov.mode, 'a')
        scov.insert_timesteps(10)
        scov.close()
        rcov = SimplexCoverage.load(scov.persistence_dir, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov._range_value.time[0] = 1

    # ############################
    # GET

    def test_domain_expansion(self):
        # Tests temporal_domain expansion and getting and setting values for all parameters
        res = False
        scov = self.get_cov(nt=0)
        tsteps = scov.num_timesteps
        res = _run_standard_tests(scov, tsteps)
        self.assertTrue(res)
        tsteps = tsteps + 10
        res = _insert_set_get(scov=scov, timesteps=tsteps, data=np.arange(tsteps), _slice=slice(scov.num_timesteps, tsteps), param='all')
        self.assertTrue(res)
        res = _run_standard_tests(scov, tsteps)
        self.assertTrue(res)
        prev_tsteps = tsteps
        tsteps = 35
        res = _insert_set_get(scov=scov, timesteps=tsteps, data=np.arange(tsteps)+prev_tsteps, _slice=slice(prev_tsteps, tsteps), param='all')
        self.assertTrue(res)
        res = _run_standard_tests(scov, tsteps+prev_tsteps)
        scov.close()
        self.assertTrue(res)

    # ############################
    # SET

    # ############################
    # INLINE & OUT OF BAND R/W

    # ############################
    # CACHING

    # ############################
    # ERRORS

    # ############################
    # SAVE
    def test_coverage_flush(self):
        # Tests that the .flush() function flushes the coverage
        scov = self.get_cov()
        scov.flush()
        self.assertTrue(not scov.has_dirty_values())
        scov.close()

    def test_coverage_save(self):
        # Tests that the .save() function flushes coverage
        scov = self.get_cov()
        scov.save(scov)
        self.assertTrue(not scov.has_dirty_values())
        scov.close()



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

    ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    ctxt.description = ''
    ctxt.axis = AxisTypeEnum.TIME
    ctxt.uom = 'seconds since 01-01-1900'
    pdict.add_context(ctxt, is_temporal=True)

    ctxt = ParameterContext('lat', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    ctxt.description = ''
    ctxt.axis = AxisTypeEnum.LAT
    ctxt.uom = 'degree_north'
    pdict.add_context(ctxt)

    ctxt = ParameterContext('lon', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    ctxt.description = ''
    ctxt.axis = AxisTypeEnum.LON
    ctxt.uom = 'degree_east'
    pdict.add_context(ctxt)

    # Temperature - values expected to be the decimal results of conversion from hex
    ctxt = ParameterContext('TEMPWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'deg_C'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Conductivity - values expected to be the decimal results of conversion from hex
    ctxt = ParameterContext('CONDWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Pressure - values expected to be the decimal results of conversion from hex
    ctxt = ParameterContext('PRESWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'dbar'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # TEMPWAT_L1 = (TEMPWAT_L0 / 10000) - 10
    tl1_func = '(T / 10000) - 10'
    tl1_pmap = {'T': 'TEMPWAT_L0'}
    expr = NumexprFunction('TEMPWAT_L1', tl1_func, ['T'], param_map=tl1_pmap)
    ctxt = ParameterContext('TEMPWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'deg_C'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # CONDWAT_L1 = (CONDWAT_L0 / 100000) - 0.5
    cl1_func = '(C / 100000) - 0.5'
    cl1_pmap = {'C': 'CONDWAT_L0'}
    expr = NumexprFunction('CONDWAT_L1', cl1_func, ['C'], param_map=cl1_pmap)
    ctxt = ParameterContext('CONDWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
    #   PRESWAT_L1 = (PRESWAT_L0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
    pl1_func = '(P * p_range / (0.85 * 65536)) - (0.05 * p_range)'
    pl1_pmap = {'P': 'PRESWAT_L0', 'p_range': 679.34040721}
    expr = NumexprFunction('PRESWAT_L1', pl1_func, ['P', 'p_range'], param_map=pl1_pmap)
    ctxt = ParameterContext('PRESWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
    #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

    # PRACSAL = gsw.SP_from_C((CONDWAT_L1 * 10), TEMPWAT_L1, PRESWAT_L1)
    owner = 'gsw'
    sal_func = 'SP_from_C'
    sal_arglist = ['C', 't', 'p']
    sal_pmap = {'C': NumexprFunction('CONDWAT_L1*10', 'C*10', ['C'], param_map={'C': 'CONDWAT_L1'}), 't': 'TEMPWAT_L1', 'p': 'PRESWAT_L1'}
    sal_kwargmap = None
    expr = PythonFunction('PRACSAL', owner, sal_func, sal_arglist, sal_kwargmap, sal_pmap)
    ctxt = ParameterContext('PRACSAL', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'g kg-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # absolute_salinity = gsw.SA_from_SP(PRACSAL, PRESWAT_L1, longitude, latitude)
    # conservative_temperature = gsw.CT_from_t(absolute_salinity, TEMPWAT_L1, PRESWAT_L1)
    # DENSITY = gsw.rho(absolute_salinity, conservative_temperature, PRESWAT_L1)
    owner = 'gsw'
    abs_sal_expr = PythonFunction('abs_sal', owner, 'SA_from_SP', ['PRACSAL', 'PRESWAT_L1', 'LON','LAT'])
    cons_temp_expr = PythonFunction('cons_temp', owner, 'CT_from_t', [abs_sal_expr, 'TEMPWAT_L1', 'PRESWAT_L1'])
    dens_expr = PythonFunction('DENSITY', owner, 'rho', [abs_sal_expr, cons_temp_expr, 'PRESWAT_L1'])
    ctxt = ParameterContext('DENSITY', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'kg m-3'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    # from coverage_model.test.test_parameter_functions import _create_all_params
    # contexts = _create_all_params()
    # pdict.add_context(contexts.pop('time'), is_temporal=True)  # Add time
    # map(pdict.add_context, contexts.values())  # Add others

    return pdict

MASTER_PDICT = _make_master_parameter_dict()

def _run_standard_tests(scov, timesteps):
    # A suite of standard tests to run against a SimplexCoverage
    results = []
    # Check basic metadata
    results.append(scov.name == 'sample coverage_model')
    results.append(scov.num_timesteps == timesteps)
    results.append(list(scov.temporal_domain.shape.extents) == [timesteps])
    params = scov.list_parameters()
    for param in params:
        pc = scov.get_parameter_context(param)
        results.append(len(pc.dom.identifier) == 36)

    return False not in results

def _insert_set_get(scov, timesteps, data, _slice, param='all'):
    # Function to test variable occurances of getting and setting values across parameter(s)
    data = data[_slice]
    ret = []

    scov.insert_timesteps(timesteps)
    param_list = []
    if param == 'all':
        param_list = scov.list_parameters()
    else:
        param_list.append(param)

    for param in param_list:
        scov.set_parameter_values(param, data, _slice)
        scov.get_dirty_values_async_result().get(timeout=60)
        # TODO: Is the res = assignment below correct?
        ret = scov.get_parameter_values(param, _slice)
    return (ret == data).all()

def _make_tcrs():
    # Construct temporal Coordinate Reference System object
    tcrs = CRS([AxisTypeEnum.TIME])
    return tcrs

def _make_scrs():
    # Construct spatial Coordinate Reference System object
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])
    return scrs

def _make_tdom(tcrs):
    # Construct temporal domain object
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    return tdom

def _make_sdom(scrs):
    # Create spatial domain object
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
    return sdom

# @attr('INT', group='jdc')
# def test_pdict_helper():
#     pdict = get_parameter_dict()
#     self.assertEqual(MASTER_PDICT.keys(), pdict.keys())
#
#     pname_filter = ['TIME','CONDUCTIVITY','TEMPWAT_L0']
#     pdict = get_parameter_dict(parameter_list=pname_filter)
#     self.assertIsInstance(pdict, ParameterDictionary)
#     self.assertEqual(pname_filter, pdict.keys())