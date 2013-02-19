
#!/usr/bin/env python

"""
@package coverage_model.test.test_parameter_functions
@file coverage_model/test/test_parameter_functions.py
@author Christopher Mueller
@brief UNIT tests for the parameter_functions module
"""

from nose.plugins.attrib import attr
from coverage_model import *
from coverage_model.parameter_functions import ParameterFunctionException
import numpy as np
from unittest import TestCase

def errfunc(val1, val2):
    raise StandardError()

def pyfunc(val1, val2):
    return val1 * val2

def _get_vals(name, slice_):
    if name == 'VALS':
        return np.array([1, 3, 5, 6, 23])[slice_]
    elif name == 'first':
        return np.array([1, 2, 3, 4, 5])[slice_]
    elif name == 'second':
        return np.array([1, 4, 6, 8, 10])[slice_]
    else:
        return np.zeros(5)[slice_]

@attr('UNIT',group='cov')
class TestParameterFunctionsUnit(TestCase):

    def test_numexpr_function(self):
        func = NumexprFunction('v*10', 'v*10', {'v': 'VALS'})

        ret = func.evaluate(_get_vals, slice(None))
        self.assertTrue(np.array_equal(ret, np.array([1, 3, 5, 6, 23]) * 10))

    def test_numexpr_function_slice(self):
        func = NumexprFunction('v*10', 'v*10', {'v': 'VALS'})

        ret = func.evaluate(_get_vals, slice(3, None))
        self.assertTrue(np.array_equal(ret, np.array([6, 23]) * 10))

    def test_nested_numexpr_function(self):
        func1 = NumexprFunction('v*10', 'v*10', {'v': 'VALS'})
        func2 = NumexprFunction('v*10', 'v*10', {'v': func1})

        ret = func2.evaluate(_get_vals, slice(None))
        self.assertTrue(np.array_equal(ret, np.array([100, 300, 500, 600, 2300])))

    def test_python_function(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'pyfunc', ['first', 'second'])

        ret = func.evaluate(_get_vals, slice(None))
        self.assertTrue(np.array_equal(ret, np.array([1*1, 2*4, 3*6, 4*8, 5*10])))

    def test_python_function_exception(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'errfunc', ['first', 'second'])

        self.assertRaises(StandardError, func.evaluate, _get_vals, slice(None))

    def test_python_function_slice(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'pyfunc', ['first','second'])

        ret = func.evaluate(_get_vals, slice(1, 4))
        self.assertTrue(np.array_equal(ret, np.array([2*4, 3*6, 4*8])))

    def test_nested_python_function(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func1 = PythonFunction('square', owner, 'pyfunc', ['first','first'])
        func2 = PythonFunction('quartic', owner, 'pyfunc', [func1, func1])

        ret = func2.evaluate(_get_vals, slice(None))
        self.assertTrue(np.array_equal(ret, np.array([1,  16,  81, 256, 625])))


@attr('INT',group='cov')
class TestParameterFunctionsInt(TestCase):

    def setUp(self):
        self.contexts = None
        self.value_classes = None

    def _get_param_vals(self, name, slice_):
        shp = utils.slice_shape(slice_, (10,))
        def _getarr(vmin, shp, vmax=None,):
            if vmax is None:
                return np.empty(shp).fill(vmin)
            return np.arange(vmin, vmax, (vmax - vmin) / int(utils.prod(shp)), dtype='float32').reshape(shp)

        if name == 'LAT':
            ret = np.empty(shp)
            ret.fill(45)
        elif name == 'LON':
            ret = np.empty(shp)
            ret.fill(-71)
        elif name == 'TEMPWAT_L0':
            ret = _getarr(280000, shp, 350000)
        elif name == 'CONDWAT_L0':
            ret = _getarr(100000, shp, 750000)
        elif name == 'PRESWAT_L0':
            ret = _getarr(3000, shp, 10000)
        elif name in self.value_classes: # Non-L0 parameters
            ret = self.value_classes[name][:]
        else:
            return np.zeros(shp)

        return ret

    def _ctxt_callback(self, context_name):
        return self.contexts[context_name]

    def test_L1_params(self):
        self.contexts = _get_pc_dict('TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        self.value_classes = {}

        dom_set = SimpleDomainSet((10,))

        # Add the callback for retrieving values
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pval_callback'):
                p._pval_callback = self._get_param_vals
                p._ctxt_callback = self._ctxt_callback
                self.value_classes[n] = get_value_class(p.param_type, dom_set)

        # Get the L1 data
        t1val = get_value_class(self.contexts['TEMPWAT_L1'].param_type, dom_set)
        c1val = get_value_class(self.contexts['CONDWAT_L1'].param_type, dom_set)
        p1val = get_value_class(self.contexts['PRESWAT_L1'].param_type, dom_set)

        # Perform assertions - involves "manual" calculation of values
        # Get the L0 data needed for validating output
        t0vals = self._get_param_vals('TEMPWAT_L0', slice(None))
        c0vals = self._get_param_vals('CONDWAT_L0', slice(None))
        p0vals = self._get_param_vals('PRESWAT_L0', slice(None))

        # TEMPWAT_L1 = (TEMPWAT_L0 / 10000) - 10
        t1 = (t0vals / 10000) - 10
        self.assertTrue(np.allclose(t1val[:], t1))

        # CONDWAT_L1 = (CONDWAT_L0 / 100000) - 0.5
        c1 = (c0vals / 100000) - 0.5
        self.assertTrue(np.allclose(c1val[:], c1))

        # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
        #   PRESWAT_L1 = (PRESWAT_L0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
        p1 = (p0vals * 679.34040721 / (0.85 * 65536)) - (0.05 * 679.34040721)
        self.assertTrue(np.allclose(p1val[:], p1))

    def test_L2_params(self):
        self.contexts = _get_pc_dict('TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1',
                                     'PRACSAL', 'DENSITY')

        self.value_classes = {}

        dom_set = SimpleDomainSet((10,))

        # Add the callback for retrieving values
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pval_callback'):
                p._pval_callback = self._get_param_vals
                p._ctxt_callback = self._ctxt_callback
                self.value_classes[n] = get_value_class(p.param_type, dom_set)

        # Get the L2 data
        psval = get_value_class(self.contexts['PRACSAL'].param_type, dom_set)
        rhoval = get_value_class(self.contexts['DENSITY'].param_type, dom_set)

        # Perform assertions - involves "manual" calculation of values
        # Get the L0 data needed for validating output
        latvals = self._get_param_vals('LAT', slice(None))
        lonvals = self._get_param_vals('LON', slice(None))

        # Get the L1 data needed for validating output
        t1val = get_value_class(self.contexts['TEMPWAT_L1'].param_type, dom_set)
        c1val = get_value_class(self.contexts['CONDWAT_L1'].param_type, dom_set)
        p1val = get_value_class(self.contexts['PRESWAT_L1'].param_type, dom_set)

        # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
        #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

        # PRACSAL = gsw.SP_from_C((CONDWAT_L1 * 10), TEMPWAT_L1, PRESWAT_L1)
        import gsw
        ps = gsw.SP_from_C((c1val[:] * 10.), t1val[:], p1val[:])
        self.assertTrue(np.allclose(psval[:], ps))

        # absolute_salinity = gsw.SA_from_SP(PRACSAL, PRESWAT_L1, longitude, latitude)
        # conservative_temperature = gsw.CT_from_t(absolute_salinity, TEMPWAT_L1, PRESWAT_L1)
        # DENSITY = gsw.rho(absolute_salinity, conservative_temperature, PRESWAT_L1)
        abs_sal = gsw.SA_from_SP(psval[:], p1val[:], lonvals, latvals)
        cons_temp = gsw.CT_from_t(abs_sal, t1val[:], p1val[:])
        rho = gsw.rho(abs_sal, cons_temp, p1val[:])
        self.assertTrue(np.allclose(rhoval[:], rho))

import networkx as nx
@attr('INT',group='cov')
class TestParameterValidatorInt(TestCase):

    def setUp(self):
        pass

    def test_simple_passthrough(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        in_contexts = in_values
        out_contexts = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L1_from_L0(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        in_contexts = in_values
        out_contexts = _get_pc_list('TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L2_from_L0(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        in_contexts = in_values
        out_contexts = _get_pc_list('DENSITY', 'PRACSAL', 'TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        out_values = ['DENSITY', 'PRACSAL']

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_Ln_from_L0(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        in_contexts = in_values
        out_contexts = _get_pc_list('DENSITY', 'PRACSAL', 'TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L2_from_L1(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        in_contexts = in_values
        out_contexts = _get_pc_list('DENSITY', 'PRACSAL')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L1_from_L0_fail_missing_L0(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON')
        in_contexts = in_values
        out_contexts = _get_pc_list('TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            self.assertRaises(ParameterFunctionException, pfv.validate, o)

    def test_L2_from_L0_fail_missing_L0(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON')
        in_contexts = in_values
        out_contexts = _get_pc_list('DENSITY', 'PRACSAL', 'TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')
        out_values = ['DENSITY', 'PRACSAL']

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            self.assertRaises(ParameterFunctionException, pfv.validate, o)

    def test_L2_from_L0_fail_missing_L1(self):
        in_values = _get_pc_list('TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0')
        in_contexts = in_values
        out_contexts = _get_pc_list('DENSITY', 'PRACSAL')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            self.assertRaises(ParameterFunctionException, pfv.validate, o)

def _get_pc_dict(*pnames):
    all_pc = _create_all_params()
    return {x: all_pc[x] for x in pnames}

def _get_pc_list(*pnames):
    all_pc = _create_all_params()
    return [all_pc[x] for x in pnames]

def _create_all_params():
    '''
     [
     'DENSITY',
     'TIME',
     'LON',
     'TEMPWAT_L1',
     'TEMPWAT_L0',
     'CONDWAT_L1',
     'CONDWAT_L0',
     'PRESWAT_L1',
     'PRESWAT_L0',
     'LAT',
     'PRACSAL'
     ]
    @return:
    '''

    contexts = {}

    t_ctxt = ParameterContext('TIME', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.uom = 'seconds since 01-01-1900'
    contexts['TIME'] = t_ctxt

    lat_ctxt = ParameterContext('LAT', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    lat_ctxt.axis = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    contexts['LAT'] = lat_ctxt

    lon_ctxt = ParameterContext('LON', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    lon_ctxt.axis = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    contexts['LON'] = lon_ctxt

    # Independent Parameters

    # Temperature - values expected to be the decimal results of conversion from hex
    temp_ctxt = ParameterContext('TEMPWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    temp_ctxt.uom = 'deg_C'
    contexts['TEMPWAT_L0'] = temp_ctxt

    # Conductivity - values expected to be the decimal results of conversion from hex
    cond_ctxt = ParameterContext('CONDWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    cond_ctxt.uom = 'S m-1'
    contexts['CONDWAT_L0'] = cond_ctxt

    # Pressure - values expected to be the decimal results of conversion from hex
    press_ctxt = ParameterContext('PRESWAT_L0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    press_ctxt.uom = 'dbar'
    contexts['PRESWAT_L0'] = press_ctxt


    # Dependent Parameters

    # TEMPWAT_L1 = (TEMPWAT_L0 / 10000) - 10
    tl1_func = '(T_L0 / 10000) - 10'
    tl1_pmap = {'T_L0':'TEMPWAT_L0'}
    expr = NumexprFunction('TEMPWAT_L1', tl1_func, tl1_pmap)
    tempL1_ctxt = ParameterContext('TEMPWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    tempL1_ctxt.uom = 'deg_C'
    contexts['TEMPWAT_L1'] = tempL1_ctxt

    # CONDWAT_L1 = (CONDWAT_L0 / 100000) - 0.5
    cl1_func = '(C_L0 / 100000) - 0.5'
    cl1_pmap = {'C_L0':'CONDWAT_L0'}
    expr = NumexprFunction('CONDWAT_L1', cl1_func, cl1_pmap)
    condL1_ctxt = ParameterContext('CONDWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    condL1_ctxt.uom = 'S m-1'
    contexts['CONDWAT_L1'] = condL1_ctxt

    # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
    #   PRESWAT_L1 = (PRESWAT_L0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
    pl1_func = '(P_L0 * 679.34040721 / (0.85 * 65536)) - (0.05 * 679.34040721)'
    pl1_pmap = {'P_L0':'PRESWAT_L0'}
    expr = NumexprFunction('PRESWAT_L1', pl1_func, pl1_pmap)
    presL1_ctxt = ParameterContext('PRESWAT_L1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    presL1_ctxt.uom = 'S m-1'
    contexts['PRESWAT_L1'] = presL1_ctxt

    # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
    #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

    # PRACSAL = gsw.SP_from_C((CONDWAT_L1 * 10), TEMPWAT_L1, PRESWAT_L1)
    owner = 'gsw'
    sal_func = 'SP_from_C'
    sal_arglist = [NumexprFunction('CONDWAT_L1*10', 'C*10', {'C':'CONDWAT_L1'}), 'TEMPWAT_L1', 'PRESWAT_L1']
    sal_kwargmap = None
    expr = PythonFunction('PRACSAL', owner, sal_func, sal_arglist, sal_kwargmap)
    sal_ctxt = ParameterContext('PRACSAL', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)
    sal_ctxt.uom = 'g kg-1'
    contexts['PRACSAL'] = sal_ctxt

    # absolute_salinity = gsw.SA_from_SP(PRACSAL, PRESWAT_L1, longitude, latitude)
    # conservative_temperature = gsw.CT_from_t(absolute_salinity, TEMPWAT_L1, PRESWAT_L1)
    # DENSITY = gsw.rho(absolute_salinity, conservative_temperature, PRESWAT_L1)
    owner = 'gsw'
    abs_sal_expr = PythonFunction('abs_sal', owner, 'SA_from_SP', ['PRACSAL', 'PRESWAT_L1', 'LON','LAT'], None)
    cons_temp_expr = PythonFunction('cons_temp', owner, 'CT_from_t', [abs_sal_expr, 'TEMPWAT_L1', 'PRESWAT_L1'], None)
    dens_expr = PythonFunction('DENSITY', owner, 'rho', [abs_sal_expr, cons_temp_expr, 'PRESWAT_L1'], None)
    dens_ctxt = ParameterContext('DENSITY', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
    dens_ctxt.uom = 'kg m-3'
    contexts['DENSITY'] = dens_ctxt

    return contexts



