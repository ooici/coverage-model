
#!/usr/bin/env python

"""
@package coverage_model.test.test_parameter_functions
@file coverage_model/test/test_parameter_functions.py
@author Christopher Mueller
@brief UNIT tests for the parameter_functions module
"""

from nose.plugins.attrib import attr
from coverage_model import *
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


def _get_param_vals(name, slice_):
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
    else:
        ret = np.zeros(shp)

    return ret

@attr('INT',group='cov')
class TestParameterFunctionsInt(TestCase):

    def setUp(self):
        self.contexts = None

    def _get_pc_dict(self, *pnames):
        all_pc = self._create_all_params()
        return {x: all_pc[x] for x in pnames}

    def _get_pc_list(self, *pnames):
        all_pc = self._create_all_params()
        return [all_pc[x] for x in pnames]

    def _ctxt_callback(self, context_name):
        return self.contexts[context_name]

    def test_L1_params(self):
        self.contexts = self._get_pc_dict(
            'TIME', 'LAT', 'LON', 'TEMPWAT_L0', 'CONDWAT_L0', 'PRESWAT_L0',
            'TEMPWAT_L1', 'CONDWAT_L1', 'PRESWAT_L1')

        # Add the callback for retrieving values
        for p in self.contexts.values():
            if hasattr(p, '_pval_callback'):
                p._pval_callback = _get_param_vals
                p._ctxt_callback = self._ctxt_callback

        dom_set = SimpleDomainSet((10,))

        # Not really necessary....only needed for validating output
        # latval = get_value_class(self.contexts['LAT'].param_type, dom_set)
        # lonval = get_value_class(self.contexts['LON'].param_type, dom_set)
        # t0val = get_value_class(self.contexts['TEMPWAT_L0'].param_type, dom_set)
        # c0val = get_value_class(self.contexts['CONDWAT_L0'].param_type, dom_set)
        # p0val = get_value_class(self.contexts['PRESWAT_L0'].param_type, dom_set)
        # latval[:] = _get_param_vals('LAT', slice(None))
        # lonval[:] = _get_param_vals('LON', slice(None))
        # t0val[:] = _get_param_vals('TEMPWAT_L0', slice(None))
        # c0val[:] = _get_param_vals('CONDWAT_L0', slice(None))
        # p0val[:] = _get_param_vals('PRESWAT_L0', slice(None))

        t1val = get_value_class(self.contexts['TEMPWAT_L1'].param_type, dom_set)
        c1val = get_value_class(self.contexts['CONDWAT_L1'].param_type, dom_set)
        p1val = get_value_class(self.contexts['PRESWAT_L1'].param_type, dom_set)

        self.assertTrue(
            np.allclose(t1val[:], np.array(
                [18.0, 18.70000076, 19.39999962, 20.10000038, 20.79999924,
                 21.5, 22.20000076, 22.90000153, 23.59999847, 24.29999924]
            ))
        )
        self.assertTrue(
            np.allclose(c1val[:], np.array(
                [0.5, 1.14999998, 1.79999995, 2.45000005, 3.0999999,
                 3.75, 4.4000001, 5.05000019, 5.69999981, 6.3499999]
            ))
        )
        self.assertTrue(
            np.allclose(p1val[:], np.array(
                [2.61855132, 11.15518471, 19.6918181,  28.22845149, 36.76508488,
                 45.30171827, 53.83835167, 62.37498506, 70.91161845, 79.44825184]
            ))
        )

    def _create_all_params(self):
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
