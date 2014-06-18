
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

def errfunc(val1, val2):
    raise StandardError()

def pyfunc(val1, val2):
    return val1 * val2

def _get_vals(name, time_segment=(None,None), stride_length=None):
    if name == 'VALS':
        return np.array([1, 3, 5, 6, 23])[time_segment[0]:time_segment[1]:stride_length]
    elif name == 'first':
        return np.array([1, 2, 3, 4, 5])[time_segment[0]:time_segment[1]:stride_length]
    elif name == 'second':
        return np.array([1, 4, 6, 8, 10])[time_segment[0]:time_segment[1]:stride_length]
    else:
        return np.zeros(5)[time_segment[0]:time_segment[1]:stride_length]

def callback_arg_func(pv_callback):
    a = pv_callback('tempwat_l0')
    b = pv_callback('preswat_l0')
    return a + b

@attr('UNIT',group='cov')
class TestParameterFunctionsUnit(CoverageModelUnitTestCase):

    def test_numexpr_function(self):
        func = NumexprFunction('v*10', 'v*10', ['v'], {'v': 'VALS'})

        ret = func.evaluate(_get_vals, (None,None))
        np.testing.assert_array_equal(ret, np.array([1, 3, 5, 6, 23]) * 10)

    def test_numexpr_function_slice(self):
        func = NumexprFunction('v*10', 'v*10', ['v'], {'v': 'VALS'})

        ret = func.evaluate(_get_vals, (3, None))
        np.testing.assert_array_equal(ret, np.array([6, 23]) * 10)

    def test_nested_numexpr_function(self):
        func1 = NumexprFunction('v*10', 'v*10', ['v'], {'v': 'VALS'})
        func2 = NumexprFunction('v*10', 'v*10', ['v'], {'v': func1})

        ret = func2.evaluate(_get_vals, time_segment=(None,None))
        np.testing.assert_array_equal(ret, np.array([100, 300, 500, 600, 2300]))

    def test_numexpr_function_splat(self):
        func = NumexprFunction('v*a', 'v*a', ['v', 'a*'], {'v': 'VALS', 'a*': 'VALS'})

        ret = func.evaluate(_get_vals, (None,None))
        np.testing.assert_array_equal(ret, np.array([1*23, 3*23, 5*23, 6*23, 23*23]))

    def test_python_function(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'pyfunc', ['first', 'second'])

        ret = func.evaluate(_get_vals, (None,None))
        np.testing.assert_array_equal(ret, np.array([1*1, 2*4, 3*6, 4*8, 5*10]))

    def test_python_function_splat(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'pyfunc', ['first', 'second*'], param_map={'first': 'first', 'second*': 'second'})

        ret = func.evaluate(_get_vals, (None,None))
        np.testing.assert_array_equal(ret, np.array([1*10, 2*10, 3*10, 4*10, 5*10]))

    def test_python_function_exception(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'errfunc', ['first', 'second'])

        self.assertRaises(StandardError, func.evaluate, _get_vals, slice(None))

    def test_python_function_slice(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func = PythonFunction('multiplier', owner, 'pyfunc', ['first', 'second'])

        ret = func.evaluate(_get_vals, (1, 4))
        np.testing.assert_array_equal(ret, np.array([2*4, 3*6, 4*8]))

    def test_nested_python_function(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func1 = PythonFunction('square', owner, 'pyfunc', ['first', 'first'])
        func2 = PythonFunction('quartic', owner, 'pyfunc', [func1, func1])

        ret = func2.evaluate(_get_vals, (None,None))
        np.testing.assert_array_equal(ret, np.array([1,  16,  81, 256, 625]))

    def get_module_dependencies(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func1 = PythonFunction('square', owner, 'pyfunc', ['first', 'first'])
        func2 = PythonFunction('quartic', owner, 'pyfunc', [func1, func1])

        deps = func2.get_module_dependencies()
        self.assertEqual(({'coverage_model.test.test_parameter_functions'}), deps)

        func = NumexprFunction('v*10', 'v*10', ['v'], {'v': func1})
        deps = func.get_module_dependencies()
        self.assertEqual(({'numexpr', 'coverage_model.test.test_parameter_functions'}), deps)

    def test_get_function_map(self):
        owner = 'coverage_model.test.test_parameter_functions'
        func1 = PythonFunction('square', owner, 'pyfunc', ['first', 'first'])
        func = NumexprFunction('v*10', 'v*10', ['v'], {'v': func1})

        efm = {'[v*10]': {'arg_0': {'[v :|: square]': {'arg_0': '!first :|: first!', 'arg_1': '!first :|: first!'}}}}
        self.assertEqual(func.get_function_map(), efm)

        efm = {'[square]': {'arg_0': '!first :|: first!', 'arg_1': '!first :|: first!'}}
        self.assertEqual(func1.get_function_map(), efm)

@attr('INT',group='cov')
class TestParameterFunctionsInt(CoverageModelIntTestCase):

    def setUp(self):
        self.contexts = None
        self.value_classes = None

    def _get_param_vals(self, name, time_segment, stride_length=None):
        shp = utils.slice_shape(time_segment, (10,))

        def _getarr(vmin, shp, vmax=None,):
            if vmax is None:
                ret = np.empty(shp)
                ret.fill(vmin)
                return ret
            sp = np.prod(shp)
            while vmax-vmin < sp:
                vmax *= 2
            return np.arange(vmin, vmax, dtype='float32')[:10]

        if name == 'lat':
            ret = _getarr(45, shp)
        elif name == 'lon':
            ret = _getarr(-71, shp)
        elif name == 'tempwat_l0':
            ret = _getarr(280000, shp, 350000)
        elif name == 'condwat_l0':
            ret = _getarr(100000, shp, 750000)
        elif name == 'preswat_l0':
            ret = _getarr(3000, shp, 10000)
        elif name in self.value_classes: # Non-L0 parameters
            ret = self.value_classes[name][:]
        else:
            return np.zeros(shp)

        return ret[time_segment]

    def _ctxt_callback(self, context_name):
        return self.contexts[context_name]

    def test_get_function_map(self):
        self.contexts = _get_pc_dict('tempwat_l0', 'condwat_l0', 'preswat_l0',
                                     'tempwat_l1', 'condwat_l1', 'preswat_l1',
                                     'pracsal')

        # Add the callback for retrieving contexts
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pctxt_callback'):
                p._pctxt_callback = self._ctxt_callback

        fmap = {'<tempwat_l0>': None}
        self.assertEqual(fmap, self.contexts['tempwat_l0'].param_type.get_function_map())

        fmap = {'tempwat_l1': {'arg_0': {'<tempwat_l0>': None}}}
        self.assertEqual(fmap, self.contexts['tempwat_l1'].param_type.get_function_map())

        fmap = {'pracsal': {'arg_0': {'[C :|: condwat_l1*10]': {'arg_0': {'C :|: condwat_l1': {'arg_0': {'<condwat_l0>': None}}}}},
                            'arg_1': {'t :|: tempwat_l1': {'arg_0': {'<tempwat_l0>': None}}},
                            'arg_2': {'p :|: preswat_l1': {'arg_0': {'<preswat_l0>': None}, 'arg_1': '<p_range :|: 679.34040721>'}}}}
        self.assertEqual(fmap, self.contexts['pracsal'].param_type.get_function_map())

    def test_get_independent_parameters(self):
        self.contexts = _get_pc_dict('tempwat_l0', 'condwat_l0', 'preswat_l0',
                                     'tempwat_l1', 'condwat_l1', 'preswat_l1',
                                     'pracsal')

        # Add the callback for retrieving contexts
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pctxt_callback'):
                p._pctxt_callback = self._ctxt_callback

        ips = ('tempwat_l0',)
        self.assertEqual(ips, self.contexts['tempwat_l0'].param_type.get_independent_parameters())

        ips = ('tempwat_l0',)
        self.assertEqual(ips, self.contexts['tempwat_l1'].param_type.get_independent_parameters())

        ips = ('condwat_l0', '679.34040721', 'preswat_l0', 'tempwat_l0')
        self.assertEqual(ips, self.contexts['pracsal'].param_type.get_independent_parameters())

    def test_get_dependent_parameters(self):
        self.contexts = _get_pc_dict('tempwat_l0', 'condwat_l0', 'preswat_l0',
                                     'tempwat_l1', 'condwat_l1', 'preswat_l1',
                                     'pracsal')

        # Add the callback for retrieving contexts
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pctxt_callback'):
                p._pctxt_callback = self._ctxt_callback

        dps = ()
        self.assertEqual(dps, self.contexts['tempwat_l0'].param_type.get_dependent_parameters())

        dps = ('tempwat_l1',)
        self.assertEqual(dps, self.contexts['tempwat_l1'].param_type.get_dependent_parameters())

        dps = ('condwat_l1', 'preswat_l1', 'pracsal', 'tempwat_l1')
        self.assertEqual(dps, self.contexts['pracsal'].param_type.get_dependent_parameters())

    def test_get_module_dependencies(self):
        self.contexts = _get_pc_dict('tempwat_l0', 'condwat_l0', 'preswat_l0',
                                     'tempwat_l1', 'condwat_l1', 'preswat_l1',
                                     'pracsal')

        # Add the callback for retrieving contexts
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pctxt_callback'):
                p._pctxt_callback = self._ctxt_callback

        self.assertEqual((), self.contexts['tempwat_l0'].get_module_dependencies())

        self.assertEqual(('numexpr',), self.contexts['tempwat_l1'].get_module_dependencies())

        self.assertEqual(('numexpr', 'gsw'), self.contexts['pracsal'].get_module_dependencies())

    def test_L1_params(self):
        self.contexts = _get_pc_dict('tempwat_l1', 'condwat_l1', 'preswat_l1')
        self.value_classes = {}

        dom_set = SimpleDomainSet((10,))

        # Add the callback for retrieving values
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pval_callback'):
                p.param_type.callback = self._get_param_vals
                p._ctxt_callback = self._ctxt_callback
                self.value_classes[n] = get_value_class(p.param_type, dom_set)

        # Get the L1 data
        t1val = get_value_class(self.contexts['tempwat_l1'].param_type, dom_set)
        c1val = get_value_class(self.contexts['condwat_l1'].param_type, dom_set)
        p1val = get_value_class(self.contexts['preswat_l1'].param_type, dom_set)

        # Perform assertions - involves "manual" calculation of values
        # Get the L0 data needed for validating output
        t0vals = self._get_param_vals('tempwat_l0', slice(None))
        c0vals = self._get_param_vals('condwat_l0', slice(None))
        p0vals = self._get_param_vals('preswat_l0', slice(None))

        # tempwat_l1 = (tempwat_l0 / 10000) - 10
        t1 = (t0vals / 10000) - 10
        np.testing.assert_allclose(t1val[:], t1, rtol=1e-05)

        # condwat_l1 = (condwat_l0 / 100000) - 0.5
        c1 = (c0vals / 100000) - 0.5
        np.testing.assert_allclose(c1val[:], c1, rtol=1e-05)

        # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
        #   preswat_l1 = (preswat_l0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
        p1 = (p0vals * 679.34040721 / (0.85 * 65536)) - (0.05 * 679.34040721)
        np.testing.assert_allclose(p1val[:], p1, rtol=1e-05)

    def test_L2_params(self):
        self.contexts = _get_pc_dict('tempwat_l1', 'condwat_l1', 'preswat_l1',
                                     'pracsal', 'density')

        self.value_classes = {}

        dom_set = SimpleDomainSet((10,))

        # Add the callback for retrieving values
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pval_callback'):
                p.param_type.callback = self._get_param_vals
                p._ctxt_callback = self._ctxt_callback
                self.value_classes[n] = get_value_class(p.param_type, dom_set)

        # Get the L2 data
        psval = get_value_class(self.contexts['pracsal'].param_type, dom_set)
        rhoval = get_value_class(self.contexts['density'].param_type, dom_set)

        # Perform assertions - involves "manual" calculation of values
        # Get the L0 data needed for validating output
        latvals = self._get_param_vals('lat', slice(None))
        lonvals = self._get_param_vals('lon', slice(None))

        # Get the L1 data needed for validating output
        t1val = get_value_class(self.contexts['tempwat_l1'].param_type, dom_set)
        c1val = get_value_class(self.contexts['condwat_l1'].param_type, dom_set)
        p1val = get_value_class(self.contexts['preswat_l1'].param_type, dom_set)

        # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
        #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

        # pracsal = gsw.SP_from_C((condwat_l1 * 10), tempwat_l1, preswat_l1)
        import gsw
        ps = gsw.SP_from_C((c1val[:] * 10.), t1val[:], p1val[:])
        np.testing.assert_allclose(psval[:], ps)

        # absolute_salinity = gsw.SA_from_SP(pracsal, preswat_l1, longitude, latitude)
        # conservative_temperature = gsw.CT_from_t(absolute_salinity, tempwat_l1, preswat_l1)
        # density = gsw.rho(absolute_salinity, conservative_temperature, preswat_l1)
        abs_sal = gsw.SA_from_SP(psval[:], p1val[:], lonvals, latvals)
        cons_temp = gsw.CT_from_t(abs_sal, t1val[:], p1val[:])
        rho = gsw.rho(abs_sal, cons_temp, p1val[:])
        np.testing.assert_allclose(rhoval[:], rho)

    def test_pv_callback_argument(self):
        self.contexts = _get_pc_dict('pv_callback')

        self.value_classes = {}

        dom_set = SimpleDomainSet((10,))

        # Add the callback for retrieving values
        for n, p in self.contexts.iteritems():
            if hasattr(p, '_pval_callback'):
                p.param_type.callback = self._get_param_vals
                p._ctxt_callback = self._ctxt_callback
                # self.value_classes[n] = get_value_class(p.param_type, dom_set)

        pvcb_val = get_value_class(self.contexts['pv_callback'].param_type, dom_set)

        expect = np.array([283000.,  283002.,  283004.,  283006.,  283008.,  283010., 283012.,  283014.,  283016.,  283018.])
        np.testing.assert_array_equal(pvcb_val[:], expect)
        np.testing.assert_array_equal(pvcb_val[2::2], expect[2::2])
        np.testing.assert_array_equal(pvcb_val[:-3:4], expect[:-3:4])


import networkx as nx
@attr('INT',group='cov')
class TestParameterValidatorInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def test_simple_passthrough(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        in_contexts = in_values
        out_contexts = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L1_from_L0(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        in_contexts = in_values
        out_contexts = _get_pc_list('tempwat_l1', 'condwat_l1', 'preswat_l1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L2_from_L0(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        in_contexts = in_values
        out_contexts = _get_pc_list('density', 'pracsal', 'tempwat_l1', 'condwat_l1', 'preswat_l1')
        out_values = ['density', 'pracsal']

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_Ln_from_L0(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        in_contexts = in_values
        out_contexts = _get_pc_list('density', 'pracsal', 'tempwat_l1', 'condwat_l1', 'preswat_l1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L2_from_L1(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l1', 'condwat_l1', 'preswat_l1')
        in_contexts = in_values
        out_contexts = _get_pc_list('density', 'pracsal')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            g = pfv.validate(o)
            self.assertIsInstance(g, nx.DiGraph)

    def test_L1_from_L0_fail_missing_L0(self):
        in_values = _get_pc_list('time', 'lat', 'lon')
        in_contexts = in_values
        out_contexts = _get_pc_list('tempwat_l1', 'condwat_l1', 'preswat_l1')
        out_values = [p.name for p in out_contexts]

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            self.assertRaises(ParameterFunctionException, pfv.validate, o)

    def test_L2_from_L0_fail_missing_L0(self):
        in_values = _get_pc_list('time', 'lat', 'lon')
        in_contexts = in_values
        out_contexts = _get_pc_list('density', 'pracsal', 'tempwat_l1', 'condwat_l1', 'preswat_l1')
        out_values = ['density', 'pracsal']

        pfv = ParameterFunctionValidator(in_values, in_contexts, out_contexts)

        for o in out_values:
            self.assertRaises(ParameterFunctionException, pfv.validate, o)

    def test_L2_from_L0_fail_missing_L1(self):
        in_values = _get_pc_list('time', 'lat', 'lon', 'tempwat_l0', 'condwat_l0', 'preswat_l0')
        in_contexts = in_values
        out_contexts = _get_pc_list('density', 'pracsal')
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
     'density',
     'time',
     'lon',
     'tempwat_l1',
     'tempwat_l0',
     'condwat_l1',
     'condwat_l0',
     'preswat_l1',
     'preswat_l0',
     'lat',
     'pracsal'
     ]
    @return:
    '''

    contexts = {}

    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1900'
    contexts['time'] = t_ctxt

    lat_ctxt = ParameterContext('lat', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    lat_ctxt.axis = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    contexts['lat'] = lat_ctxt

    lon_ctxt = ParameterContext('lon', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    lon_ctxt.axis = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    contexts['lon'] = lon_ctxt

    # Independent Parameters

    # Temperature - values expected to be the decimal results of conversion from hex
    temp_ctxt = ParameterContext('tempwat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    temp_ctxt.uom = 'deg_C'
    contexts['tempwat_l0'] = temp_ctxt

    # Conductivity - values expected to be the decimal results of conversion from hex
    cond_ctxt = ParameterContext('condwat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    cond_ctxt.uom = 'S m-1'
    contexts['condwat_l0'] = cond_ctxt

    # Pressure - values expected to be the decimal results of conversion from hex
    press_ctxt = ParameterContext('preswat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    press_ctxt.uom = 'dbar'
    contexts['preswat_l0'] = press_ctxt


    # Dependent Parameters

    # tempwat_l1 = (tempwat_l0 / 10000) - 10
    tl1_func = '(T / 10000) - 10'
    tl1_pmap = {'T': 'tempwat_l0'}
    expr = NumexprFunction('tempwat_l1', tl1_func, ['T'], param_map=tl1_pmap)
    tempL1_ctxt = ParameterContext('tempwat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    tempL1_ctxt.uom = 'deg_C'
    contexts['tempwat_l1'] = tempL1_ctxt

    # condwat_l1 = (condwat_l0 / 100000) - 0.5
    cl1_func = '(C / 100000) - 0.5'
    cl1_pmap = {'C': 'condwat_l0'}
    expr = NumexprFunction('condwat_l1', cl1_func, ['C'], param_map=cl1_pmap)
    condL1_ctxt = ParameterContext('condwat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    condL1_ctxt.uom = 'S m-1'
    contexts['condwat_l1'] = condL1_ctxt

    # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
    #   preswat_l1 = (preswat_l0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
    pl1_func = '(P * p_range / (0.85 * 65536)) - (0.05 * p_range)'
    pl1_pmap = {'P': 'preswat_l0', 'p_range': 679.34040721}
    expr = NumexprFunction('preswat_l1', pl1_func, ['P', 'p_range'], param_map=pl1_pmap)
    presL1_ctxt = ParameterContext('preswat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    presL1_ctxt.uom = 'S m-1'
    contexts['preswat_l1'] = presL1_ctxt

    # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
    #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

    # pracsal = gsw.SP_from_C((condwat_l1 * 10), tempwat_l1, preswat_l1)
    owner = 'gsw'
    sal_func = 'SP_from_C'
    sal_arglist = ['C', 't', 'p']
    sal_pmap = {'C': NumexprFunction('condwat_l1*10', 'C*10', ['C'], param_map={'C': 'condwat_l1'}), 't': 'tempwat_l1', 'p': 'preswat_l1'}
    sal_kwargmap = None
    expr = PythonFunction('pracsal', owner, sal_func, sal_arglist, sal_kwargmap, sal_pmap)
    sal_ctxt = ParameterContext('pracsal', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)
    sal_ctxt.uom = 'g kg-1'
    contexts['pracsal'] = sal_ctxt

    # absolute_salinity = gsw.SA_from_SP(pracsal, preswat_l1, longitude, latitude)
    # conservative_temperature = gsw.CT_from_t(absolute_salinity, tempwat_l1, preswat_l1)
    # density = gsw.rho(absolute_salinity, conservative_temperature, preswat_l1)
    owner = 'gsw'
    abs_sal_expr = PythonFunction('abs_sal', owner, 'SA_from_SP', ['pracsal', 'preswat_l1', 'lon','lat'])
    cons_temp_expr = PythonFunction('cons_temp', owner, 'CT_from_t', [abs_sal_expr, 'tempwat_l1', 'preswat_l1'])
    dens_expr = PythonFunction('density', owner, 'rho', [abs_sal_expr, cons_temp_expr, 'preswat_l1'])
    dens_ctxt = ParameterContext('density', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
    dens_ctxt.uom = 'kg m-3'
    contexts['density'] = dens_ctxt

    owner = 'coverage_model.test.test_parameter_functions'
    pvcb_func = PythonFunction('pv_callback', owner, 'callback_arg_func', ['pv_callback'])
    pvcb_ctxt = ParameterContext('pv_callback', param_type=ParameterFunctionType(pvcb_func), variability=VariabilityEnum.TEMPORAL)
    contexts['pv_callback'] = pvcb_ctxt

    return contexts



