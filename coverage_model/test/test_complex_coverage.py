#!/usr/bin/env python

"""
@package coverage_model.test.test_complex_coverage
@file coverage_model/test/test_R2_complex_coverage.py
@author Christopher Mueller
@brief Unit & Integration tests for NewComplexCoverage
"""

from ooi.logging import log
from copy import deepcopy
import mock
from nose.plugins.attrib import attr
import numpy as np
import os
import random
import time
import unittest
from coverage_model import *
from coverage_model.coverages.complex_coverage import ComplexCoverage
from coverage_model.coverages.coverage_extents import ReferenceCoverageExtents, ExtentsDict
from coverage_model.parameter_functions import ExternalFunction
from coverage_model.hdf_utils import HDFLockingFile
from coverage_test_base import CoverageIntTestBase, get_props


def _make_param_dict(params, make_temporal=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    if make_temporal:
        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('float32')))
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt, is_temporal=True)

    for p in params:
        if isinstance(p, ParameterContext):
            pdict.add_context(p)
        elif isinstance(params, tuple):
            pdict.add_context(ParameterContext(p[0], param_type=QuantityType(value_encoding=np.dtype(p[1]))))
        else:
            pdict.add_context(ParameterContext(p, param_type=QuantityType(value_encoding=np.dtype('float32'))))

    return pdict


def _make_cov(root_dir, params, nt=10, data_dict=None, make_temporal=True):
    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    if isinstance(params, ParameterDictionary):
        pdict = params
    else:
        # Instantiate a ParameterDictionary
        pdict = _make_param_dict(params, make_temporal)

    scov = SimplexCoverage(root_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)

    if nt == 0 and data_dict is None:
        pass
    else:
        p_dict = {}
        for p in scov.list_parameters():
            if p == scov.ingest_time_parameter_name:
                continue
            if data_dict is not None and p in data_dict:
                if data_dict[p] is None:
                    continue
                dat = data_dict[p]
            else:
                dat = range(nt)

            try:
                p_dict[p] = np.array(dat)
            except Exception as ex:
                import sys
                raise Exception('Error setting values for {0}: {1}'.format(p, data_dict[p])), None, sys.exc_traceback
        scov.set_parameter_values(make_parameter_data_dict(p_dict))

    scov.close()

    return os.path.realpath(scov.persistence_dir)

class CoverageEnvironment(CoverageModelIntTestCase, CoverageIntTestBase):
    @attr('UTIL', group='cov')
    def test_cov_params(self):
        contexts = create_all_params()
        del contexts['time']
        del contexts['density']
        contexts = contexts.values()
        cova_pth = _make_cov(self.working_dir, contexts, nt=2)
        cov = SimplexCoverage.load(cova_pth, mode='r+')
        cov._range_value['time'][:] = [1, 2]
        cov._range_value['temperature'][:] = [205378, 289972]
        cov._range_value['conductivity'][:] = [410913, 417588]
        cov._range_value['pressure'][:] = [3939, 13616]
        cov._range_value['p_range'][:] = 1000.
        cov._range_value['lat'][:] = 40.
        cov._range_value['lon'][:] = -70.


        # Make a new function
        owner = 'ion_functions.data.ctd_functions'
        dens_func = 'ctd_density'
        dens_arglist = ['SP', 't', 'p', 'lat', 'lon']
        dens_pmap = {'SP':'pracsal', 't':'seawater_temperature', 'p':'seawater_pressure', 'lat':'lat', 'lon':'lon'}
        dens_expr = PythonFunction('density', owner, dens_func, dens_arglist, None, dens_pmap)
        dens_ctxt = ParameterContext('density', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
        dens_ctxt.uom = 'kg m-3'

        cov.append_parameter(dens_ctxt)

        # Make sure it worked
        np.testing.assert_array_equal(cov.get_parameter_values('density').get_data()['density'],
                np.array([ 1024.98205566,  1019.4932251 ], dtype=np.float32))

    @attr('UTIL', group='cov')
    def test_something(self):

        # Create a large dataset spanning a year
        # Each coverage represents a week
    

        cova_pth = _make_cov(self.working_dir, ['value_set'], nt=1000, data_dict={'time': np.arange(1000,2000),'value_set':np.arange(1000)})
        cov = AbstractCoverage.load(cova_pth)

        results = self.simple_search(cov, 1212, 1390)
        np.testing.assert_array_equal(results['time'], np.arange(1212, 1391))
        np.testing.assert_array_equal(results['value_set'], np.arange(212, 391))

        from pyon.util.breakpoint import breakpoint
        breakpoint(locals(), globals())

    def simple_search(self, coverage, start, stop):
        from coverage_model.search.search_parameter import ParamValueRange, ParamValue, SearchCriteria
        from coverage_model.search.coverage_search import CoverageSearch
        from coverage_model.search.search_constants import IndexedParameters
        pdir, guid = os.path.split(coverage.persistence_dir)
        time_param = ParamValueRange(IndexedParameters.Time, (start, stop))
        criteria = SearchCriteria(time_param)
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        cov = results.get_view_coverage(guid, pdir)
        retval = cov.get_observations()
        return retval

    @attr('UTIL', group='cov')
    def test_aggregates(self):

        array_stuff = ParameterContext('array_stuff', param_type=ArrayType(inner_encoding='float32'))
        x, y = np.mgrid[0:10, 0:10]

        cova_pth = _make_cov(self.working_dir, ['value_set', array_stuff], data_dict={'time': np.arange(10),'value_set' : np.ones(10), 'array_stuff' : x})
        covb_pth = _make_cov(self.working_dir, ['value_set', array_stuff], data_dict={'time': np.arange(20,30), 'value_set': np.ones(10) * 2, 'array_stuff' : y})
        covc_pth = _make_cov(self.working_dir, ['value_set', array_stuff], data_dict={'time': np.arange(15,25), 'value_set' : np.ones(10) * 3, 'array_stuff' : x})

        cov = SimplexCoverage.load(cova_pth, mode='r+')

        cov_pths = [cova_pth, covb_pth]


        ccov = ComplexCoverage(self.working_dir, create_guid(), 'complex coverage',
                reference_coverage_locs=[covb_pth],
                reference_coverage_extents=TestNewComplexCoverageInt.get_extents([covb_pth]),
                parameter_dictionary=ParameterDictionary(),
                complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        cova = AbstractCoverage.load(cova_pth)
        covb = AbstractCoverage.load(covb_pth)

        ccov.append_reference_coverage(cova_pth, extents=ReferenceCoverageExtents('a', cova.persistence_guid, time_extents=(None,None)))
        ccov.append_reference_coverage(covb_pth, extents=ReferenceCoverageExtents('b', covb.persistence_guid, time_extents=(None,None)))

        # TODO: correct this once ViewCoverage is worked out
        # View coverage construction doesn't work for DB-based metadata.  View Coverage will be modified in the future
        # vcov = ViewCoverage(self.working_dir, create_guid(), 'view coverage', reference_coverage_location = ccov.persistence_dir)

        

@attr('INT',group='cov')
class TestComplexCoverageInt(CoverageModelIntTestCase, CoverageIntTestBase):

    # Make a deep copy of the base TESTING_PROPERTIES dict and then modify for this class
    TESTING_PROPERTIES = deepcopy(CoverageIntTestBase.TESTING_PROPERTIES)
    TESTING_PROPERTIES['test_props_decorator'] = {'test_props': 10}
    TESTING_PROPERTIES['test_get_time_data_metrics'] = {'time_data_size': 0.01907348}

    @get_props()
    def test_props_decorator(self):
        props = self.test_props_decorator.props
        self.assertIsInstance(props, dict)
        expected = {'time_steps': 30, 'test_props': 10, 'brick_size': 1000}
        self.assertEqual(props, expected)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def get_extents(cls, rcov_locs):
        extents = {}
        i = 0
        for cov_pth in rcov_locs:
            i += 1
            cov = AbstractCoverage.load(cov_pth)
            cov_id = cov.persistence_guid
            extents[cov_id] = ReferenceCoverageExtents(str(i), cov_id, cov.get_data_bounds('time'))
        return extents

    @classmethod
    def get_no_extents(cls, rcov_locs):
        extents = {}
        i = 0
        for cov_pth in rcov_locs:
            i += 1
            cov = AbstractCoverage.load(cov_pth)
            cov_id = cov.persistence_guid
            extents[cov_id] = ReferenceCoverageExtents(str(i), cov_id, time_extents=None)
        return extents

    @classmethod
    def get_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=30, auto_flush_values=True):
        # Many tests assume nt is the 'total' number of timesteps, must divide between the 3 coverages
        sz1 = sz2 = sz3 = int(nt) / 3
        sz3 += nt - sum([sz1, sz2, sz3])

        first_times = np.arange(0, sz1, dtype='float32')
        first_data = np.arange(0, sz1, dtype='float32')

        second_times = np.arange(sz1, sz1+sz2, dtype='float32')
        second_data = np.arange(sz1, sz1+sz2, dtype='float32')

        third_times = np.arange(sz1+sz2, nt, dtype='float32')
        third_data = np.arange(sz1+sz2, nt, dtype='float32')

        cova_pth = _make_cov(cls.working_dir, ['data_all', 'data_a'], nt=sz1,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(cls.working_dir, ['data_all', 'data_b'], nt=sz2,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(cls.working_dir, ['data_all', 'data_c'], nt=sz3,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(cls.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                      parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b', 'data_c']),
                                      reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                      reference_coverage_extents=cls.get_no_extents([cova_pth, covb_pth, covc_pth]),
                                      complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        return comp_cov, 'TestNewComplexCoverageInt'

    ######################
    # Overridden base tests
    ######################

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        # Cannot set values against a NewComplexCoverage - just return True
        return True

    def test_append_parameter(self):
        nt = 60
        ccov, cov_name = self.get_cov(inline_data_writes=True, nt=nt)

        parameter_name = 'a*b'
        func = NumexprFunction('a*b', 'a*b', ['a', 'b'], {'a': 'data_a', 'b': 'data_b'})
        pc_in = ParameterContext(parameter_name, param_type=ParameterFunctionType(function=func, value_encoding=np.dtype('float32')))

        ccov.append_parameter(pc_in)
        self.assertIn(parameter_name, ccov.list_parameters())

        with self.assertRaises(ValueError):
            ccov.append_parameter(pc_in)

    @unittest.skip('Functionality verified in \'test_temporal_aggregation\'')
    def test_refresh(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_create_multi_bricks(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_coverage_pickle_and_in_memory(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_coverage_mode_expand_domain(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_coverage_mode_set_value(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_pickle_problems_in_memory(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_allparams_five_bricks(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_allparams_one_brick(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_time_five_bricks(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_time_five_bricks_strided(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_time_one_brick(self):
        pass

    @unittest.skip('Does not apply to NewComplexCoverage')
    def test_set_time_one_brick_strided(self):
        pass

    ######################
    # Additional tests specific to Complex Coverage
    ######################

    def test_file_mode(self):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        nt = 10
        rcov_locs = [_make_cov('test_data', ['first_param'], nt=nt),
                     _make_cov('test_data', ['second_param'], nt=nt),
                     _make_cov('test_data', ['third_param', 'fourth_param'], nt=nt),
                     ]

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        func = NumexprFunction('a*b', 'a*b', ['a', 'b'], {'a': 'first_param', 'b': 'second_param'})
        val_ctxt = ParameterContext('a*b', param_type=ParameterFunctionType(function=func, value_encoding=np.dtype('float32')))
        pdict.add_context(val_ctxt)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        ccov = ComplexCoverage('test_data', create_guid(), 'sample complex coverage',
                                  parameter_dictionary=_make_param_dict(['first_param', 'second_param', 'third_param', 'fourth_param']),
                                  mode='w', reference_coverage_locs=rcov_locs, reference_coverage_extents=self.get_no_extents(rcov_locs))

        ccov_pth = ccov.persistence_dir
        ccov_masterfile_pth = ccov._persistence_layer.master_manager.file_path

        storage_type = ccov._persistence_layer.master_manager.storage_type()
        # Close the CC
        ccov.close()
        del(ccov)

        # Open NewComplexCoverage in write mode
        w_ccov = AbstractCoverage.load(ccov_pth)

        # Loop over opening and reading data out of CC 10 times
        rpt = 20
        expected_array = np.empty(3*3, dtype=np.float32)
        for i in range(3):
            expected_array[i*3:i*3+3] = i
        while rpt > 0:
            read_ccov = AbstractCoverage.load(ccov_pth, mode='r')
            self.assertIsInstance(read_ccov, AbstractCoverage)
            time_value = read_ccov.get_parameter_values('time', time_segment=(0,2))
            np.testing.assert_array_equal(time_value.get_data()['time'], expected_array)
            read_ccov.close()
            del(read_ccov)
            rpt = rpt - 1

        w_ccov.close()
        del(w_ccov)

        if storage_type == 'db':
            # Only for file-based metadata
            # Open NewComplexCoverage's master file using locking
            # with HDFLockingFile(ccov_masterfile_pth, 'r+') as f:

            # Test ability to read from NewComplexCoverage in readonly mode
            locked_ccov = AbstractCoverage.load(ccov_pth, mode='r')
            self.assertIsInstance(locked_ccov, AbstractCoverage)
            time_value = locked_ccov.get_parameter_values('time', time_segment=(1,1)).get_data()['time']
            np.testing.assert_array_equal(time_value, np.array([1]*3, dtype=np.float32))

            # Test inability to load coverage again
            AbstractCoverage.load(ccov_pth)
            AbstractCoverage.load(ccov_pth, mode='w')
            AbstractCoverage.load(ccov_pth, mode='a')
            AbstractCoverage.load(ccov_pth, mode='r+')

            locked_ccov.close()
            del(locked_ccov)

    def test_temporal_aggregation(self):
        size = 100000
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        third_times = np.arange(size*2, size*3, dtype='float32')
        third_data = np.arange(size*7, size*8, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(self.working_dir, ['data_all', 'data_c'], nt=size,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b', 'data_c']),
                                   reference_coverage_locs=[cova_pth, covb_pth, covc_pth],
                                   reference_coverage_extents=self.get_no_extents([cova_pth, covb_pth, covc_pth]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        self.assertEqual(3*size, comp_cov.num_timesteps())
        tvals = comp_cov.get_time_values()
        np.testing.assert_array_equal(tvals, np.arange(3*size, dtype='float32'))
        self.assertEqual(tvals.dtype, np.dtype('float32'))  # np.array_equal does NOT check dtype!!

        all_data = np.empty(0, dtype='float32')
        all_data = np.append(all_data, first_data)
        all_data = np.append(all_data, second_data)
        all_data = np.append(all_data, third_data)
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_all').get_data()['data_all'], all_data)
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_all', time_segment=(0, size-1)).get_data()['data_all'], first_data)

        fill_arr = np.empty(size*3, dtype='float32')
        fill_arr[:] = -9999.0
        first = fill_arr.copy()
        first[0:size] = first_data
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_a').get_data()['data_a'], first)

        second = fill_arr.copy()
        second[size:size*2] = second_data
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_b').get_data()['data_b'], second)

        third = fill_arr.copy()
        third[size*2:size*3] = third_data
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_c').get_data()['data_c'], third)

        # Check that the head_coverage_path is correct
        self.assertEqual(os.path.relpath(comp_cov.head_coverage_path), os.path.relpath(covc_pth))

        # Add some data to the last coverage (covc) and make sure it comes in
        cov_c = AbstractCoverage.load(covc_pth, mode='a')
        addnl_c_data = np.arange(size*8, size*9, dtype='float32')
        p_dict = {
            'time': np.arange(size*3, size*4),
            'data_all': addnl_c_data,
            'data_c': addnl_c_data
        }
        cov_c.set_parameter_values(make_parameter_data_dict(p_dict))
        cov_c.close()

        # # Refresh the complex coverage
        # comp_cov.refresh()

        all_data = np.append(all_data, addnl_c_data)
        # np.testing.assert_array_equal(comp_cov.get_parameter_values('data_all').get_data()['data_all'], all_data)

        third = np.append(third, addnl_c_data)
        comp_cov.refresh()
        np.testing.assert_array_equal(comp_cov.get_parameter_values('data_c').get_data()['data_c'], third)

        # Check that the head_coverage_path is still correct
        self.assertEqual(os.path.abspath(comp_cov.head_coverage_path), os.path.abspath(covc_pth))

    def _setup_allparams(self, size=10, num_covs=2, sequential_covs=True):
        # Setup types
        types = []
        types.append(('qtype', QuantityType()))
        types.append(('atype_n', ArrayType()))
        # types.append(('atype_s', ArrayType()))
        letts='abcdefghijklmnopqrstuvwxyz'
        while len(letts) < size:
            letts += letts
        types.append(('rtype', RecordType()))
        types.append(('btype', BooleanType()))
        types.append(('ctype_n', ConstantType(QuantityType(value_encoding=np.dtype('int32')))))
        types.append(('ctype_s', ConstantType(QuantityType(value_encoding=np.dtype('S21')))))
        types.append(('crtype', ConstantRangeType(QuantityType(value_encoding=np.dtype('int16')))))
        types.append(('pftype', ParameterFunctionType(NumexprFunction('v*10', 'v*10', ['v'], {'v': 'ctype_n'}))))
        cat = {99:'empty',0:'turkey',1:'duck',2:'chicken'}
        catkeys = cat.keys()
        types.append(('cattype', CategoryType(categories=cat)))
        types.append(('sctype', SparseConstantType(fill_value=-998, value_encoding='int32')))

        # Make coverages
        covs = []
        cov_data = []
        for i in xrange(num_covs):
            ii = i + 1
            # Make parameters
            pdict = ParameterDictionary()
            tpc = ParameterContext('time', param_type=QuantityType(value_encoding='float32'))
            tpc.axis = AxisTypeEnum.TIME
            pdict.add_context(tpc)
            for t in types:
                pdict.add_context(ParameterContext(t[0], param_type=t[1], variability=VariabilityEnum.TEMPORAL))

            # Make the data
            data_dict = {}
            if sequential_covs:
                tmax = ii * size
                tmin = i * size
                tdata = np.random.random_sample(size) * (tmax - tmin) + tmin
                tdata.sort()
            else:
                tdata = np.random.random_sample(size) * (200 - 0) + 0
                tdata.sort()
            data_dict['time'] = tdata
            data_dict['atype_n'] = [[ii for a in xrange(random.choice(range(1,size)))] for r in xrange(size)]
            data_dict['qtype'] = np.random.random_sample(size) * (50 - 10) + 10
            data_dict['rtype'] = [{letts[r]: letts[r:]} for r in xrange(size)]
            data_dict['btype'] = [random.choice([True, False]) for r in xrange(size)]
            data_dict['ctype_n'] = [ii*20] * size
            data_dict['cattype'] = [random.choice(catkeys) for r in xrange(size)]
            data_dict['sctype'] = [ii*30] * size

            # Create the coverage
            covs.append(_make_cov(self.working_dir, pdict, nt=size, data_dict=data_dict))

            # Now add values for pftype, for later comparison
            data_dict['pftype'] = [x*10 for x in data_dict['ctype_n']]
            # And update the values for cattype
            data_dict['cattype'] = [cat[k] for k in data_dict['cattype']]

            # Add the data_dict to the cov_data list
            cov_data.append(data_dict)

        return covs, cov_data

    def test_temporal_aggregation_all_param_types(self):
        size = 10

        covs, cov_data = self._setup_allparams(size=size)

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=covs,
                                   reference_coverage_extents=self.get_extents(covs),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        for p in comp_cov.list_parameters():
            for i in xrange(len(covs)):
                ddict = cov_data[i]
                if p in ['qtype', 'time']:
                    self.assertTrue(np.allclose(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)), ddict[p]))
                elif p == 'ctype_s':
                    self.assertTrue(np.atleast_1d(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)) == ddict[p]).all())
                else:
                    self.assertTrue(np.array_equal(comp_cov.get_parameter_values(p, slice(i*size, (i+1)*size)), ddict[p]))

    def test_temporal_interleaved(self):
        num_times = 6
        tpc = num_times / 2

        first_times = np.random.random_sample(tpc) * (20 - 0) + 0
        # first_times =  np.array([0,1,2,5,6,10,11,13,14,16], dtype='float32')
        first_times.sort()
        first_data = np.arange(tpc, dtype='float32') * 0.2
        first_full = np.random.random_sample(tpc) * (80 - 60) + 60

        second_times = np.random.random_sample(tpc) * (20 - 0) + 0
        # second_times = np.array([3,4,7,8,9,12,15,17,18,19], dtype='float32')
        second_times.sort()
        second_data = np.random.random_sample(tpc) * (50 - 10) + 10
        second_full = np.random.random_sample(tpc) * (80 - 60) + 60

        log.debug('\nCov A info:\n%s\n%s\n%s\n---------', first_times, first_data, first_full)
        log.debug('\nCov B info:\n%s\n%s\n%s\n---------', second_times, second_data, second_full)

        cova_pth = _make_cov(self.working_dir,
                             ['first_param', 'full_param'], nt=tpc,
                             data_dict={'time': first_times, 'first_param': first_data, 'full_param': first_full})
        covb_pth = _make_cov(self.working_dir,
                             ['second_param', 'full_param'], nt=tpc,
                             data_dict={'time': second_times, 'second_param': second_data, 'full_param': second_full})

        # Instantiate the NewComplexCoverage
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample complex coverage',
                               parameter_dictionary=_make_param_dict(['first_param', 'full_param', 'second_param']),
                               reference_coverage_locs=[cova_pth, covb_pth],
                               reference_coverage_extents=self.get_no_extents([cova_pth, covb_pth]),
                               complex_type=ComplexCoverageType.TEMPORAL_INTERLEAVED)

        self.assertEqual(ccov.list_parameters(), ['first_param', 'full_param', ccov.ingest_time_parameter_name, 'second_param', 'time'])

        self.assertEqual(ccov.temporal_parameter_name, 'time')

        time_interleave = np.append(first_times, second_times)
        sort_i = np.argsort(time_interleave)
        time_interleave = np.array(time_interleave, dtype='float32')
        np.testing.assert_array_equal(ccov.get_time_values(), time_interleave[sort_i])
        self.assertTrue(np.allclose(ccov.get_time_values(), time_interleave[sort_i]))

        cova = AbstractCoverage.load(cova_pth)
        full_interleave = np.append(first_full, second_full)
        full_interleave = np.array(full_interleave, dtype='float32')
        np.testing.assert_array_equal(ccov.get_parameter_values('full_param', sort_parameter='time').get_data()['full_param'], full_interleave[sort_i])

        first_interleave = np.empty((num_times,), dtype='float32')
        first_interleave.fill(ccov.get_parameter_context('first_param').fill_value)
        first_interleave[:tpc] = first_data
        np.testing.assert_array_equal(ccov.get_parameter_values(['first_param', 'time'], sort_parameter='time').get_data()['first_param'], first_interleave[sort_i])

        second_interleave = np.empty((num_times,), dtype='float32')
        second_interleave.fill(ccov.get_parameter_context('second_param').fill_value)
        second_interleave[tpc:] = second_data
        np.testing.assert_array_equal(ccov.get_parameter_values('second_param', sort_parameter='time').get_data()['second_param'], second_interleave[sort_i])

    def test_temporal_interleaved_all_param_types(self):
        size = 10
        covs, cov_data = self._setup_allparams(size=size, num_covs=5, sequential_covs=False)

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=covs,
                                   reference_coverage_extents=self.get_extents(covs),
                                   complex_type=ComplexCoverageType.TEMPORAL_INTERLEAVED)

        time_interleave = np.empty(0)
        for c in cov_data:
            time_interleave = np.append(time_interleave, c['time'])
        sort_i = np.argsort(time_interleave)

        self.assertTrue(np.allclose(comp_cov.get_time_values(), time_interleave[sort_i]))

    def test_append_reference_coverage(self):
        size = 100000
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        third_times = np.arange(size*2, size*3, dtype='float32')
        third_data = np.arange(size*7, size*8, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})
        covc_pth = _make_cov(self.working_dir, ['data_all', 'data_c'], nt=size,
                             data_dict={'time': third_times, 'data_all': third_data, 'data_c': third_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   reference_coverage_extents=self.get_extents([cova_pth, covb_pth]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        # Verify stuff worked normally...
        tvals = comp_cov.get_time_values()
        self.assertTrue(np.array_equal(tvals, np.arange(2*size, dtype='float32')))

        covc = AbstractCoverage.load(covc_pth)
        covc_id = covc.persistence_guid
        # Append the new coverage
        comp_cov.append_reference_coverage(covc_pth, extents=ReferenceCoverageExtents('c', covc_id, time_extents=(size*2, size*3)))

        # Now make sure the new data is there!
        tvals = comp_cov.get_time_values()
        self.assertTrue(np.array_equal(tvals, np.arange(3*size, dtype='float32')))

    def test_head_coverage_path(self):
        size = 10
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   reference_coverage_extents=self.get_extents([cova_pth, covb_pth]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        if comp_cov._persistence_layer.master_manager.storage_type() != 'db':
            # TODO: correct this once ViewCoverage is worked out
            # View coverage construction doesn't work for DB-based metadata.  View Coverage will be modified in the future
            self.assertTrue(True)
        else:
            comp_cov2 = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                        reference_coverage_locs=[cova_pth],
                                        reference_coverage_extents=self.get_extents([cova_pth]),
                                        complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

            comp_cov3 = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal broadcast coverage',
                                         reference_coverage_locs=[comp_cov2.persistence_dir, covb_pth],
                                         reference_coverage_extents=self.get_extents([comp_cov2.persistence_dir, covb_pth]),
                                         complex_type=ComplexCoverageType.TEMPORAL_BROADCAST)

            # Ensure the correct path is returned from NewComplexCoverage.head_coverage_path in CC --> [SC & SC] scenario
            self.assertEqual(os.path.abspath(comp_cov.head_coverage_path), os.path.abspath(covb_pth))

            # Ensure the correct path is returned from NewComplexCoverage.head_coverage_path in CC --> [SC & VC] scenario
            self.assertEqual(os.path.abspath(comp_cov2.head_coverage_path), os.path.abspath(cova_pth))
            self.assertEqual(os.path.abspath(comp_cov3.head_coverage_path), os.path.abspath(covb_pth))

            # Ensure the correct path is returned from NewComplexCoverage.head_coverage_path in CC --> [SC & CC --> [VC & SC]] scenario
            self.assertEqual(os.path.abspath(comp_cov3.head_coverage_path), os.path.abspath(covb_pth))

    def make_timeseries_cov(self):
        cova_pth = _make_cov(self.working_dir, ['value_set'], data_dict={'time': np.arange(10,20),'value_set':np.ones(10)})
        cov = AbstractCoverage.load(cova_pth)
        pdict = cov.parameter_dictionary

        ccov = ComplexCoverage(self.working_dir, create_guid(), 'complex coverage',
                reference_coverage_locs=[],
                reference_coverage_extents=self.get_extents([]),
                parameter_dictionary=pdict,
                complex_type=ComplexCoverageType.TIMESERIES)
        return ccov

    def test_striding(self):
        pass

    def test_get_all_parameters(self):
        size = 10
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b']),
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   reference_coverage_extents=self.get_no_extents([cova_pth, covb_pth]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        pvals = comp_cov.get_parameter_values().get_data()
        expected_data_all = np.empty(size*2, dtype='float32')
        expected_data_all[0:size] = first_data
        expected_data_all[size:size*2] = second_data
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        expected_times = np.empty(size*2)
        expected_times[0:size] = first_times
        expected_times[size:size*2] = second_times
        np.testing.assert_array_equal(expected_times, pvals['time'])
        expected_data_a = np.empty(size*2)
        expected_data_a[0:size] = first_data
        expected_data_a[size:size*2] = -9999.0
        np.testing.assert_array_equal(expected_data_a, pvals['data_a'])
        expected_data_b = np.empty(size*2)
        expected_data_b[0:size] = -9999.0
        expected_data_b[size:size*2] = second_data
        np.testing.assert_array_equal(expected_data_b, pvals['data_b'])

    def test_get_parameters_with_limiting_extents(self):
        size = 10
        first_times = np.arange(0, size, dtype='float32')
        first_data = np.arange(size, size*2, dtype='float32')

        second_times = np.arange(size, size*2, dtype='float32')
        second_data = np.arange(size*4, size*5, dtype='float32')

        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=size,
                             data_dict={'time': first_times, 'data_all': first_data, 'data_a': first_data})
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=size,
                             data_dict={'time': second_times, 'data_all': second_data, 'data_b': second_data})

        cova = AbstractCoverage.load(cova_pth)
        covb = AbstractCoverage.load(covb_pth)
        cova_id = cova.persistence_guid
        covb_id = covb.persistence_guid
        rcov_extents = {}
        # cova is referenced once
        rcov_extents[cova_id] = ReferenceCoverageExtents('a', cova_id, time_extents=(0,4))
        # covb is referenced once but with the possibility of being referenced again (hence the list)
        rcov_extents[covb_id] = [ReferenceCoverageExtents('b', covb_id, time_extents=(10,12))]
        cova.close()
        covb.close()

        comp_cov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b']),
                                   reference_coverage_locs=[cova_pth, covb_pth],
                                   reference_coverage_extents=rcov_extents,
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        self.assertEqual(8, comp_cov.num_timesteps())
        times = np.empty(8, dtype='float32')
        times[0:5] = first_times[0:5]
        times[5:8] = second_times[0:3]
        fill_data = np.empty(8, dtype='float32')
        fill_data.fill(-9999.0)
        adata = fill_data.copy()
        adata[0:5] = first_data[0:5]
        pvals = comp_cov.get_parameter_values('data_a', sort_parameter='time').get_data()
        np.testing.assert_array_equal(times, pvals['time'])
        np.testing.assert_array_equal(adata, pvals['data_a'])

        bdata = fill_data.copy()
        bdata[5:8] = second_data[0:3]
        pvals = comp_cov.get_parameter_values('data_b').get_data()
        np.testing.assert_array_equal(times, pvals['time'])
        np.testing.assert_array_equal(bdata, pvals['data_b'])

        pvals = comp_cov.get_parameter_values(['data_b', 'data_all']).get_data()
        expected_data_all = np.empty(8)
        expected_data_all[0:5] = first_data[0:5]
        expected_data_all[5:8] = second_data[0:3]
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        np.testing.assert_array_equal(times, pvals['time'])
        np.testing.assert_array_equal(bdata, pvals['data_b'])

        pvals = comp_cov.get_parameter_values(['data_b', 'data_all'], time_segment=(4,11)).get_data()
        expected_data_all = np.empty(3)
        expected_data_all[0] = first_data[4]
        expected_data_all[1:3] = second_data[0:2]
        expected_times = np.empty(3)
        expected_times[0] = first_times[4]
        expected_times[1:3] = second_times[0:2]
        expected_b = np.empty(3)
        expected_b[0] = -9999.0
        expected_b[1:3] = second_data[0:2]
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        np.testing.assert_array_equal(expected_times, pvals['time'])
        np.testing.assert_array_equal(expected_b, pvals['data_b'])

        comp_cov.set_reference_coverage_extents(covb_id, ReferenceCoverageExtents('b2', covb_id, time_extents=(18,19)), append=True)

        pvals = comp_cov.get_parameter_values(['data_b', 'data_all']).get_data()
        expected_data_all = np.empty(10)
        expected_data_all[0:5] = first_data[0:5]
        expected_data_all[5:8] = second_data[0:3]
        expected_data_all[8:10] = second_data[8:10]
        expected_times = np.empty(10)
        expected_times[0:5] = first_times[0:5]
        expected_times[5:8] = second_times[0:3]
        expected_times[8:10] = second_times[8:10]
        expected_b = np.empty(10)
        expected_b[:] = -9999.0
        expected_b[5:8] = second_data[0:3]
        expected_b[8:10] = second_data[8:10]
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        np.testing.assert_array_equal(expected_times, pvals['time'])
        np.testing.assert_array_equal(expected_b, pvals['data_b'])

        comp_cov.set_reference_coverage_extents(covb_id, ReferenceCoverageExtents('b2', covb_id, time_extents=(18,19)), append=False)

        pvals = comp_cov.get_parameter_values(['data_b', 'data_all']).get_data()
        expected_data_all = np.empty(7)
        expected_data_all[0:5] = first_data[0:5]
        expected_data_all[5:7] = second_data[8:10]
        expected_times = np.empty(7)
        expected_times[0:5] = first_times[0:5]
        expected_times[5:7] = second_times[8:10]
        expected_b = np.empty(7)
        expected_b[:] = -9999.0
        expected_b[5:7] = second_data[8:10]
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        np.testing.assert_array_equal(expected_times, pvals['time'])
        np.testing.assert_array_equal(expected_b, pvals['data_b'])

        comp_cov_id = comp_cov.persistence_guid
        comp_cov.close()

        ccov = AbstractCoverage.load(self.working_dir, comp_cov_id)
        pvals = ccov.get_parameter_values(['data_b', 'data_all']).get_data()
        expected_data_all = np.empty(7)
        expected_data_all[0:5] = first_data[0:5]
        expected_data_all[5:7] = second_data[8:10]
        expected_times = np.empty(7)
        expected_times[0:5] = first_times[0:5]
        expected_times[5:7] = second_times[8:10]
        expected_b = np.empty(7)
        expected_b[:] = -9999.0
        expected_b[5:7] = second_data[8:10]
        np.testing.assert_array_equal(expected_data_all, pvals['data_all'])
        np.testing.assert_array_equal(expected_times, pvals['time'])
        np.testing.assert_array_equal(expected_b, pvals['data_b'])

    def test_attributes(self):
        # Complex coverages are read only
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b']),
                                   reference_coverage_locs=[],
                                   reference_coverage_extents=self.get_extents([]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        data_dict = {
            'time': NumpyParameterData('time', np.arange(10))
        }

        self.assertRaises(NotImplementedError, ccov.set_parameter_values, data_dict)

        # A Complex Coverage has it's own parameter dictionary
        for param_name in ['data_all', 'data_a', 'data_b']:
            pc = ccov.get_parameter_context(param_name)
            self.assertIsInstance(pc, ParameterContext)

        # A Complex Coverage comprises windows of other datasets
        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=10,
                             data_dict={'time': np.arange(10), 'data_all': np.arange(100,110), 'data_a': np.arange(50,60)})
        cova = AbstractCoverage.load(cova_pth)
        cova_id = cova.persistence_guid
        cova.close()

        # Should raise without a window
        self.assertRaises(ValueError, ccov.append_reference_coverage, cova_pth)

        # Append the first window of a dataset, that window doesn't encompass the entire first dataset
        ccov.append_reference_coverage(cova_pth, ReferenceCoverageExtents('first-deployment', cova_id, time_extents=(2,8)))

        # Get the data and make sure we can see values 2-8
        data = ccov.get_parameter_values(fill_empty_params=True, as_record_array=False).get_data()
        np.testing.assert_allclose(data['time'], np.arange(2,9))
        np.testing.assert_allclose(data['data_all'], np.arange(102,109))
        np.testing.assert_allclose(data['data_a'], np.arange(52,59))

        # A Complex Coverage fills in missing parameters
        np.testing.assert_allclose(data['data_b'], np.array([-9999] * 7))

        
        covb_pth = _make_cov(self.working_dir, ['data_all', 'data_b'], nt=20,
                             data_dict={'time': np.arange(20), 'data_all': np.arange(100,120), 'data_b': np.arange(20,40)})
        covb = AbstractCoverage.load(covb_pth)
        covb_id = covb.persistence_guid
        covb.close()
        ccov.append_reference_coverage(covb_pth, ReferenceCoverageExtents('second-deployment', covb_id, time_extents=(15,19)))

        data = ccov.get_parameter_values(time_segment=(None,None), fill_empty_params=True, as_record_array=False).get_data()
        time_dense = np.concatenate((np.arange(2,9), np.arange(15,20)))
        np.testing.assert_allclose(data['time'], time_dense)

        data_b_dense = np.concatenate(([-9999] * 7, np.arange(35,40)))
        np.testing.assert_allclose(data['data_b'], data_b_dense)
        
        # Test slicing
        data = ccov.get_parameter_values('time', fill_empty_params=True, as_record_array=False, stride_length=3).get_data()
        # Stretch goal
        np.testing.assert_allclose(data['time'], time_dense[::3])

    def test_empty_coverages(self):
        # Complex coverages are read only
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict(['data_all', 'data_a', 'data_b']),
                                   reference_coverage_locs=[],
                                   reference_coverage_extents=self.get_extents([]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        data_dict = {
            'time': NumpyParameterData('time', np.arange(10))
        }

        self.assertRaises(NotImplementedError, ccov.set_parameter_values, data_dict)

        # A Complex Coverage has it's own parameter dictionary
        for param_name in ['data_all', 'data_a', 'data_b']:
            pc = ccov.get_parameter_context(param_name)
            self.assertIsInstance(pc, ParameterContext)

        # A Complex Coverage comprises windows of other datasets
        cova_pth = _make_cov(self.working_dir, ['data_all', 'data_a'], nt=0)
        cova = AbstractCoverage.load(cova_pth)
        cova_id = cova.persistence_guid
        self.assertEqual(cova.num_timesteps(), 0)
        cova.close()

        # Should raise without a window
        self.assertRaises(ValueError, ccov.append_reference_coverage, cova_pth)

        # Append the first window of a dataset, that window doesn't encompass the entire first dataset
        ccov.append_reference_coverage(cova_pth, ReferenceCoverageExtents('first-deployment', cova_id, time_extents=(2,8)))

        # Make sure we can get the data (should be empty)
        data = ccov.get_parameter_values(fill_empty_params=True, as_record_array=False).get_data()
        np.testing.assert_allclose(data['time'], np.array([]))
        np.testing.assert_allclose(data['data_a'], np.array([]))
        np.testing.assert_allclose(data['data_b'], np.array([]))

        ccov.refresh()

        cova = AbstractCoverage.load(cova_pth, mode='a')
        cova.set_parameter_values({'time': np.arange(10), 'data_all': np.arange(100,110), 'data_a': np.arange(50,60)})

        # Get the data and make sure we can see values 2-8
        data = ccov.get_parameter_values(fill_empty_params=True, as_record_array=False).get_data()
        np.testing.assert_allclose(data['time'], np.arange(2,9))
        np.testing.assert_allclose(data['data_all'], np.arange(102,109))
        np.testing.assert_allclose(data['data_a'], np.arange(52,59))

    def test_array_coverages(self):
        # Complex coverages are read only
        array_stuff = ParameterContext('array_stuff', param_type=ArrayType(inner_encoding='int32', inner_fill_value=-9999))
        ccov = ComplexCoverage(self.working_dir, create_guid(), 'sample temporal aggregation coverage',
                                   parameter_dictionary=_make_param_dict([array_stuff]),
                                   reference_coverage_locs=[],
                                   reference_coverage_extents=self.get_extents([]),
                                   complex_type=ComplexCoverageType.TEMPORAL_AGGREGATION)

        data_dict = {
            'time': NumpyParameterData('time', np.arange(10))
        }

        self.assertRaises(NotImplementedError, ccov.set_parameter_values, data_dict)

        # A Complex Coverage has it's own parameter dictionary
        for param_name in ['array_stuff']:
            pc = ccov.get_parameter_context(param_name)
            self.assertIsInstance(pc, ParameterContext)

        # A Complex Coverage comprises windows of other datasets
        arr2 = ParameterContext('array_stuff', param_type=ArrayType(inner_encoding='int32', inner_length=2))
        cov2_pth = _make_cov(self.working_dir, [arr2], nt=0)
        cov2 = AbstractCoverage.load(cov2_pth)
        cov2_id = cov2.persistence_guid
        self.assertEqual(cov2.num_timesteps(), 0)
        cov2.close()

        arr4 = ParameterContext('array_stuff', param_type=ArrayType(inner_encoding='int32', inner_length=4))
        cov4_pth = _make_cov(self.working_dir, [arr4], nt=0)
        cov4 = AbstractCoverage.load(cov4_pth)
        cov4_id = cov4.persistence_guid
        self.assertEqual(cov4.num_timesteps(), 0)
        cov4.close()
        # Should raise without a window
        self.assertRaises(ValueError, ccov.append_reference_coverage, cov2_pth)

        # Append the first window of a dataset, that window doesn't encompass the entire first dataset
        ccov.append_reference_coverage(cov2_pth, ReferenceCoverageExtents('first-deployment', cov2_id, time_extents=(0,4)))
        ccov.append_reference_coverage(cov4_pth, ReferenceCoverageExtents('second-deployment', cov4_id, time_extents=(6,8)))

        # Make sure we can get the data (should be empty)
        data = ccov.get_parameter_values(fill_empty_params=True, as_record_array=False).get_data()
        np.testing.assert_allclose(data['time'], np.array([]))
        np.testing.assert_allclose(data['array_stuff'], np.array([]))

        ccov.refresh()

        array = np.arange(20, dtype='int32')
        array2 = array.reshape((10,2))
        array4 = array.reshape((5,4))
        cov2 = AbstractCoverage.load(cov2_pth, mode='a')
        cov2.set_parameter_values({'time': np.arange(10), 'array_stuff': array2})
        cov4 = AbstractCoverage.load(cov4_pth, mode='a')
        cov4.set_parameter_values({'time': np.arange(5,10), 'array_stuff': array4})

        ccov.refresh()
        # Get the data and make sure we can see values 2-8
        data = ccov.get_parameter_values(fill_empty_params=True, as_record_array=False).get_data()
        np.testing.assert_allclose(data['time'], np.array([0,1,2,3,4,6,7,8], dtype='int32'))
        expected_array_stuff = np.array( [ [0,1,-9999,-9999], [2,3,-9999,-9999], [4,5,-9999,-9999], [6,7,-9999,-9999], [8,9,-9999,-9999], [4,5,6,7], [8,9,10,11], [12,13,14,15]], dtype='int32')
        np.testing.assert_allclose(data['array_stuff'], expected_array_stuff)


    @attr('INT', group='cov')
    def test_external_refs(self):

        # Create a three param coverage
        offset = NumexprFunction('offset', arg_list=['x'], expression='x + 1')
        offset.param_map = {'x':'value_set'}
        ctx = ParameterContext('offset', param_type=ParameterFunctionType(offset, value_encoding='<f4'))

        cova_pth = _make_cov(self.working_dir, ['value_set', ctx], data_dict={'time':np.arange(0,10,0.7), 'value_set':np.arange(20,30, 0.7)})
        cova = SimplexCoverage.load(cova_pth, mode='r')

        with self.assertRaises(AttributeError): # method doesn't exist
            pfunc = ExternalFunction('example', cova.persistence_guid, 'value_set', 'coverage_model.parameter_functions', 'linear_map', [])
            ctx = ParameterContext('example', param_type=ParameterFunctionType(pfunc, value_encoding='<f4'))
            covb_pth = _make_cov(self.working_dir, [ctx], data_dict={'time':np.arange(0.5, 10.5, 1)})
            cov = SimplexCoverage.load(covb_pth, mode='r')
            data = cov.get_parameter_values().get_data()

        # Create another coverage that references the above
        pfunc = ExternalFunction('example', cova.persistence_guid, 'value_set')
        ctx = ParameterContext('example', param_type=ParameterFunctionType(pfunc, value_encoding='<f4'))
        covb_pth = _make_cov(self.working_dir, [ctx], data_dict={'time':np.arange(0.5, 10.5, 1)})
        cov = SimplexCoverage.load(covb_pth, mode='r')
        # Assert that the values are correctly interpolated
        data = cov.get_parameter_values().get_data()
        np.testing.assert_allclose(data['example'], np.arange(20.5, 30.5, 1))
        np.testing.assert_array_equal(data['time'], np.arange(0.5, 10.5, 1))

        cov.close()

        pfunc_explicit = ExternalFunction('example', cova.persistence_guid, 'value_set', 'coverage_model.util.external_parameter_methods', 'linear_map', [])
        ctx_explicit = ParameterContext('example', param_type=ParameterFunctionType(pfunc_explicit, value_encoding='<f4'))
        covc_pth = _make_cov(self.working_dir, [ctx_explicit], data_dict={'time':np.arange(0.5, 10.5, 1)})
        covc = SimplexCoverage.load(covc_pth, mode='r')
        # Assert that the values are correctly interpolated
        data = covc.get_parameter_values().get_data()
        np.testing.assert_allclose(data['example'], np.arange(20.5, 30.5, 1))
        np.testing.assert_array_equal(data['time'], np.arange(0.5, 10.5, 1))


def create_all_params():
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
    
    p_range_ctxt = ParameterContext('p_range', param_type=ConstantType(QuantityType(value_encoding=np.dtype('float32'))), fill_value=-9999)
    p_range_ctxt.uom = '1'
    contexts['p_range'] = p_range_ctxt

    # Independent Parameters

    # Temperature - values expected to be the decimal results of conversion from hex
    temp_ctxt = ParameterContext('temperature', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    temp_ctxt.uom = 'counts'
    contexts['temperature'] = temp_ctxt

    # Conductivity - values expected to be the decimal results of conversion from hex
    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    cond_ctxt.uom = 'counts'
    contexts['conductivity'] = cond_ctxt

    # Pressure - values expected to be the decimal results of conversion from hex
    press_ctxt = ParameterContext('pressure', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    press_ctxt.uom = 'counts'
    contexts['pressure'] = press_ctxt


    # Dependent Parameters

    # tempwat_l1 = (tempwat_l0 / 10000) - 10
    tl1_func = '(T / 10000) - 10'
    tl1_pmap = {'T': 'temperature'}
    expr = NumexprFunction('seawater_temperature', tl1_func, ['T'], param_map=tl1_pmap)
    tempL1_ctxt = ParameterContext('seawater_temperature', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    tempL1_ctxt.uom = 'deg_C'
    contexts['seawater_temperature'] = tempL1_ctxt

    # condwat_l1 = (condwat_l0 / 100000) - 0.5
    cl1_func = '(C / 100000) - 0.5'
    cl1_pmap = {'C': 'conductivity'}
    expr = NumexprFunction('seawater_conductivity', cl1_func, ['C'], param_map=cl1_pmap)
    condL1_ctxt = ParameterContext('seawater_conductivity', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    condL1_ctxt.uom = 'S m-1'
    contexts['seawater_conductivity'] = condL1_ctxt

    # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
    #   preswat_l1 = (preswat_l0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
    pl1_func = '(P * p_range / (0.85 * 65536)) - (0.05 * p_range)'
    pl1_pmap = {'P': 'pressure', 'p_range': 'p_range'}
    expr = NumexprFunction('seawter_pressure', pl1_func, ['P', 'p_range'], param_map=pl1_pmap)
    presL1_ctxt = ParameterContext('seawater_pressure', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    presL1_ctxt.uom = 'S m-1'
    contexts['seawater_pressure'] = presL1_ctxt

    # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
    #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

    # pracsal = gsw.SP_from_C((condwat_l1 * 10), tempwat_l1, preswat_l1)
    owner = 'ion_functions.data.ctd_functions'
    sal_func = 'ctd_pracsal'
    sal_arglist = ['C', 't', 'p']
    sal_pmap = {'C': 'seawater_conductivity', 't': 'seawater_temperature', 'p': 'seawater_pressure'}
    sal_kwargmap = None
    expr = PythonFunction('pracsal', owner, sal_func, sal_arglist, sal_kwargmap, sal_pmap)
    sal_ctxt = ParameterContext('pracsal', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)
    sal_ctxt.uom = 'g kg-1'
    contexts['pracsal'] = sal_ctxt

    # absolute_salinity = gsw.SA_from_SP(pracsal, preswat_l1, longitude, latitude)
    # conservative_temperature = gsw.CT_from_t(absolute_salinity, tempwat_l1, preswat_l1)
    # density = gsw.rho(absolute_salinity, conservative_temperature, preswat_l1)
    owner = 'ion_functions.data.ctd_functions'
    dens_func = 'ctd_density'
    dens_arglist = ['SP', 't', 'p', 'lat', 'lon']
    dens_pmap = {'SP':'pracsal', 't':'seawater_temperature', 'p':'seawater_pressure', 'lat':'lat', 'lon':'lon'}
    dens_expr = PythonFunction('density', owner, dens_func, dens_arglist, None, dens_pmap)
    dens_ctxt = ParameterContext('density', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
    dens_ctxt.uom = 'kg m-3'
    contexts['density'] = dens_ctxt

    return contexts

