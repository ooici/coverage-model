#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_coverage.py
@author Christopher Mueller
@brief Test cases for the coverage_model module
"""
import os
import shutil
import tempfile

from pyon.public import log
from pyon.util.int_test import IonIntegrationTestCase
import numpy as np
#import time
from coverage_model.basic_types import AxisTypeEnum, MutabilityEnum, create_guid, VariabilityEnum
from coverage_model.coverage import CRS, GridDomain, GridShape, SimplexCoverage
from coverage_model.numexpr_utils import make_range_expr
from coverage_model.parameter import ParameterDictionary, ParameterContext
from coverage_model.parameter_types import QuantityType, ConstantType, ArrayType, RecordType
from nose.plugins.attrib import attr
#from mock import patch, Mock

import unittest

@attr('INT', group='cov')
class TestCoverageModelBasicsInt(IonIntegrationTestCase):

    def setUp(self):
        # Create temporary working directory for the persisted coverage tests
        self.working_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        # Comment this out if you need to inspect the HDF5 files.
        shutil.rmtree(self.working_dir)
#        pass
        
    # Loading Coverage Tests
    # Test normal load of a SimplexCoverage
    def test_load_succeeds(self):
        # Depends on a valid coverage existing, so let's make one
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage(base_path, guid)
        lcov.close()
        self.assertIsInstance(lcov, SimplexCoverage)

    # Test load when path is invalid
    def test_load_fails_not_found(self):
        guid = create_guid()
        base_path = '/'
        with self.assertRaises(SystemError):
            SimplexCoverage(base_path, guid)

    # Test load when path is invali
    def test_load_fails_bad_path(self):
        # Depends on a valid coverage existing, so let's make one
        scov = self._make_samplecov()
        guid = scov.persistence_guid
        base_path = 'some_path_that_dne'
        scov.close()
        with self.assertRaises(SystemError):
            SimplexCoverage(base_path, guid)

    def test_load_fails_bad_guid(self):
        # Depends on a valid coverage existing, so let's make one
        scov = self._make_samplecov()
        pl = scov._persistence_layer
        guid = 'some_guid_that_dne'
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        with self.assertRaises(SystemError):
            SimplexCoverage(base_path, guid)

    def test_load_succeeds_with_options(self):
        # Depends on a valid coverage existing, so let's make one
        scov = self._make_samplecov()
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        scov.close()
        base_path = root_path.replace(guid,'')
        name = 'whatever'
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

    #CREATES
    # 'dir, 'guid', 'name', 'pd', 'tdom, 'sdom', 'in-memory'

    def test_create_multi_bricks(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        self.assertIsInstance(scov, SimplexCoverage)


    def test_create_succeeds(self):
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = 'sample coverage_model'
        bricking_scheme = {'brick_size':1000, 'chunk_size':True}
        scov = SimplexCoverage(
            root_dir=self.working_dir,
            persistence_guid=create_guid(),
            name=name,
            parameter_dictionary=pdict,
            temporal_domain=tdom,
            spatial_domain=sdom,
            in_memory_storage=in_memory,
            bricking_scheme=bricking_scheme)
        log.trace(scov.persistence_guid)
        self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='all')
        scov.close()
        self.assertIsInstance(scov, SimplexCoverage)

    def test_create_dir_not_exists(self):
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        with self.assertRaises(SystemError):
            SimplexCoverage('bad_path', create_guid(), 'sample coverage_model', pdict, tdom, sdom, in_memory)

    def test_create_guid_valid(self):
        self.assertTrue(len(create_guid()) == 36)

    # The name field can be just about anything and work right now...not sure what would really break this??
    # Basically, if it is dictable then it is valid
    def test_create_name_invalid(self):
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = np.arange(10)
        with self.assertRaises(AttributeError):
            SimplexCoverage(self.working_dir, create_guid(), name, pdict, tdom, sdom, in_memory)

    def test_create_pdict_invalid(self):
        pdict = 1
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = 'sample coverage_model'
        with self.assertRaises(TypeError):
            SimplexCoverage(self.working_dir, create_guid(), name, pdict, tdom, sdom, in_memory)

    def test_create_tdom_invalid(self):
        pdict = self._make_parameter_dict()
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = 'sample coverage_model'
        with self.assertRaises(TypeError):
            SimplexCoverage(
                root_dir=self.working_dir,
                persistence_guid=create_guid(),
                name=name,
                parameter_dictionary=pdict,
                temporal_domain=1,
                spatial_domain=sdom,
                in_memory_storage=in_memory,
                bricking_scheme=None)

    def test_create_sdom_invalid(self):
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = 1
        in_memory = False
        name = 'sample coverage_model'
        with self.assertRaises(TypeError):
            SimplexCoverage(
                root_dir=self.working_dir,
                persistence_guid=create_guid(),
                name=name,
                parameter_dictionary=pdict,
                temporal_domain=tdom,
                spatial_domain=sdom,
                in_memory_storage=in_memory,
                bricking_scheme=None)

    def test_samplecov_create(self):
        res = False
        tsteps = 0
        scov = self._make_samplecov()
        res = self._run_standard_tests(scov, tsteps)
        tsteps = 10
        res = self._insert_set_get(scov=scov, timesteps=tsteps, data=np.arange(tsteps), _slice=slice(0, tsteps), param='all')
        res = self._run_standard_tests(scov, tsteps)
        prev_tsteps = tsteps
        tsteps = 35
        res = self._insert_set_get(scov=scov, timesteps=tsteps, data=np.arange(tsteps)+prev_tsteps, _slice=slice(prev_tsteps, tsteps), param='all')
        res = self._run_standard_tests(scov, tsteps+prev_tsteps)
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_one_brick(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_one_brick(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_five_bricks(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_one_brick_strided(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10,2), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks_strided(self):
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50,5), param='time')
        scov.close()
        self.assertTrue(res)

    def test_ptypescov_create(self):
        scov = self._make_ptypescov()
        scov.close()
        self.assertTrue(scov.name == 'sample coverage_model')

    def test_ptypescov_load_data(self):
        ptypes_cov = self._make_ptypescov()
        ptypes_cov_loaded = self._load_data_ptypescov(ptypes_cov)
        ptypes_cov.close()
        ptypes_cov_loaded.close()
        self.assertTrue(ptypes_cov_loaded.temporal_domain.shape.extents == (2000,))

    def test_ptypescov_get_values(self):
        results = []
        ptypes_cov = self._make_ptypescov()
        ptypes_cov_loaded = self._load_data_ptypescov(ptypes_cov)
        log.debug(ptypes_cov_loaded._range_value.quantity_time[:])
        results.append((ptypes_cov_loaded._range_value.quantity_time[:] == np.arange(2000)).any())
        log.debug(ptypes_cov_loaded._range_value.const_int[0])
        results.append(ptypes_cov_loaded._range_value.const_int[0] == 45)
        log.debug(results)
        ptypes_cov.close()
        ptypes_cov_loaded.close()
        self.assertTrue(False not in results)

    def test_nospatial_create(self):
        scov = self._make_nospatialcov()
        scov.close()
        self.assertTrue(scov.name == 'sample coverage_model')

    def test_emptysamplecov_create(self):
        scov = self._make_emptysamplecov()
        scov.close()
        self.assertTrue(scov.name == 'empty sample coverage_model')

    def test_close_coverage_before_done_using_it(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        scov.close()
        with self.assertRaises(ValueError):
                scov.get_time_values()

    def test_slice_and_dice(self):
        params, _slices, results, index_errors = self._slice_and_dice(brick_size=1000, time_steps=5000)
        log.debug('slices per parameter: %s', len(_slices))
        log.debug('total slices ran: %s', len(_slices)*len(params))
        log.debug('data failure slices: %s', len(results))
        log.debug('IndexError slices: %s\n%s', len(index_errors), index_errors)
        self.assertTrue(len(results)+len(index_errors) == 0)

    def test_slice_raises_index_error_in_out(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(4999, 5020, None)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', _slice)

    def test_slice_raises_index_error_out_out(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(5010, 5020, None)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', _slice)

    def test_slice_raises_index_error_in_out_step(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(4000, 5020, 5)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', _slice)

    def test_int_raises_index_error(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', 9000)

    def test_array_raises_index_error(self):
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', [[5,9000]])

    def test_get_by_slice(self):
        results = []
        brick_size = 10
        time_steps = 30
        cov = self._create_multi_bricks_cov(brick_size, time_steps)
        dat = cov.get_parameter_values('time')
        for s in range(len(dat)):
            for e in range(len(dat)):
                e+=1
                if s < e:
                    for st in range(e-s):
                        sl = slice(s, e, st+1)
                        mock_data = np.array(range(*sl.indices(sl.stop)))
                        data = cov.get_parameter_values('time', sl)
                        results.append(np.array_equiv(mock_data, data))
        self.assertTrue(False not in results)

#    def test_hdf5_file_missing(self):
#        brick_size = 1000
#        time_steps = 5000
#        cov = self._create_multi_bricks_cov(brick_size, time_steps)
#        pl = cov._persistence_layer
#        root_dir = pl.master_manager.root_dir
#        log.trace(root_dir)
#        cov.insert_timesteps(5000)
#        shutil.rmtree(root_dir)
#        with self.assertRaises(SystemError):
#            try:
#                cov.set_parameter_values('temp', 56.6)
#            except Exception:
#                raise SystemError

    def test_coverage_pickle_and_in_memory(self):
        self._make_oneparamcov(True, True)
        self.assertTrue(os.path.exists(os.path.join(self.working_dir, 'oneparamsample.cov')))

    def test_bad_pc_from_dict(self):
        pc1 = ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius'))
        with self.assertRaises(TypeError):
            pc1._fromdict('junk', pc1.dump())

        pc2 = pc1._fromdict(pc1.dump())
        self.assertTrue(pc1 == pc2)

    def test_params(self):
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

#        print 'Should be equal and compare \'one-to-one\' with nothing in the None list'
#        print pdict_1 == pdict_2
#        print pdict_1.compare(pdict_2)
        self.assertTrue(pdict_1 == pdict_2)
        self.assertTrue(pdict_1.compare(pdict_2) == {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp'], 'time': ['time']})

#        print '\nShould be unequal and compare with an empty list for \'temp\' and \'temp2\' in the None list'
#        print pdict_1 == pdict_3
#        print pdict_1.compare(pdict_3)
        self.assertTrue(pdict_1 != pdict_3)
        self.assertTrue(pdict_1.compare(pdict_3) == {'lat': ['lat'], 'lon': ['lon'], None: ['temp2'], 'temp': [], 'time': ['time']})

#        print '\nShould be unequal and compare with both \'temp\' and \'temp2\' in \'temp\' and nothing in the None list'
#        print pdict_1 == pdict_4
#        print pdict_1.compare(pdict_4)
        self.assertTrue(pdict_1 != pdict_4)
        self.assertTrue(pdict_1.compare(pdict_4) == {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp', 'temp2'], 'time': ['time']})
    def test_pickle_problems(self):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        bricking_scheme = {'brick_size':1000,'chunk_size':500}
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', pdict, tdom, sdom, True, bricking_scheme=bricking_scheme)

        # Insert some timesteps (automatically expands other arrays)
        nt = 2000
        scov.insert_timesteps(nt)

        # Add data for the parameter
        scov.set_parameter_values('time', value=np.arange(nt))

        SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'oneparamsample.cov'))

        ncov = SimplexCoverage.pickle_load(os.path.join(self.working_dir, 'oneparamsample.cov'))
        self.assertIsInstance(ncov, SimplexCoverage)

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_load('some_bad_file_location.cov')

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_save('nat_a_SimplexCoverage', os.path.join(self.working_dir, 'oneparamsample.cov'))

    def _slice_and_dice(self, brick_size, time_steps):
        results = []
        index_errors = []
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        params = scov.list_parameters()
        _slices = []
        # TODO: Automatically calulate the start, stops and strides based on the brick size and time_steps
        starts = [0, 1, 10, 500, 1000, 1001, 3000, 4999]
        stops = [1, 2, 11, 501, 1001, 1002, 3001, 5000]
        strides = [None, 1, 2, 3, 4, 5, 50, 100, 500, 750, 999, 1000, 1001, 1249, 1250, 1500, 2000, 3000, 4000, 5000]
        for stride in strides:
            for start in starts:
                for stop in stops:
                    if stop > start and (stop-start) > stride:
                        _slices.append(slice(start, stop, stride))
        for param in params:
            for _slice in _slices:
                log.trace('working on _slice: %s', _slice)
                sliced_data = np.arange(5000)[_slice]
                try:
                    ret = scov.get_parameter_values(param, _slice)
                    if not (ret == sliced_data).all():
                        results.append(_slice)
                        log.trace('failed _slice: %s', _slice)
                except IndexError as ie:
                    log.trace('%s; moving to next slice', ie.message)
                    index_errors.append(_slice)
                    continue
        scov.close()
        return params, _slices, results, index_errors

    def _run_standard_tests(self, scov, timesteps):
        results = []
        # Check basic metadata
        results.append(scov.name == 'sample coverage_model')
        results.append(scov.num_timesteps == timesteps)
        results.append(list(scov.temporal_domain.shape.extents) == [timesteps])
        req_params = ['conductivity', 'lat', 'lon', 'temp', 'time']
        params = scov.list_parameters()
        for param in params:
            results.append(param in req_params)
            pc = scov.get_parameter_context(param)
            results.append(len(pc.dom.identifier) == 36)
            pc_dom_dict = pc.dom.dump()
        pdict = scov.parameter_dictionary.dump()
        return False not in results

    def _create_multi_bricks_cov(self, brick_size, time_steps):
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = 'multiple bricks coverage'
        bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        scov = SimplexCoverage(
            root_dir=self.working_dir,
            persistence_guid=create_guid(),
            name=name,
            parameter_dictionary=pdict,
            temporal_domain=tdom,
            spatial_domain=sdom,
            in_memory_storage=in_memory,
            bricking_scheme=bricking_scheme)
        log.trace(os.path.join(self.working_dir, scov.persistence_guid))
        self._insert_set_get(scov=scov, timesteps=time_steps, data=np.arange(time_steps), _slice=slice(0,time_steps), param='all')
#        scov.close()
        return scov

    def _insert_set_get(self, scov, timesteps, data, _slice, param='all'):
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
            scov.get_dirty_values_async_result().get(timeout=30)
            ret = scov.get_parameter_values(param, _slice)
        return (ret == data).all()

    def _load_data_ptypescov(self, scov):
        # Insert some timesteps (automatically expands other arrays)
        scov.insert_timesteps(2000)

        # Add data for each parameter
        scov.set_parameter_values('quantity_time', value=np.arange(2000))
        scov.set_parameter_values('const_int', value=45.32) # Set a constant directly, with incorrect data type (fixed under the hood)
        scov.set_parameter_values('const_float', value=make_range_expr(-71.11)) # Set with a properly formed constant expression
        scov.set_parameter_values('quantity', value=np.random.random_sample(2000)*(26-23)+23)

        #    # Setting three range expressions such that indices 0-2 == 10, 3-7 == 15 and >=8 == 20
        #    scov.set_parameter_values('function', value=make_range_expr(10, 0, 3, min_incl=True, max_incl=False, else_val=-999.9))
        #    scov.set_parameter_values('function', value=make_range_expr(15, 3, 8, min_incl=True, max_incl=False, else_val=-999.9))
        #    scov.set_parameter_values('function', value=make_range_expr(20, 8, min_incl=True, max_incl=False, else_val=-999.9))

        arrval = []
        recval = []
        letts='abcdefghijklmnopqrstuvwxyz'
        for x in xrange(len(letts)):
            arrval.append(np.random.bytes(np.random.randint(1,len(letts)))) # One value (which is a byte string) for each member of the domain
            d = {letts[x]: letts[x:]}
            recval.append(d) # One value (which is a dict) for each member of the domain
#        scov.set_parameter_values('array', value=arrval)
#        scov.set_parameter_values('record', value=recval)
        scov.get_dirty_values_async_result().get(timeout=30)
        return scov

    def _make_ptypescov(self, save_coverage=False, in_memory=False):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        quant_t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
        quant_t_ctxt.reference_frame = AxisTypeEnum.TIME
        quant_t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(quant_t_ctxt)

        cnst_int_ctxt = ParameterContext('const_int', param_type=ConstantType(QuantityType(value_encoding=np.dtype('int32'))), variability=VariabilityEnum.NONE)
        cnst_int_ctxt.long_name = 'example of a parameter of type ConstantType, base_type int32'
        cnst_int_ctxt.reference_frame = AxisTypeEnum.LAT
        cnst_int_ctxt.uom = 'degree_north'
        pdict.add_context(cnst_int_ctxt)

        cnst_flt_ctxt = ParameterContext('const_float', param_type=ConstantType(), variability=VariabilityEnum.NONE)
        cnst_flt_ctxt.long_name = 'example of a parameter of type QuantityType, base_type float (default)'
        cnst_flt_ctxt.reference_frame = AxisTypeEnum.LON
        cnst_flt_ctxt.uom = 'degree_east'
        pdict.add_context(cnst_flt_ctxt)

        #    func_ctxt = ParameterContext('function', param_type=FunctionType(QuantityType(value_encoding=np.dtype('float32'))))
        #    func_ctxt.long_name = 'example of a parameter of type FunctionType'
        #    pdict.add_context(func_ctxt)

        quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
        quant_ctxt.long_name = 'example of a parameter of type QuantityType'
        quant_ctxt.uom = 'degree_Celsius'
        pdict.add_context(quant_ctxt)

        arr_ctxt = ParameterContext('array', param_type=ArrayType())
        arr_ctxt.long_name = 'example of a parameter of type ArrayType, will be filled with variable-length \'byte-string\' data'
        pdict.add_context(arr_ctxt)

        rec_ctxt = ParameterContext('record', param_type=RecordType())
        rec_ctxt.long_name = 'example of a parameter of type RecordType, will be filled with dictionaries'
        pdict.add_context(rec_ctxt)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', pdict, tdom, sdom, in_memory)

        return scov

    def _make_parameter_dict(self):
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.reference_frame = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.reference_frame = AxisTypeEnum.LON
        lon_ctxt.uom = 'degree_east'
        pdict.add_context(lon_ctxt)

        temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
        temp_ctxt.uom = 'degree_Celsius'
        pdict.add_context(temp_ctxt)

        cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
        cond_ctxt.uom = 'unknown'
        pdict.add_context(cond_ctxt)

        return pdict

    def _make_tcrs(self):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        return tcrs

    def _make_scrs(self):
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])
        return scrs

    def _make_tdom(self, tcrs):
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        return tdom

    def _make_sdom(self, scrs):
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
        return sdom

    def _make_samplecov(self, save_coverage=False, in_memory=False):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.reference_frame = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.reference_frame = AxisTypeEnum.LON
        lon_ctxt.uom = 'degree_east'
        pdict.add_context(lon_ctxt)

        temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
        temp_ctxt.uom = 'degree_Celsius'
        pdict.add_context(temp_ctxt)

        cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
        cond_ctxt.uom = 'unknown'
        pdict.add_context(cond_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', pdict, tdom, sdom, in_memory)
        return scov

    def _make_oneparamcov(self, save_coverage=False, in_memory=False):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        bricking_scheme = {'brick_size':1000,'chunk_size':500}
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', pdict, tdom, sdom, in_memory, bricking_scheme=bricking_scheme)

        # Insert some timesteps (automatically expands other arrays)
        nt = 2000
        scov.insert_timesteps(nt)

        # Add data for the parameter
        scov.set_parameter_values('time', value=np.arange(nt))

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(self.working_dir, 'oneparamsample.cov'))

        return scov

    def _make_nospatialcov(self, save_coverage=False, in_memory=False):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)

        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
        quant_ctxt.long_name = 'example of a parameter of type QuantityType'
        quant_ctxt.uom = 'degree_Celsius'
        pdict.add_context(quant_ctxt)

        const_ctxt = ParameterContext('constant', param_type=ConstantType())
        const_ctxt.long_name = 'example of a parameter of type ConstantType'
        pdict.add_context(const_ctxt)

        arr_ctxt = ParameterContext('array', param_type=ArrayType())
        arr_ctxt.long_name = 'example of a parameter of type ArrayType with base_type ndarray (resolves to \'object\')'
        pdict.add_context(arr_ctxt)

        arr2_ctxt = ParameterContext('array2', param_type=ArrayType())
        arr2_ctxt.long_name = 'example of a parameter of type ArrayType with base_type object'
        pdict.add_context(arr2_ctxt)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', pdict, temporal_domain=tdom, in_memory_storage=in_memory)

        # Insert some timesteps (automatically expands other arrays)
        scov.insert_timesteps(10)

        # Add data for each parameter
        scov.set_parameter_values('quantity_time', value=np.arange(10))
        scov.set_parameter_values('quantity', value=np.random.random_sample(10)*(26-23)+23)
        scov.set_parameter_values('constant', value=20)

        arrval = []
        arr2val = []
        for x in xrange(scov.num_timesteps): # One value (which IS an array) for each member of the domain
            arrval.append(np.random.bytes(np.random.randint(1,20)))
            arr2val.append(np.random.random_sample(np.random.randint(1,10)))
        scov.set_parameter_values('array', value=arrval)
        scov.set_parameter_values('array2', value=arr2val)
        scov.get_dirty_values_async_result().get(timeout=30)
        return scov
#
#    def _make_parameter_context(self, name, ptype, dtype, uom):
#        ctxt = ParameterContext(name, type_string)
#
#    def _make_parameter_dictionary(self, params):
#        pdict = ParameterDictionary()
#
#        for param_context in params:
#            pdict.add_context(param_context)
#        return pdict

    def _make_emptysamplecov(self, save_coverage=False, in_memory=False):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.reference_frame = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.reference_frame = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.reference_frame = AxisTypeEnum.LON
        lon_ctxt.uom = 'degree_east'
        pdict.add_context(lon_ctxt)

        temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
        temp_ctxt.uom = 'degree_Celsius'
        pdict.add_context(temp_ctxt)

        cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
        cond_ctxt.uom = 'unknown'
        pdict.add_context(cond_ctxt)

        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'empty sample coverage_model', pdict, tdom, sdom, in_memory)

        return scov
