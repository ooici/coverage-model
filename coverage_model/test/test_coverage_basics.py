#!/usr/bin/env python

"""
@package coverage_model.test.test_coverage
@file coverage_model/test/test_coverage.py
@author James Case
@brief Test cases for the coverage_model module
"""
import random
import os
import shutil
import tempfile

from pyon.public import log

import numpy as np
from coverage_model import *
from nose.plugins.attrib import attr

@attr('INT', group='cov')
class TestCoverageModelBasicsInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    # Loading Coverage Tests
    def test_load_succeeds(self):
        # Creates a valid coverage, inserts data and loads coverage back up from the HDF5 files.
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        pl = scov._persistence_layer
        guid = scov.persistence_guid
        root_path = pl.master_manager.root_dir
        base_path = root_path.replace(guid,'')
        scov.close()
        lcov = SimplexCoverage(base_path, guid)
        self.assertIsInstance(lcov, SimplexCoverage)
        lcov.close()

    def test_dot_load_succeeds(self):
        # Creates a valid coverage, inserts data and .load coverage back up from the HDF5 files.
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
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
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
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
        scov = self._make_samplecov()
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
        scov = self._make_samplecov()
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertTrue(os.path.exists(scov.persistence_dir))
        guid = 'some_incorrect_guid'
        base_path = scov.persistence_dir
        scov.close()
        with self.assertRaises(SystemError) as se:
            SimplexCoverage.load(base_path, guid)
            self.assertEquals(se.message, 'Cannot find specified coverage: {0}'.format(os.path.join(base_path, guid)))

    def test_load_only_pd_raises_error(self):
        scov = self._make_samplecov()
        scov.close()
        with self.assertRaises(TypeError):
            SimplexCoverage(scov.persistence_dir)

    def test_load_options_pd_pg(self):
        scov = self._make_samplecov()
        scov.close()
        cov = SimplexCoverage(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd(self):
        scov = self._make_samplecov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd_pg(self):
        scov = self._make_samplecov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_coverage_mode_expand_domain(self):
        scov = self._make_samplecov()
        self.assertEqual(scov.mode, 'a')
        scov.close()
        rcov = SimplexCoverage.load(self.working_dir, scov.persistence_guid, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov.insert_timesteps(10)

    def test_coverage_mode_set_value(self):
        scov = self._make_samplecov()
        self.assertEqual(scov.mode, 'a')
        scov.insert_timesteps(10)
        scov.close()
        rcov = SimplexCoverage.load(self.working_dir, scov.persistence_guid, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov._range_value.time[0] = 1

    def test_load_succeeds_with_options(self):
        # Tests loading a SimplexCoverage using init parameters
        scov = self._make_samplecov()
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

    def test_coverage_flush(self):
        # Tests that the .flush() function flushes the coverage
        scov = self._make_samplecov()
        scov.flush()
        self.assertTrue(not scov.has_dirty_values())
        scov.close()

    def test_coverage_save(self):
        # Tests that the .save() function flushes coverage
        scov = self._make_samplecov()
        scov.save(scov)
        self.assertTrue(not scov.has_dirty_values())
        scov.close()

    def test_create_multi_bricks(self):
        # Tests creation of multiple (5) bricks
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertTrue(len(scov._persistence_layer.master_manager.brick_list) == 5)

    def test_create_succeeds(self):
        # Tests creation of SimplexCoverage succeeds
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        self.assertIsInstance(tcrs.lat_lon(), CRS)
        self.assertIsInstance(tcrs.lat_lon_height(), CRS)
        self.assertIsInstance(tcrs.x_y_z(), CRS)
        self.assertIsInstance(tcrs.standard_temporal(), CRS)
        self.assertTrue(tcrs.axes == {'TIME': None})
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        self.assertTrue(scrs.axes == {'LAT': None, 'LON': None})
        self.assertTrue(str(scrs) == " ID: None\n Axes: {'LAT': None, 'LON': None}")
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
        # Tests creation of SimplexCoverage fails using an incorrect path
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        in_memory = False
        name = 'sample coverage_model'
        with self.assertRaises(SystemError):
            SimplexCoverage(
                root_dir='bad_dir',
                persistence_guid=create_guid(),
                name=name,
                parameter_dictionary=pdict,
                temporal_domain=tdom,
                spatial_domain=sdom,
                in_memory_storage=in_memory)

    def test_create_guid_valid(self):
        # Tests that the create_guid() function outputs a properly formed GUID
        self.assertTrue(len(create_guid()) == 36)

    def test_create_name_invalid(self):
        # Tests condition where the coverage name is an invalid type
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
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

    def test_create_pdict_invalid(self):
        # Tests condition where the ParameterDictionary is invalid
        pdict = 1 # ParameterDictionary cannot be an int
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
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
                temporal_domain=tdom,
                spatial_domain=sdom,
                in_memory_storage=in_memory)

    def test_create_tdom_invalid(self):
        # Tests condition where the temporal_domain parameter is invalid
        pdict = self._make_parameter_dict()
        scrs = self._make_scrs()
        sdom = self._make_sdom(scrs)
        tdom = 1 # temporal_domain cannot be of type int
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

    def test_create_sdom_invalid(self):
        # Tests condition where the spatial_domain parameter is invalid
        pdict = self._make_parameter_dict()
        tcrs = self._make_tcrs()
        tdom = self._make_tdom(tcrs)
        scrs = self._make_scrs()
        sdom = 1 # spatial_domain cannot be of type int
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
        # Tests temporal_domain expansion and getting and setting values for all parameters
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
        # Tests setting and getting one brick's worth of data for the 'time' parameter
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_one_brick(self):
        # Tests setting and getting one brick's worth of data for all parameters in the coverage
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks(self):
        # Tests setting and getting five brick's worth of data for the 'time' parameter
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_five_bricks(self):
        # Tests setting and getting five brick's worth of data for all parameters
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_one_brick_strided(self):
        # Tests setting and getting one brick's worth of data with a stride of two for the 'time' parameter
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10,2), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks_strided(self):
        # Tests setting and getting five brick's worth of data with a stride of five for the 'time' parameter
        scov = self._make_samplecov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50,5), param='time')
        scov.close()
        self.assertTrue(res)

    def test_ptypescov_create(self):
        # Tests creation of types QuantityType, ConstantType, ArrayType and RecordType
        # TODO: FunctionType not yet supported and has been skipped in the test
        scov = self._make_ptypescov()
        scov.close()
        self.assertIsInstance(scov, SimplexCoverage)

    def test_ptypescov_load_data(self):
        # Tests loading of data into QuantityType, ConstantType, ArrayType and RecordType
        ptypes_cov = self._make_ptypescov()
        ptypes_cov_loaded = self._load_data_ptypescov(ptypes_cov)
        ptypes_cov.close()
        ptypes_cov_loaded.close()
        self.assertEqual(ptypes_cov_loaded.temporal_domain.shape.extents, (2000,))

    def test_ptypescov_get_values(self):
        # Tests retrieval of values from QuantityType, ConstantType,
        # TODO: Implement getting values from FunctionType, ArrayType and RecordType
        results = []
        ptypes_cov = self._make_ptypescov()
        ptypes_cov_loaded = self._load_data_ptypescov(ptypes_cov)
        # QuantityType
        results.append((ptypes_cov_loaded._range_value.quantity_time[:] == np.arange(2000)).any())
        # ConstantType
        results.append(ptypes_cov_loaded._range_value.const_int[0] == 45)
        # ConstantRangeType
        results.append(ptypes_cov_loaded._range_value.const_rng_int[0] == (-10, 10))
        ptypes_cov.close()
        ptypes_cov_loaded.close()
        self.assertTrue(False not in results)

    def test_nospatial_create(self):
        # Tests creation of a SimplexCoverage with only a temporal domain
        scov = self._make_nospatialcov()
        scov.close()
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertEquals(scov.spatial_domain, None)

    def test_emptysamplecov_create(self):
        # Tests creation of SimplexCoverage with zero values in the temporal domain
        scov = self._make_emptysamplecov()
        scov.close()
        self.assertIsInstance(scov, SimplexCoverage)

    def test_close_coverage_before_done_using_it(self):
        # Tests closing a coverage and then attempting to retrieve values.
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        scov.close()
        with self.assertRaises(ValueError):
            scov.get_time_values()

    def test_slice_and_dice(self):
        # Tests for slice and index errors across mutliple bricks
        params, _slices, results, index_errors = self._slice_and_dice(brick_size=1000, time_steps=5000)
        log.debug('slices per parameter: %s', len(_slices))
        log.debug('total slices ran: %s', len(_slices)*len(params))
        log.debug('data failure slices: %s', len(results))
        log.debug('IndexError slices: %s\n%s', len(index_errors), index_errors)
        self.assertTrue(len(results)+len(index_errors) == 0)

    def test_slice_stop_greater_than_size(self):
        # Tests that a slice defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(4998, 5020, None)
        ret = scov.get_parameter_values('temp', _slice)
        self.assertTrue(np.array_equal(ret, np.arange(4998, 5000, dtype=scov.get_parameter_context('temp').param_type.value_encoding)))

    def test_slice_stop_greater_than_size_with_step(self):
        # Tests that a slice (with step) defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(4000, 5020, 5)
        ret = scov.get_parameter_values('temp', _slice)
        self.assertTrue(np.array_equal(ret, np.arange(4000, 5000, 5, dtype=scov.get_parameter_context('temp').param_type.value_encoding)))

    def test_slice_raises_index_error_out_out(self):
        # Tests that an array defined totally outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        _slice = slice(5010, 5020, None)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', _slice)

    def test_int_raises_index_error(self):
        # Tests that an integer defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', 9000)

    def test_array_raises_index_error(self):
        # Tests that an array defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov = self._create_multi_bricks_cov(brick_size, time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('temp', [[5,9000]])

    def test_get_by_slice(self):
        # Tests retrieving data across multiple bricks for a variety of slices
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

    def test_coverage_pickle_and_in_memory(self):
        # Tests creating a SimplexCoverage in memory and saving it to a pickle object
        cov = self._make_oneparamcov(True, True)
        cov.close()
        self.assertTrue(os.path.exists(os.path.join(self.working_dir, 'oneparamsample.cov')))

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

    def test_parameter_properties(self):
        pc = ParameterContext('pcname')
        self.assertEquals(pc.name, 'pcname')
        self.assertFalse(pc.is_coordinate)

    def test_params(self):
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

        log.debug('Should be equal and compare \'one-to-one\' with nothing in the None list')
        self.assertEquals(pdict_1, pdict_2)
        self.assertEquals(pdict_1.compare(pdict_2), {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp'], 'time': ['time']})

        log.debug('Should be unequal and compare with an empty list for \'temp\' and \'temp2\' in the None list')
        self.assertNotEquals(pdict_1, pdict_3)
        self.assertEquals(pdict_1.compare(pdict_3), {'lat': ['lat'], 'lon': ['lon'], None: ['temp2'], 'temp': [], 'time': ['time']})

        log.debug('Should be unequal and compare with both \'temp\' and \'temp2\' in \'temp\' and nothing in the None list')
        self.assertNotEquals(pdict_1,  pdict_4)
        self.assertEquals(pdict_1.compare(pdict_4), {'lat': ['lat'], 'lon': ['lon'], None: [], 'temp': ['temp', 'temp2'], 'time': ['time']})

    def test_append_parameter(self):
        results = []
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')

        parameter_name = 'turbidity'
        pc_in = ParameterContext(parameter_name, param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_in.uom = 'FTU'

        scov.append_parameter(pc_in)

        nt = 50
        scov.set_parameter_values('turbidity', value=np.random.random_sample(nt)*20)

        ret_data = scov.get_parameter_values('turbidity')
        self.assertTrue(np.array_equal(sample_values, ret_data))

        scov.insert_timesteps(100)
        self.assertTrue(len(scov.get_parameter_values('turbidity')) == 150)

        scov.set_parameter_values('turbidity', value=np.arange(150, dtype='f'))
        self.assertTrue(np.array_equal(np.arange(150, dtype='f'), scov.get_parameter_values('turbidity')))

        with self.assertRaises(ValueError):
            scov.append_parameter(pc_in)

    def test_append_parameter_invalid_pc(self):
        scov = self._make_samplecov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        with self.assertRaises(TypeError):
            scov.append_parameter('junk')

    def test_get_all_data_metadata(self):
        scov = self._make_samplecov(in_memory=True, in_line=True, auto_flush=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')
        scov.get_data_bounds(parameter_name='time')
        scov.get_data_bounds_by_axis(axis=AxisTypeEnum.TIME)
        scov.get_data_extents(parameter_name='time')
        scov.get_data_extents_by_axis(axis=AxisTypeEnum.TIME)
        scov.get_data_size(parameter_name='time', slice_=None, in_bytes=False)

    def test_persistence_variation1(self):
        scov = self._make_samplecov(in_memory=False, in_line=False, auto_flush=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='all')

    def test_persistence_variation2(self):
        scov = self._make_samplecov(in_memory=True, in_line=False, auto_flush=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='all')

    def test_persistence_variation3(self):
        scov = self._make_samplecov(in_memory=True, in_line=True, auto_flush=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='all')

    def test_persistence_variation4(self):
        scov = self._make_samplecov(in_memory=False, in_line=True, auto_flush=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='all')

    def test_persistence_variation5(self):
        scov = self._make_samplecov(in_memory=False, in_line=False, auto_flush=True, mode='a')
        cov_info_str = scov.info
        self.assertIsInstance(cov_info_str, basestring)

    def test_run_test_dispatcher(self):
        from coverage_model.brick_dispatch import run_test_dispatcher
        disp=run_test_dispatcher(work_count=20, num_workers=1)
        self.assertTrue(disp.is_single_worker)
        self.assertEquals(disp.num_workers, 1)
        self.assertFalse(disp.has_active_work())
        self.assertFalse(disp.has_pending_work())
        self.assertFalse(disp.has_stashed_work())
        self.assertFalse(disp.is_dirty())
        self.assertTrue(disp.is_single_worker)

    def test_pickle_problems_in_memory(self):
        # Tests saving and loading with both successful and unsuccessful test scenarios
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
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
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', pdict, tdom, sdom, bricking_scheme=bricking_scheme, in_memory_storage=True)

        # Insert some timesteps (automatically expands other arrays)
        nt = 2000
        scov.insert_timesteps(nt)

        # Add data for the parameter
        scov.set_parameter_values('time', value=np.arange(nt))
        pickled_coverage_file = os.path.join(self.working_dir, 'oneparamsample.cov')
        SimplexCoverage.pickle_save(scov, pickled_coverage_file)
        self.assertTrue(os.path.join(self.working_dir, 'oneparamsample.cov'))

        ncov = SimplexCoverage.pickle_load(pickled_coverage_file)
        self.assertIsInstance(ncov, SimplexCoverage)

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_load('some_bad_file_location.cov')

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_save('nat_a_SimplexCoverage', pickled_coverage_file)

    def _slice_and_dice(self, brick_size, time_steps):
        # Tests retrieving data for a variety of slice conditions
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
        # A suite of standard tests to run against a SimplexCoverage
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
        # Constructs SimplexCoverage containing multiple bricks with loaded data
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
        return scov

    def _insert_set_get(self, scov, timesteps, data, _slice, param='all'):
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
            ret = scov.get_parameter_values(param, _slice)
        return (ret == data).all()

    def _load_data_ptypescov(self, scov):
        # Loads data into ptypescov parameters, returns SimplexCoverage

        # Insert some timesteps (automatically expands other arrays)
        nt = 2000
        scov.insert_timesteps(nt)

        # Add data for each parameter
        scov.set_parameter_values('quantity_time', value=np.arange(nt))
        scov.set_parameter_values('const_float', value=-71.11) # Set a constant with correct data type
        scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
        scov.set_parameter_values('const_str', value='constant string value')
        scov.set_parameter_values('quantity', value=np.random.random_sample(nt)*(26-23)+23)
        scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
        scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list

        #    # Setting three range expressions such that indices 0-2 == 10, 3-7 == 15 and >=8 == 20
        #    scov.set_parameter_values('function', value=make_range_expr(10, 0, 3, min_incl=True, max_incl=False, else_val=-999.9))
        #    scov.set_parameter_values('function', value=make_range_expr(15, 3, 8, min_incl=True, max_incl=False, else_val=-999.9))
        #    scov.set_parameter_values('function', value=make_range_expr(20, 8, min_incl=True, max_incl=False, else_val=-999.9))

        cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
        arrval = []
        recval = []
        catval = []
        fstrval = []
        catkeys = cat.keys()
        letts='abcdefghijklmnopqrstuvwxyz'
        while len(letts) < nt:
            letts += 'abcdefghijklmnopqrstuvwxyz'
        for x in xrange(nt):
            arrval.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
            d = {letts[x]: letts[x:]}
            recval.append(d) # One value (which is a dict) for each member of the domain
            catval.append(random.choice(catkeys))
            fstrval.append(''.join([random.choice(letts) for x in xrange(8)])) # A random string of length 8
        scov.set_parameter_values('array', value=arrval)
        scov.set_parameter_values('record', value=recval)
        scov.set_parameter_values('category', value=catval)
        scov.set_parameter_values('fixed_str', value=fstrval)
        scov.get_dirty_values_async_result().get(timeout=60)
        return scov

    def _make_ptypescov(self, save_coverage=False, in_memory=False):
        # Construct SimplexCoverage containing types QuantityType, ConstantType, ArrayType and RecordType
        # TODO: FunctionType not yet implemented

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
        quant_t_ctxt.axis = AxisTypeEnum.TIME
        quant_t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(quant_t_ctxt)

        cnst_flt_ctxt = ParameterContext('const_float', param_type=ConstantType(), variability=VariabilityEnum.NONE)
        cnst_flt_ctxt.long_name = 'example of a parameter of type ConstantType, base_type float (default)'
        cnst_flt_ctxt.axis = AxisTypeEnum.LON
        cnst_flt_ctxt.uom = 'degree_east'
        pdict.add_context(cnst_flt_ctxt)

        cnst_int_ctxt = ParameterContext('const_int', param_type=ConstantType(QuantityType(value_encoding=np.dtype('int32'))), variability=VariabilityEnum.NONE)
        cnst_int_ctxt.long_name = 'example of a parameter of type ConstantType, base_type int32'
        cnst_int_ctxt.axis = AxisTypeEnum.LAT
        cnst_int_ctxt.uom = 'degree_north'
        pdict.add_context(cnst_int_ctxt)

        cnst_str_ctxt = ParameterContext('const_str', param_type=ConstantType(QuantityType(value_encoding=np.dtype('S21'))), fill_value='', variability=VariabilityEnum.NONE)
        cnst_str_ctxt.long_name = 'example of a parameter of type ConstantType, base_type fixed-len string'
        pdict.add_context(cnst_str_ctxt)

        cnst_rng_flt_ctxt = ParameterContext('const_rng_flt', param_type=ConstantRangeType(), variability=VariabilityEnum.NONE)
        cnst_rng_flt_ctxt.long_name = 'example of a parameter of type ConstantRangeType, base_type float (default)'
        pdict.add_context(cnst_rng_flt_ctxt)

        cnst_rng_int_ctxt = ParameterContext('const_rng_int', param_type=ConstantRangeType(QuantityType(value_encoding='int16')), variability=VariabilityEnum.NONE)
        cnst_rng_int_ctxt.long_name = 'example of a parameter of type ConstantRangeType, base_type int16'
        pdict.add_context(cnst_rng_int_ctxt)

        cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
        cat_ctxt = ParameterContext('category', param_type=CategoryType(categories=cat), variability=VariabilityEnum.TEMPORAL)
        pdict.add_context(cat_ctxt)

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

        fstr_ctxt = ParameterContext('fixed_str', param_type=QuantityType(value_encoding=np.dtype('S8')), fill_value='')
        fstr_ctxt.long_name = 'example of a fixed-length string parameter'
        pdict.add_context(fstr_ctxt)

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, in_memory_storage=in_memory)

        return scov

    def _make_parameter_dict(self):
        # Construct ParameterDictionary of various QuantityTypes
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.axis = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.axis = AxisTypeEnum.LON
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
        # Construct temporal Coordinate Reference System object
        tcrs = CRS([AxisTypeEnum.TIME])
        return tcrs

    def _make_scrs(self):
        # Construct spatial Coordinate Reference System object
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])
        return scrs

    def _make_tdom(self, tcrs):
        # Construct temporal domain object
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        return tdom

    def _make_sdom(self, scrs):
        # Create spatial domain object
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
        return sdom

    def _make_samplecov(self, in_memory=False, in_line=False, auto_flush=True, mode=None):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.axis = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.axis = AxisTypeEnum.LON
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
        #scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, in_memory_storage=in_memory)
        scov = SimplexCoverage(self.working_dir, create_guid(), name='sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, mode=mode, in_memory_storage=in_memory, bricking_scheme=None, inline_data_writes=in_line, auto_flush_values=auto_flush)
        return scov

    def _make_oneparamcov(self, save_coverage=False, in_memory=False):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
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
        scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

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
        t_ctxt.axis = AxisTypeEnum.TIME
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
        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, mode='a', in_memory_storage=in_memory)

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
        scov.get_dirty_values_async_result().get(timeout=60)
        return scov

    def _make_emptysamplecov(self, save_coverage=False, in_memory=False):
        # Instantiate a ParameterDictionary
        pdict = ParameterDictionary()

        # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
        t_ctxt.axis = AxisTypeEnum.TIME
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt)

        lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lat_ctxt.axis = AxisTypeEnum.LAT
        lat_ctxt.uom = 'degree_north'
        pdict.add_context(lat_ctxt)

        lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
        lon_ctxt.axis = AxisTypeEnum.LON
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
        scov = SimplexCoverage(self.working_dir, create_guid(), 'empty sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, mode='a', in_memory_storage=in_memory)

        return scov
