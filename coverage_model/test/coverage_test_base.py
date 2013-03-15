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

    def setUp(self):
        pass

    def get_cov(self, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        raise NotImplementedError()

    def _insert_set_get(self, scov=None, timesteps=None, data=None, _slice=None, param='all'):
        raise NotImplementedError()

    # ############################
    # METADATA
    def test_get_all_data_metadata(self):
        scov, cov_name = self.get_cov(only_time=True, in_memory=True, inline_data_writes=True, auto_flush_values=True, nt=5000)
        # self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')
        res = scov.get_data_bounds(parameter_name='time')
        self.assertEqual(res, (0, 4999))
        res = scov.get_data_bounds_by_axis(axis=AxisTypeEnum.TIME)
        self.assertEqual(res, (0, 4999))
        res = scov.get_data_extents(parameter_name='time')
        self.assertEqual(res, (5000,))
        res = scov.get_data_extents_by_axis(axis=AxisTypeEnum.TIME)
        self.assertEqual(res, (5000,))
        res = scov.get_data_size(parameter_name='time', slice_=None, in_bytes=False)
        self.assertEqual(res, 0.03814696)

    # ############################
    # CONSTRUCTION
    def test_create_cov(self):
        time_steps = 30
        cov, cov_name = self.get_cov(nt=time_steps)
        self.assertIsInstance(cov, SimplexCoverage)
        cov_info_str = cov.info
        self.assertIsInstance(cov_info_str, basestring)
        self.assertEqual(cov.name, 'sample coverage_model')

        if cov_name not in OMIT_TEST_NO_DATA:
            self.assertEqual(cov.num_timesteps, time_steps)
            self.assertEqual(list(cov.temporal_domain.shape.extents), [time_steps])

        params = cov.list_parameters()
        for param in params:
            pc = cov.get_parameter_context(param)
            self.assertEqual(len(pc.dom.identifier), 36)

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

    def test_create_multi_bricks(self):
        # Tests creation of multiple (5) bricks
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        self.assertIsInstance(scov, SimplexCoverage)

        if scov.num_timesteps != time_steps:
            log.warn('Must be an empty coverage!')
        else:
            self.assertTrue(len(scov._persistence_layer.master_manager.brick_list) == 5)

    def test_create_dir_not_exists(self):
        # Tests creation of SimplexCoverage fails using an incorrect path
        pdict = get_parameter_dict()
        tcrs = _make_tcrs()
        tdom = _make_tdom(tcrs)
        scrs = _make_scrs()
        sdom = _make_sdom(scrs)
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

    def test_create_pdict_invalid(self):
        # Tests condition where the ParameterDictionary is invalid
        pdict = 1 # ParameterDictionary cannot be an int
        tcrs = _make_tcrs()
        tdom = _make_tdom(tcrs)
        scrs = _make_scrs()
        sdom = _make_sdom(scrs)
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
        pdict = get_parameter_dict()
        scrs = _make_scrs()
        sdom = _make_sdom(scrs)
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
        pdict = get_parameter_dict()
        tcrs = _make_tcrs()
        tdom = _make_tdom(tcrs)
        scrs = _make_scrs()
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

    def test_close_coverage_before_done_using_it(self):
        # Tests closing a coverage and then attempting to retrieve values.
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        scov.close()
        with self.assertRaises(ValueError):
            scov.get_time_values()


    # ############################
    # LOADING
    def test_load_succeeds(self):
        # Creates a valid coverage, inserts data and loads coverage back up from the HDF5 files.
        scov, cov_name = self.get_cov()
        self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
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
        scov, cov_name = self.get_cov()
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
        scov, cov_name = self.get_cov(nt=50)
        # self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
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
        scov, cov_name = self.get_cov()
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
        scov, cov_name = self.get_cov()
        self.assertIsInstance(scov, SimplexCoverage)
        self.assertTrue(os.path.exists(scov.persistence_dir))
        guid = 'some_incorrect_guid'
        base_path = scov.persistence_dir
        scov.close()
        with self.assertRaises(SystemError) as se:
            SimplexCoverage.load(base_path, guid)
            self.assertEquals(se.message, 'Cannot find specified coverage: {0}'.format(os.path.join(base_path, guid)))

    def test_load_only_pd_raises_error(self):
        scov, cov_name = self.get_cov()
        scov.close()
        with self.assertRaises(TypeError):
            SimplexCoverage(scov.persistence_dir)

    def test_load_options_pd_pg(self):
        scov, cov_name = self.get_cov()
        scov.close()
        cov = SimplexCoverage(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd(self):
        scov, cov_name = self.get_cov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_dot_load_options_pd_pg(self):
        scov, cov_name = self.get_cov()
        scov.close()
        cov = SimplexCoverage.load(scov.persistence_dir, scov.persistence_guid)
        self.assertIsInstance(cov, SimplexCoverage)
        cov.close()

    def test_load_succeeds_with_options(self):
        # Tests loading a SimplexCoverage using init parameters
        scov, cov_name = self.get_cov()
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
        scov, cov_name = self.get_cov()
        self.assertEqual(scov.mode, 'a')
        scov.close()
        rcov = SimplexCoverage.load(scov.persistence_dir, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov.insert_timesteps(10)

    def test_coverage_mode_set_value(self):
        scov, cov_name = self.get_cov()
        self.assertEqual(scov.mode, 'a')
        scov.insert_timesteps(10)
        scov.close()
        rcov = SimplexCoverage.load(scov.persistence_dir, mode='r')
        self.assertEqual(rcov.mode, 'r')
        with self.assertRaises(IOError):
            rcov._range_value.time[0] = 1

    def test_persistence_variation1(self):
        scov, cov_name = self.get_cov(only_time=True, in_memory=False, inline_data_writes=False, auto_flush_values=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')

    def test_persistence_variation2(self):
        scov, cov_name = self.get_cov(only_time=True, in_memory=True, inline_data_writes=False, auto_flush_values=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')

    def test_persistence_variation3(self):
        scov, cov_name = self.get_cov(only_time=True, in_memory=True, inline_data_writes=True, auto_flush_values=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')

    def test_persistence_variation4(self):
        scov, cov_name = self.get_cov(only_time=True, in_memory=False, inline_data_writes=True, auto_flush_values=True)
        res = self._insert_set_get(scov=scov, timesteps=5000, data=np.arange(5000), _slice=slice(0,5000), param='time')

    # ############################
    # GET
    def test_ptypescov_get_values(self):
        pass

    # def test_slice_and_dice(self):
    #     # Tests for slice and index errors across multiple bricks
    #     time_steps = 5000
    #     scov, cov_name = self.get_cov(brick_size=1000, nt=time_steps)
    #     if cov_name not in OMIT_TEST_NO_DATA:
    #         params, _slices, results, index_errors = _slice_and_dice(scov)
    #         log.debug('slices per parameter: %s', len(_slices))
    #         log.debug('total slices ran: %s', len(_slices) * len(params))
    #         log.debug('data failure slices: %s', len(results))
    #         log.debug('IndexError slices: %s\n%s', len(index_errors), index_errors)
    #         scov.close()
    #         self.assertTrue(len(results) + len(index_errors) == 0)

    def test_slice_stop_greater_than_size(self):
        # Tests that a slice defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        if scov.num_timesteps > 0:
            _slice = slice(4998, 5020, None)
            ret = scov.get_parameter_values('time', _slice)
            self.assertTrue(np.array_equal(ret, np.arange(4998, 5000, dtype=scov.get_parameter_context('time').param_type.value_encoding)))

    def test_slice_stop_greater_than_size_with_step(self):
        # Tests that a slice (with step) defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        if cov_name not in OMIT_TEST_NO_DATA:
            _slice = slice(4000, 5020, 5)
            ret = scov.get_parameter_values('time', _slice)
            self.assertTrue(np.array_equal(ret, np.arange(4000, 5000, 5, dtype=scov.get_parameter_context('time').param_type.value_encoding)))

    def test_slice_raises_index_error_out_out(self):
        # Tests that an array defined totally outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        _slice = slice(5010, 5020, None)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('time', _slice)

    def test_int_raises_index_error(self):
        # Tests that an integer defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('time', 9000)

    def test_array_raises_index_error(self):
        # Tests that an array defined outside the coverage data bounds raises an error when attempting retrieval
        brick_size = 1000
        time_steps = 5000
        scov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
        with self.assertRaises(IndexError):
            scov.get_parameter_values('time', [[5,9000]])

    def test_get_by_slice(self):
        # Tests retrieving data across multiple bricks for a variety of slices
        results = []
        brick_size = 10
        time_steps = 30
        cov, cov_name = self.get_cov(brick_size=brick_size, nt=time_steps)
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

    # ############################
    # SET
    def test_samplecov_time_one_brick(self):
        # Tests setting and getting one brick's worth of data for the 'time' parameter
        scov, cov_name = self.get_cov(only_time=True)
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_one_brick(self):
        # Tests setting and getting one brick's worth of data for all parameters in the coverage
        scov, cov_name = self.get_cov()
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks(self):
        # Tests setting and getting five brick's worth of data for the 'time' parameter
        scov, cov_name = self.get_cov(only_time=True)
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_allparams_five_bricks(self):
        # Tests setting and getting five brick's worth of data for all parameters
        scov, cov_name = self.get_cov()
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50), param='all')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_one_brick_strided(self):
        # Tests setting and getting one brick's worth of data with a stride of two for the 'time' parameter
        scov, cov_name = self.get_cov(only_time=True)
        res = self._insert_set_get(scov=scov, timesteps=10, data=np.arange(10), _slice=slice(0,10,2), param='time')
        scov.close()
        self.assertTrue(res)

    def test_samplecov_time_five_bricks_strided(self):
        # Tests setting and getting five brick's worth of data with a stride of five for the 'time' parameter
        scov, cov_name = self.get_cov(only_time=True)
        res = self._insert_set_get(scov=scov, timesteps=50, data=np.arange(50), _slice=slice(0,50,5), param='time')
        scov.close()
        self.assertTrue(res)

    # ############################
    # INLINE & OUT OF BAND R/W
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

    # ############################
    # CACHING

    # ############################
    # ERRORS

    # ############################
    # SAVE
    def test_coverage_flush(self):
        # Tests that the .flush() function flushes the coverage
        scov, cov_name = self.get_cov()
        scov.flush()
        self.assertTrue(not scov.has_dirty_values())
        scov.close()

    def test_coverage_save(self):
        # Tests that the .save() function flushes coverage
        scov, cov_name = self.get_cov()
        scov.save(scov)
        self.assertTrue(not scov.has_dirty_values())
        scov.close()

    def test_coverage_pickle_and_in_memory(self):
        # Tests creating a SimplexCoverage in memory and saving it to a pickle object
        cov, cov_name = self.get_cov(only_time=True, in_memory=True, save_coverage=True, nt=2000)
        cov.close()
        self.assertTrue(os.path.exists(os.path.join(self.working_dir, 'sample.cov')))

    def test_pickle_problems_in_memory(self):
        # Tests saving and loading with both successful and unsuccessful test scenarios
        nt = 2000
        scov, cov_name = self.get_cov(only_time=True, brick_size=1000, in_memory=True, nt=nt)

        # Add data for the parameter
        #TODO: This gets repeated, create separate function
        if scov.num_timesteps != nt:
            log.warn('Must be an empty coverage!')
        else:
            scov.set_parameter_values('time', value=np.arange(nt))

        pickled_coverage_file = os.path.join(self.working_dir, 'sample.cov')
        SimplexCoverage.pickle_save(scov, pickled_coverage_file)
        self.assertTrue(os.path.join(self.working_dir, 'sample.cov'))

        ncov = SimplexCoverage.pickle_load(pickled_coverage_file)
        self.assertIsInstance(ncov, SimplexCoverage)

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_load('some_bad_file_location.cov')

        with self.assertRaises(StandardError):
            SimplexCoverage.pickle_save('nat_a_SimplexCoverage', pickled_coverage_file)

    # ############################
    # PARAMETERS
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

    def test_get_parameter(self):
        cov, cov_name = self.get_cov()
        param = cov.get_parameter('time')
        self.assertEqual(param.name, 'time')

        cov.close()
        with self.assertRaises(ValueError):
            cov.get_parameter('time')

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
        scov, cov_name = self.get_cov(inline_data_writes=True, nt=50)

        parameter_name = 'turbidity'
        pc_in = ParameterContext(parameter_name, param_type=QuantityType(value_encoding=np.dtype('float32')))
        pc_in.uom = 'FTU'

        scov.append_parameter(pc_in)

        nt = 50
        sample_values = np.arange(nt, dtype='f')
        scov.set_parameter_values('turbidity', value=sample_values)

        ret_data = scov.get_parameter_values('turbidity')
        self.assertTrue(np.array_equal(sample_values, ret_data))

        scov.insert_timesteps(100)
        self.assertTrue(len(scov.get_parameter_values('turbidity')) == 150)

        scov.set_parameter_values('turbidity', value=np.arange(150, dtype='f'))
        self.assertTrue(np.array_equal(np.arange(150, dtype='f'), scov.get_parameter_values('turbidity')))

        with self.assertRaises(ValueError):
            scov.append_parameter(pc_in)

    def test_append_parameter_invalid_pc(self):
        scov, cov_name = self.get_cov(only_time=True, nt=50)
        with self.assertRaises(TypeError):
            scov.append_parameter('junk')


def get_parameter_dict_info():
    pdict_info = {}

    for pname in MASTER_PDICT:
        p = MASTER_PDICT.get_context(pname)
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

OMIT_TEST_NO_DATA = ['TestEmptySampleCovInt']
EXEMPLAR_CATEGORIES = {0:'turkey',1:'duck',2:'chicken',99:'None'}

def _make_master_parameter_dict():
    # Construct ParameterDictionary of all supported types
    # TODO: Ensure all description attributes are filled in properly
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
    cnst_rng_int_ctxt.description = ''

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
    ctxt = ParameterContext('tempwat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'deg_C'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Conductivity - values expected to be the decimal results of conversion from hex
    ctxt = ParameterContext('condwat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Pressure - values expected to be the decimal results of conversion from hex
    ctxt = ParameterContext('preswat_l0', param_type=QuantityType(value_encoding=np.dtype('float32')), fill_value=-9999)
    ctxt.uom = 'dbar'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # tempwat_l1 = (tempwat_l0 / 10000) - 10
    tl1_func = '(T / 10000) - 10'
    tl1_pmap = {'T': 'tempwat_l0'}
    expr = NumexprFunction('tempwat_l1', tl1_func, ['T'], param_map=tl1_pmap)
    ctxt = ParameterContext('tempwat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'deg_C'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # condwat_l1 = (condwat_l0 / 100000) - 0.5
    cl1_func = '(C / 100000) - 0.5'
    cl1_pmap = {'C': 'condwat_l0'}
    expr = NumexprFunction('condwat_l1', cl1_func, ['C'], param_map=cl1_pmap)
    ctxt = ParameterContext('condwat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Equation uses p_range, which is a calibration coefficient - Fixing to 679.34040721
    #   preswat_l1 = (preswat_l0 * p_range / (0.85 * 65536)) - (0.05 * p_range)
    pl1_func = '(P * p_range / (0.85 * 65536)) - (0.05 * p_range)'
    pl1_pmap = {'P': 'preswat_l0', 'p_range': 679.34040721}
    expr = NumexprFunction('preswat_l1', pl1_func, ['P', 'p_range'], param_map=pl1_pmap)
    ctxt = ParameterContext('preswat_l1', param_type=ParameterFunctionType(function=expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'S m-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # Density & practical salinity calucluated using the Gibbs Seawater library - available via python-gsw project:
    #       https://code.google.com/p/python-gsw/ & http://pypi.python.org/pypi/gsw/3.0.1

    # pracsal = gsw.SP_from_C((condwat_l1 * 10), tempwat_l1, preswat_l1)
    owner = 'gsw'
    sal_func = 'SP_from_C'
    sal_arglist = ['C', 't', 'p']
    sal_pmap = {'C': NumexprFunction('condwat_l1*10', 'C*10', ['C'], param_map={'C': 'condwat_l1'}), 't': 'tempwat_l1', 'p': 'preswat_l1'}
    sal_kwargmap = None
    expr = PythonFunction('pracsal', owner, sal_func, sal_arglist, sal_kwargmap, sal_pmap)
    ctxt = ParameterContext('pracsal', param_type=ParameterFunctionType(expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'g kg-1'
    ctxt.description = ''
    pdict.add_context(ctxt)

    # absolute_salinity = gsw.SA_from_SP(pracsal, preswat_l1, longitude, latitude)
    # conservative_temperature = gsw.CT_from_t(absolute_salinity, tempwat_l1, preswat_l1)
    # density = gsw.rho(absolute_salinity, conservative_temperature, preswat_l1)
    owner = 'gsw'
    abs_sal_expr = PythonFunction('abs_sal', owner, 'SA_from_SP', ['pracsal', 'preswat_l1', 'LON','LAT'])
    cons_temp_expr = PythonFunction('cons_temp', owner, 'CT_from_t', [abs_sal_expr, 'tempwat_l1', 'preswat_l1'])
    dens_expr = PythonFunction('density', owner, 'rho', [abs_sal_expr, cons_temp_expr, 'preswat_l1'])
    ctxt = ParameterContext('density', param_type=ParameterFunctionType(dens_expr), variability=VariabilityEnum.TEMPORAL)
    ctxt.uom = 'kg m-3'
    ctxt.description = ''
    pdict.add_context(ctxt)

    return pdict

MASTER_PDICT = _make_master_parameter_dict()

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

# def _slice_and_dice(scov):
#     # Tests retrieving data for a variety of slice conditions
#     results = []
#     index_errors = []
#     params = scov.list_parameters()
#     _slices = []
#     # TODO: Automatically calculate the start, stops and strides based on the brick size and time_steps
#     starts = [0, 1, 10, 500, 1000, 1001, 3000, 4999]
#     stops = [1, 2, 11, 501, 1001, 1002, 3001, 5000]
#     strides = [None, 1, 2, 3, 4, 5, 50, 100, 500, 750, 999, 1000, 1001, 1249, 1250, 1500, 2000, 3000, 4000, 5000]
#     for stride in strides:
#         for start in starts:
#             for stop in stops:
#                 if stop > start and (stop-start) > stride:
#                     _slices.append(slice(start, stop, stride))
#     for param in params:
#         for _slice in _slices:
#             log.trace('working on _slice: %s', _slice)
#             sliced_data = np.arange(5000)[_slice]
#             try:
#                 ret = scov.get_parameter_values(param, _slice)
#                 if not (ret == sliced_data).all():
#                     results.append(_slice)
#                     log.trace('failed _slice: %s', _slice)
#             except IndexError as ie:
#                 log.trace('%s; moving to next slice', ie.message)
#                 index_errors.append(_slice)
#                 continue
#     scov.close()
#     return params, _slices, results, index_errors


@attr('INT', group='cov')
class TestSupportingCoverageObjectsInt(CoverageModelIntTestCase):

    def setUp(self):
        pass

    def test_pdict_helper(self):
        pdict = get_parameter_dict()
        self.assertEqual(MASTER_PDICT.keys(), pdict.keys())

        pname_filter = ['time','conductivity','tempwat_l0']
        pdict = get_parameter_dict(parameter_list=pname_filter)
        self.assertIsInstance(pdict, ParameterDictionary)
        self.assertEqual(pname_filter, pdict.keys())

    def test_create_supporting_objects_succeeds(self):
        # Tests creation of SimplexCoverage succeeds
        pdict = get_parameter_dict()
        self.assertIsInstance(pdict, ParameterDictionary)
        tcrs = _make_tcrs()
        self.assertIsInstance(tcrs.lat_lon(), CRS)
        self.assertIsInstance(tcrs.lat_lon_height(), CRS)
        self.assertIsInstance(tcrs.x_y_z(), CRS)
        self.assertIsInstance(tcrs.standard_temporal(), CRS)
        self.assertTrue(tcrs.axes == {'TIME': None})
        tdom = _make_tdom(tcrs)
        self.assertIsInstance(tdom, GridDomain)
        scrs = _make_scrs()
        self.assertTrue(scrs.axes == {'LAT': None, 'LON': None})
        self.assertTrue(str(scrs) == " ID: None\n Axes: {'LAT': None, 'LON': None}")
        sdom = _make_sdom(scrs)
        self.assertIsInstance(sdom, GridDomain)