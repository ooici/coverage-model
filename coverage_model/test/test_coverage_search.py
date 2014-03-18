__author__ = 'casey'


from nose.plugins.attrib import attr
import numpy as np
import os, shutil, tempfile
import unittest
import shutil, tempfile
from pyon.core.bootstrap import CFG
from pyon.datastore.datastore_common import DatastoreFactory
import psycopg2
import psycopg2.extras
from coverage_model import *
from coverage_model.address import *
from coverage_model.search.coverage_search import *
from coverage_model.search.search_parameter import *
from coverage_model.search.search_constants import *

from coverage_test_base import CoverageIntTestBase, get_props, get_parameter_dict, EXEMPLAR_CATEGORIES


@attr('INT',group='cov')
class TestCoverageSearchInt(CoverageModelUnitTestCase):
    working_dir = os.path.join(tempfile.gettempdir(), 'cov_mdl_tests')
    coverages = set()

    @classmethod
    def setUpClass(cls):
        if os.path.exists(cls.working_dir):
            shutil.rmtree(cls.working_dir)

        os.mkdir(cls.working_dir)

    @classmethod
    def tearDownClass(cls):
        # Removes temporary files
        # Comment this out if you need to inspect the HDF5 files.
        shutil.rmtree(cls.working_dir)
        if len(cls.coverages) > 0:
            try:
                span_store = DatastoreFactory.get_datastore(datastore_name='coverage_spans', config=CFG)
                coverage_store = DatastoreFactory.get_datastore(datastore_name='coverage', config=CFG)
                if span_store is None:
                    raise RuntimeError("Unable to load datastore for coverage_spans")
                if coverage_store is None:
                    raise RuntimeError("Unable to load datastore for coverages")
                for guid in cls.coverages:
                    with span_store.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                        cur.execute("DELETE FROM %s WHERE coverage_id='%s'" % (span_store._get_datastore_name(), guid))
                        cur.execute("DELETE FROM %s WHERE id='%s'" % (coverage_store._get_datastore_name(), guid))
            except:
                pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def create_cov(cls, only_time=False, in_memory=False, inline_data_writes=True, brick_size=None, auto_flush_values=True):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        pname_filter = ['time',
                            'boolean',
                            'const_int',
                            'const_rng_flt',
                            'const_rng_int',
                            'numexpr_func',
                            'category',
                            'record',
                            'sparse',
                            'lat',
                            'lon',
                            'depth',
                            'salinity']

        if only_time:
            pname_filter = ['time']

        pdict = get_parameter_dict(parameter_list=pname_filter)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        cov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)
        cls.coverages.add(cov.persistence_guid)
        return cov

    @classmethod
    def construct_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        scov = cls.create_cov(only_time=only_time, in_memory=in_memory, inline_data_writes=inline_data_writes, brick_size=brick_size, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestTestSpanUnit'
        else:
            # Add data for each parameter
            if only_time:
                scov.insert_timesteps(nt)
                scov.set_parameter_values('time', value=np.arange(1, nt))
            else:
                scov.set_parameter_values('sparse', [[[2, 4, 6], [8, 10, 12]]])
                scov.insert_timesteps(nt/2)

                scov.set_parameter_values('sparse', [[[4, 8], [16, 20]]])
                scov.insert_timesteps(nt/2)

                scov.set_parameter_values('time', value=100 * np.random.random_sample(nt))
                scov.append_parameter(ParameterContext('depth'))
                scov.set_parameter_values('depth', value=1000 * np.random.random_sample(nt))
                scov.append_parameter(ParameterContext('salinity'))
                scov.set_parameter_values('salinity', value=1000 * np.random.random_sample(nt))

                scov.append_parameter(ParameterContext('m_lon'))
                scov.append_parameter(ParameterContext('m_lat'))
                scov.set_parameter_values('m_lon', value=160 * np.random.random_sample(nt))
                scov.set_parameter_values('m_lat', value=70 * np.random.random_sample(nt))

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        return scov, 'TestCoverageInt'

    @unittest.skipIf(False, 'Not Automated. Requires setup.')
    def test_coverage_conversion(self):
        cov_id = '1fd8b69e63744f9e9705fbfd003102fa'
        cov_dir = '/Users/casey/Desktop/datasets'
        cov = AbstractCoverage.load(cov_dir, cov_id)
        self.assertIsNotNone(cov)
        mm = cov._persistence_layer.master_manager
        self.assertIsNotNone(mm)

        search = CoverageSearch(SearchCriteria([ParamValue(IndexedParameters.CoverageId, cov_id)]))
        retrieved_cov = search.find(cov_id, cov_dir)
        self.assertIsNotNone(retrieved_cov)

        new_mm = retrieved_cov._persistence_layer.master_manager
        self.assertNotEqual(id(mm), id(new_mm))
        self.assertEqual(0, len(mm.hdf_conversion_key_diffs(new_mm)))

    def test_calculate_statistics(self):
        scov, cov_name = self.construct_cov(nt=100)
        self.assertIsNotNone(scov)

        stats, not_calculated = scov.calculate_statistics()

    def test_coverage_creation(self):
        scov, cov_name = self.construct_cov(nt=100)
        self.assertIsNotNone(scov)

        mm = scov._persistence_layer.master_manager
        time_param = ParamValueRange(IndexedParameters.Time, (1, 100))
        coverage_id_param = ParamValue(IndexedParameters.CoverageId, mm.guid)
        criteria = SearchCriteria([time_param, coverage_id_param])
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        self.assertIsNotNone(results)
        self.assertIn( mm.guid, results.get_found_coverage_ids())
        cov = results.get_view_coverage(mm.guid, self.working_dir)
        self.assertIsNotNone(cov)
        npa = cov.get_observations(start_index=10, end_index=25, order_by=['time'])
        self.assertEqual(len(npa), 15)
        index = 0
        for field in npa.dtype.names:
            if field == 'time':
                break;
            index = index + 1
        last = -1000
        for tup in npa:
            current = tup[index]
            self.assertGreaterEqual(current, last)
            last = current

    def test_search_by_id(self):
        scov, cov_name = self.construct_cov(nt=100)
        self.assertIsNotNone(scov)

        mm = scov._persistence_layer.master_manager
        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(ParamValue(IndexedParameters.CoverageId, mm.guid))
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        self.assertIsNotNone(results)
        self.assertIn( mm.guid, results.get_found_coverage_ids())

        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(ParamValueRange(IndexedParameters.CoverageId, (mm.guid, "hello")))
        search = CoverageSearch(criteria, order_by=['time'])
        self.assertRaises(TypeError, search.select)

        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(ParamValue(IndexedParameters.CoverageId, 10))
        search = CoverageSearch(criteria, order_by=['time'])
        self.assertRaises(TypeError, search.select)

    def test_search_by_depth(self):
        scov, cov_name = self.construct_cov(nt=100)
        self.assertIsNotNone(scov)

        mm = scov._persistence_layer.master_manager
        if hasattr(mm, 'span_collection'):
            for k, span in mm.span_collection.span_dict.iteritems():
                if 'time' in span.params:
                    time_min, time_max = span.params['time']
                if 'depth' in span.params:
                    lat_min, lat_max = span.params['depth']

        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(ParamValueRange(IndexedParameters.Vertical, (time_min, time_max)))
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        self.assertIsNotNone(results)
        self.assertIn( mm.guid, results.get_found_coverage_ids())

        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(ParamValue(IndexedParameters.CoverageId, (time_min+time_max)/2))
        search = CoverageSearch(criteria, order_by=['time'])
        self.assertIsNotNone(results)
        self.assertIn( mm.guid, results.get_found_coverage_ids())

        criteria = SearchCriteria((ParamValueRange(IndexedParameters.Time, (1, 100))))
        criteria.append(Param2DValueRange(IndexedParameters.CoverageId, ((10,15), (time_min, time_max))))
        search = CoverageSearch(criteria, order_by=['time'])
        self.assertRaises(TypeError, search.select)

    def test_data_arrives_out_of_order(self):
        cov = self.create_cov()

        nt = 10
        cov.insert_timesteps(nt)
        cov.set_parameter_values('time', value=np.arange(21, 31))
        cov.append_parameter(ParameterContext('depth'))
        cov.set_parameter_values('depth', value=200 * np.random.random_sample(nt))
        id_ = cov.persistence_guid
        cov.close()

        cov2 = AbstractCoverage.load(self.working_dir, id_, mode='w')
        self.assertIsNotNone(cov2)

        cov2.insert_timesteps(10)
        cov2.set_parameter_values('time', value=np.arange(11, 21), tdoa=slice(10,None))
        cov2.set_parameter_values('depth', value=200 * np.random.random_sample(nt), tdoa=slice(10,None))

        cov2.insert_timesteps(10)
        cov2.set_parameter_values('time', value=np.arange(1, 11), tdoa=slice(20,30))
        cov2.set_parameter_values('depth', value=200 * np.random.random_sample(nt), tdoa=slice(20,30))

        time_param = ParamValueRange(IndexedParameters.Time, (1, 100))
        coverage_id_param = ParamValue(IndexedParameters.CoverageId, id_)
        criteria = SearchCriteria([time_param, coverage_id_param])
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        self.assertIsNotNone(results)
        self.assertIn(id_, results.get_found_coverage_ids())
        cov = results.get_view_coverage(id_, self.working_dir)
        self.assertIsNotNone(cov)
        npa = cov.get_observations()
        self.assertEqual(len(npa), 30)

        index = 0
        for field in npa.dtype.names:
            if field == 'time':
                break;
            index = index + 1
        last = -1000
        for tup in npa:
            current = tup[index]
            self.assertGreaterEqual(current, last)
            last = current

    def test_concurrent_read_and_write(self):
        cov1, cov_name = self.construct_cov(nt=100)
        self.assertIsNotNone(cov1)
        id1 = cov1.persistence_guid
        cov1_dict = cov1.get_value_dictionary()

        cov2, cov_name = self.construct_cov(nt=50)
        self.assertIsNotNone(cov2)
        id2 = cov2.persistence_guid
        cov2_dict = cov2.get_value_dictionary()

        cov3, cov_name = self.construct_cov(nt=20)
        self.assertIsNotNone(cov3)
        id3 = cov3.persistence_guid
        cov3_dict = cov3.get_value_dictionary()

        for cov in [cov1, cov2, cov3]:
            cov.close()

        def check_equal_param_dicts(p_dict_1, p_dict_2):
            self.assertEqual(p_dict_1.keys(), p_dict_2.keys())
            for key in p_dict_1.keys():
                self.assertTrue(np.array_equal(p_dict_1[key], p_dict_2[key]))

        cov1r = AbstractCoverage.load(self.working_dir, id1)
        cov2r = AbstractCoverage.load(self.working_dir, id2)
        cov3r = AbstractCoverage.load(self.working_dir, id3)
        self.assertIsNotNone(cov1r)
        self.assertIsNotNone(cov2r)
        self.assertIsNotNone(cov3r)
        for cov_tup in [(cov1_dict, cov1r), (cov2_dict, cov2r), (cov3_dict, cov3r)]:
            check_equal_param_dicts(cov_tup[0], cov_tup[1].get_value_dictionary())

        cov1a = AbstractCoverage.load(self.working_dir, id1)
        cov2a = AbstractCoverage.load(self.working_dir, id2)
        cov3a = AbstractCoverage.load(self.working_dir, id3)
        self.assertIsNotNone(cov1a)
        self.assertIsNotNone(cov2a)
        self.assertIsNotNone(cov3a)
        for cov_tup in [(cov1_dict, cov1a), (cov2_dict, cov2a), (cov3_dict, cov3a)]:
            check_equal_param_dicts(cov_tup[0], cov_tup[1].get_value_dictionary())

        from multiprocessing import Process, Queue, queues

        def read_cov_and_evaluate_values(dir_, id_, value_dict, q, slp):
            cov_ = None
            message = id_
            try:
                cov_ = AbstractCoverage.load(dir_, id_)
            except:
                message = 'Caught exception'
            if cov_ is None and message == id_:
                message = 'Construction failed'
            elif cov_ is not None:
                try:
                    check_equal_param_dicts(value_dict, cov_.get_value_dictionary())
                except:
                    message = 'Values inconsistent'
            q.put(message)
            import time
            time.sleep(slp)


        threads = []
        q = Queue()
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id1, cov1_dict, q, 5)))
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id1, cov1_dict, q, 4)))
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id2, cov2_dict, q, 3)))
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id2, cov2_dict, q, 2)))
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id3, cov3_dict, q, 1)))
        threads.append(Process(target=read_cov_and_evaluate_values, args=(self.working_dir, id3, cov3_dict, q, 0)))
        import time
        for thread in threads:
            thread.start()
            time.sleep(1)
        for thread in threads:
            thread.join()
        message_count = 0
        while True:
            try:
                message = q.get_nowait()
                self.assertIn(message, [id1, id2, id3])
                message_count += 1
            except queues.Empty:
                break
        self.assertEqual(message_count, len(threads))