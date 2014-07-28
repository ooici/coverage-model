__author__ = 'casey'


from nose.plugins.attrib import attr
import numpy as np
import os, shutil, tempfile
import unittest
from pyon.core.bootstrap import CFG
from pyon.datastore.datastore_common import DatastoreFactory
import psycopg2
import psycopg2.extras
from coverage_model import *
from coverage_model.address import *
from coverage_model.data_span import *
from coverage_model.db_connectors import DBFactory, DB
from coverage_model.search.coverage_search import *
from coverage_model.search.search_parameter import *
from coverage_model.search.search_constants import *
from coverage_model.storage.span_storage_factory import SpanStorageFactory
from coverage_model.storage.postgres_span_storage import PostgresSpanStorage

from coverage_test_base import CoverageIntTestBase, get_props, get_parameter_dict, EXEMPLAR_CATEGORIES

@attr('UNIT',group='cov')
class TestSpanUnit(CoverageModelUnitTestCase):

    def test_address_equality_and_serialization(self):
        addr_list = [ [Address('some_id'), [Address('other')]],
                      [FileAddress('id2', 'file_name', 0, 100), [FileAddress('is', 'file_name', 0, 100),
                       FileAddress('id2', 'bad_file', 0, 100), FileAddress('id2', 'file_name', 1, 100),
                       FileAddress('id2', 'file_name', 0, 101)] ],
                      [BrickAddress('id3', 'brick1', (2, 200)), [BrickAddress('ir', 'brick1', (2, 200)),
                       BrickAddress('id3', 'bad_brick', (2, 200)), BrickAddress('id3', 'brick1', (-1, 200)),
                       BrickAddress('id3', 'brick1', (2, -1))] ],
                      [BrickFileAddress('id4', 'brick2'), [BrickFileAddress('iq', 'brick2'),
                       BrickFileAddress('id4', 'bad_brick')] ] ]

        for addr_type in addr_list:
            base_addr = addr_type[0]
            addr_str = str(base_addr)
            new_addr = AddressFactory.from_str(addr_str)
            self.assertEqual(base_addr, new_addr)
            for address in addr_type[1]:
                self.assertNotEqual(base_addr, address)

    def test_span_equality_and_serialization(self):
        addr = BrickFileAddress('id4', 'brick2')
        other_addr = BrickFileAddress('bad', 'brick')
        base_span = SpanStats(addr, {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)})
        s = str(base_span)
        new_span = SpanStats.from_str(s)
        self.assertEqual(base_span, new_span)
        d = base_span.as_dict()
        new_span = SpanStats.from_dict(d)
        self.assertEqual(base_span, new_span)

        bad_spans = [SpanStats(other_addr, {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}),
                     SpanStats(addr, {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9), 'dummy': (1,1)}),
                     SpanStats(addr, {'time': (1,1), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}) ]

        for span in bad_spans:
            self.assertNotEqual(span, base_span)

    def test_span_collection_equality_and_serialization(self):
        spans = [SpanStats(BrickFileAddress('id2', 'Brick1'), {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}),
                 SpanStats(BrickFileAddress('id2', 'Brick2'), {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9), 'dummy': (1,1)}),
                 SpanStats(BrickFileAddress('id2', 'Brick3'), {'time': (1,1), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}) ]

        #spans = [ParamSpan(BrickFileAddress('id2', 'Brink1'), {})]
        spans_collection = SpanCollectionByFile()
        for span in spans:
            spans_collection.add_span(span)

        s = str(spans_collection)
        new_col = SpanCollectionByFile.from_str(s)
        self.assertEqual(spans_collection, new_col)
        d = spans_collection.as_dict()
        new_col = SpanCollectionByFile.from_dict(d)
        self.assertEqual(spans_collection, new_col)

        spans = [SpanStats(BrickFileAddress('id2', 'Brick'), {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}),
                 SpanStats(BrickFileAddress('id2', 'Brick2'), {'time': (1,2), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9), 'dummy': (1,1)}),
                 SpanStats(BrickFileAddress('id2', 'Brick3'), {'time': (1,1), 'lat': (0.0, 0.1), 'lon': (0.0, 179.9)}) ]

        bad_col = SpanCollectionByFile()
        self.assertNotEqual(spans_collection, bad_col)
        for span in spans:
            bad_col.add_span(span)

        self.assertNotEqual(spans_collection, bad_col)

    def test_span_serialization(self):
        import random
        import string

        span_uuid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        cov_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
        param_dict = {'one': np.array([1234567.115,1,1,1,1,1], dtype=np.float32), 'two': np.array([2,2,2,2,2,2], dtype=np.float64), 'ingest_time': np.array([1,2,2,3,3,1], dtype=np.float32)}
        span = Span(span_uuid, cov_id, param_dict, ingest_time=None, compressors=None, mutable=False)

        txt = span.serialize()
        json = span.as_json()
        msgpack_span = Span.deserialize(txt)
        json_span = Span.from_json(json)

        self.assertEqual(msgpack_span, span)
        self.assertEqual(json_span, msgpack_span)


@attr('INT',group='cov')
class TestSpanInt(CoverageModelUnitTestCase):
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

    def setUp(self):
        pass

    def tearDown(self):
        pass

    @classmethod
    def construct_cov(cls, only_time=False, save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None, make_empty=False, nt=None, auto_flush_values=True):
        """
        Construct coverage
        """
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

        pname_filter = ['time',
                            'boolean',
                            'const_float',
                            'const_int',
                            'const_str',
                            'const_rng_flt',
                            'const_rng_int',
                            'numexpr_func',
                            'category',
                            'quantity',
                            'array',
                            'record',
                            'fixed_str',
                            'sparse',
                            'lat',
                            'lon',
                            'depth']

        if only_time:
            pname_filter = ['time']

        pdict = get_parameter_dict(parameter_list=pname_filter)

        if brick_size is not None:
            bricking_scheme = {'brick_size':brick_size, 'chunk_size':True}
        else:
            bricking_scheme = None

        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

        # Insert some timesteps (automatically expands other arrays)
        if (nt is None) or (nt == 0) or (make_empty is True):
            return scov, 'TestTestSpanUnit'
        else:
            # Add data for each parameter
            if only_time:
                scov.set_parameter_values(make_parameter_data_dict({scov.temporal_parameter_name: np.arange(1000, 10000, nt+1)}))
            else:
                parameter_values = {}
                # scov.set_parameter_values('sparse', [[[2, 4, 6], [8, 10, 12]]])
                # scov.insert_timesteps(nt/2)
                #
                # scov.set_parameter_values('sparse', [[[4, 8], [16, 20]]])
                # scov.insert_timesteps(nt/2)
                scov.append_parameter(ParameterContext('m_lon'))
                scov.append_parameter(ParameterContext('m_lat'))
                scov.append_parameter(ParameterContext('depth'))

                parameter_values['time']= np.arange(1000, 1000+nt)
                parameter_values['depth']= 1000 * np.random.random_sample(nt)
                parameter_values['m_lon'] = 160 * np.random.random_sample(nt)
                parameter_values['m_lat'] = 70 * np.random.random_sample(nt)
                scov.set_parameter_values(make_parameter_data_dict(parameter_values))

        cls.coverages.add(scov.persistence_guid)
        return scov, 'TestSpanInt'

    def test_spans_in_coverage(self):
        #Coverage construction will write data to bricks, create spans, and write spans to the db.
        #Retrieve the parameter values from a brick, get the spans from the master manager.
        #Make sure the min/max from the brick values match the min/max from master manager spans.
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        cov_data = scov.get_parameter_values(['time', 'm_lat', 'm_lon']).get_data()
        pmin, pmax = scov.get_data_bounds('time')
        self.assertEqual(np.float32(pmin), cov_data['time'].min())
        self.assertEqual(np.float32(pmax), cov_data['time'].max())
        pmin, pmax = scov.get_data_bounds('m_lat')
        self.assertEqual(np.float32(pmin), np.float32(cov_data['m_lat'].min()))
        self.assertEqual(np.float32(pmax), np.float32(cov_data['m_lat'].max()))
        pmin, pmax = scov.get_data_bounds('m_lon')
        self.assertEqual(np.float32(pmin), np.float32(cov_data['m_lon'].min()))
        self.assertEqual(np.float32(pmax), np.float32(cov_data['m_lon'].max()))

    def test_span_insert(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.coverages.add(scov.persistence_guid)
        self.assertIsNotNone(scov)
        span_store = DatastoreFactory.get_datastore(datastore_name='coverage_spans', config=CFG)
        if span_store is None:
            raise RuntimeError("Unable to load datastore for coverage_spans")
        span_addr = []
        with span_store.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT span_address from %s where coverage_id='%s'" % ('ion_coverage_span_stats', scov.persistence_guid))
            self.assertGreater(cur.rowcount, 0)
            for row in cur:
                span_addr.append(row['span_address'])
        with span_store.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            for addr in span_addr:
                cur.execute("DELETE FROM %s WHERE span_address='%s'" % (span_store._get_datastore_name(), addr))

    def test_get_coverage(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        orig_mm = scov._persistence_layer.master_manager
        cov_id = orig_mm.guid
        retrieved_cov = SimplexCoverage(TestSpanInt.working_dir, cov_id, 'sample coverage_model')
        new_mm = retrieved_cov._persistence_layer.master_manager

        self.assertEqual(new_mm, orig_mm)
        self.assertNotEqual(id(new_mm), id(orig_mm))

    def test_search_for_span(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_min-1, lat_max+1),(lon_min-1, lon_max+1)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_time_fails(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+20, time_max+30))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_min-1, lat_max+1),(lon_min-1, lon_max+1)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertFalse(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_lat_fails(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_max+1, lat_max+2),(lon_min-1, lon_max+1)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertFalse(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_lon_fails(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_min+1, lat_max+2),(lon_max+0.5, lon_max+1)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertFalse(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_that_barely_overlaps_searched_box(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_max, lat_max+20), (lon_min-0.1, lon_max+20)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_using_lat_and_lon(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Latitude, (lat_min-1, lat_max+20))
        criteria.append(param)
        param = ParamValueRange(IndexedParameters.Longitude, (lon_min-1, lon_max+20))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

        lat = ParamValueRange(IndexedParameters.Latitude, (lat_min-1, lat_min-0.5))
        lon = ParamValueRange(IndexedParameters.Longitude, (lon_min-1, lon_min-0.5))
        criteria = SearchCriteria([lat, lon])
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertFalse(scov.persistence_guid in results.get_found_coverage_ids())

        lat = ParamValue(IndexedParameters.Latitude, (lat_min+lat_max)/2)
        lon = ParamValue(IndexedParameters.Longitude, (lon_min+lon_max)/2)
        criteria = SearchCriteria([lat, lon])
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

        lat = ParamValueRange(IndexedParameters.Latitude, (lat_min-1, lat_max+1))
        lon = ParamValueRange(IndexedParameters.Longitude, (lon_min-150, lon_max+10))
        criteria = SearchCriteria([lat, lon])
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

        lat = ParamValueRange(IndexedParameters.Latitude, (lat_min-1, lat_max+1))
        criteria = SearchCriteria(lat)
        search = CoverageSearch(criteria)
        self.assertRaises(ValueError, search.select)

    def test_search_for_span_that_contains_searched_box(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')
        lat_min, lat_max = scov.get_data_bounds('m_lat')
        lon_min, lon_max = scov.get_data_bounds('m_lon')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min-1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((lat_min-0.01, lat_max+0.01),(lon_min-0.5, lon_max+0.5)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_search_for_span_contained_inside_large_box(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max+1))
        criteria.append(param)
        param = Param2DValueRange(IndexedParameters.GeoBox, ((-15.5, 85.5), (0.5, 170.5)))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_for_searched_time_range_smaller_than_span_time_range(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min+1, time_max-1))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_for_searched_time_range_larger_than_span_time_range(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        time_min, time_max = scov.get_data_bounds('time')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Time, (time_min-1, time_max+1))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_for_searched_vertical_range(self):
        scov, cov_name = self.construct_cov(nt=10)
        self.assertIsNotNone(scov)

        depth_min, depth_max = scov.get_data_bounds('depth')

        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Vertical, (depth_min-1, depth_max-1))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

        del criteria
        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Vertical, (depth_max+0.00000001, depth_max+10.1))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertFalse(scov.persistence_guid in results.get_found_coverage_ids())

        del criteria
        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Vertical, (depth_max-0.00000001, depth_max-0.000000001))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

        del criteria
        criteria = SearchCriteria()
        param = ParamValueRange(IndexedParameters.Vertical, (depth_min-1, depth_max+1))
        criteria.append(param)
        search = CoverageSearch(criteria)
        results = search.select()
        self.assertTrue(scov.persistence_guid in results.get_found_coverage_ids())

    def test_minimum_search_criteria(self):
        param = ParamValue('dummy', 10)
        criteria = SearchCriteria(search_params=[param])
        search = CoverageSearch(criteria)
        self.assertRaises(ValueError, search.select)
        criteria.append(ParamValue(IndexedParameters.Time, 5))
        search = CoverageSearch(criteria)
        self.assertRaises(ValueError, search.select)
