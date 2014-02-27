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

from coverage_test_base import CoverageIntTestBase, get_props, get_parameter_dict, EXEMPLAR_CATEGORIES

@attr('UNIT',group='cov')
#class TestCoverageSearchUnit(CoverageModelUnitTestCase):

    #def test_coverage_search_initialization(self):
    #    time_min, time_max = (-1.0,-1.0)
    #    lat_min, lat_max = (1000.0, 1000.0)
    #    lon_min, lon_max = (1000.0, 1000.0)
    #
    #    search_constants = SearchParameterNames()
    #    criteria = SearchCriteria()
    #    param = SearchParameter(search_constants.TIME, (time_min, time_max), ParamValueRange())
    #    criteria.append(param)
    #    param = SearchParameter(search_constants.GEO_BOX, ((lat_min, lat_max),(lon_min, lon_max)), Param2DValueRange())
    #    criteria.append(param)
    #    results = None
    #    results = CoverageSearchResults(criteria)
    #    self.assertIsNotNone(results)
    #
    #def test_coverage_search_initialization_fails_for_invalid_parameters(self):
    #    time_min, time_max = (-1.0,-1.0)
    #    lat_min, lat_max = (1000.0, 1000.0)
    #    lon_min, lon_max = (1000.0, 1000.0)
    #
    #    search_constants = SearchParameterNames()
    #    criteria = SearchCriteria()
    #    param = SearchParameter(search_constants.INTERNAL_TIME_KEY, (time_min, time_max), ParamValueRange())
    #    criteria.append(param)
    #    param = SearchParameter(search_constants.GPS_LAT_KEY, (lat_min, lat_max), ParamValueRange())
    #    criteria.append(param)
    #    results = None
    #    try:
    #        results = CoverageSearchResults(criteria)
    #    except ValueError:
    #        pass
    #    self.assertIsNone(results)

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
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme, auto_flush_values=auto_flush_values)

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
                #scov.set_parameter_values('boolean', value=[True, True, True], tdoa=[[2,4,14]])
                #scov.set_parameter_values('const_ft', value=-71.11) # Set a constant with correct data type
                #scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
                #scov.set_parameter_values('const_str', value='constant string value') # Set with a string
                #scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
                #scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list

                scov.set_parameter_values('lon', value=160 * np.random.random_sample(nt))
                scov.set_parameter_values('lat', value=70 * np.random.random_sample(nt))

                #arrval = []
                #recval = []
                #catval = []
                #fstrval = []
                #catkeys = EXEMPLAR_CATEGORIES.keys()
                #letts='abcdefghijklmnopqrstuvwxyz'
                #while len(letts) < nt:
                #    letts += 'abcdefghijklmnopqrstuvwxyz'
                #for x in xrange(nt):
                #    arrval.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
                #    d = {letts[x]: letts[x:]}
                #    recval.append(d) # One value (which is a dict) for each member of the domain
                #    catval.append(random.choice(catkeys))
                #    fstrval.append(''.join([random.choice(letts) for x in xrange(8)])) # A random string of length 8
                #scov.set_parameter_values('array', value=arrval)
                #scov.set_parameter_values('record', value=recval)
                #scov.set_parameter_values('category', value=catval)
                #scov.set_parameter_values('fixed_str', value=fstrval)

        if in_memory and save_coverage:
            SimplexCoverage.pickle_save(scov, os.path.join(cls.working_dir, 'sample.cov'))

        cls.coverages.add(scov.persistence_guid)
        return scov, 'TestCoverageInt'

    def test_coverage_creation(self):
        scov, cov_name = self.construct_cov(nt=16777216)
        self.assertIsNotNone(scov)

        time_min, time_max = (-1.0,-1.0)
        lat_min, lat_max = (1000.0, 1000.0)
        lon_min, lon_max = (1000.0, 1000.0)
        mm = scov._persistence_layer.master_manager
        if hasattr(mm, 'span_collection'):
            for k, span in mm.span_collection.span_dict.iteritems():
                if 'time' in span.params:
                    time_min, time_max = span.params['time']
                if 'lat' in span.params:
                    lat_min, lat_max = span.params['lat']
                if 'lat' in span.params:
                    lon_min, lon_max = span.params['lon']

        search_constants = SearchParameterNames()
        criteria = SearchCriteria()
        param = SearchParameter(search_constants.TIME, (1, 100), ParamValueRange())
        criteria.append(param)
        #param = SearchParameter(search_constants.GEO_BOX, ((lat_min, lat_max),(lon_min, lon_max)), Param2DValueRange())
        #criteria.append(param)
        search = CoverageSearch(criteria, order_by=['time'])
        results = search.select()
        self.assertIsNotNone(results)
        self.assertIsNotNone( mm.guid, results.get_found_coverage_ids())
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
