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
from coverage_model.storage.parameter_data import *
from coverage_test_base import CoverageIntTestBase, get_props, get_parameter_dict, EXEMPLAR_CATEGORIES

@attr('UNIT',group='cov')
class TestPostgresStorageUnit(CoverageModelUnitTestCase):

    @classmethod
    def nope(cls):
        pass


@attr('INT',group='cov')
class TestPostgresStorageInt(CoverageModelUnitTestCase):
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
                            # 'const_str',
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
            np_dict = {}
            # Add data for each parameter
            time_array = np.arange(10000, 10000+nt)
            np_dict[scov.temporal_parameter_name] = NumpyParameterData(scov.temporal_parameter_name, time_array, time_array)
            if not only_time:
                scov.append_parameter(ParameterContext('depth', fill_value=9999.0))
                # scov.append_parameter(ParameterContext('lon'))
                # scov.append_parameter(ParameterContext('lat'))
                scov.append_parameter(ParameterContext('const_str', fill_value='Nope'))
                np_dict['depth'] = NumpyParameterData('depth', np.random.uniform(0,200,[nt]), time_array)
                np_dict['lon'] = NumpyParameterData('lon', np.random.uniform(-180,180,[nt]), time_array)
                np_dict['lat'] = NumpyParameterData('lat', np.random.uniform(-90,90,[nt]), time_array)
                np_dict['const_float'] = ConstantOverTime('const_float', 88.8, time_start=10000, time_end=10000+nt)
                np_dict['const_str'] = ConstantOverTime('const_str', 'Jello', time_start=10000, time_end=10000+nt)

                #scov.set_parameter_values('boolean', value=[True, True, True], tdoa=[[2,4,14]])
                #scov.set_parameter_values('const_ft', value=-71.11) # Set a constant with correct data type
                #scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
                #scov.set_parameter_values('const_str', value='constant string value') # Set with a string
                #scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
                #scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list

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

                scov.set_parameter_values(np_dict)

        cls.coverages.add(scov.persistence_guid)
        return scov, 'TestSpanInt'

    def test_store_data(self):
        #Coverage construction will write data to bricks, create spans, and write spans to the db.
        #Retrieve the parameter values from a brick, get the spans from the master manager.
        #Make sure the min/max from the brick values match the min/max from master manager spans.
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)
        np_dict={}
        initial_depth_array = scov.get_parameter_values('depth').get_data()['depth']
        initial_time_array = scov.get_parameter_values(scov.temporal_parameter_name).get_data()[scov.temporal_parameter_name]
        time_array = np.array([9050, 10051, 10052, 10053, 10054, 10055])
        depth_array = np.array([1.0, np.NaN, 0.2, np.NaN, 1.01, 9.0])

        total_time_array = np.append(initial_time_array, time_array)
        total_depth_array = np.append(initial_depth_array, depth_array)

        sort_order = np.argsort(total_time_array)
        total_time_array = total_time_array[sort_order]
        total_depth_array = total_depth_array[sort_order]


        np_dict[scov.temporal_parameter_name] = NumpyParameterData(scov.temporal_parameter_name, time_array, time_array)
        np_dict['depth'] = NumpyParameterData('depth', depth_array, time_array)
        scov.set_parameter_values(np_dict)


        param_data = scov.get_parameter_values(scov.temporal_parameter_name, sort_parameter=scov.temporal_parameter_name)
        rec_arr = param_data.get_data()
        self.assertEqual(1, len(rec_arr.shape))
        self.assertTrue(scov.temporal_parameter_name in rec_arr.dtype.names)
        np.testing.assert_array_equal(total_time_array, rec_arr[scov.temporal_parameter_name])


        params_to_get = [scov.temporal_parameter_name, 'depth', 'const_float', 'const_str']
        param_data = scov.get_parameter_values(params_to_get, sort_parameter=scov.temporal_parameter_name)
        rec_arr = param_data.get_data()

        self.assertEqual(len(params_to_get), len(rec_arr.dtype.names))

        for param_name in params_to_get:
            self.assertTrue(param_name in rec_arr.dtype.names)

        self.assertTrue(scov.temporal_parameter_name in rec_arr.dtype.names)

        np.testing.assert_array_equal(total_time_array, rec_arr[scov.temporal_parameter_name])
        np.testing.assert_array_equal(total_depth_array, rec_arr['depth'])

        def f(x, t_a, val, fill_val):
            if t_a >= 10000 and t_a <= 10000+ts:
                self.assertEqual(x, val)
            else:
                self.assertEqual(x, fill_val)

        f = np.vectorize(f)

        f(rec_arr['const_float'], rec_arr[scov.temporal_parameter_name], 88.8, scov._persistence_layer.value_list['const_float'].fill_value)
        f(rec_arr['const_str'], rec_arr[scov.temporal_parameter_name], 'Jello', scov._persistence_layer.value_list['const_str'].fill_value)


    def test_add_constant(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)
        time_array = np.array([9050, 10010, 10011, 10012, 10013, 10014])
        np_dict = {}
        np_dict[scov.temporal_parameter_name] = NumpyParameterData(scov.temporal_parameter_name, time_array, time_array)
        scov.set_parameter_values(np_dict)

        initial_floats = scov.get_parameter_values('const_float').get_data()['const_float']
        scov.set_parameter_values({'const_float': ConstantOverTime('const_float', 99.9, time_start=10009.0)})
        scov.set_parameter_values({'const_float': ConstantOverTime('const_float', 1.0, time_start=10008.0)})
        scov.set_parameter_values({'const_float': ConstantOverTime('const_float', 17.0)})
        new_float_arr = scov.get_parameter_values((scov.temporal_parameter_name, 'const_float')).get_data()
        # self.assertTrue(False)

    def test_reconstruct_coverage(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        val_names = ['depth', 'lat', 'lon']
        id = scov.persistence_guid
        vals = scov.get_parameter_values(val_names).get_data()

        rcov = AbstractCoverage.load(self.working_dir, id, mode='r')
        rvals = rcov.get_parameter_values(val_names).get_data()

        np.testing.assert_array_equal(vals, rvals)
