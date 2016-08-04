__author__ = 'casey'

import shutil
import tempfile
import calendar
import time
from datetime import datetime
import unittest

from nose.plugins.attrib import attr
from pyon.core.bootstrap import CFG
from pyon.datastore.datastore_common import DatastoreFactory
import psycopg2
import psycopg2.extras
from coverage_model import *
from coverage_model.address import *
from coverage_model.parameter_data import *
from coverage_test_base import get_parameter_dict
from coverage_model.parameter_types import *


@attr('UNIT',group='cov')
class TestPostgresStorageUnit(CoverageModelUnitTestCase):

    @classmethod
    def nope(cls):
        pass

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
                pdict.add_context(ParameterContext(p, param_type=QuantityType(value_encoding=np.dtype('float64'))))

    scov = SimplexCoverage(root_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)
    if not data_dict:
        return scov


    data_dict = _easy_dict(data_dict)
    scov.set_parameter_values(data_dict)
    return scov

def _easy_dict(data_dict):
    if 'time' in data_dict:
        time_array = data_dict['time']
    else:
        elements = data_dict.values()[0]
        time_array = np.arange(len(elements))

    for k,v in data_dict.iteritems():
        data_dict[k] = NumpyParameterData(k, v, time_array)
    return data_dict

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
        scov = SimplexCoverage(cls.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict,
                               temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes,
                               in_memory_storage=in_memory, bricking_scheme=bricking_scheme,
                               auto_flush_values=auto_flush_values, value_caching=False)


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
                scov.append_parameter(ParameterContext('const_str', param_type=ConstantType(value_encoding='S10'), fill_value='Nope'))
                np_dict['depth'] = NumpyParameterData('depth', np.random.uniform(0,200,[nt]), time_array)
                np_dict['lon'] = NumpyParameterData('lon', np.random.uniform(-180,180,[nt]), time_array)
                np_dict['lat'] = NumpyParameterData('lat', np.random.uniform(-90,90,[nt]), time_array)
                np_dict['const_float'] = ConstantOverTime('const_float', 88.8, time_start=10000, time_end=10000+nt)
                np_dict['const_str'] = ConstantOverTime('const_str', 'Jello', time_start=10000, time_end=10000+nt)

                scov.set_parameter_values(np_dict)

        cls.coverages.add(scov.persistence_guid)
        return scov, 'TestSpanInt'

    def test_store_data(self):
        #Coverage construction will write data, create spans, and write spans to the db.
        #Retrieve the parameter values from a brick, get the spans from the master manager.
        #Make sure the min/max from the brick values match the min/max from master manager spans.
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)
        np_dict={}
        initial_depth_array = scov.get_parameter_values('depth').get_data()['depth']
        initial_time_array = scov.get_parameter_values(scov.temporal_parameter_name).get_data()[scov.temporal_parameter_name]
        time_array = np.array([9050, 10051, 10052, 10053, 10054, 10055])
        depth_array = np.array([1.0, np.NaN, 0.2, np.NaN, 1.01, 9.0], dtype=np.dtype('float32'))

        total_time_array = np.append(initial_time_array, time_array)
        total_depth_array = np.append(initial_depth_array, depth_array)

        sort_order = np.argsort(total_time_array)
        total_time_array = total_time_array[sort_order]
        total_depth_array = total_depth_array[sort_order]


        np_dict[scov.temporal_parameter_name] = NumpyParameterData(scov.temporal_parameter_name, time_array, time_array)
        np_dict['depth'] = NumpyParameterData('depth', depth_array, time_array)
        param_data = scov.get_parameter_values(scov.temporal_parameter_name, sort_parameter=scov.temporal_parameter_name)
        scov.set_parameter_values(np_dict)


        cov_id = scov.persistence_guid
        scov.close()
        scov = AbstractCoverage.resurrect(cov_id,mode='r')
        param_data = scov.get_parameter_values(scov.temporal_parameter_name, sort_parameter=scov.temporal_parameter_name)
        param_data = scov.get_parameter_values(scov.temporal_parameter_name, sort_parameter=scov.temporal_parameter_name, as_record_array=True)
        param_data.convert_to_record_array()
        rec_arr = param_data.get_data()
        self.assertEqual(1, len(rec_arr.shape))
        self.assertTrue(scov.temporal_parameter_name in rec_arr.dtype.names)
        np.testing.assert_array_equal(total_time_array, rec_arr[scov.temporal_parameter_name])


        params_to_get = [scov.temporal_parameter_name, 'depth', 'const_float', 'const_str']
        param_data = scov.get_parameter_values(params_to_get, sort_parameter=scov.temporal_parameter_name)
        param_data.convert_to_record_array()
        rec_arr = param_data.get_data()

        self.assertEqual(len(params_to_get), len(rec_arr.dtype.names)-1) #-1 for ingest time

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

        f(rec_arr['const_float'], rec_arr[scov.temporal_parameter_name], np.float32(88.8), scov._persistence_layer.value_list['const_float'].fill_value)
        f(rec_arr['const_str'], rec_arr[scov.temporal_parameter_name], 'Jello', scov._persistence_layer.value_list['const_str'].fill_value)

    def test_add_constant(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)
        time_array = np.array([9050, 10010, 10011, 10012, 10013, 10014])
        np_dict = {}
        np_dict[scov.temporal_parameter_name] = NumpyParameterData(scov.temporal_parameter_name, time_array, time_array)
        scov.set_parameter_values(np_dict)

        scov.append_parameter(ParameterContext('sparseness', fill_value=9999))

        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 2048)})
        rv = scov.get_parameter_values(scov.temporal_parameter_name)
        expected_array = np.empty(len(rv.get_data()[scov.temporal_parameter_name]))
        expected_array.fill(2048)
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness'], as_record_array=True).get_data()['sparseness']
        np.testing.assert_array_equal(expected_array, returned_array)

        expected_array[1:4] = 4096
        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 4096, time_start=10000, time_end=10002)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness'], as_record_array=True).get_data()
        print returned_array
        np.testing.assert_array_equal(expected_array, returned_array['sparseness'])

        expected_array[-3:] = 17
        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 17, time_start=10012)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness'], as_record_array=True).get_data()['sparseness']
        np.testing.assert_array_equal(expected_array, returned_array)

        expected_array[0:1] = -10
        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', -10, time_end=9999)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness']).get_data()['sparseness']
        np.testing.assert_array_equal(expected_array, returned_array)

    def test_fill_unfound_data(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        fill_value = 'Empty'
        param_name = 'dummy'
        scov.append_parameter(ParameterContext(param_name, param_type=ConstantType(value_encoding='object'), fill_value=fill_value))
        vals = scov.get_parameter_values([param_name, 'const_float'], fill_empty_params=True)

        retrieved_dummy_array = vals.get_data()[param_name]
        expected_dummy_array = np.empty(ts, dtype=np.array([param_name]).dtype)
        expected_dummy_array.fill(fill_value)

        np.testing.assert_array_equal(retrieved_dummy_array, expected_dummy_array)

    def test_get_all_parameters(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        vals = scov.get_parameter_values(fill_empty_params=True)
        vals.convert_to_record_array()

        if vals.is_record_array:
            params_retrieved = vals.get_data().dtype.names
        else:
            params_retrieved = vals.get_data().keys()

        params_expected = scov._range_dictionary.keys()

        self.assertEqual(sorted(params_expected), sorted(params_retrieved))

    def test_reconstruct_coverage(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        val_names = ['depth', 'lat', 'lon']
        id = scov.persistence_guid
        vals = scov.get_parameter_values(val_names)

        rcov = AbstractCoverage.load(self.working_dir, id, mode='r')
        rvals = rcov.get_parameter_values(val_names)

        vals.convert_to_record_array()
        rvals.convert_to_record_array()
        np.testing.assert_array_equal(vals.get_data(), rvals.get_data())

    def test_striding(self):
        ts = 1000
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        for stride_length in [2,3]:
            expected_data = np.arange(10000,10000+ts,stride_length, dtype=np.float64)
            returned_data = scov.get_parameter_values(scov.temporal_parameter_name, stride_length=stride_length).get_data()[scov.temporal_parameter_name]
            np.testing.assert_array_equal(expected_data, returned_data)

    def test_open_interval(self):
         ts = 0
         scov, cov_name = self.construct_cov(nt=ts)
         time_array = np.arange(10000, 10003)
         data_dict = {
             'time' : NumpyParameterData('time', time_array, time_array),
             'quantity' : NumpyParameterData('quantity', np.array([30, 40, 50]), time_array)
         }
         scov.set_parameter_values(data_dict)

         # Get data on open interval (-inf, 10002]
         data_dict = scov.get_parameter_values(param_names=['time', 'quantity'], time_segment=(None, 10002)).get_data()
         np.testing.assert_array_equal(data_dict['time'], np.array([10000., 10001., 10002.]))
         np.testing.assert_array_equal(data_dict['quantity'], np.array([30., 40., 50.]))

         # Get data on open interval [10001, inf)
         data_dict = scov.get_parameter_values(param_names=['time', 'quantity'], time_segment=(10001, None)).get_data()
         np.testing.assert_array_equal(data_dict['time'], np.array([10001., 10002.]))
         np.testing.assert_array_equal(data_dict['quantity'], np.array([40., 50.]))

         # Get all data on open interval (-inf, inf)
         scov.set_parameter_values({'time': NumpyParameterData('time', np.arange(10003, 10010))})
         data_dict = scov.get_parameter_values(param_names=['time', 'quantity']).get_data()
         np.testing.assert_array_equal(data_dict['time'], np.arange(10000, 10010))
         np.testing.assert_array_equal(data_dict['quantity'],
                 np.array([30., 40., 50., -9999., -9999., -9999., -9999., -9999., -9999., -9999.]))

         data_dict = scov.get_parameter_values(param_names=['time', 'quantity'], time_segment=(None, None)).get_data()
         np.testing.assert_array_equal(data_dict['time'], np.arange(10000, 10010))
         np.testing.assert_array_equal(data_dict['quantity'],
                 np.array([30., 40., 50., -9999., -9999., -9999., -9999., -9999., -9999., -9999.]))

    def test_category_get_set(self):

        param_type = CategoryType(categories={0:'port_timestamp', 1:'driver_timestamp', 2:'internal_timestamp', 3:'time', -99:'empty'})
        param = ParameterContext('category', param_type=param_type)

        scov = _make_cov(self.working_dir, ['dat', param], nt=0)
        self.addCleanup(scov.close)

        data_dict = _easy_dict({
            'time' : np.array([0, 1]),
            'dat' : np.array([20, 20]),
            'category' : np.array([1, 1], dtype='int64')
            })

        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        scov.get_parameter_values().get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

    def create_numpy_object_array(self, array):
        if isinstance(array, np.ndarray):
            array = array.tolist()
        arr = np.empty(len(array), dtype=object)
        arr[:] = array
        return arr

    def test_learn_array_inner_length(self):
        param_type = ArrayType(inner_encoding='int32', inner_fill_value='-9999')
        param_ctx = ParameterContext("array_type", param_type=param_type)

        # test well-formed
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[0,0,0], [1,1,1]], dtype=np.dtype('int32'))
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test ragged
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[0,0,0], [1,1,1,1]])
        }

        data_dict = _easy_dict(data_dict)
        with self.assertRaises(TypeError):
            scov.set_parameter_values(data_dict)

            returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
            for k,v in data_dict.iteritems():
                np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test one element
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0], dtype='<f8'),
            'array_type' : np.array([[0,0,0]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        if isinstance(scov, SimplexCoverage):
            il = scov._persistence_layer.parameter_metadata['array_type'].parameter_context.param_type.inner_length
            self.assertEqual(il, 3)

        d = scov.persistence_dir
        scov.close()
        cov = AbstractCoverage.load(d)
        if isinstance(cov, SimplexCoverage):
            il = cov._persistence_layer.parameter_metadata['array_type'].parameter_context.param_type.inner_length
            self.assertEqual(il, 3)

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }
        data_dict = _easy_dict(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

    def test_array_types(self):
        param_type = ArrayType(inner_length=3, inner_encoding='int32', inner_fill_value='-9999')
        param_ctx = ParameterContext("array_type", param_type=param_type)

        # test well-formed
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[0,0,0], [1,1,1]], dtype=np.dtype('int32'))
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test ragged
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[0,0,0], [1,1,1,1]])
        }

        data_dict = _easy_dict(data_dict)
        with self.assertRaises(ValueError):
            scov.set_parameter_values(data_dict)

            returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
            for k,v in data_dict.iteritems():
                np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test one element
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0], dtype='<f8'),
            'array_type' : np.array([[0,0,0]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[1,1,1], [2,2,2]], dtype='i4')
        }
        data_dict = _easy_dict(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

    def test_ragged_array_types(self):
        param_type = RaggedArrayType()
        param_ctx = ParameterContext("array_type", param_type=param_type, fill_value='')

        # test well-formed
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        with self.assertRaises(ValueError):
            data_dict = {
                'time' : np.array([0,1], dtype='<f8'),
                'array_type' : np.array([[0,0,0], [1,1,1]], dtype=np.dtype('object'))
            }

            data_dict = _easy_dict(data_dict)
            scov.set_parameter_values(data_dict)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : np.array([[0,0,0], [1,1,1,1]], dtype=np.dtype('object'))
        }
        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test one element
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        from coverage_model.util.numpy_utils import  NumpyUtils
        data_dict = {
            'time' : np.array([0], dtype='<f8'),
            'array_type' : RaggedArrayType.create_ragged_array(np.array([[0,0,0]]))
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : NumpyUtils.create_numpy_object_array(np.array([[1,1,1], [2,2,2]], dtype='i4'))
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test numpy array
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : NumpyUtils.create_numpy_object_array(np.array([[1,1,1], [2,2,2,2]]))
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        data_dict = {
            'time' : np.array([2,3], dtype='<f8'),
            'array_type' : RaggedArrayType.create_ragged_array(np.array([[1,1,1], [2,2,2]]))
        }
        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(time_segment=(2,3), param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])
            self.assertEqual(2, returned_dict[k].size)

        data_dict = {
            'time' : np.array([0,1,2,3], dtype='<f8'),
            'array_type' : NumpyUtils.create_numpy_object_array(np.array([[1,1,1], [2,2,2,2], [1,1,1], [2,2,2]]))
        }
        data_dict = _easy_dict(data_dict)
        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])
            self.assertEqual(4, returned_dict[k].size)

    def test_invalid_persistence_name(self):
        # Construct temporal and spatial Coordinate Reference System objects
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
        pdict = ParameterDictionary()
        t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('float32')))
        t_ctxt.uom = 'seconds since 01-01-1970'
        pdict.add_context(t_ctxt, is_temporal=True)
        data_dict = {
            'time' : np.array([0,1])
        }
        data_dict = _easy_dict(data_dict)

        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)
        scov.set_parameter_values(data_dict)

        scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, persistence_name='postgres_span_storage')
        scov.set_parameter_values(data_dict)

        with self.assertRaises(RuntimeError):
            scov = SimplexCoverage(self.working_dir, create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, persistence_name='Hello')
            scov.set_parameter_values(data_dict)


    def test_lon_bug(self):
        lon = ParameterContext('lon', param_type=SparseConstantType(value_encoding='float64'))
        lat = ParameterContext('lat', param_type=SparseConstantType(value_encoding='float64'))

        cov = _make_cov(self.working_dir, ['quantity', lon, lat], nt=0)
        import time
        ntp_now = time.time() + 2208988800

        data_dict = {

            #'quality_flag' : NumpyParameterData('quality_flag', np.array([None]), np.array([ntp_now])),
            #'preferred_timestamp' : NumpyParameterData('preferred_timesatmp', np.array(['driver_timestamp']), np.array([ntp_now])),
            #'temp' : NumpyParameterData('temp', np.array([300000]), np.array([ntp_now])),
            #'port_timestamp' : NumpyParameterData('port_timestamp', np.array([ntp_now]), np.array([ntp_now])),
            'lon' : ConstantOverTime('lon', -71., time_start = ntp_now, time_end=None),
            #'pressure' : NumpyParameterData('pressure', np.array([256.8]), np.array([ntp_now])),
            #'internal_timestamp' : NumpyParameterData('internal_timestamp', np.array([ntp_now]), np.array([ntp_now])),
            'time' : NumpyParameterData('time', np.array([ntp_now])),
            #'lat' : ConstantOverTime('lat', 45, time_start = ntp_now, time_end=None),
            #'driver_timestamp' : NumpyParameterData('driver_timestamp', np.array([ntp_now]), np.array([ntp_now])),
            #'conductivity' : NumpyParameterData('conductivity', np.array([4341400]), np.array([ntp_now])),
            }
        cov.set_parameter_values(data_dict)

    def test_has_data(self):
        lon = ParameterContext('lon', param_type=SparseConstantType(value_encoding='float64'))
        lat = ParameterContext('lat', param_type=SparseConstantType(value_encoding='float64'))

        cov = _make_cov(self.working_dir, ['quantity', lon, lat], nt=0)
        import time
        ntp_now = time.time() + 2208988800

        self.assertTrue(cov.is_empty())
        self.assertFalse(cov.has_parameter_data())
        data_dict = {

            'lon' : ConstantOverTime('lon', -71., time_start = ntp_now, time_end=None),
            'time' : NumpyParameterData('time', np.array([ntp_now])),
            'lat' : ConstantOverTime('lat', 45, time_start = ntp_now, time_end=None),
            }
        cov.set_parameter_values(data_dict)
        self.assertFalse(cov.is_empty())
        self.assertTrue(cov.has_parameter_data())
        cov.set_parameter_values(data_dict)
        self.assertFalse(cov.is_empty())
        self.assertTrue(cov.has_parameter_data())

    def test_sparse_arrays(self):
        sparse_array = ParameterContext('sparse_array', param_type=ArrayType(inner_encoding='float32', inner_length=3), fill_value=0.0)
        cov = _make_cov(self.working_dir, ['quantity', sparse_array], nt=0)
        import time
        ntp_now = time.time() + 2208988800
        data_dict = {
            'sparse_array' : ConstantOverTime('sparse_array', [0.0, 1.2, 3.12], time_start=ntp_now)
            # 'time' : NumpyParameterData('time', np.arange(0, 100))
        }
        cov.set_parameter_values(data_dict)

        data_dict = {
            'time' : NumpyParameterData('time', np.arange(ntp_now , ntp_now + 10)),
            'quantity' : NumpyParameterData('quantity', np.arange(10))
        }
        cov.set_parameter_values(data_dict)
        retrieved_data = cov.get_parameter_values().get_data()
        sparse_data = retrieved_data['sparse_array']

        np.testing.assert_array_equal(sparse_data,
                np.array([[ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989],
                          [ 0.        ,  1.20000005,  3.11999989]], dtype=np.float32))


    def test_array_insert_and_return(self):
        array_type = ParameterContext('array_type', param_type=ArrayType(inner_length=3, inner_encoding='float32'), fill_value=-9999)
        cov = _make_cov(self.working_dir, [array_type], nt=0)

        data = np.array([[0, 1.2, 3.2] * 10], dtype=np.float32).reshape(10,3)

        cov.set_parameter_values({
            'time' : NumpyParameterData('time', np.arange(10)),
            'array_type' : NumpyParameterData('array_type', data)
        })

        return_dict = cov.get_parameter_values(as_record_array=False).get_data()
        np.testing.assert_array_equal(data, return_dict['array_type'])

    def test_array_multiset(self):
        array_type = ParameterContext('array_type', param_type=ArrayType(inner_encoding='float32'))
        cov = _make_cov(self.working_dir, [array_type], nt=0)


        cov.set_parameter_values({
            'time' : NumpyParameterData('time', np.array([0])),
            'array_type' : NumpyParameterData('array_type', np.array([[1,2,3,4,5]], dtype=np.float32))
        })

        data = cov.get_parameter_values(as_record_array=False, fill_empty_params=True).get_data()


        cov.set_parameter_values({
            'time' : NumpyParameterData('time', np.array([1,2,3])),
            'array_type' : NumpyParameterData('array_type', np.array([[2, 2, 2, 2, 2], [3,3,3,3,3], [4,4,4,4,4]], dtype=np.float32))
        })

        data = cov.get_parameter_values(as_record_array=False, fill_empty_params=True).get_data()
        np.testing.assert_allclose(data['array_type'],
                np.array([[1,2,3,4,5],
                          [2,2,2,2,2],
                          [3,3,3,3,3],
                          [4,4,4,4,4]], dtype=np.float32))


    def test_pfs(self):
        pf_example = PythonFunction(name='identity',
                                    owner='coverage_model.test.test_postgres_storage',
                                    func_name='identity',
                                    arg_list=['x'],
                                    param_map={'x':'quantity'})

        pf_context = ParameterContext('identity', param_type=ParameterFunctionType(pf_example))

        ne_example = NumexprFunction(name = 'sum',
                                     expression= 'x+2',
                                     arg_list=['x'],
                                     param_map={'x':'quantity'})

        ne_context = ParameterContext('sum', param_type=ParameterFunctionType(ne_example))

        cov = _make_cov(self.working_dir, ['quantity', pf_context, ne_context], nt=0)
        import time
        ntp_now = time.time() + 2208988800

        data_dict = {
            'time' : NumpyParameterData('time', np.arange(ntp_now , ntp_now + 10)),
            'quantity' : NumpyParameterData('quantity', np.arange(10))
        }
        cov.set_parameter_values(data_dict)


        retval = cov.get_parameter_values(['identity']).get_data()['identity']
        np.testing.assert_allclose(retval, np.arange(10)*3)

        retval = cov.get_parameter_values(['sum']).get_data()['sum']
        np.testing.assert_allclose(retval, np.arange(10) + 2)


    def test_order_by_ingest(self):
        data_ctx = ParameterContext('data', param_type=RecordType())
        cov = _make_cov(self.working_dir, ['ingestion_timestamp',data_ctx], nt=0)
        data = [
            [3612185233.2, 1403196433.83, '\r\x00\n'],
            [3612185233.21, 1403196433.89, '\r\nS>'],
            [3612185234.14, 1403196434.84, 'd\x00s\x00\r\x00\n'],
            [3612185234.15, 1403196434.9, 'SBE37-SMP V 2.6 SERIAL NO. 2165   19 Jun 2014  16:47:14\r\nnot logging: received stop command\r\nsample interval = 489 seconds\r\nsamplenumber = 0, free = 200000\r\ntransmit real-time data\r\ndo not output salinity with each sample\r\ndo not output sound velocity with each sample\r\ndo not store time with each sample\r\nnumber of samples to average = 0\r\nreference pressure = 0.0 db\r\nserial sync mode disabled\r\nwait time after serial sync sampling = 0 seconds\r\ninternal pump is installed\r\ntemperature = 7.54 deg C\r\nWARNING: LOW BATTERY VOLTAGE!!\r\n\r\nS>'],
            [3612185234.23, 1403196435.14, '\r\x00\n'],
            [3612185234.24, 1403196435.18, '\r\nS>'],
            [3612185235.23, 1403196435.89, 'I\x00N\x00T\x00E\x00R\x00V\x00A\x00L\x00=\x001\x00\r\x00\n'],
            [3612185235.24, 1403196435.95, '\r\nS>'],
            [3612185235.36, 1403196436.43, '\r\x00\n'],
            [3612185235.37, 1403196436.47, '\r\nS>'],
            [3612185236.36, 1403196436.97, 'd\x00s\x00\r\x00\n'],
            [3612185236.37, 1403196437.01, 'SBE37-SMP V 2.6 SERIAL NO. 2165   19 Jun 2014  16:47:16\r\nnot logging: received stop command\r\nsample interval = 1 seconds\r\nsamplenumber = 0, free = 200000\r\ntransmit real-time data\r\ndo not output salinity with each sample\r\ndo not output sound velocity with each sample\r\ndo not store time with each sample\r\nnumber of samples to average = 0\r\nreference pressure = 0.0 db\r\nserial sync mode disabled\r\nwait time after serial sync sampling = 0 seconds\r\ninternal pump is installed\r\ntemperature = 7.54 deg C\r\nWARNING: LOW BATTERY VOLTAGE!!\r\n\r\nS>'],
            [3612185237.69, 1403196437.5, '\r\x00\n'],
            [3612185237.73, 1403196437.54, '\r\nS>'],
            [3612185238.67, 1403196438.53, 'd\x00c\x00\r\x00\n'],
            [3612185238.71, 1403196438.6, 'SBE37-SM V 2.6b  3464\r\ntemperature:  08-nov-05\r\n    TA0 = -2.572242e-04\r\n    TA1 = 3.138936e-04\r\n    TA2 = -9.717158e-06\r\n    TA3 = 2.138735e-07\r\nconductivity:  08-nov-05\r\n    G = -9.870930e-01\r\n    H = 1.417895e-01\r\n    I = 1.334915e-04\r\n    J = 3.339261e-05\r\n    CPCOR = 9.570000e-08\r\n    CTCOR = 3.250000e-06\r\n    WBOTC = 1.202400e-05\r\npressure S/N 4955, range = 10778.3700826 psia:  12-aug-05\r\n    PA0 = 5.916199e+00\r\n    PA1 = 4.851819e-01\r\n    PA2 = 4.596432e-07\r\n    PTCA0 = 2.762492e+02\r\n    PTCA1 = 6.603433e-01\r\n    PTCA2 = 5.756490e-03\r\n    PTCSB0 = 2.461450e+01\r\n    PTCSB1 = -9.000000e-04\r\n    PTCSB2 = 0.000000e+00\r\n    POFFSET = 0.000000e+00\r\nrtc:  08-nov-05\r\n    RTCA0 = 9.999862e-01\r\n    RTCA1 = 1.686132e-06\r\n    RTCA2 = -3.022745e-08\r\n\r\nS>'],
            [3612185239.34, 1403196440.1, '\r\x00\n'],
            [3612185239.34, 1403196440.14, '\r\nS>'],
            [3612185240.33, 1403196441.13, 't\x00s\x00\r\x00\n'],
            [3612185240.34, 1403196441.25, '\r\n13.5705,9.96891, 563.855,   16.7048, 1506.486, %s\r\n\r\nS>'],
            [3612185241.17, 1403196441.68, '\r\x00\n'],
            [3612185241.17, 1403196441.72, '\r\nS>'],
            [3612185242.17, 1403196442.71, 't\x00s\x00\r\x00\n'],
            [3612185242.17, 1403196442.8, '\r\n78.4297,94.67386, 705.741,   7.7154, 1506.692, %s\r\n\r\nS>'],
            [3612185242.38, 1403196443.26, '\r\x00\n'],
            [3612185242.39, 1403196443.31, '\r\nS>'],
            [3612185243.39, 1403196444.29, 't\x00s\x00\r\x00\n'],
            [3612185243.39, 1403196444.38, '\r\n-0.7999,99.38759, 174.620,   1.9935, 1506.788, %s\r\n\r\nS>'],
            [3612185244.22, 1403196444.84, '\r\x00\n'],
            [3612185244.22, 1403196444.88, '\r\nS>'],
            [3612185245.21, 1403196445.87, 't\x00s\x00\r\x00\n'],
            [3612185245.22, 1403196445.97, '\r\n79.0312,49.82681, 937.259,   14.3514, 1506.451, %s\r\n\r\nS>'],
            [3612185246.49, 1403196446.42, '\r\x00\n'],
            [3612185246.54, 1403196446.46, '\r\nS>'],
            [3612185247.48, 1403196447.45, 't\x00s\x00\r\x00\n'], # For some reason THIS ONE winds up out of order
            [3612185247.54, 1403196447.55, '\r\n19.0626,36.04380, 561.725,   1.8966, 1506.278, %s\r\n\r\nS>'], # This one too!
            [3612185247.31, 1403196447.99, '\r\x00\n'],
            [3612185247.31, 1403196448.04, '\r\nS>'],
            [3612185248.31, 1403196449.02, 't\x00s\x00\r\x00\n'],
            [3612185248.31, 1403196449.12, '\r\n93.5314,38.22850, 244.773,   4.6265, 1506.445, %s\r\n\r\nS>'],
            [3612185249.97, 1403196449.57, '\r\x00\n'],
            [3612185249.1, 1403196449.62, '\r\nS>'],
            [3612185250.98, 1403196450.61, 't\x00s\x00\r\x00\n'],
            [3612185250.1, 1403196450.7, '\r\n54.6971,0.73575, 615.201,   15.2357, 1506.851, %s\r\n\r\nS>'],
            [3612185250.35, 1403196451.16, '\r\x00\n'],
            [3612185250.36, 1403196451.2, '\r\nS>'],
            [3612185251.36, 1403196452.19, 't\x00s\x00\r\x00\n'],
            [3612185251.36, 1403196452.28, '\r\n77.1542,83.33314, 514.924,   2.5421, 1506.993, %s\r\n\r\nS>'],
            [3612185252.19, 1403196452.74, '\r\x00\n'],
            [3612185252.19, 1403196452.78, '\r\nS>'],
            [3612185253.19, 1403196453.77, 't\x00s\x00\r\x00\n'],
            [3612185253.19, 1403196453.86, '\r\n-0.8058,0.90317, 422.635,   3.5081, 1505.670, %s\r\n\r\nS>'],
            [3612185253.41, 1403196454.31, '\r\x00\n'],
            [3612185253.41, 1403196454.36, '\r\nS>'],
            [3612185254.4, 1403196455.34, 't\x00s\x00\r\x00\n'],
            [3612185254.41, 1403196455.43, '\r\n13.6863,36.22584, 470.093,   15.1094, 1506.268, %s\r\n\r\nS>']
        ]
        
        for granule in data:
            timestamp, ingestion_timestamp, raw_data = granule
            cov.set_parameter_values({
                'time' : NumpyParameterData('time', np.atleast_1d(timestamp)),
                'ingestion_timestamp' : NumpyParameterData('ingestion_timestamp', np.atleast_1d(ingestion_timestamp)),
                'data' : NumpyParameterData('data', np.atleast_1d(raw_data))
            })

        return_values = cov.get_parameter_values(sort_parameter='ingestion_timestamp').get_data()
        sorted_values = np.sort(return_values['ingestion_timestamp'])
        np.testing.assert_array_equal(sorted_values, return_values['ingestion_timestamp'])



    @unittest.skip('Skip for now.  Needs to be fixed.')
    def test_calibrations(self):
        FILLIN = None
        functions = {
            'secondary_interpolation' : {
                'owner' : 'coverage_model.test.example_functions',
                'func_name' : 'secondary_interpolation',
                'arg_list' : ['x','range0','range1','starts','ends'],
            },
            'polyval_calibration' : {
                'owner' : 'coverage_model.test.example_functions',
                'func_name' : 'polyval_calibration',
                'arg_list' : ['coefficients', 'x']
            }
        }

        params = [
            ParameterContext('conductivity',
                             param_type=QuantityType(value_encoding='float32'),
                             uom='Sm-1'),
            ParameterContext('condwat_l1b_pd_cals',
                             param_type=SparseConstantType(value_encoding='float32,float32,float32,float32,float32'),
                             fill_value=-9999,
                             uom='1'),
            ParameterContext('condwat_l1b_pd',
                             param_type=ParameterFunctionType(PythonFunction(name='polyval_calibration',
                                     param_map={
                                         'coefficients':'condwat_l1b_pd_cals',
                                         'x' : 'conductivity'
                                     },
                                     **functions['polyval_calibration'])),
                             uom='Sm-1'),
            ParameterContext('condwat_l1b_start',
                             param_type=SparseConstantType(value_encoding='float64'),
                             uom='seconds since 1900-01-01'),
            ParameterContext('condwat_l1b_pr_cals',
                             param_type=SparseConstantType(value_encoding='float32,float32,float32,float32,float32'),
                             uom='1'),
            ParameterContext('condwat_l1b_pr',
                             param_type=ParameterFunctionType(PythonFunction(name='polyval_calibration',
                                     param_map={
                                         'coefficients':'condwat_l1b_pr_cals',
                                         'x' : 'conductivity'
                                     },
                                     **functions['polyval_calibration'])),
                             uom='Sm-1'),
            ParameterContext('condwat_l1b_end',
                             param_type=SparseConstantType(value_encoding='float64'),
                             uom='seconds since 1900-01-01'),
            ParameterContext('condwat_l1b_interp',
                             param_type=ParameterFunctionType(PythonFunction(name='interpolation',
                                     param_map={
                                         'x' : 'time',
                                         'range0' : 'condwat_l1b_pd',
                                         'range1' : 'condwat_l1b_pr',
                                         'starts' : 'condwat_l1b_start',
                                         'ends' : 'condwat_l1b_end'
                                     },
                                     **functions['secondary_interpolation'])),
                             uom='Sm-1')
        ]

        cov = _make_cov(self.working_dir, [params[0]], nt=0)

        ntp_now = time.time() + 2208988800
        x = np.arange(ntp_now, ntp_now + 3600)
        # Ingestion gets a data granule with time,conf
        granule = {
            'time' : NumpyParameterData('time', x),
            'conductivity' : NumpyParameterData('conductivity', -3.08641975308642e-07 * (x - ntp_now) * (x - ntp_now - 3600))
        }
        cov.set_parameter_values(granule)

        # Should work
        cov.get_parameter_values(['conductivity'], fill_empty_params=True, as_record_array=False).get_data()

        # Post-deployment calibrations are uploaded
        for param in params[1:4]:
            cov.append_parameter(param)

        cov.set_parameter_values({
            'condwat_l1b_pd_cals' : ConstantOverTime('condwat_l1b_pd_cals', (0.0, 0.0, 0.0, 1.20, 1.0)),
            'condwat_l1b_start' : ConstantOverTime('condwat_l1b_start', ntp_now+10)
        })

        # Ensure that the polyval works
        data = cov.get_parameter_values(['conductivity', 'condwat_l1b_pd'], sort_parameter='time', fill_empty_params=True, as_record_array=False).get_data()
        cond = data['conductivity']
        calibrated = data['condwat_l1b_pd']
        np.testing.assert_array_equal(calibrated, cond * 1.2 + 1.0)


        # Post-recover calibrations are uploaded
        # Post-deployment calibrations are uploaded
        for param in params[4:]:
            cov.append_parameter(param)

        cov.set_parameter_values({
            'condwat_l1b_pr_cals' : ConstantOverTime('condwat_l1b_pr_cals', (0.0, 0.0, 0.0, 1.20, 2.0)),
            'condwat_l1b_end' : ConstantOverTime('condwat_l1b_end', ntp_now+1800)
        })


        data = cov.get_parameter_values(['conductivity', 'condwat_l1b_pr', 'condwat_l1b_pr_cals'], fill_empty_params=True, as_record_array=False).get_data()
        cond = data['conductivity']
        calibrated = data['condwat_l1b_pr']
        np.testing.assert_allclose(calibrated, cond * 1.2 + 2.0)


        data = cov.get_parameter_values(['conductivity', 'condwat_l1b_interp'], fill_empty_params=True, as_record_array=False).get_data()

    def test_order_by_ingest(self):
        data_ctx = ParameterContext('data', param_type=RecordType())
        cov = _make_cov(self.working_dir, ['ingestion_timestamp',data_ctx], nt=0)
        data = [
            [3612185233.2, 1403196433.83, '\r\x00\n'],
            [3612185233.21, 1403196433.89, '\r\nS>'],
            [3612185234.14, 1403196434.84, 'd\x00s\x00\r\x00\n'],
            [3612185234.15, 1403196434.9, 'SBE37-SMP V 2.6 SERIAL NO. 2165   19 Jun 2014  16:47:14\r\nnot logging: received stop command\r\nsample interval = 489 seconds\r\nsamplenumber = 0, free = 200000\r\ntransmit real-time data\r\ndo not output salinity with each sample\r\ndo not output sound velocity with each sample\r\ndo not store time with each sample\r\nnumber of samples to average = 0\r\nreference pressure = 0.0 db\r\nserial sync mode disabled\r\nwait time after serial sync sampling = 0 seconds\r\ninternal pump is installed\r\ntemperature = 7.54 deg C\r\nWARNING: LOW BATTERY VOLTAGE!!\r\n\r\nS>'],
            [3612185234.23, 1403196435.14, '\r\x00\n'],
            [3612185234.24, 1403196435.18, '\r\nS>'],
            [3612185235.23, 1403196435.89, 'I\x00N\x00T\x00E\x00R\x00V\x00A\x00L\x00=\x001\x00\r\x00\n'],
            [3612185235.24, 1403196435.95, '\r\nS>'],
            [3612185235.36, 1403196436.43, '\r\x00\n'],
            [3612185235.37, 1403196436.47, '\r\nS>'],
            [3612185236.36, 1403196436.97, 'd\x00s\x00\r\x00\n'],
            [3612185236.37, 1403196437.01, 'SBE37-SMP V 2.6 SERIAL NO. 2165   19 Jun 2014  16:47:16\r\nnot logging: received stop command\r\nsample interval = 1 seconds\r\nsamplenumber = 0, free = 200000\r\ntransmit real-time data\r\ndo not output salinity with each sample\r\ndo not output sound velocity with each sample\r\ndo not store time with each sample\r\nnumber of samples to average = 0\r\nreference pressure = 0.0 db\r\nserial sync mode disabled\r\nwait time after serial sync sampling = 0 seconds\r\ninternal pump is installed\r\ntemperature = 7.54 deg C\r\nWARNING: LOW BATTERY VOLTAGE!!\r\n\r\nS>'],
            [3612185237.69, 1403196437.5, '\r\x00\n'],
            [3612185237.73, 1403196437.54, '\r\nS>'],
            [3612185238.67, 1403196438.53, 'd\x00c\x00\r\x00\n'],
            [3612185238.71, 1403196438.6, 'SBE37-SM V 2.6b  3464\r\ntemperature:  08-nov-05\r\n    TA0 = -2.572242e-04\r\n    TA1 = 3.138936e-04\r\n    TA2 = -9.717158e-06\r\n    TA3 = 2.138735e-07\r\nconductivity:  08-nov-05\r\n    G = -9.870930e-01\r\n    H = 1.417895e-01\r\n    I = 1.334915e-04\r\n    J = 3.339261e-05\r\n    CPCOR = 9.570000e-08\r\n    CTCOR = 3.250000e-06\r\n    WBOTC = 1.202400e-05\r\npressure S/N 4955, range = 10778.3700826 psia:  12-aug-05\r\n    PA0 = 5.916199e+00\r\n    PA1 = 4.851819e-01\r\n    PA2 = 4.596432e-07\r\n    PTCA0 = 2.762492e+02\r\n    PTCA1 = 6.603433e-01\r\n    PTCA2 = 5.756490e-03\r\n    PTCSB0 = 2.461450e+01\r\n    PTCSB1 = -9.000000e-04\r\n    PTCSB2 = 0.000000e+00\r\n    POFFSET = 0.000000e+00\r\nrtc:  08-nov-05\r\n    RTCA0 = 9.999862e-01\r\n    RTCA1 = 1.686132e-06\r\n    RTCA2 = -3.022745e-08\r\n\r\nS>'],
            [3612185239.34, 1403196440.1, '\r\x00\n'],
            [3612185239.34, 1403196440.14, '\r\nS>'],
            [3612185240.33, 1403196441.13, 't\x00s\x00\r\x00\n'],
            [3612185240.34, 1403196441.25, '\r\n13.5705,9.96891, 563.855,   16.7048, 1506.486, %s\r\n\r\nS>'],
            [3612185241.17, 1403196441.68, '\r\x00\n'],
            [3612185241.17, 1403196441.72, '\r\nS>'],
            [3612185242.17, 1403196442.71, 't\x00s\x00\r\x00\n'],
            [3612185242.17, 1403196442.8, '\r\n78.4297,94.67386, 705.741,   7.7154, 1506.692, %s\r\n\r\nS>'],
            [3612185242.38, 1403196443.26, '\r\x00\n'],
            [3612185242.39, 1403196443.31, '\r\nS>'],
            [3612185243.39, 1403196444.29, 't\x00s\x00\r\x00\n'],
            [3612185243.39, 1403196444.38, '\r\n-0.7999,99.38759, 174.620,   1.9935, 1506.788, %s\r\n\r\nS>'],
            [3612185244.22, 1403196444.84, '\r\x00\n'],
            [3612185244.22, 1403196444.88, '\r\nS>'],
            [3612185245.21, 1403196445.87, 't\x00s\x00\r\x00\n'],
            [3612185245.22, 1403196445.97, '\r\n79.0312,49.82681, 937.259,   14.3514, 1506.451, %s\r\n\r\nS>'],
            [3612185246.49, 1403196446.42, '\r\x00\n'],
            [3612185246.54, 1403196446.46, '\r\nS>'],
            [3612185247.48, 1403196447.45, 't\x00s\x00\r\x00\n'], # For some reason THIS ONE winds up out of order
            [3612185247.54, 1403196447.55, '\r\n19.0626,36.04380, 561.725,   1.8966, 1506.278, %s\r\n\r\nS>'], # This one too!
            [3612185247.31, 1403196447.99, '\r\x00\n'],
            [3612185247.31, 1403196448.04, '\r\nS>'],
            [3612185248.31, 1403196449.02, 't\x00s\x00\r\x00\n'],
            [3612185248.31, 1403196449.12, '\r\n93.5314,38.22850, 244.773,   4.6265, 1506.445, %s\r\n\r\nS>'],
            [3612185249.97, 1403196449.57, '\r\x00\n'],
            [3612185249.1, 1403196449.62, '\r\nS>'],
            [3612185250.98, 1403196450.61, 't\x00s\x00\r\x00\n'],
            [3612185250.1, 1403196450.7, '\r\n54.6971,0.73575, 615.201,   15.2357, 1506.851, %s\r\n\r\nS>'],
            [3612185250.35, 1403196451.16, '\r\x00\n'],
            [3612185250.36, 1403196451.2, '\r\nS>'],
            [3612185251.36, 1403196452.19, 't\x00s\x00\r\x00\n'],
            [3612185251.36, 1403196452.28, '\r\n77.1542,83.33314, 514.924,   2.5421, 1506.993, %s\r\n\r\nS>'],
            [3612185252.19, 1403196452.74, '\r\x00\n'],
            [3612185252.19, 1403196452.78, '\r\nS>'],
            [3612185253.19, 1403196453.77, 't\x00s\x00\r\x00\n'],
            [3612185253.19, 1403196453.86, '\r\n-0.8058,0.90317, 422.635,   3.5081, 1505.670, %s\r\n\r\nS>'],
            [3612185253.41, 1403196454.31, '\r\x00\n'],
            [3612185253.41, 1403196454.36, '\r\nS>'],
            [3612185254.4, 1403196455.34, 't\x00s\x00\r\x00\n'],
            [3612185254.41, 1403196455.43, '\r\n13.6863,36.22584, 470.093,   15.1094, 1506.268, %s\r\n\r\nS>']
        ]

        for granule in data:
            timestamp, ingestion_timestamp, raw_data = granule
            cov.set_parameter_values({
                'time' : NumpyParameterData('time', np.atleast_1d(timestamp)),
                'ingestion_timestamp' : NumpyParameterData('ingestion_timestamp', np.atleast_1d(ingestion_timestamp)),
                'data' : NumpyParameterData('data', np.atleast_1d(raw_data))
            })

        return_values = cov.get_parameter_values(sort_parameter='ingestion_timestamp').get_data()
        sorted_values = np.sort(return_values['ingestion_timestamp'])

        for x in range(len(return_values['ingestion_timestamp'])):
            if sorted_values[x] != return_values['ingestion_timestamp'][x]:
                print ''
                print "Differ at index %i" % x
                print return_values['ingestion_timestamp'][x-1], sorted_values[x-1]
                print return_values['ingestion_timestamp'][x], sorted_values[x]
                print return_values['ingestion_timestamp'][x+1], sorted_values[x+1]
        np.testing.assert_array_equal(sorted_values, return_values['ingestion_timestamp'])


def identity(x):
    return np.copy(x)*3

