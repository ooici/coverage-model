__author__ = 'casey'

import shutil
import tempfile
import calendar
from datetime import datetime

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

        rv = scov.get_parameter_values(scov.temporal_parameter_name)
        expected_array = np.empty(len(rv.get_data()))
        expected_array.fill(2048)
        scov.append_parameter(ParameterContext('sparseness', fill_value=9999))

        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 2048)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness']).get_data()['sparseness']
        np.testing.assert_array_equal(expected_array, returned_array)

        expected_array[1:4] = 4096
        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 4096, time_start=10000, time_end=10002)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness']).get_data()['sparseness']
        np.testing.assert_array_equal(expected_array, returned_array)

        expected_array[-3:] = 17
        scov.set_parameter_values({'sparseness': ConstantOverTime('sparseness', 17, time_start=10012)})
        returned_array = scov.get_parameter_values([scov.temporal_parameter_name, 'sparseness']).get_data()['sparseness']
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
        scov.append_parameter(ParameterContext(param_name, fill_value=fill_value))
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

        params_retrieved = vals.get_data().dtype.names
        params_expected = scov._range_dictionary.keys()

        self.assertEqual(sorted(params_expected), sorted(params_retrieved))

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
            'category' : np.array(['driver_timestamp', 'driver_timestamp'], dtype='O')
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

    def test_array_types(self):
        param_type = ArrayType(inner_encoding='<f4')
        param_ctx = ParameterContext("array_type", param_type=param_type)

        # test well-formed
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : self.create_numpy_object_array([[0,0,0], [1,1,1]])
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
            'array_type' : self.create_numpy_object_array([[0,0,0], [1,1,1,1]])
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

        # test one element
        scov = _make_cov(self.working_dir, ['quantity', param_ctx], nt = 0)

        data_dict = {
            'time' : np.array([0], dtype='<f8'),
            'array_type' : self.create_numpy_object_array([[0,0,0]])
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
            'array_type' : self.create_numpy_object_array(np.array([[1,1,1], [2,2,2]]))
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
            'array_type' : np.array([[1,1,1], [2,2,2]])
        }

        data_dict = _easy_dict(data_dict)
        scov.set_parameter_values(data_dict)

        data_dict = {
            'time' : np.array([0,1], dtype='<f8'),
            'array_type' : self.create_numpy_object_array(np.array([[1,1,1], [2,2,2]]))
        }
        data_dict = _easy_dict(data_dict)

        returned_dict = scov.get_parameter_values(param_names=data_dict.keys()).get_data()
        for k,v in data_dict.iteritems():
            np.testing.assert_array_equal(v.get_data(), returned_dict[k])

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
        sparse_array = ParameterContext('sparse_array', param_type=QuantityType(value_encoding='float32, float32, float32'), fill_value=(0.0,0.0,0.0))
        cov = _make_cov(self.working_dir, ['quantity', sparse_array], nt=0)
        import time
        ntp_now = time.time() + 2208988800
        data_dict = {
            'sparse_array' : ConstantOverTime('sparse_array', (0.0, 1.2, 3.12), time_start=ntp_now)
            # 'time' : NumpyParameterData('time', np.arange(0, 100))
        }
        cov.set_parameter_values(data_dict)

        data_dict = {
            'time' : NumpyParameterData('time', np.arange(ntp_now , ntp_now + 10)),
            'quantity' : NumpyParameterData('quantity', np.arange(10))
        }
        cov.set_parameter_values(data_dict)
        retrieved_data = cov.get_parameter_values().get_data()
        self.assertEqual(10, retrieved_data['sparse_array'].size)
        for x in range(10):
            self.assertEqual(np.float32(0.0), retrieved_data['sparse_array'][x][0])
            self.assertEqual(np.float32(1.2), retrieved_data['sparse_array'][x][1])
            self.assertEqual(np.float32(3.12), retrieved_data['sparse_array'][x][2])


    def test_pfs(self):
        print "Verified"
        pf_example = PythonFunction(name='identity', 
                                    owner='coverage_model.test.test_postgres_storage',
                                    func_name='identity',
                                    arg_list=['x'],
                                    param_map={'x':'quantity'})
        print "Made it this far too"

        pf_context = ParameterContext('identity', param_type=ParameterFunctionType(pf_example))

        cov = _make_cov(self.working_dir, ['quantity', pf_context], nt=0)
        import time
        ntp_now = time.time() + 2208988800

        data_dict = {
            'time' : NumpyParameterData('time', np.arange(ntp_now , ntp_now + 10)),
            'quantity' : NumpyParameterData('quantity', np.arange(10))
        }
        cov.set_parameter_values(data_dict)


        retval = cov.get_parameter_values(['identity']).get_data()['identity']
        np.testing.assert_allclose(retval, np.arange(10))



def identity(x):
    return np.copy(x)
