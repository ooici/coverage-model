__author__ = 'casey'

import shutil
import tempfile

from nose.plugins.attrib import attr
from pyon.core.bootstrap import CFG
from pyon.datastore.datastore_common import DatastoreFactory
import psycopg2
import psycopg2.extras
from coverage_model import *
from coverage_model.address import *
from coverage_model.parameter_data import *
from coverage_test_base import get_parameter_dict


@attr('INT',group='cov')
class TestSpanValidation(CoverageModelUnitTestCase):
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
            return scov, 'TestSpanValidationUnit'
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

                scov.set_parameter_values(np_dict)

        cls.coverages.add(scov.persistence_guid)
        return scov, 'TestSpanValidationInt'

    def test_store_data(self):
        ts = 10
        scov, cov_name = self.construct_cov(nt=ts)
        self.assertIsNotNone(scov)

        valid_spans, invalid_spans = scov.validate_span_data()

        self.assertEqual(0, len(invalid_spans))