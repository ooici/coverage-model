__author__ = 'casey'

from ooi.logging import log
from pyon.util.config import Config
import datetime


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class CoverageConfig(object):
    __metaclass__ = Singleton
    _default_ordered_time_key_preferences = ['internal_timestamp', 'time', 'driver_timestamp',
                                             'm_present_time', 'preferred_timestamp']
    _default_ordered_lat_key_preferences = ['m_lat', 'm_gps_lat', 'c_wpt_lat', 'lat']
    _default_ordered_lon_key_preferences = ['m_lon', 'm_gps_lon', 'c_wpt_lon', 'lon']
    _default_ordered_vertical_key_preferences = ['m_depth', 'depth']
    _default_time_db_key = 'time_range'
    _default_geo_db_key = 'spatial_geometry'
    _default_vertical_db_key = 'vertical_range'
    _default_span_id_db_key = 'span_address'
    _default_span_coverage_id_db_key = 'coverage_id'
    _default_storage_location = None

    def __init__(self):
        print 'setting defaults'
        self.ordered_time_key_preferences = self._default_ordered_time_key_preferences
        self.ordered_lat_key_preferences = self._default_ordered_lat_key_preferences
        self.ordered_lon_key_preferences = self._default_ordered_lon_key_preferences
        self.ordered_vertical_key_preferences = self._default_ordered_vertical_key_preferences
        self.time_db_key = self._default_time_db_key
        self.geo_db_key = self._default_geo_db_key
        self.vertical_db_key = self._default_vertical_db_key
        self.span_id_db_key = self._default_span_id_db_key
        self.span_coverage_id_db_key = self._default_span_coverage_id_db_key
        self.top_level_storage_location = self._default_storage_location
        self.using_default_config = True
        self.config_time = 0
        self.read_and_set_config()

    def read_and_set_config(self):
        one_from_config = False
        try:
            data = Config(["res/config/coverage.yml"]).data['CoverageConfig']
            for k, v in data.iteritems():
                self.__setattr__(k, v)
                one_from_config = True
            self.using_default_config = False
            self.config_time = datetime.datetime.utcnow().time()
        except Exception as ex:
            if one_from_config:
                log.info("Load from config failed with '%s'.  Using hybrid default/config file configuration" % ex.message)
                return False
            else:
                log.info("load from config failed with '%s'.  Using default config" % ex.message)
        return True

    def __setattr__(self, key, value):
        if key.endswith('_'):
            for k, v in value.iteritems():
                self.__setattr__(k, v)
        else:
            super(CoverageConfig, self).__setattr__(key,value)

    def get_preferred_key(self, options, preferences):
        for key in preferences:
            if key in options:
                return key
        return None

    def get_lon_key(self, options):
        return self.get_preferred_key(options, self.ordered_lon_key_preferences)

    def get_lat_key(self, options):
        return self.get_preferred_key(options, self.ordered_lat_key_preferences)

    def get_time_key(self, options):
        return self.get_preferred_key(options, self.ordered_time_key_preferences)

    def get_vertical_key(self, options):
        return self.get_preferred_key(options, self.ordered_vertical_key_preferences)

    def get_coverage_class(self, type, version=None):
        if type == 'complex':
            module = __import__('coverage_model.coverages.complex_coverage')
            return getattr(module, 'ComplexCoverage')
        if type == 'simplex':
            module = __import__('coverage_model.coverage')
            return getattr(module, 'SimplexCoverage')
        return None
