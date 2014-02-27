__author__ = 'casey'

from ooi.logging import log
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

    def __init__(self):
        self.ordered_time_key_preferences = self._default_ordered_time_key_preferences
        self.ordered_lat_key_preferences = self._default_ordered_lat_key_preferences
        self.ordered_lon_key_preferences = self._default_ordered_lon_key_preferences
        self.ordered_vertical_key_preferences = self._default_ordered_vertical_key_preferences
        self.using_default_config = True
        self.config_time = 0
        self.read_and_set_config()

    def read_and_set_config(self):
        is_success = False
        if is_success is True:
            self.using_default_config = False
            self.config_time = datetime.datetime.utcnow().time()
        return is_success