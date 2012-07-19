#!/usr/bin/env python

"""
@package 
@file common
@author Christopher Mueller
@brief 
"""

import uuid
def create_guid():
    """
    @retval Return global unique id string
    """
    # guids seem to be more readable if they are UPPERCASE
    return str(uuid.uuid4()).upper()

class AbstractBase(object):
    """

    """
    def __init__(self):
        self.mutable = False
        self.extension = {}

class AbstractIdentifiable(AbstractBase):
    # TODO: Need to decide if identity is assigned on creation or on demand (when asked for)
    """

    """
    def __init__(self):
        AbstractBase.__init__(self)
        self._id = create_guid()
        self.label = ''
        self.description = ''

    @property
    def id(self):
        return self._id


class AxisTypeEnum(object):
    """
    Temporarily taken from: http://www.unidata.ucar.edu/software/netcdf-java/v4.2/javadoc/ucar/nc2/constants/AxisType.html
    """
    # Ensemble represents the ensemble coordinate
    ENSEMBLE = 0

    # GeoX represents a x coordinate
    GEO_X = 1

    # GeoY represents a y coordinate
    GEO_Y = 2

    # GeoZ represents a z coordinate
    GEO_Z = 3

    # Height represents a vertical height coordinate
    HEIGHT = 4

    # Lat represents a latitude coordinate
    LAT = 5

    # Lon represents a longitude coordinate
    LON = 6

    # Pressure represents a vertical pressure coordinate
    PRESSURE = 7

    # RadialAzimuth represents a radial azimuth coordinate
    RADIAL_AZIMUTH = 8

    # RadialDistance represents a radial distance coordinate
    RADIAL_DISTANCE = 9

    # RadialElevation represents a radial elevation coordinate
    RADIAL_ELEVATION = 10

    # RunTime represents the runTime coordinate
    RUNTIME = 11

    # Time represents the time coordinate
    TIME = 12

    _value_map = {'ENSAMBLE':0, 'GEO_X':1 , 'GEO_Y':2, 'GEO_Z':3, 'HEIGHT':4, 'LAT':5, 'LON':6, 'PRESSURE':7, 'RADIAL_AZIMUTH':8, 'RADIAL_DISTANCE':9, 'RADIAL_ELEVATION':10, 'RUNTIME':11, 'TIME':12, }
    _str_map = {0:'ENSAMBLE', 1:'GEO_X' , 2:'GEO_Y', 3:'GEO_Z', 4:'HEIGHT', 5:'LAT', 6:'LON', 7:'PRESSURE', 8:'RADIAL_AZIMUTH', 9:'RADIAL_DISTANCE', 10:'RADIAL_ELEVATION', 11:'RUNTIME', 12:'TIME', }

    @classmethod
    def get_member(cls, value):
        if isinstance(value, int):
            return AxisTypeEnum.__getattribute__(cls, AxisTypeEnum._str_map[value])
        elif isinstance(value, (str,unicode)):
            return AxisTypeEnum.__getattribute__(cls, value.upper())
        else:
            raise TypeError('AxisTypeEnum has no member: {0}'.format(value))

    @classmethod
    def is_member(cls, value, want):
        v=AxisTypeEnum.get_member(value)
        return v == want

class MutabilityEnum(object):
    IMMUTABLE = 1
    EXTENSIBLE = 2
    MUTABLE = 3
    _value_map = {'IMMUTABLE': 1, 'EXTENSIBLE': 2, 'MUTABLE': 3,}
    _str_map = {1: 'IMMUTABLE', 2: 'EXTENSIBLE', 3: 'MUTABLE'}

    @classmethod
    def get_member(cls, value):
        if isinstance(value, int):
            return MutabilityEnum.__getattribute__(cls, MutabilityEnum._str_map[value])
        elif isinstance(value, (str,unicode)):
            return MutabilityEnum.__getattribute__(cls, value.upper())
        else:
            raise TypeError('AxisTypeEnum has no member: {0}'.format(value))

    @classmethod
    def is_member(cls, value, want):
        v=MutabilityEnum.get_member(value)
        return v == want
