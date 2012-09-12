#!/usr/bin/env python

"""
@package coverage_model.basic_types
@file coverage_model/basic_types.py
@author Christopher Mueller
@brief Base classes for Coverage Model and Parameter classes
"""

from pyon.public import log
import numpy as np
import uuid

def create_guid():
    """
    @retval Return global unique id string
    """
    # guids seem to be more readable if they are UPPERCASE
    return str(uuid.uuid4()).upper()

class Dictable(object):
    def _todict(self):
        ret = dict((k,v._todict() if hasattr(v, '_todict') else v) for k, v in self.__dict__.iteritems())

        ret['cm_type'] = (self.__module__, self.__class__.__name__)
        return ret

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        arg_masks = arg_masks or {}
        log.debug('_fromdict: cls=%s',cls)
        if isinstance(cmdict, dict) and 'cm_type' in cmdict and cmdict['cm_type']:
            import inspect
            mod = inspect.getmodule(cls)
            ptmod_s, ptcls_s=cmdict.pop('cm_type')
            ptcls=getattr(mod, ptcls_s)

            # Get the argument specification for the initializer
            spec = inspect.getargspec(ptcls.__init__)
            args = spec.args[1:] # get rid of 'self'
            # Remove any optional arguments
            if spec.defaults: # if None, all are required
                for i in spec.defaults:
                    args.remove(args[-1])
            kwa={}
            for a in args:
                # Apply any argument masking
                am = arg_masks[a] if a in arg_masks else a
                if am in cmdict:
                    val = cmdict.pop(am)
                    if isinstance(val, dict) and 'cm_type' in val:
                        ms, cs = val['cm_type']
                        module = __import__(ms, fromlist=[cs])
                        classobj = getattr(module, cs)
                        kwa[a] = classobj._fromdict(val)
                    else:
                        kwa[a] = val
                else:
                    kwa[a] = None

            ret = ptcls(**kwa)
            for k,v in cmdict.iteritems():
                if isinstance(v, dict) and 'cm_type' in v:
                    ms, cs = v['cm_type']
                    module = __import__(ms, fromlist=[cs])
                    classobj = getattr(module, cs)
                    setattr(ret,k,classobj._fromdict(v))
                else:
                    setattr(ret,k,v)


            return ret
        else:
            raise TypeError('cmdict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(cmdict))

class AbstractBase(Dictable):
    """
    Base class for all coverage model objects

    Provides id, mutability and extension attributes
    """
    def __init__(self, id=None, mutable=None, extension=None):
        """
        Construct a new AbstractBase object

        @param id   The ID of this object type
        @param mutable  The mutability of the object; defaults to False
        """
        self._id = id
        self.mutable = mutable or False
        self.extension = extension or {}

    @property
    def id(self):
        return self._id

class AbstractIdentifiable(AbstractBase):
    """
    Base identifiable class for all coverage model objects

    Provides identifier, label and description attributes
    """
    def __init__(self, identifier=None, label=None, description=None, **kwargs):
        """
        Construct a new AbstractIdentifiable object

        @param identifier   The UUID of this 'instance'
        @param label    The short description of the object
        @param description  The full description of the object
        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractBase; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractBase.__init__(self, **kwc)
        # TODO: Need to decide if identifier is assigned on creation or on demand (when asked for); using the latter for the time being
        self._identifier = identifier
        self.label = label or ''
        self.description = description or ''

    @property
    def identifier(self):
        """
        The UUID of the object.  Generated on first request.
        """
        if self._identifier is None:
            self._identifier = create_guid()

        return self._identifier

class DomainOfApplication(object):
    # CBM: Document this!!
    def __init__(self, slices, topoDim=None):
        if slices is None:
            raise StandardError('\'slices\' cannot be None')
        self.topoDim = topoDim or 0

        if is_valid_constraint(slices):
            if not np.iterable(slices):
                slices = [slices]

            self.slices = slices
        else:
            raise StandardError('\'slices\' must be either single, tuple, or list of slice or int objects')

    def __iter__(self):
        return self.slices.__iter__()

    def __len__(self):
        return len(self.slices)

def is_valid_constraint(v):
    ret = False
    if isinstance(v, (slice, int)) or\
       (isinstance(v, (list,tuple)) and np.array([is_valid_constraint(e) for e in v]).all()):
        ret = True

    return ret

def get_valid_DomainOfApplication(v, valid_shape):
    """
    Takes the value to validate and a tuple representing the valid_shape
    """

    if v is None:
        if len(valid_shape) == 1:
            v = slice(None)
        else:
            v = [slice(None) for x in valid_shape]

    if not isinstance(v, DomainOfApplication):
        v = DomainOfApplication(v)

    return v

class AxisTypeEnum(object):
    """
    Enumeration of Axis Types used when building CRS objects and assigning the ParameterContext.reference_frame

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
    """
    Enumeration of mutability types used in creation of concrete AbstractDomain objects
    """
    # NTK: mutable indicates ...

    ## Indicates the domain cannot be changed
    IMMUTABLE = 1
    ## Indicates the domain can be expanded in any of it's dimensions
    EXTENSIBLE = 2
    ## NTK: Indicates ...
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
            raise TypeError('MutabilityEnum has no member: {0}'.format(value))

    @classmethod
    def is_member(cls, value, want):
        v=MutabilityEnum.get_member(value)
        return v == want

class VariabilityEnum(object):
    """
    Enumeration of Variability used when building ParameterContext objects
    """
    ## Indicates that the associated object is variable temporally, but not spatially
    TEMPORAL = 1
    ## Indicates that the associated object is variable spatially, but not temporally
    SPATIAL = 2
    ## Indicates that the associated object is variable both temporally and spatially
    BOTH = 3
    ## Indicates that the associated object has NO variability; is constant in both space and time
    NONE = 99
    _value_map = {'TEMPORAL': 1, 'SPATIAL': 2, 'MUTABLE': 3, 'NONE': 99}
    _str_map = {1: 'TEMPORAL', 2: 'SPATIAL', 3: 'BOTH', 99: 'NONE'}

    @classmethod
    def get_member(cls, value):
        if isinstance(value, int):
            return VariabilityEnum.__getattribute__(cls, VariabilityEnum._str_map[value])
        elif isinstance(value, (str,unicode)):
            return VariabilityEnum.__getattribute__(cls, value.upper())
        else:
            raise TypeError('VariabilityEnum has no member: {0}'.format(value))

    @classmethod
    def is_member(cls, value, want):
        v=VariabilityEnum.get_member(value)
        return v == want