#!/usr/bin/env python

"""
@package 
@file coverage
@author Christopher Mueller
@brief 
"""

# http://collabedit.com/kkacp


# Notes:
# cell_map[cell_handle]=(0d_handles,)
# 0d_map[0d_handle]=(cell_handles,)
# ** Done via an association table of 2 arrays - indices are the alignment
#
# parameter values implicitly aligned to 0d handle array
#
# shape applied to 0d handle array
#
# cell is 'top level dimension'
#
# parameters can be implicitly aligned to cell array

# TODO: All implementation is 'pre-prototype' - only intended to flesh out the API

#CBM: see next line
#TODO: Add type checking throughout all classes as determined appropriate, ala:
#@property
#def spatial_domain(self):
#    return self.__spatial_domain
#
#@spatial_domain.setter
#def spatial_domain(self, value):
#    if isinstance(value, AbstractDomain):
#        self.__spatial_domain = value

from coverage_model.basic_types import *
import numpy as np
import pickle

#################
# Coverage Objects
#################

class AbstractCoverage(AbstractIdentifiable):
    """
    Core data model, persistence, etc
    TemporalTopology
    SpatialTopology
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)

class ViewCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1 AbstractCoverage and applies a Filter
    """
    def __init__(self):
        AbstractCoverage.__init__(self)

        self.reference_coverage = ''
        self.structure_filter = StructureFilter()
        self.parameter_filter = ParameterFilter()

class ComplexCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1-n coverages
    """
    def __init__(self):
        AbstractCoverage.__init__(self)
        self.coverages = []

class SimplexCoverage(AbstractCoverage):
    """
    
    """
    def __init__(self, range_dictionary, spatial_domain, temporal_domain=None):
        AbstractCoverage.__init__(self)

        self.range_dictionary = range_dictionary
        self.spatial_domain = spatial_domain
        self.temporal_domain = temporal_domain or GridDomain(GridShape('temporal',[0]), None, None)
        self.range_type = {}
        self.range_ = RangeGroup()
        self._pcmap = {}
        self._temporal_param_name = None

    @classmethod
    def save(cls, cov_obj, file_path, use_ascii=False):
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('Object must be an instance or subclass of AbstractCoverage, not {0}'.format(type(cov_obj)))
        with open(file_path, 'w') as f:
            pickle.dump(cov_obj, f, 0 if use_ascii else 2)

        print 'Saved coverage to \'{0}\''.format(file_path)

    @classmethod
    def load(cls, file_path):
        with open(file_path, 'r') as f:
            obj = pickle.load(f)

        print 'Loaded coverage from {0}'.format(file_path)
        return obj

    # TODO: If we are going to call out separate functions for time, we should have a corresponding way to add the temporal parameter
    def append_parameter(self, parameter_context, temporal=False):
        pname = parameter_context.name

        if temporal:
            if self._temporal_param_name is None:
                self._temporal_param_name = pname
            else:
                raise StandardError("temporal_parameter already defined.")

        # Get the correct domain and array shape
        if temporal:
            dom = self.temporal_domain
            shp = self.temporal_domain.shape.extents
        else:
            dom = self.spatial_domain
            if len(self.spatial_domain.shape.extents) == 1:
                shp = self.temporal_domain.shape.extents
            else:
                shp = self.temporal_domain.shape.extents + self.spatial_domain.shape.extents

        # Assign the pname to the CRS (if applicable)
        if not parameter_context.axis is None:
            if parameter_context.axis in dom.crs.axes:
                dom.crs.axes[parameter_context.axis] = parameter_context.name

        self._pcmap[pname] = (len(self._pcmap), parameter_context, dom)
        self.range_type[pname] = parameter_context
        self.range_[pname] = RangeMember(np.zeros(shp, parameter_context.param_type))
        setattr(self.range_, pname, self.range_[pname])

    def get_parameter(self, param_name):
        if param_name in self.range_:
            return self.range_type[param_name], self.range_[param_name].content

    def insert_timesteps(self, count, origin=None):
        if not origin is None:
            raise SystemError('Only append is currently supported')

        for n in self._pcmap:
            arr = self.range_[n].content
            arr = np.append(arr, np.zeros((count,) + arr.shape[1:]), 0)
            self.range_[n].content = arr

#    def set_time_value(self, time_index, time_value):
#        pass
#
#    def get_time_value(self, time_index):
#        pass

    def set_time_values(self, tdoa, values):
        return self.set_parameter_values(self._temporal_param_name, tdoa, None, values)

    def get_time_values(self, tdoa=None, return_array=None):
        return self.get_parameter_values(self._temporal_param_name, tdoa, None, return_array)

#    def set_parameter_values_at_time(self, param_name, time_index, values, doa):
#        pass
#
#    def get_parameter_values_at_time(self, param_name, time_index, doa):# doa?
#        pass

    def set_parameter_values(self, param_name, tdoa, sdoa, values):
        if not param_name in self.range_:
            raise StandardError('Parameter \'{0}\' not found in coverage'.format(param_name))

        tdoa = _get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        sdoa = _get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)

        print 'temporal doa: {0}'.format(tdoa.slices)
        print 'spatial doa: {0}'.format(sdoa.slices)

        slice_ = [tdoa.slices, sdoa.slices]

        print 'Setting slice: {0}'.format(slice_)

        #TODO: Do we need some validation that slice_ is the same rank and/or shape as values?
        self.range_[param_name][slice_] = values

    def get_parameter_values(self, param_name, tdoa=None, sdoa=None, return_array=None):
        if not param_name in self.range_:
            raise StandardError('Parameter \'{0}\' not found in coverage'.format(param_name))

        return_array = return_array or np.zeros([0])

        tdoa = _get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        sdoa = _get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)

        print 'temporal doa: {0}'.format(tdoa.slices)
        print 'spatial doa: {0}'.format(sdoa.slices)

        slice_ = [tdoa.slices, sdoa.slices]

        print 'Getting slice: {0}'.format(slice_)

        return_array = self.range_[param_name][slice_]
        return return_array


#################
# Range Objects
#################

#TODO: Consider usage of Ellipsis in all this slicing stuff as well

class DomainOfApplication(object):

    def __init__(self, topoDim, slices):
        self.topoDim = topoDim
        if _is_valid_constraint(slices):
            if not isinstance(slices, slice) and not np.iterable(slices):
                slices = [slices]

            self.slices = slices

    def __iter__(self):
        return self.slices.__iter__()

    def __len__(self):
        return len(self.slices)

def _is_valid_constraint(v):
    ret = False
    if isinstance(v, (slice, int)) or \
       (isinstance(v, (list,tuple)) and np.array([_is_valid_constraint(e) for e in v]).all()):
            ret = True

    return ret

def _get_valid_DomainOfApplication(v, valid_shape):
    """
    Takes the value to validate and a tuple representing the valid_shape
    """

    if v is None:
        if len(valid_shape) == 1:
            v = slice(None)
        else:
            v = [slice(None) for x in valid_shape]

    if not isinstance(v, DomainOfApplication):
        if _is_valid_constraint(v):
            if not isinstance(v, slice) and not np.iterable(v):
                v = [v,]
            v = DomainOfApplication(0, v)
        else:
            raise StandardError('value must be either a DomainOfApplication or a single, tuple, or list of slice or int objects')

    return v

class RangeGroup(dict):
    """
    All the functionality of a built_in dict, plus the ability to use setattr to allow dynamic addition of params (RangeMember)
    """
    pass

class RangeMember(object):
    """
    This is what would provide the "abstraction" between the in-memory model and the underlying persistence - all through __getitem__ and __setitem__
    Mapping between the "complete" domain and the storage strategy can happen here
    """

    def __init__(self, arr_obj):
        self._arr_obj = arr_obj

    @property
    def shape(self):
        return self._arr_obj.shape

    @property
    def content(self):
        return self._arr_obj

    @content.setter
    def content(self, value):
        self._arr_obj=value

    # CBM: First swack - see this for more possible checks: http://code.google.com/p/netcdf4-python/source/browse/trunk/netCDF4_utils.py
    def __getitem__(self, slice_):
        if not _is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a tuple
        if not np.iterable(slice_):
            slice_ = (slice_,)
        elif not isinstance(slice_,tuple):
            slice_ = tuple(slice_)

        # Then make it's the correct shape TODO: Should reference the shape of the Domain object
        alen = len(self._arr_obj.shape)
        slen = len(slice_)
        if not slen == alen:
            if slen > alen:
                slice_ = slice_[:alen]
            else:
                for n in range(slen, alen):
                    slice_ += (slice(None,None,None),)

        return self._arr_obj[slice_]

    def __setitem__(self, slice_, value):
        if not _is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a tuple
        if not np.iterable(slice_):
            slice_ = (slice_,)
        elif not isinstance(slice_,tuple):
            slice_ = tuple(slice_)

        # Then make it's the correct shape TODO: Should reference the shape of the Domain object
        alen = len(self._arr_obj.shape)
        slen = len(slice_)
        if not slen == alen:
            if slen > alen:
                slice_ = slice_[:alen]
            else:
                for n in range(slen, alen):
                    slice_ += (slice(None,None,None),)

        self._arr_obj[slice_] = value

    def __str__(self):
        print self._arr_obj

class RangeDictionary(AbstractIdentifiable):
    """
    Currently known as Taxonomy with respect to the Granule & RecordDictionary
    May be synonymous with RangeType
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


#################
# Abstract Parameter Value Objects
#################

class AbstractParameterValue(AbstractBase):
    """

    """
    def __init__(self):
        AbstractBase.__init__(self)

class AbstractSimplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)

class AbstractComplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)


#################
# Abstract Parameter Type Objects
#################

class AbstractParameterType(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)

class AbstractSimplexParameterType(AbstractParameterType):
    """

    """
    def __init__(self):
        AbstractParameterType.__init__(self)

class AbstractComplexParameterType(AbstractParameterType):
    """

    """
    def __init__(self):
        AbstractParameterType.__init__(self)


#################
# CRS Objects
#################
class CRS(AbstractIdentifiable):
    """

    """
    def __init__(self, axis_labels):
        AbstractIdentifiable.__init__(self)
        self.axes={}
        for l in axis_labels:
            if l in AxisTypeEnum._str_map or l in AxisTypeEnum._value_map:
                self.axes[l]=None
            else:
                raise SystemError('Unknown AxisType: {0}'.format(l))

    @classmethod
    def standard_temporal(cls):
        return CRS([AxisTypeEnum.TIME])

    @classmethod
    def lat_lon_height(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    @classmethod
    def lat_lon(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    @classmethod
    def x_y_z(cls):
        return CRS([AxisTypeEnum.GEO_X, AxisTypeEnum.GEO_Y, AxisTypeEnum.GEO_Z])

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

#################
# Domain Objects
#################

class MutabilityEnum(object):
    IMMUTABLE = 1
    EXTENSIBLE = 2
    MUTABLE = 3
    _value_map = {'IMMUTABLE': 1, 'EXTENSIBLE': 2, 'MUTABLE': 3,}
    _str_map = {1: 'IMMUTABLE', 2: 'EXTENSIBLE', 3: 'MUTABLE'}

class AbstractDomain(AbstractIdentifiable):
    """

    """
    def __init__(self, shape, crs, mutability):
        AbstractIdentifiable.__init__(self)
        self.shape = shape
        self.crs = crs
        self.mutability = mutability or MutabilityEnum.IMMUTABLE

#    def get_mutability(self):
#        return self.mutability

    def get_max_dimension(self):
        pass

    def get_num_elements(self, dim_index):
        pass

#    def get_shape(self):
#        return self.shape

    def insert_elements(self, dim_index, count, doa):
        pass

class GridDomain(AbstractDomain):
    """

    """
    def __init__(self, shape, crs, mutability=None):
        AbstractDomain.__init__(self, shape, crs, mutability)

class TopologicalDomain(AbstractDomain):
    """

    """
    def __init__(self):
        AbstractDomain.__init__(self)


#################
# Shape Objects
#################

class AbstractShape(AbstractIdentifiable):
    """

    """
    def __init__(self, name, extents):
        AbstractIdentifiable.__init__(self)
        self.name = name
        self.extents = extents

    @property
    def rank(self):
        return len(self.extents)

#    def rank(self):
#        return len(self.extents)

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, extents):
        AbstractShape.__init__(self, name, extents)

    #CBM: Make extents type-safe


#################
# Filter Objects
#################

class AbstractFilter(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


class StructureFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)

class ParameterFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)



#################
# Possibly OBE ?
#################

#class Topology():
#    """
#    Sets of topological entities
#    Mappings between topological entities
#    """
#    pass
#
#class TopologicalEntity():
#    """
#
#    """
#    pass
#
#class ConstructionType():
#    """
#    Lattice or not (i.e. 'unstructured')
#    """
#    pass