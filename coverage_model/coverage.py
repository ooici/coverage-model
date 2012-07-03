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
        self._tparam_name = None

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
            if self._tparam_name is None:
                self._tparam_name = pname
            else:
                raise StandardError("temporal_parameter already defined.")

        # Get the correct domain
        dom = self.temporal_domain if temporal else self.spatial_domain

        # Assign the pname to the CRS (if applicable)
        if not parameter_context.axis is None:
            if parameter_context.axis in dom.crs.axes:
                dom.crs.axes[parameter_context.axis] = parameter_context.name

        self._pcmap[pname] = (len(self._pcmap), parameter_context, dom)
        self.range_type[pname] = parameter_context
        self.range_[pname] = RangeMember(np.zeros(dom.shape.dims, parameter_context.param_type))
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

    def set_time_value(self, time_index, time_value):
        pass

    def get_time_value(self, time_index):
        pass

    def set_time_values(self, doa, values):
        pass

    def get_time_values(self, doa, return_array):
        pass

    def set_parameter_values_at_time(self, param_name, time_index, values, doa):
        pass

    def get_parameter_values_at_time(self, param_name, time_index, doa):# doa?
        pass

    def set_parameter_values(self, param_name, tdoa, sdoa, values):
        pass

    def get_parameter_values(self, param_name, tdoa, sdoa, return_array):
        pass


#################
# Range Objects
#################

#TODO: Consider usage of Ellipsis in all this slicing stuff as well

class DomainOfApplication(object):

    def __init__(self, topoDim, slices):
        self.topoDim = topoDim
        if _is_valid_constraint(slices):
            self.slices = slices

    def __iter__(self):
        return self.slices.__iter__()

    def __len__(self):
        return len(self.slices)

def _is_valid_constraint(v):
    ret = False
    if isinstance(v, slice) or \
       isinstance(v, int) or \
       (isinstance(v, (list,tuple)) and np.array([_is_valid_constraint(e) for e in v]).all()):
            ret = True

    return ret

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
            self.axes[l]=None

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
    def __init__(self, name, dims):
        AbstractIdentifiable.__init__(self)
        self.name = name
        self.dims = dims

    @property
    def rank(self):
        return len(self.dims)

#    def rank(self):
#        return len(self.dims)

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, dims):
        AbstractShape.__init__(self, name, dims)

    #CBM: Make dims type-safe


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