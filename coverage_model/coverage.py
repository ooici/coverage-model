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

class AbstractBase():
    """

    """
    extension = {}

class AbstractIdentifiable(AbstractBase):
    """

    """
    identifier = None
    label = ''
    description = ''

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
    _reference_coverage = AbstractCoverage()
    _structure_filter = StructureFilter()
    _parameter_filter = ParameterFilter()

    def __init__(self, reference_coverage, structure_filter=None, parameter_filter=None):
        AbstractCoverage.__init__(self)
        self._reference_coverage = reference_coverage
        self._structure_filter = structure_filter or StructureFilter()
        self._parameter_filter = parameter_filter or ParameterFilter()

class ComplexCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1-n coverages
    """
    _coverages = []
    def __init__(self, coverages=None):
        AbstractCoverage.__init__(self)
        self._coverages.extend([x for x in coverages if isinstance(x, AbstractCoverage)])


class SimplexCoverage(AbstractCoverage):
    """
    
    """

    _range_dictionary = RangeDictionary()
    _spatial_domain = AbstractDomain(None)
    _temporal_domain = GridDomain(None)
    _range_type = RangeType()
    _range_dictionary =RangeDictionary()

    def __init__(self, range_dictionary, spatial_domain, temporal_domain=None):
        AbstractCoverage.__init__(self)

        if not isinstance(range_dictionary, RangeDictionary):
            raise TypeError('record_dictionary must be a RangeDictionary object')
        self._range_dictionary = range_dictionary

        if not isinstance(spatial_domain, AbstractDomain):
            raise TypeError('spatial_domain must be an AbstractDomain object')
        self._spatial_domain = spatial_domain

        if not temporal_domain is None and not isinstance(temporal_domain, AbstractDomain):
            raise TypeError('spatial_domain must be an AbstractDomain object')
        self._temporal_domain = temporal_domain or GridDomain(None)


class Parameter(AbstractIdentifiable):
    """

    """
    def __init__(self, name, param_type, param_value, isCoordinate=False):
        AbstractIdentifiable.__init__(self)

        self.name = name
        self.isCoordinate = isCoordinate
        if not isinstance(param_type, AbstractParameterType):
            raise TypeError('param_type must be a ParameterType object') # I agree
        self._type = param_type
        if not isinstance(param_value, AbstractParameterValue):
            raise TypeError('param_value must be a ParameterValue object')
        self._value = param_value


#################
# Range Objects
#################

class AbstractRange():
    """
    NOT an AbstractIdentifiable!!
    """
    pass

class RangeType(AbstractIdentifiable):
    """
    May be synonymous with RangeDictionary
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)

class RangeDictionary(AbstractIdentifiable):
    """
    Currently known as Taxonomy with respect to the Granule & RecordDictionary
    May be synonymous with RangeType
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


#################
# Parameter Value Objects
#################

class AbstractParameterValue(AbstractRange):
    """

    """
    def __init__(self):
        AbstractRange.__init__(self)

class AbstractSimplexComponent(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)

class AbstractComplexComponent(AbstractParameterValue):
    """

    """
    def __init__(self):
        AbstractParameterValue.__init__(self)

class Boolean(AbstractSimplexComponent):
    """

    """
    def __init__(self):
        AbstractSimplexComponent.__init__(self)

class Category(AbstractSimplexComponent):
    """

    """
    pass

class CategoryRange(AbstractSimplexComponent):
    """

    """
    pass

class Count(AbstractSimplexComponent):
    """

    """
    pass

class CountRange(AbstractSimplexComponent):
    """

    """
    pass

class Quantity(AbstractSimplexComponent):
    """

    """
    pass

class QuantityRange(AbstractSimplexComponent):
    """

    """
    pass

class Text(AbstractSimplexComponent):
    """

    """
    pass

class Time(AbstractSimplexComponent):
    """

    """
    pass

class TimeRange(AbstractSimplexComponent):
    """

    """
    pass

class DataRecord(AbstractComplexComponent):
    """
    Heterogeneous set of named things (dict)
    """
    pass

class Vector(AbstractComplexComponent):
    """
    Heterogeneous set of unnamed things (tuple)
    """
    pass

class DataArray(AbstractComplexComponent):
    """
    Homogeneous set of unnamed things (array)
    """
    pass



#################
# Parameter Type Objects
#################

class AbstractParameterType(RangeType):
    """

    """
    abstract_data_component = None
    pass

class AbstractSimplexComponentType(AbstractParameterType):
    """

    """
    pass

class AbstractComplexComponentType(AbstractParameterType):
    """

    """
    pass

class BooleanType(AbstractSimplexComponentType):
    """

    """
    pass

class CategoryType(AbstractSimplexComponentType):
    """

    """
    pass

class CategoryRangeType(AbstractSimplexComponentType):
    """

    """
    pass

class CountType(AbstractSimplexComponentType):
    """

    """
    pass

class CountRangeType(AbstractSimplexComponentType):
    """

    """
    pass

class QuantityType(AbstractSimplexComponentType):
    """

    """
    pass

class QuantityRangeType(AbstractSimplexComponentType):
    """

    """
    pass

class TextType(AbstractSimplexComponentType):
    """

    """
    pass

class TimeType(AbstractSimplexComponentType):
    """

    """
    pass

class TimeRangeType(AbstractSimplexComponentType):
    """

    """
    pass

class DataRecordType(AbstractComplexComponentType):
    """

    """
    pass

class VectorType(AbstractComplexComponentType):
    """

    """
    pass

class DataArrayType(AbstractComplexComponentType):
    """

    """
    pass


#################
# Domain Objects
#################

class AbstractDomain(AbstractIdentifiable):
    """

    """
    def __init__(self, abstract_shape):
        AbstractIdentifiable.__init__(self)
        self.shape = []
        self.shape.append(abstract_shape)

    def extend(self, extend_by_count, dim_index=None):
        raise NotImplementedError()

class GridDomain(AbstractDomain):
    """

    """
    def __init__(self, abstract_shape):
        AbstractDomain.__init__(self, abstract_shape)

    def extend(self, extend_by_count, dim_index=None):
        dim_index = dim_index or 0

class TopologicalDomain(AbstractDomain):
    """

    """
    def __init__(self, abstract_shape):
        AbstractDomain.__init__(self, abstract_shape)

class AbstractShape():
    """

    """
    pass

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims

class AbstractFilter(AbstractIdentifiable):
    """

    """
    def __init__(self, abstract_shape):
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

class Topology():
    """
    Sets of topological entities
    Mappings between topological entities
    """
    pass

class TopologicalEntity():
    """

    """
    pass

class ConstructionType():
    """
    Lattice or not (i.e. 'unstructured')
    """
    pass