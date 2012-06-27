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



class AbstractCoverage():
    """
    Core data model, persistence, etc
    TemporalTopology
    SpatialTopology
    """
    def __init__(self):
        pass

class ViewCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1 AbstractCoverage and applies a Filter
    """
    def __init__(self, reference_coverage, structure_filter=None, parameter_filter=None):
        AbstractCoverage.__init__(self)
        self.reference_coverage = reference_coverage
        self.structure_filter = structure_filter or StructureFilter()
        self.parameter_filter = parameter_filter or ParameterFilter()

class ComplexCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1-n coverages
    """
    def __init__(self, coverages=None):
        AbstractCoverage.__init__(self)
        self.coverages=[]
        self.coverages.extend([x for x in coverages if isinstance(x, AbstractCoverage)])


class SimplexCoverage(AbstractCoverage):
    """
    
    """

    def __init__(self, record_dictionary, spatial_domain, temporal_domain=None):
        AbstractCoverage.__init__(self)

        if not isinstance(record_dictionary, RecordDictionary):
            raise TypeError('record_dictionary must be a RecordDictionary object')
        self._record_dictionary = record_dictionary

        if not isinstance(spatial_domain, AbstractDomain):
            raise TypeError('spatial_domain must be an AbstractDomain object')
        self._spatial_domain = spatial_domain

        if not temporal_domain is None and not isinstance(temporal_domain, AbstractDomain):
            raise TypeError('spatial_domain must be an AbstractDomain object')
        self._temporal_domain = temporal_domain or GridDomain()


class Parameter():
    """

    """
    def __init__(self, name, param_type, param_value, isCoordinate=False, atts=None, value=None):
        self.name = name
        self.isCoordinate = isCoordinate
        if not isinstance(param_type, ParameterType):
            raise TypeError('param_type must be a ParameterType object') # I agree
        self._type = param_type
        if not isinstance(param_value, ParameterValue):
            raise TypeError('param_value must be a ParameterValue object')
        self._value = param_value


class ParameterType():
    """

    """
    pass

class ParameterValue():
    """

    """
    pass

class Range():
    pass

class RangeType():
    pass

class RangeDictionary():
    """
    Currently known as Taxonomy with respect to the Granule & RecordDictionary
    """
    pass




class AbstractDomain():
    """

    """
    def __init__(self, abstract_shape):
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

class CoverageConstruction():
    """
    Presents higher level construct on top of a base coverage
    """
    def __init__(self, ConstructionType, construction_map, Coverage):
        pass

class Filter():
    """

    """
    pass

class StructureFilter(Filter):
    """

    """
    pass

class ParameterFilter(Filter):
    """

    """
    pass
    