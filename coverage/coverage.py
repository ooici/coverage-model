#!/usr/bin/env python

"""
@package 
@file coverage
@author Christopher Mueller
@brief 
"""


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


#    def __init__(self, temporal_topology, spatial_topology, parameters=None, attributes=None, taxonomy=None, crs=None):
#        AbstractCoverage.__init__(self)
#        # TODO: promote some (all?) of these fields to AbstractCoverage
#        if not isinstance(temporal_topology, Topology):
#            raise TypeError('temporal_topology must be a Topology object')
#        self.temporal_topology = temporal_topology
#
#        if not isinstance(spatial_topology, Topology):
#            raise TypeError('spatial_topology must be a Topology object')
#        self.spatial_topology = spatial_topology
#
#        self._parameters = {}
#        self._parameters.update({x.name : x for x in parameters if isinstance(x, Parameter)})
#        self._attributes = {}
#        self._parameters.update({x.name : x for x in attributes if isinstance(x, Attribute)})
#        self.taxonomy = taxonomy #or Taxonomy # TODO import this
#        self.crs = crs
#
#    def add_parameter(self, parameter, overwrite=False):
#        ret=False
#        if isinstance(parameter, Parameter):
#            if parameter.name in self._parameters:
#                if overwrite:
#                    ret=True
#            else:
#                ret=True
#
#        if ret:
#            self._parameters[parameter.name] = parameter
#
#        return ret
#
#    def remove_parameter(self, parameter):
#        del self._parameters[parameter.name]
#
#    def get_parameters(self):
#        return self._parameters.copy()
#
#    def add_attribute(self, attribute, overwrite=False):
#        ret=False
#        if isinstance(attribute,Attribute):
#            if attribute.name in self._attributes:
#                if overwrite:
#                    ret=True
#            else:
#                ret=True
#
#        if ret:
#            self._attributes[attribute.name] = attribute
#
#        return ret
#
#    def remove_attribute(self, attribute):
#        del self._attributes[attribute.name]
#
#    # TODO: if _parameters is promoted to AbstractCoverage, these can be too
#    @property
#    def domain(self):
#        return [x for x in self._parameters if x.isDomain]
#
#    @property
#    def range(self):
#        return [x for x in self._parameters if not x.isDomain]

class Parameter():
    """

    """
    def __init__(self, name, type, isDomain=False, atts=None, value=None):
        self.name = name
        self.type = type
        self.isDomain = isDomain
        self.atts = {}
        if isinstance(atts, dict):
            self.atts.update({x.name : Attribute for x in atts if isinstance(x, Attribute)})
        self.value = value

    # TODO: decide on the desired equivalency behavior
    def __eq__(self, other):
        return self.name is other.name and \
               self.type is other.type and \
               self.isDomain is other.isDomain and \
               self.atts is other.atts and \
               self.value is other.value

class AbstractDomain():
    def __init__(self, abstract_shape):
        self.shape = []
        self.shape.append(abstract_shape)

    def extend(self, extend_by_count, dim_index=None):
        raise NotImplementedError()

class GridDomain(AbstractDomain):

    def __init__(self, abstract_shape):
        AbstractDomain.__init__(self, abstract_shape)

    def extend(self, extend_by_count, dim_index=None):
        dim_index = dim_index or 0

class TopologicalDomain(AbstractDomain):

    def __init__(self, abstract_shape):
        AbstractDomain.__init__(self, abstract_shape)

class AbstractShape():
    pass

class GridShape(AbstractShape):

    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class Attribute():
    """

    """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    # TODO: decide on the desired equivalency behavior
    def __eq__(self, other):
        return self.name is other.name and \
               self.value is other.value

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
    pass

class StructureFilter(Filter):
    pass

class ParameterFilter(Filter):
    pass

