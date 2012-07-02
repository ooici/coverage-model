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
    def __init__(self):
        AbstractCoverage.__init__(self)

        self.spatial_domain = AbstractDomain()
        self.temporal_domain = AbstractDomain()
        self.range_dictionary = RangeDictionary()
        self.range_type = {}
        self.range = {}


#################
# Range Dictionary
#################

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
# Domain Objects
#################

class AbstractDomain(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)
        self.shape = AbstractShape()

class GridDomain(AbstractDomain):
    """

    """
    def __init__(self):
        AbstractDomain.__init__(self)

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
    def __init__(self):
        AbstractIdentifiable.__init__(self)

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, dims):
        AbstractShape.__init__(self)
        self.name = name
        self.dims = dims


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