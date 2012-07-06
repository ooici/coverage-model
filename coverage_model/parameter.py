#!/usr/bin/env python

"""
@package 
@file parameter
@author Christopher Mueller
@brief 
"""


#CBM: see next line
#TODO: Add type checking throughout all classes as determined appropriate, ala:
#@property
#def is_coordinate(self):
#    return self.__is_coordinate
#
#@is_coordinate.setter
#def is_coordinate(self, value):
#    if isinstance(value, bool):
#        self.__is_coordinate = value


from coverage_model.basic_types import *
from coverage_model.coverage import AbstractParameterType, AbstractSimplexParameterType, AbstractSimplexParameterValue, AbstractComplexParameterType, AbstractComplexParameterValue

class Parameter(AbstractIdentifiable):
    """

    """
    def __init__(self, parameter_context, shape, value):
        AbstractIdentifiable.__init__(self)
        self.context = parameter_context
        self.value = value
        self.shape = shape

    # Expose a couple of the context attributes at this level as "read only"
    @property
    def name(self):
        return self.context.name

    @property
    def is_coordinate(self):
        return self.context.is_coord



class ParameterContext(AbstractIdentifiable):
    """

    """
    def __init__(self, name, is_coord=False, param_type=None, fill_value=None, axis=None):
        AbstractIdentifiable.__init__(self)
        self.name = name
        self.is_coord = is_coord
        self.param_type = param_type or AbstractParameterType()
        self.fill_value = fill_value or -999
        self.axis = axis or None

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Name: {1}'.format(indent, self.name))
        if self.is_coord:
            lst.append('{0}Is Coordinate: {1}'.format(indent, self.axis))
        lst.append('{0}Type: {1}'.format(indent, self.param_type))
        lst.append('{0}Fill Value: {1}'.format(indent, self.fill_value))

        return '\n'.join(lst)

#################
# Parameter Type Objects
#################

class ReferenceType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class BooleanType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class CategoryType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class CountType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class QuantityType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class TextType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class TimeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class CategoryRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class CountRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class QuantityRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class TimeRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self):
        AbstractSimplexParameterType.__init__(self)

class RecordType(AbstractComplexParameterType):
    """

    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)

class VectorType(AbstractComplexParameterType):
    """

    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)

class ArrayType(AbstractComplexParameterType):
    """

    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)


#################
# Parameter Value Objects
#################

class Reference(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Boolean(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Category(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Count(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Quantity(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Text(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Time(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class CategoryRange(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class CountRange(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class QuantityRange(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class TimeRange(AbstractSimplexParameterValue):
    """

    """
    def __init__(self):
        AbstractSimplexParameterValue.__init__(self)

class Record(AbstractComplexParameterValue):
    """
    Heterogeneous set of named things (dict)
    """
    def __init__(self):
        AbstractComplexParameterValue.__init__(self)

class Vector(AbstractComplexParameterValue):
    """
    Heterogeneous set of unnamed things (tuple)
    """
    def __init__(self):
        AbstractComplexParameterValue.__init__(self)

class Array(AbstractComplexParameterValue):
    """
    Homogeneous set of unnamed things (array)
    """
    def __init__(self):
        AbstractComplexParameterValue.__init__(self)

