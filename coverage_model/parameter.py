#!/usr/bin/env python

"""
@package 
@file parameter
@author Christopher Mueller
@brief 
"""

from coverage_model.basic_types import *
from coverage_model.coverage import AbstractShape, AbstractParameterType, AbstractParameterValue, AbstractSimplexParameterType, AbstractSimplexParameterValue, AbstractComplexParameterType, AbstractComplexParameterValue

class Parameter(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)

        self.name = ''
        self.is_coordinate = False
        self.context = ParameterContext()
        self.value = AbstractParameterValue()
        self.shape = AbstractShape()

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self.__name = value

    @property
    def is_coordinate(self):
        return self.__is_coordinate

    @is_coordinate.setter
    def is_coordinate(self, value):
        if isinstance(value, bool):
            self.__is_coordinate = value

    @property
    def context(self):
        return self.__context

    @context.setter
    def context(self, value):
        if isinstance(value, ParameterContext):
            self.__context = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, value):
        if isinstance(value, AbstractParameterValue):
            self.__value = value

    @property
    def shape(self):
        return self.__shape

    @shape.setter
    def shape(self, value):
        if isinstance(value, AbstractShape):
            self.__shape = value


class ParameterContext(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)
        self.param_type = AbstractParameterType()

    @property
    def param_type(self):
        return self.__param_type

    @param_type.setter
    def param_type(self, value):
        if isinstance(value, AbstractParameterType):
            self.__param_type = value

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

