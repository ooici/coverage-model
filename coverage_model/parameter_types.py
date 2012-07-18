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

from coverage_model.coverage import AbstractSimplexParameterType, AbstractComplexParameterType
import numpy as np

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
    def __init__(self, value_encoding, uom=None):
        AbstractSimplexParameterType.__init__(self)
        try:
            dt = np.dtype(value_encoding)
            if dt.isbuiltin:
                self.value_encoding = dt
            else:
                raise TypeError()
        except TypeError:
            raise TypeError('\'value_encoding\' must be a valid numpy dtype: {0}'.format(value_encoding))

        self.template_attrs['uom'] = uom or 'unspecified'

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
    Heterogeneous set of named things (dict)
    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)

class VectorType(AbstractComplexParameterType):
    """
    Heterogeneous set of unnamed things (tuple)
    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)

class ArrayType(AbstractComplexParameterType):
    """
    Homogeneous set of unnamed things (array)
    """
    def __init__(self):
        AbstractComplexParameterType.__init__(self)
