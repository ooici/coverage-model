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
import numpy as np

#################
# Abstract Parameter Type Objects
#################

class AbstractParameterType(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)
        self.template_attrs = {}

    def _dump(self):
        ret = dict((k,v) for k,v in self.__dict__.iteritems())
        ret['p_type'] = self.__class__.__name__
        return ret

    @classmethod
    def _load(cls, tdict):
        if isinstance(tdict, dict) and 'p_type' in tdict and tdict['p_type']:
            import inspect
            mod = inspect.getmodule(cls)
            ptcls=getattr(mod, tdict['p_type'])

            args = inspect.getargspec(ptcls.__init__).args
            del args[0] # get rid of 'self'
            kwa={}
            for a in args:
                kwa[a] = tdict[a] if a in tdict else None

            ret = ptcls(**kwa)
            for k,v in tdict.iteritems():
                if not k in kwa.keys() and not k is 'p_type':
                    setattr(ret,k,v)

            return ret
        else:
            raise TypeError('tdict is not properly formed, must be of type dict and contain a \'p_type\' key: {0}'.format(tdict))




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
                self.value_encoding = dt.char
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
