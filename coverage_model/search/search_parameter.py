#!/usr/bin/env python

"""
@package coverage_model.search_parameter
@file coverage_model/search_parameter.py
@author Casey Bryant
@brief Interfaces and implementations for finding coverages that match criteria
"""

__author__ = 'casey'

from ooi.logging import log

import collections


# This is a workaround to create Enum-like functionality.  It is by no means comprehensive.  It attempts to
# define a limited set of acceptable types, allowing a simpler Param interface.
class ParamValue(object):
    @staticmethod
    def getValueType(self):
        return "Value"

    def __init__(self):
        pass


class ParamValueRange(object):
    @staticmethod
    def getValueType(self):
        return "ValueRange"

    def __init__(self):
        pass


class Param2DValueRange(object):
    @staticmethod
    def getValueType(self):
        return "2DValueRange"

    def __init__(self):
        pass


class Param(object):
    @classmethod
    def from_range(cls, param_name, start, stop):
        return Param(param_name, (start, stop), ParamValueRange)

    @classmethod
    def from_value(cls, param_name, value):
        return Param(param_name, value, ParamValue())

    def __init__(self, param_name, value, param_type=ParamValue()):
        if not isinstance(param_name, basestring):
            raise ValueError("param_name must be string type")
        if isinstance(param_type, ParamValueRange):
            if not isinstance(value, tuple) or not 2 == len(value) or not type(value[0]) == type(value[1]):
                raise ValueError("".join(["value for ParamValueRange type must be a 2 element tuple with objects of the same type.  Found ", str(value)]))
        elif isinstance(param_type, ParamValue):
            if isinstance(value, (tuple, list, dict, set)):
                raise ValueError("".join(["value for ParamRangeValue cannot be a collection.  Found ", value.__class__.__name__]))
        elif isinstance(param_type, Param2DValueRange):
            if not isinstance(value, tuple) or not 2 == len(value) \
                or not isinstance(value[0], tuple) or not 2 == len(value[0]) or not type(value[0][0]) == type(value[0][1]) \
                    or not isinstance(value[1], tuple) or not 2 == len(value[1]) or not type(value[1][0]) == type(value[1][1]):

                raise ValueError("".join(["value for Param2DRangeValue must be a 2 element tuple with each element a 2 element tuple.  Found ", str(value)]))
        else:
            raise ValueError("".join(["param_type must be implement ParamValue. Found ", param_type.__class__.__name__]))

        self.param_type = param_type
        self.param_name = param_name
        self.value = value

    def tuplize(self):
        return self.param_type.__class__.__name__, self.param_name, self.value

    def __repr__(self):
        return str(self.tuplize())

    def __str__(self):
        return str(self.tuplize())

#class DirectValue(Param):
#    def __init__(self, id, value):
#        if not isinstance(id, basestring):
#            raise ValueError("Id must be string type")
#        elif isinstance(value, (tuple, list, dict, set)):
#            raise ValueError("Values of arguments DirectPosition cannot be a collection")
#        else:
#            setattr(self, id, value)
#
#
#class RangeValue(Param):
#    def __init__(self, id, value_range):
#        if not isinstance(id, basestring):
#            raise ValueError("Id must be string type")
#        elif not isinstance(value_range, tuple) or not 2 == len(value_range) or not type(value_range[0]) == type(value_range[1]):
#            raise ValueError("Values of arguments RangePosition must be a tuple of 2 elements of the same type")
#        else:
#            setattr(self, id, value_range)


class SearchCriteria:
    @classmethod
    def _evaluate_type_(cls, something):
        if not isinstance(something, Param):
            raise ValueError("".join(["Search criteria parameters must be of type ", Param.__name__, " found ",
                                      something.__class__.__name__]))

    def __init__(self, with_params=None):
        self.criteria = []
        if isinstance(with_params, collections.Iterable):
            for val in with_params:
                self.append(val)

    def append(self, param):
        self._evaluate_type_(param)
        self.criteria.append(param)

