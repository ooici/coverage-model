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
import sys
from json import JSONEncoder, JSONDecoder


# These classes define a limited set of acceptable types, with type checking, allowing a simpler
# SearchParameter interface.
class SearchParameter(object):
    def __init__(self, param_name, value):
        if self.__class__.__name__ == SearchParameter.__name__:
            raise NotImplementedError(''.join(['Cannot instantiate and abstract class: ', self.__class__.__name__]))
        if not isinstance(param_name, basestring):
            raise ValueError('param_name must be string type')
        self.param_name = param_name
        self.value = value

    def tuplize(self):
        return self.__class__.__name__, self.param_name, self.value

    def __repr__(self):
        return str(self.tuplize())

    def __str__(self):
        return str(self.tuplize())

    def to_json(self):
        v = dict(vars(self))
        v['class'] = self.__class__.__name__
        return JSONEncoder().encode(v)

    @staticmethod
    def from_json(json_str):
        py_dict = JSONDecoder().decode(json_str)
        param_name = None
        value = None
        class_ = None
        for k in py_dict.keys():
            if k == 'param_name':
                param_name = py_dict[k]
            elif k == 'value':
                v = py_dict[k]
                if isinstance(v, (tuple, list, dict, set)):
                    if isinstance(v[0], (tuple, list, dict, set)):
                        value = tuple(zip(*v))
                    else:
                        value = tuple(v)
                else:
                    value = v
            elif k == 'class':
                class_ = py_dict[k]
        if param_name is None or value is None or class_ is None:
            raise ValueError("Unable to reconstruct Param object from JSON")
        return getattr(sys.modules[__name__], class_)(param_name, value)


class ParamValue(SearchParameter):
    def __init__(self, param_name, value):
        if isinstance(value, (tuple, list, dict, set)):
            raise ValueError(''.join(['value for ParamValue cannot be a collection.  Found ',
                                      value.__class__.__name__]))

        super(ParamValue, self).__init__(param_name, value)


class ParamValueRange(SearchParameter):
    def __init__(self, param_name, value):
        if not isinstance(value, tuple) or not 2 == len(value) or not type(value[0]) == type(value[1]):
            raise ValueError(''.join(['value for ParamValueRange type must be a 2 element tuple with objects'
                                      ' of the same type.  Found ', str(value)]))
        if isinstance(value[0], (tuple, list, dict, set)):
            raise ValueError(''.join(['Elements in ParamValueRange cannot be a collection.  Found ',
                                      str(value)]))
        if value[0] > value[1]:
            raise ValueError(''.join(['First element of tuple must be less than or equal to the second '
                                      'element.  Found ', str(value)]))
        super(ParamValueRange, self).__init__(param_name, value)


class Param2DValueRange(SearchParameter):
    def __init__(self, param_name, value):
        if not isinstance(value, tuple) or not 2 == len(value) \
            or not isinstance(value[0], tuple) or not 2 == len(value[0]) or not type(value[0][0]) == type(value[0][1]) \
                or not isinstance(value[1], tuple) or not 2 == len(value[1]) or not type(value[1][0]) == type(value[1][1]):

            raise ValueError(''.join(['value for Param2DRangeValue must be a 2 element tuple with each element '
                                      'a 2 element tuple.  Found ', str(value)]))
        if value[0][0] > value[0][1] or value[1][0] > value[1][1]:
            raise ValueError(''.join(['First element of each tuple must be less than or equal to the the second '
                                      'element of the tuple. Found ', str(value)]))
        super(Param2DValueRange, self).__init__(param_name, value)


class SearchCriteria():
    @classmethod
    def _evaluate_type_(cls, something):
        if not isinstance(something, SearchParameter):
            raise ValueError("".join(["Search criteria parameters must be of type ", SearchParameter.__name__, " found ",
                                      something.__class__.__name__]))

    def __init__(self, search_params=None):
        self.criteria = {}
        if isinstance(search_params, collections.Iterable):
            for val in search_params:
                self.append(val)
        elif search_params is not None:
            self.append(search_params)

    def append(self, param):
        self._evaluate_type_(param)
        self.criteria[param.param_name] = param
