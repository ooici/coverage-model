#!/usr/bin/env python

"""
@package coverage_model.postgres_persisted_storage
@file coverage_model.postgres_persisted_storage
@author Casey Bryant
@brief Persistence Layer specialized classes for storing persisted data to Postgres
"""

import numpy as np
from coverage_model.parameter import ParameterContext


class ParameterDataDict(object):
    pass


class ParameterData(object):

    def __init__(self, param_name, param_data):
        if not isinstance(param_name, basestring):
            raise TypeError("param_name must implement type %s" % (basestring.__name__))
        self.param_name = param_name
        self._data = param_data

    def get_data(self):
        return self._data

    def get_data_as_numpy_array(self, alignment_array, fill_value=None):
        pass


class NumpyParameterData(ParameterData):

    def tmp(self, param_dict, alignment_key, param_context_dict=None):
        if not isinstance(param_dict, dict):
            raise TypeError("param_dict must implement type %s" % dict.__name__)
        if not alignment_key in param_dict:
            raise ValueError("alignment array missing from param_dict.  Looking for key: %s" % alignment_key)
        alignment_array = param_dict[alignment_key]
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s" % np.ndarray.__name__)
        for key, val in param_dict.iteritems():
            if not isinstance(key, basestring):
                raise TypeError("dict keys must implement type %s" % (basestring.__name__))
            if not isinstance(val, np.ndarray):
                raise TypeError("param_array must implement type %s" % np.ndarray.__name__)
            if not val.size == alignment_array.size:
                raise ValueError("param_array, %s,  and alignment_array must have the same number of elements." % key)
        super(NumpyParameterData, self).__init__(param_dict)

        self.alignment_key = alignment_key
        self.param_context_dict=None

    def __init__(self, param_name, param_array, alignment_array=None):
        if not isinstance(param_array, np.ndarray):
            raise TypeError("param_array must implement type %s" % np.ndarray.__name__)
        if alignment_array is not None:
            if not isinstance(alignment_array, np.ndarray):
                raise TypeError("alignment_array must implement type %s" % np.ndarray.__name__)
            if not param_array.size == alignment_array.size:
                raise ValueError("param_array and alignment_array must have the same number of elements")
        super(NumpyParameterData, self).__init__(param_name, param_array)
        self._alignment = alignment_array

    def get_alignment(self):
        return self._alignment

    def get_mask_for_alignment(self, alignment_array):
        if self._alignment is None:
            raise IndexError('Native object alignment not specified')
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s", np.ndarray)
        # local_alignment_arr = self.get_data()[self._alignment]
        return np.in1d(self._alignment, alignment_array)

    def get_data_as_numpy_array(self, alignment_array, fill_value=None):
        if self._alignment is None:
            raise IndexError('Native object alignment not specified')
        if id(alignment_array) == id(self._alignment):
            return self.get_data()
        return self._data[self.get_mask_for_alignment(alignment_array)]


class ConstantOverRange(ParameterData):

    def __init__(self, param_name, value, range_start=None, range_end=None):
        super(ConstantOverRange, self).__init__(param_name, value)
        self.start = range_start
        self.stop = range_end

    def get_data_as_numpy_array(self, alignment_array, fill_value=None):
        """
        NaN fill value causes array equivalency checks to fail
        Using arrays created by numpy.arange() does not include elements that are
         equivalent to start_range.
        """
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s", np.ndarray)

        dtype = type(fill_value)
        if isinstance(fill_value, basestring):
            dtype = object
        arr = np.empty(alignment_array.size, dtype)
        arr.fill(fill_value)
        if self.start is None and self.stop is None:
            arr.fill(self._data)
            pass
        elif self.start is None:
            arr[alignment_array <= self.stop] = self._data
        elif self.stop is None:
            arr[alignment_array >= self.start] = self._data
        else:
            arr[np.logical_and(alignment_array >= self.start, alignment_array <= self.stop)] = self._data

        return arr
