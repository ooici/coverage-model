#!/usr/bin/env python

"""
@package coverage_model.postgres_persisted_storage
@file coverage_model.postgres_persisted_storage
@author Casey Bryant
@brief Persistence Layer specialized classes for storing persisted data to Postgres
"""

import numpy as np
from coverage_model.util.numpy_utils import NumpyUtils


def make_parameter_data_dict(data_dict):
    if 'time' in data_dict:
        time_array = data_dict['time']
    elif len(data_dict) > 0:
        elements = data_dict.values()[0]
        time_array = np.arange(len(elements))

        for k,v in data_dict.iteritems():
            data_dict[k] = NumpyParameterData(k, v, time_array)
    return data_dict


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


class NumpyDictParameterData(ParameterData):

    def __init__(self, param_dict, alignment_key=None, param_context_dict=None, as_rec_array=False):
        if len(param_dict) < 1:
            self.size = 0
            self.alignment_key = alignment_key
            self.param_context_dict = param_context_dict
            super(NumpyDictParameterData, self).__init__('none', {})
            return

        if not isinstance(param_dict, dict):
            raise TypeError("param_dict must implement type %s" % dict.__name__)
        alignment_array = None
        if alignment_key is not None:
            if not alignment_key in param_dict:
                raise ValueError("alignment array missing from param_dict.  Looking for key: %s" % alignment_key)
            alignment_array = param_dict[alignment_key]
        else:
            for key, val in param_dict.iteritems():
                if val is not None and isinstance(val, np.ndarray):
                    alignment_array = val
                    break
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s, found %s" % (np.ndarray.__name__, type(alignment_array)))
        self.size = alignment_array.size
        data = param_dict
        super(NumpyDictParameterData, self).__init__('none', data)

        self.sort_data_by_parameter(alignment_key)

        self.is_record_array = False
        if as_rec_array:
            self.convert_to_record_array()

        self.alignment_key = alignment_key
        self.param_context_dict = param_context_dict

    def sort_data_by_parameter(self, sort_parameter):
        if sort_parameter is not None:
            self._data = NumpyUtils.sort_flat_arrays(self._data, sort_parameter)

    def convert_to_record_array(self):
        if not self.is_record_array:
            try:
                name_arr = []
                data_arr = []
                format_arr = []
                for param_name, param_data in self._data.iteritems():
                    self._validate_param_data(param_name, param_data, self._data[self.alignment_key].size)
                    name_arr.append(param_name)
                    if len(param_data.shape) > 1:
                        param_data = np.core.records.fromrecords(param_data)
                    data_arr.append(param_data)
                    # param_dict[param_name] = param_data
                    format_arr.append(np.dtype(param_data.dtype).str)
                format_arr = ', '.join(format_arr)

                self._data = np.core.records.fromarrays(data_arr, names=name_arr, formats=format_arr)
            except Exception as e:
                # print "Record array creation failed, just return numpy arrays. Failure: %s" % e.message
                #TODO log something
                self.is_record_array = False

        return self.is_record_array

    def _validate_param_data(self, param_name, param_data, size, param_context=None):
            if not isinstance(param_name, basestring):
                raise TypeError("dict keys must implement type %s" % basestring.__name__)
            if not isinstance(param_data, np.ndarray):
                raise TypeError("param_array must implement type %s" % np.ndarray.__name__)
            if not param_data.shape[0] == size:
                raise ValueError("param_array, %s, and alignment_array must have the same number of elements." % param_name)

    # def add_param_data(self, param_name, param_data, param_context):
    #     self._validate_param_data(param_name, param_data, param_context)
    #     self._data[param_name] = param_data
    #     self.param_context_dict[param_name] = param_context

    def get_mask_for_alignment(self, alignment_array):
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s", np.ndarray)
        if not alignment_array.size == self.size:
            raise TypeError("alignment_array size mismatch - found %i, expected %i" % (alignment_array.size, self._data[self.alignment_key].size))
        return np.in1d(self._alignment, alignment_array)

    def get_data_as_numpy_array(self, alignment_array, fill_value=None):
        if self.alignment_key is not None and id(alignment_array) == id(self._data[self.alignment_key]):
            return self.get_data()
        mask = self.get_mask_for_alignment(alignment_array)
        return self._data[self.get_mask_for_alignment(alignment_array)]

    def __eq__(self, other):
        if self.__dict__ == other.__dict__:
            return True
        return False


class NumpyParameterData(ParameterData):

    def __init__(self, param_name, param_array, alignment_array=None):
        if not isinstance(param_array, np.ndarray):
            raise TypeError("param_array must implement type %s" % np.ndarray.__name__)
        if alignment_array is not None:
            if not isinstance(alignment_array, np.ndarray):
                raise TypeError("alignment_array must implement type %s" % np.ndarray.__name__)
            if not param_array.shape[0] == alignment_array.shape[0]:
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

    def min(self):
        return np.nanmin(self._data)

    def max(self):
        return np.nanmax(self._data)

    def __eq__(self, other):
        if self.param_name == other.param_name and np.array_equal(self._data, other._data):
            return True
        return False


class ConstantOverTime(ParameterData):

    def __init__(self, param_name, value, time_start=None, time_end=None):
        super(ConstantOverTime, self).__init__(param_name, value)
        if time_end is not None and time_start is not None and not time_end > time_start:
            raise RuntimeError("time_start, %s, must be before time_end, %s." % (time_start, time_end))
        self.start = time_start
        self.stop = time_end

    def min(self):
        return self._data

    def max(self):
        return self._data

    @classmethod
    def merge_data_as_numpy_array(cls, alignment_array, obj_dict, param_type=None, arr=None):
        order = sorted(obj_dict.keys())
        for key in order:
            arr = obj_dict[key].get_data_as_numpy_array(alignment_array, param_type, arr=arr)
        return arr

    def get_data_as_numpy_array(self, alignment_array, param_type, arr=None):
        """
        NaN fill value causes array equivalency checks to fail
        Using arrays created by numpy.arange() does not include elements that are
         equivalent to start_range.
        """
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s", np.ndarray)

        if arr is None:
            arr = param_type.create_filled_array(alignment_array.size)

        if self.start is None and self.stop is None:
            arr[:] = self._data
        elif self.start is None:
            arr[alignment_array <= self.stop] = self._data
        elif self.stop is None:
            arr[alignment_array >= self.start] = self._data
        else:
            arr[np.logical_and(alignment_array >= self.start, alignment_array <= self.stop)] = self._data

        return arr

    def applies(self, time):
        if self.start <= time >= self.end:
            return True
        return False

    def _greater_stop(self, other):
        if self.stop is None and other.stop is not None:
            return self
        elif self.stop is not None and other.stop is None:
            return other
        elif self.stop is None and other.stop is None:
            return None
        elif self.stop == other.stop:
            return None
        elif self.stop > other.stop:
            return self
        else:
            return other

    def _greater_start(self, other):
        if self.start is None and other.start is not None:
            return other
        elif self.start is not None and other.start is None:
            return self
        elif self.start is None and other.start is None:
            return None
        elif self.start == other.start:
            return None
        elif self.start > other.start:
            return self
        else:
            return other

    def __lt__(self, other):
        greater_stop = self._greater_stop(other)
        if greater_stop is None:
            greater_start = self._greater_start(other)
            if greater_start == self:
                return False
            elif greater_start == other:
                return True
            else:
                raise RuntimeError("Greater start type execto")
        elif greater_stop == self:
            return False
        elif greater_stop == other:
            return True
        else:
            raise RuntimeError("Invalid compare type")

    def __gt__(self, other):
        self.__lt__(other)

    def __str__(self):
        return '%s: %s %s %s %s' % (self.__class__.__name__, self.param_name, str(self._data), str(self.start), str(self.stop))

    def __eq__(self, other):
        if self.__dict__ == other.__dict__:
            return True
        return False


class RepeatOverTime(ConstantOverTime):
    def __init__(self, param_name, value, time_start=None, time_end=None):
        super(RepeatOverTime, self).__init__(param_name, value, time_start, time_end)

    def get_data_as_numpy_array(self, alignment_array, param_type, arr=None):
        """
        NaN fill value causes array equivalency checks to fail
        Using arrays created by numpy.arange() does not include elements that are
         equivalent to start_range.
        """
        if not isinstance(alignment_array, np.ndarray):
            raise TypeError("alignment_array must implement type %s", np.ndarray)

        arr = param_type.create_filled_array(alignment_array.size)

        if self.start is None and self.stop is None:
            arr[:] = self._data
        elif self.start is None:
            arr[alignment_array <= self.stop] = self._data
        elif self.stop is None:
            arr[alignment_array >= self.start] = self._data
        else:
            arr[np.logical_and(alignment_array >= self.start, alignment_array <= self.stop)] = self._data

        return arr

