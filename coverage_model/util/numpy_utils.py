__author__ = 'casey'

import numpy as np

def sort_flat_arrays(np_dict, sort_parameter):
    sorted_array_dict = {}
    sort_array = np_dict[sort_parameter]
    if len(sort_array) > 0:
        sorted_indexes = np.argsort(sort_array)
        for key, value in np_dict.iteritems():
            sorted_array_dict[key] = value[sorted_indexes]
        return sorted_array_dict
    return np_dict


def create_numpy_object_array(array):
    if isinstance(array, np.ndarray):
        array = array.tolist()
    arr = np.empty(len(array), dtype=object)
    arr[:] = array
    return arr


class NumpyUtils(object):
    @classmethod
    def create_filled_array(cls, shape, value, dtype):
        arr = np.empty(shape, dtype=dtype)
        arr[:] = value
        return arr