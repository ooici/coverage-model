__author__ = 'casey'

import numpy as np

def sort_flat_arrays(np_dict, sort_parameter):
    sorted_array_dict = {}
    sort_array = np_dict[sort_parameter]
    sorted_indexes = np.argsort(sort_array)
    for key, value in np_dict.iteritems():
        sorted_array_dict[key] = value[sorted_indexes]
    return sorted_array_dict

