__author__ = 'casey'

import numpy as np


class NumpyUtils(object):

    @classmethod
    def sort_flat_arrays(cls, np_dict, sort_parameter):
        sorted_array_dict = {}
        sort_array = np_dict[sort_parameter]
        if len(sort_array) > 0:
            sorted_indexes = np.argsort(sort_array)
            for key, value in np_dict.iteritems():
                sorted_array_dict[key] = value[sorted_indexes]
            return sorted_array_dict
        return np_dict

    @classmethod
    def create_numpy_object_array(cls, array):
        if isinstance(array, np.ndarray):
            array = array.tolist()
        arr = np.empty(len(array), dtype=object)
        arr[:] = array
        return arr

    @classmethod
    def create_filled_array(cls, shape, value, dtype):
        arr = np.empty(shape, dtype=dtype)
        arr[:] = value
        return arr

    @classmethod
    def get_duplicate_values(cls, np_arr, presorted=False):
        if not presorted:
            np_arr.sort()
        dup_vals = np.unique(np_arr[np_arr[1:] == np_arr[:-1]])
        return dup_vals


class DedupedNumpyArrayDict(object):
    """
    Accepts an aligned dictionary of numpy arrays (think numpy record array, but
    kept as separate arrays for programmatic reasons).
    Removes indexes ('records') with duplicate values in for the array found at dedupe_key.
    If add_aggregate is specified, an aggregate record, as specified by an implementing class,
    is appended.
    The resulting dictionary contains arrays that are sorted in the order determined by dedupe_key values.
    If the supplied arrays are already presorted, that can be specified to avoid additional sort processing.
    For performance reasons, the duplication identification algorithm relies on sorted values so the presorted
    flag should be used with caution.

    Implementation class supports custom non-dedupe key values for duplicate dedupe_key values, the add_aggregate
    option causes a new aggregated 'record' for each duplicate to be appended to each numpy array in the dictionary
    of numpy arrays.  This can be computationally costly because appends require a deep copy of the array.
    """
    def __init__(self, np_dict, dedupe_key, dict_arrays_are_presorted=False, add_aggregate=False):
        self.dedupe_key = dedupe_key
        self.add_aggregate = add_aggregate
        if not dict_arrays_are_presorted:
            np_dict = NumpyUtils.sort_flat_arrays(np_dict, self.dedupe_key)
        alignment_array = np_dict[self.dedupe_key]
        duplicate_values = NumpyUtils.get_duplicate_values(alignment_array, presorted=True)
        indices_to_remove = set()
        to_append = []
        for value in duplicate_values:
            indices = np.where(alignment_array==value)[0]
            prefered_indices, append_set = self.resolve_duplicate_value(np_dict, indices)
            if prefered_indices is not None:
                tmp = np.searchsorted(indices, prefered_indices)
                indices = np.delete(indices, tmp)
            indices_to_remove.update(indices)
            if self.add_aggregate:
                to_append.append(append_set)

        valid_indices = np.delete(np.arange(alignment_array.size), list(indices_to_remove))
        self.deduped_dict = {}
        for k, v in np_dict.iteritems():
            self.deduped_dict[k] = v[valid_indices]
            if len(to_append) > 0:
                append_list = []
                for append_dict in to_append:
                    append_list.append(append_dict[k])
                append_arr = np.array(append_list,dtype=self.deduped_dict[k].dtype)
                self.deduped_dict[k] = np.hstack((self.deduped_dict[k], append_arr))

        if len(append_set) > 0 and self.add_aggregate:
            self.deduped_dict = NumpyUtils.sort_flat_arrays(self.deduped_dict, self.dedupe_key)

    @property
    def np_dict(self):
        return self.deduped_dict

    def resolve_duplicate_value(self, np_dict, indices):
        raise NotImplementedError('Base class not implemented')


class MostRecentRecordNumpyDict(DedupedNumpyArrayDict):
    """
    Retains only the most recent record for a duplicate as specified by the array referenced by 'most_recent_key'.
    """
    def __init__(self, np_dict, dedupe_key, most_recent_key, dict_arrays_are_presorted=False, reverse=False):
        self.most_recent_key = most_recent_key
        self.reverse = reverse
        super(MostRecentRecordNumpyDict, self).__init__(np_dict, dedupe_key, dict_arrays_are_presorted)

    def resolve_duplicate_value(self, np_dict, indices):
        preference_array = np_dict[self.most_recent_key]
        pref_vals = preference_array[indices]
        if not self.reverse:
            prefered_indicies = pref_vals.max()
        else:
            prefered_indicies = pref_vals.min()
        return (np.where(preference_array==prefered_indicies), {})


class MostRecentValidValueNumpyDict(DedupedNumpyArrayDict):
    """
    Creates a new record for duplicates where the value of each array is the most recent valid value.  Other
    duplicate values are removed.
    """

    def __init__(self, np_dict, dedupe_key, most_recent_key, valid_values_dict, dict_arrays_are_presorted=False, add_aggregate=False):
        self.most_recent_key = most_recent_key
        self._valid_values_dict = valid_values_dict
        import random
        import string
        self.key_mutation = ''.join(random.choice(string.ascii_uppercase + string.digits) for x in range(12))
        for k, v in self.valid_values_dict.iteritems():
            np_dict[k+self.key_mutation] = v
        super(MostRecentValidValueNumpyDict, self).__init__(np_dict, dedupe_key, dict_arrays_are_presorted, add_aggregate)
        for k in self.np_dict.keys():
            v = self.deduped_dict[k]
            if k.endswith(self.key_mutation):
                nk = k[:-len(self.key_mutation)]
                self._valid_values_dict[nk] = v
                self.deduped_dict.pop(k)

    def resolve_duplicate_value(self, np_dict, indices):
        preference_array = np_dict[self.most_recent_key]
        pref_vals = preference_array[indices]
        sorted_dict = NumpyUtils.sort_flat_arrays({'i':indices, 'v':pref_vals}, 'v')
        deduped_dict_values = {}
        use_index = None
        if not self.add_aggregate:
            use_index = sorted_dict['i'][-1]
        for k, v in np_dict.iteritems():
            if k.endswith(self.key_mutation):
                continue
            new_val = None
            is_valid = False
            for index in sorted_dict['i'][::-1]:
                is_valid = np_dict[k+self.key_mutation][index]
                if is_valid:
                    new_val = np_dict[k][index]
                    is_valid = True
                    break
                elif new_val is None:
                    new_val = np_dict[k][index]
            if not self.add_aggregate:
                np_dict[k][use_index] = new_val
                np_dict[k+self.key_mutation][use_index] = is_valid
            else:
                deduped_dict_values[k] = new_val
                deduped_dict_values[k+self.key_mutation] = is_valid

        return (use_index, deduped_dict_values)

    @property
    def valid_values_dict(self):
        return self._valid_values_dict


class AggregatedDuplicatesNumpyDict(MostRecentValidValueNumpyDict):
    """
    Creates a new record for duplicates where the value of each array is the most recent valid value.  Other
    duplicate values are not removed, meaning the array must be appended.  For large arrays, this can be time consuming.
    """
    def __init__(self, np_dict, dedupe_key, most_recent_key, valid_values_dict, dict_arrays_are_presorted=False):
        super(AggregatedDuplicatesNumpyDict, self).__init__(np_dict, dedupe_key, most_recent_key, valid_values_dict, dict_arrays_are_presorted,
                                                            add_aggregate=True)

    def resolve_duplicate_value(self, np_dict, indices):
        not_used, deduped_dict_values = super(AggregatedDuplicatesNumpyDict, self).resolve_duplicate_value(np_dict, indices)
        return indices, deduped_dict_values
