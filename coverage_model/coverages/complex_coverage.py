__author__ = 'casey'

from copy import deepcopy
import os
import collections
import pickle
from collections import Iterable

import numpy as np

from ooi.logging import log
from pyon.util.async import spawn
from coverage_model.coverage import AbstractCoverage, ComplexCoverageType, SimplexCoverage
from coverage_model.coverages.aggregate_coverage import AggregateCoverage
from coverage_model.coverages.coverage_extents import ReferenceCoverageExtents, ExtentsDict
from coverage_model.parameter_data import NumpyDictParameterData


class ComplexCoverage(AggregateCoverage):
    """
    References 1-n coverages
    """
    def __init__(self, root_dir, persistence_guid, name=None, reference_coverage_locs=None, reference_coverage_extents=None, parameter_dictionary=None,
                 mode=None, complex_type=ComplexCoverageType.PARAMETRIC_STRICT, temporal_domain=None, spatial_domain=None):

        # Sets h5py file operation mode to 'w' if not specified
        # 'w' = Create file, truncate if exists
        if mode is None:
            mode = 'w'

        reference_coverage_extents = ExtentsDict(reference_coverage_extents)
        # Initializes base class with proper mode.
        super(ComplexCoverage, self).__init__(root_dir, persistence_guid, name, reference_coverage_locs, reference_coverage_extents, parameter_dictionary,
                                                 mode, complex_type, temporal_domain, spatial_domain)

        # if reference_coverage_locs is not None:
        #     for loc in reference_coverage_locs:
        #         cov = AbstractCoverage.load(loc)
        #         extents = None
        #         if reference_coverage_extents is not None and cov.persistence_guid in reference_coverage_extents.data:
        #             extents = reference_coverage_extents.data[cov.persistence_guid]
        #         self.set_reference_coverage_extents(cov.persistence_guid, extents)

    def get_parameter_values(self, param_names=None, time_segment=None, time=None,
                             sort_parameter=None, stride_length=None, return_value=None, fill_empty_params=False,
                             function_params=None, as_record_array=False):
        '''
        Obtain the value set for a given parameter over a specified domain
        '''

        if param_names is None:
            param_names = self.list_parameters()
        if not isinstance(param_names, Iterable) or isinstance(param_names, basestring):
            param_names = [param_names]
        cov_value_list = []
        all_empty = set(param_names)
        for coverage in self._reference_covs.values():
            fill_params = set()
            if isinstance(coverage, SimplexCoverage):
                for param_name in param_names:
                    if param_name not in self.list_parameters():
                        raise KeyError('No parameter named %s in Complex Coverage' % param_name)
                    if param_name in coverage.list_parameters():
                        if coverage.get_parameter_context(param_name).fill_value != self.get_parameter_context(param_name).fill_value:
                            print 'different fill values - Handle it'
                    else:
                        fill_params.add(param_name)
                if param_names is not None:
                    this_param_names = set(param_names)
                    this_param_names = this_param_names.intersection(set(coverage.list_parameters()))
                    this_param_names = list(this_param_names)
                extent_segments = None
                if coverage.persistence_guid in self._persistence_layer.rcov_extents.data:
                    extent_segments = list(self._persistence_layer.rcov_extents.data[coverage.persistence_guid])
                if extent_segments is not None:
                    for extents in extent_segments:
                        from coverage_model.util.extent_utils import get_overlap
                        if isinstance(extents, ReferenceCoverageExtents):
                            extents = extents.time_extents
                        current_time_segment = None
                        try:
                            current_time_segment = get_overlap(extents, time_segment)
                        except RuntimeError:
                            continue

                        params = coverage.get_parameter_values(this_param_names, current_time_segment, time, sort_parameter,
                                                               return_value, fill_empty_params, function_params, as_record_array=False)
                        # if len(params.get_data()) == 1 and coverage.temporal_parameter_name in params.get_data():
                        #     continue
                        cov_dict = params.get_data()
                        for param_name in param_names:
                            if param_name not in fill_params and param_name not in cov_dict:
                                fill_params.add(param_name)
                            elif param_name in cov_dict and param_name in all_empty:
                                all_empty.remove(param_name)
                        size = cov_dict[coverage.temporal_parameter_name].size
                        self._add_filled_arrays(fill_params, cov_dict, size)
                        self._add_coverage_array(cov_dict, size, coverage.persistence_guid)
                        if time is not None and time_segment is None and len(cov_value_list) > 0:
                            new = cov_dict[coverage.temporal_parameter_name][0]
                            old = cov_value_list[0][0][coverage.temporal_parameter_name][0]
                            if abs(new-time) < abs(old-time):
                                cov_value_list = [(cov_dict, coverage)]
                        else:
                            cov_value_list.append((cov_dict, coverage))
        combined_data = self._merge_value_dicts(cov_value_list, stride_length=stride_length)
        if not fill_empty_params:
            for param_name in all_empty:
                if param_name in combined_data and param_name != self.temporal_parameter_name:
                    combined_data.pop(param_name)
        if sort_parameter is None:
            sort_parameter = self.temporal_parameter_name
        if sort_parameter not in combined_data:
            sort_parameter = None
        return NumpyDictParameterData(combined_data, alignment_key=sort_parameter, as_rec_array=as_record_array)

    def _add_filled_arrays(self, params, cov_dict, size):
        new_arrays = {}
        for param in params:
            pc = self.get_parameter_context(param)
            arr = np.empty(size, dtype=pc.param_type.value_encoding)
            arr[:] = pc.fill_value
            new_arrays[param] = arr
        cov_dict.update(new_arrays)

    def append_parameter(self, parameter_context):
        # Dad doesn't store it so go to granddad
        AbstractCoverage.append_parameter(self, parameter_context)

    def append_reference_coverage(self, path, extents=None, **kwargs):
        super(ComplexCoverage, self).append_reference_coverage(path, **kwargs)
        rcov = AbstractCoverage.load(path)
        self.set_reference_coverage_extents(rcov.persistence_guid, extents, append=True)

    def set_reference_coverage_extents(self, coverage_id, extents, append=True):

        if extents is None:
            raise ValueError("Extents must be specified when appending reference coverages")

        if not isinstance(extents, (list, tuple)):
            extents = [extents]

        # Check that the extents are proper Extents and that they reference the associated coverage
        for extent in extents:
            if not isinstance(extent, ReferenceCoverageExtents):
                raise TypeError('Extents must be of type %s' % ReferenceCoverageExtents.__name__)

            if extent.cov_id != coverage_id:
                raise ValueError('Extent coverage_id, %s, does not match requested coverage id %s' % (extent.cov_id, coverage_id))

        # Make a new one
        if self._persistence_layer.rcov_extents is None:
            self._persistence_layer.rcov_extents = ExtentsDict()

        if append:
            self._persistence_layer.rcov_extents.add_extents(coverage_id, extents)

        else:
            self._persistence_layer.rcov_extents.replace_extents(coverage_id, extents)

    def get_reference_coverage_extents(self, coverage_id):
        return self._persistence_layer.rcov_extents[coverage_id]


