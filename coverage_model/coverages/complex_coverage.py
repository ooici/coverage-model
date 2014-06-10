__author__ = 'casey'

from copy import deepcopy
import os
import collections
import pickle
from collections import Iterable

import numpy as np

from ooi.logging import log
from pyon.util.async import spawn
from coverage_model.coverage import AbstractCoverage, ComplexCoverageType, RangeValues
from coverage_model.basic_types import AbstractIdentifiable, AxisTypeEnum, MutabilityEnum, VariabilityEnum, Dictable, \
    Span
from coverage_model.parameter import Parameter, ParameterDictionary, ParameterContext
from coverage_model.parameter_values import get_value_class, AbstractParameterValue
from coverage_model.persistence import InMemoryPersistenceLayer, is_persisted
from coverage_model.storage.parameter_persisted_storage import PostgresPersistenceLayer
from coverage_model.metadata_factory import MetadataManagerFactory
from coverage_model.parameter_functions import ParameterFunctionException
from coverage_model import utils
from coverage_model.utils import Interval, create_guid
from coverage_model.coverage import SimplexCoverage, GridDomain, GridShape, CRS
from coverage_model.coverages.aggregate_coverage import AggregateCoverage
from coverage_model.parameter_data import NumpyDictParameterData


class ComplexCoverage(AggregateCoverage):
    """
    References 1-n coverages
    """
    def __init__(self, root_dir, persistence_guid, name=None, reference_coverage_locs=None, parameter_dictionary=None,
                 mode=None, complex_type=ComplexCoverageType.PARAMETRIC_STRICT, temporal_domain=None, spatial_domain=None):

        # Sets h5py file operation mode to 'w' if not specified
        # 'w' = Create file, truncate if exists
        if mode is None:
            mode = 'w'

        # Initializes base class with proper mode.
        super(ComplexCoverage, self).__init__(root_dir, persistence_guid, name, reference_coverage_locs, parameter_dictionary,
                                                 mode, complex_type, temporal_domain, spatial_domain)

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
                params = coverage.get_parameter_values(this_param_names, time_segment, time, sort_parameter, stride_length,
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
                    print old, new
                    if abs(new-time) < abs(old-time):
                        cov_value_list = [(cov_dict, coverage)]
                else:
                    cov_value_list.append((cov_dict, coverage))

        combined_data = self._merge_value_dicts(cov_value_list)
        if not fill_empty_params:
            for param_name in all_empty:
                combined_data.pop(param_name)
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

    def _append_to_coverage(self, value_dictionary):
        raise NotImplementedError('Complex coverages are read-only')
        # cov = AbstractCoverage.load(self.head_coverage_path, mode='r+')
        # cov.set_parameter_values(value_dictionary)
        # cov.close()

    def new_simplex(self):
        '''
        Creates a new child simplex coverage with the same CRS information
        '''
        root_dir, guid = os.path.split(self.persistence_dir)
        name = 'generated simplex'
        pdict = deepcopy(self._range_dictionary)
        tcrs = CRS([AxisTypeEnum.TIME])
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

        # Construct temporal and spatial Domain objects
        tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
        sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)
        scov = SimplexCoverage(root_dir, 
                               utils.create_guid(),
                               name,
                               parameter_dictionary=pdict, 
                               temporal_domain=tdom,
                               spatial_domain=sdom,
                               mode='r+')
        path = scov.persistence_dir
        self.append_reference_coverage(path)
        return scov
