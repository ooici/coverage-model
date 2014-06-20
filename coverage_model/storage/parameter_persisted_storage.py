#!/usr/bin/env python

"""
@package coverage_model.parameter_persisted_storage
@file coverage_model.parameter_persisted_storage
@author Casey Bryant
@brief Persistence Layer specialized classes for abstracting parameter persistence from the underlying storage mechanism
"""

import os
import json
import base64
import numbers

import numpy as np

from ooi.logging import log
from coverage_model.metadata_factory import MetadataManagerFactory
from coverage_model.basic_types import AbstractStorage, AxisTypeEnum
from coverage_model.persistence_helpers import ParameterManager, pack, unpack
from coverage_model.parameter_data import ParameterData, NumpyParameterData, ConstantOverTime, NumpyDictParameterData
from coverage_model.data_span import Span
from coverage_model.storage.span_storage_factory import SpanStorageFactory
from coverage_model.persistence import SimplePersistenceLayer
from coverage_model.parameter_types import QuantityType


class PostgresPersistenceLayer(SimplePersistenceLayer):
    """
    The PersistenceLayer class manages the disk-level storage (and retrieval) of the Coverage Model using HDF5 files.
    """

    def __init__(self, root, guid, name=None, mode=None, inline_data_writes=True, auto_flush_values=True,
                 bricking_scheme=None, brick_dispatcher=None, value_caching=True, coverage_type=None,
                 alignment_parameter=AxisTypeEnum.TIME.lower(), storage_name=None, **kwargs):
        """
        Constructor for PersistenceLayer

        @param root The <root> component of the filesystem path for the coverage (/<root>/<guid>)
        @param guid The <guid> component of the filesystem path for the coverage (/<root>/<guid>)
        @param name CoverageModel's name persisted to the metadata attribute in the master HDF5 file
        @param auto_flush_values    True = Values flushed to HDF5 files automatically, False = Manual
        @param value_caching  if True (default), value requests should be cached for rapid duplicate retrieval
        @param kwargs
        @return None
        """

        log.debug('Persistence GUID: %s', guid)
        root = '.' if root is ('' or None) else root

        self.brick_dispatcher=brick_dispatcher
        self.bricking_scheme=bricking_scheme
        self.master_manager = MetadataManagerFactory.buildMetadataManager(root, guid, name=name, parameter_bounds=None, coverage_type=coverage_type, **kwargs)

        self.mode = mode
        if not hasattr(self.master_manager, 'auto_flush_values'):
            self.master_manager.auto_flush_values = auto_flush_values
        if not hasattr(self.master_manager, 'inline_data_writes'):
            self.master_manager.inline_data_writes = inline_data_writes
        if not hasattr(self.master_manager, 'value_caching'):
            self.master_manager.value_caching = value_caching
        if not hasattr(self.master_manager, 'coverage_type'):
            self.master_manager.coverage_type = coverage_type

        self.value_list = {}

        self.parameter_metadata = {} # {parameter_name: [brick_list, parameter_domains, rtree]}
        self.spans = {}
        self.span_list = []
        self.storage_name = storage_name

        for pname in self.param_groups:
            log.debug('parameter group: %s', pname)
            self.parameter_metadata[pname] = ParameterManager(os.path.join(self.root_dir, self.guid, pname), pname)

        if self.mode != 'r':
            if self.master_manager.is_dirty():
                self.master_manager.flush()

        self._closed = False
        if isinstance(alignment_parameter, basestring):
            self.alignment_parameter = alignment_parameter
        else:
            raise TypeError("alignment_parameter arg must implement %s.  Found %s", (basestring.__name__, type(alignment_parameter).__name__))

        log.debug('Persistence Layer Successfully Initialized')

    def __getattr__(self, key):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            return getattr(self.master_manager, key)
        else:
            return getattr(super(PostgresPersistenceLayer, self), key)

    def __setattr__(self, key, value):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            setattr(self.master_manager, key, value)
        else:
            super(PostgresPersistenceLayer, self).__setattr__(key, value)

    def update_parameter_bounds(self, parameter_name, bounds):
        dmin, dmax = bounds
        if isinstance(self.parameter_metadata[parameter_name].parameter_context.param_type, QuantityType): # TODO should we store bounds for non quantity types?
            if parameter_name in self.parameter_bounds:
                pmin, pmax = self.parameter_bounds[parameter_name]
                dmin = min(dmin, pmin)
                dmax = max(dmax, pmax)
            if dmax is None and dmin is not None:
                dmax = dmin
            elif dmin is None and dmax is not None:
                dmin = dmax
            if dmin is not None and dmax is not None:
                self.parameter_bounds[parameter_name] = (dmin, dmax)
                self.master_manager.flush()

    # CBM TODO: This needs to be improved greatly - should callback all the way to the Application layer as a "failure handler"
    def write_failure_callback(self, message, work):
        log.error('WORK DISCARDED!!!; %s: %s', message, work)

    def init_parameter(self, parameter_context, bricking_scheme):
        """
        Initializes a parameter using a ParameterContext object and a bricking
        scheme for that parameter

        @param parameter_context    ParameterContext object describing the parameter to initialize
        @param bricking_scheme  A dictionary containing the brick and chunk sizes
        @return A PersistedStorage object
        """
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

        parameter_name = parameter_context.name

        pm = ParameterManager(os.path.join(self.root_dir, self.guid, parameter_name), parameter_name, read_only=False)
        self.parameter_metadata[parameter_name] = pm

        pm.parameter_context = parameter_context

        log.debug('Initialize %s', parameter_name)

        self.master_manager.create_group(parameter_name)

        # if parameter_context.param_type._value_class == 'SparseConstantValue':
        #     v = SparsePersistedStorage(pm, self.master_manager, self.brick_dispatcher,
        #                                dtype=parameter_context.param_type.storage_encoding,
        #                                fill_value=parameter_context.param_type.fill_value,
        #                                mode=self.mode, inline_data_writes=self.inline_data_writes,
        #                                auto_flush=self.auto_flush_values)
        # else:
        v = PostgresPersistedStorage(pm, metadata_manager=self.master_manager,
                                         parameter_context=parameter_context,
                                         dtype=parameter_context.param_type.storage_encoding,
                                         fill_value=parameter_context.param_type.fill_value,
                                         mode=self.mode)
        self.value_list[parameter_name] = v

        # CBM TODO: Consider making this optional and bulk-flushing from the coverage after all parameters have been initialized
        # No need to check if they're dirty, we know they are!
        pm.flush()

        # Put the pm into read_only mode
        pm.read_only = True

        self.master_manager.flush()

        return v

    def calculate_extents(self, origin, bD, total_extents):
        """
        Calculates and returns the Rtree extents, brick extents and active brick size for the parameter

        @param origin   The origin of the brick in index space
        @param bD   The brick's domain in index space
        @param parameter_name   The parameter name
        @return rtree_extents, tuple(brick_extents), tuple(brick_active_size)
        """
        # Calculate the brick extents
        origin = list(origin)

        # Calculate the extents for the Rtree (index space)
        rtree_extents = origin + map(lambda o,s: o+s-1, origin, bD)
        # Fake out the rtree if rank == 1
        if len(origin) == 1:
            rtree_extents = [e for ext in zip(rtree_extents,[0 for x in rtree_extents]) for e in ext]
        log.debug('Rtree extents: %s', rtree_extents)

        # Calculate the extents of the brick (index space)
        brick_extents = zip(origin,map(lambda o,s: o+s-1, origin, bD))
        log.debug('Brick extents: %s', brick_extents)

        # Calculate active size using the inner extent of the domain within a brick (value space)
        brick_active_size = map(lambda o,s: min(o,s[1]+1)-s[0], total_extents, brick_extents)
        log.debug('Brick active size: %s', brick_active_size)

        # When loaded, brick_extents and brick_active_extents will be tuples...so, convert them now to allow clean comparison
        return rtree_extents, tuple(brick_extents), tuple(brick_active_size)

    def _brick_exists_master(self, brick_extents):
        do_write = True
        brick_guid = ''
        for x,v in self.master_manager.brick_list.iteritems():
            if brick_extents == v[0]:
                log.debug('Brick found with matching extents: guid=%s', x)
                do_write = False
                brick_guid = x
                break

        return do_write, brick_guid

    def has_dirty_values(self):
        """
        Checks if the master file values have been modified

        @return True if master file metadata has been modified
        """
        for v in self.value_list.itervalues():
            if v.has_dirty_values():
                return True

        return False

    def get_dirty_values_async_result(self):
        return_now = False
        if self.mode == 'r':
            log.warn('PersistenceLayer not open for writing: mode=%s', self.mode)
            return_now = True

        if self.brick_dispatcher is None:
            log.debug('\'brick_dispatcher\' is None')
            return_now = True

        if return_now:
            from gevent.event import AsyncResult
            ret = AsyncResult()
            ret.set(True)
            return ret

        return self.brick_dispatcher.get_dirty_values_async_result()

    def write_parameters(self, write_id, values):
        if not isinstance(values, dict):
            raise TypeError("values must be dict type.  Found %s" % type(values))
        arr_len = -1
        converted_dict = {}
        all_values_constant_over_time = True
        import time
        write_time = time.time()
        for key, arr in values.iteritems():
            if key not in self.value_list:
                raise KeyError("Parameter, %s, has not been initialized" % (key))
            if isinstance(arr, np.ndarray):
                arr = NumpyParameterData(key, arr)
                values[key] = arr
            if not isinstance(arr, ParameterData):
                raise TypeError("Value for %s must implement <%s>, found <%s>" % (key, ParameterData.__name__, arr.__class__.__name__))

            if not isinstance(arr, ConstantOverTime):
                if self.parameter_metadata[key].parameter_context.param_type.validate_value_set(arr.get_data()):
                    self.parameter_metadata[key].read_only = False
                    self.parameter_metadata[key].flush()
                    self.parameter_metadata[key].read_only = True

                all_values_constant_over_time = False
            if arr_len == -1 and isinstance(arr, NumpyParameterData):
                arr_len = arr.get_data().shape[0]
            elif isinstance(arr, NumpyParameterData) and arr.get_data().shape[0] != arr_len:
                raise ValueError("Array size for %s is inconsistent.  Expected %s elements, found %s." % (key, str(arr_len), str(arr.get_data().size)))
            min_val, max_val = self.value_list[key].get_statistics(arr)
            self.update_parameter_bounds(key, (min_val, max_val))

        if not all_values_constant_over_time and self.alignment_parameter not in values:
            raise LookupError("Array must be supplied for parameter, %s, to ensure alignment" % self.alignment_parameter)

        span = Span(write_id, self.master_manager.guid, values, compressors=self.value_list)
        span_table = SpanStorageFactory.get_span_storage_obj(self.storage_name)
        span_table.write_span(span)

    def get_spans_by_id(self, spans):
        return SpanStorageFactory.get_span_storage_obj(self.storage_name).get_spans(coverage_ids=self.master_manager.guid, span_ids=spans, decompressors=self.value_list)

    def _get_span_dict(self, params, time_range=None, time=None):
        return SpanStorageFactory.get_span_storage_obj(self.storage_name).get_spans(coverage_ids=self.master_manager.guid, decompressors=self.value_list)

    def read_parameters(self, params, time_range=None, time=None, sort_parameter=None, stride_length=None, fill_empty_params=False, function_params=None, as_record_array=True):
        np_dict, functions, rec_arr = self.get_data_products(params, time_range, time, sort_parameter, stride_length=stride_length, create_record_array=as_record_array, fill_empty_params=fill_empty_params)
        if function_params is not None and isinstance(function_params, dict):
            function_params.clear()
            function_params.update(functions)
        return rec_arr

    def get_data_products(self, params, time_range=None, time=None, sort_parameter=None, create_record_array=True, stride_length=None, fill_empty_params=False):
        if self.alignment_parameter not in params:
            params.append(self.alignment_parameter)

        associated_spans = self._get_span_dict(params, time_range, time)
        numpy_params, function_params = self._create_param_dict_from_spans_dict(params, associated_spans)
        dict_params = None
        np_dict = self._create_parameter_dictionary_of_numpy_arrays(numpy_params, function_params, params=dict_params)
        np_dict = self._append_parameter_fuction_data(params, np_dict)
        if fill_empty_params is True:
            np_dict = self._fill_empty_params(params, np_dict)
        rec_arr = None
        if self.alignment_parameter in np_dict:
            np_dict = self._sort_flat_arrays(np_dict, sort_parameter=sort_parameter)
            np_dict = self._trim_values_to_range(np_dict, time_range=time_range, time=time)
        if stride_length is not None:
            np_dict = {key:value[0::stride_length] for key, value in np_dict.items()}

        if len(np_dict) == 0:
            dt = np.dtype(self.parameter_metadata[self.alignment_parameter].parameter_context.param_type.value_encoding)
            np_dict = {self.alignment_parameter: np.empty(0, dtype=dt)}
        rec_arr = self._convert_to_numpy_dict_parameter(np_dict, sort_parameter=sort_parameter, as_rec_array=create_record_array)
        return np_dict, function_params, rec_arr

    def _create_param_dict_from_spans_dict(self, params, span_dict):
        numpy_params = {}
        function_params = {}
        if isinstance(span_dict, list):
            for span in span_dict:
                for param_name, data in span.param_dict.iteritems():
                    if param_name in params or params is None:
                        if isinstance(data, ConstantOverTime):
                            if param_name not in function_params:
                                function_params[param_name] = {}
                            function_params[param_name][span.ingest_time] = data
                        elif type(data) in (NumpyParameterData, NumpyDictParameterData):
                            if param_name not in numpy_params:
                                numpy_params[param_name] = {}
                            numpy_params[param_name][span.ingest_time] = data

        elif isinstance(span_dict, dict):
            for span_id, span in span_dict.iteritems():
                for param_name, data in span.iteritems():
                    if param_name in params or params is None:
                        obj = self.value_list[param_name].decompress(data[0])
                        if isinstance(obj, ConstantOverTime):
                            if param_name not in function_params:
                                function_params[param_name] = {}
                            function_params[param_name][data[3]] = obj
                        elif type(obj) in (NumpyParameterData, NumpyDictParameterData):
                            if param_name not in numpy_params:
                                numpy_params[param_name] = {}
                            numpy_params[param_name][data[3]] = obj

        return numpy_params, function_params

    @classmethod
    def _sort_flat_arrays(cls, np_dict, sort_parameter=None):
        sorted_array_dict = {}
        if sort_parameter is None or sort_parameter not in np_dict.keys():
            # sort_parameter = self.alignment_parameter
            sort_parameter = 'time'
        sort_array = np_dict[sort_parameter]
        sorted_indexes = np.argsort(sort_array)
        for key, value in np_dict.iteritems():
            sorted_array_dict[key] = value[sorted_indexes]
        return sorted_array_dict

    def _convert_to_numpy_dict_parameter(self, np_dict, sort_parameter=None, as_rec_array=False):
        param_context_dict = {}
        for key in np_dict.keys():
            if key in param_context_dict:
                param_context_dict[key] = self.value_list[key]

        if sort_parameter is None:
            sort_parameter = self.alignment_parameter

        ndpd = NumpyDictParameterData(np_dict, alignment_key=sort_parameter, param_context_dict=param_context_dict, as_rec_array=as_rec_array)

        return ndpd

    def _trim_values_to_range(self, np_dict, time_range=None, time=None):
        return_dict = {}
        time_array = np_dict[self.alignment_parameter]
        if time_range is None and time is None:
            return_dict = np_dict
        elif time_array.size == 0:
            return_dict = np_dict
        elif time_range is not None:
            if time_range[0] is not None and time_range[1] is None:
                for key, val in np_dict.iteritems():
                    return_dict[key] = val[np.where(time_range[0] <= time_array)]
            elif time_range[0] is None and time_range[1] is not None:
                for key, val in np_dict.iteritems():
                    return_dict[key] = val[np.where(time_range[1] >= time_array)]
            elif time_range[0] is not None and time_range[1] is not None:
                for key, val in np_dict.iteritems():
                    return_dict[key] = val[np.where(np.logical_and(time_range[0] <= time_array, time_range[1] >= time_array))]
            else:
                return_dict = np_dict
            return return_dict
        else:
            idx = (np.abs(time_array-time)).argmin()
            for key, val in np_dict.iteritems():
                return_dict[key] = np.array([val[idx]], dtype=val.dtype)
        return return_dict

    def _append_parameter_fuction_data(self, params, param_dict, time_segment=None, time=None):
        if time is not None and time_segment is None:
            time_segment = (time,time)
        for param in list(set(params)-set(param_dict.keys())):
            if param in self.parameter_metadata:
                param_type = self.parameter_metadata[param].parameter_context.param_type
                from coverage_model.parameter_types import ParameterFunctionType
                if isinstance(param_type, ParameterFunctionType):
                    data = param_type.function.evaluate(param_type.callback, time_segment, time)
                    param_dict[param] = data
        return param_dict

    def _create_parameter_dictionary_of_numpy_arrays(self, numpy_params, function_params=None, params=None):
        return_dict = {}
        arr_size = -1
        shape_outer_dimmension = 0
        span_order = []
        span_size_dict = {}
        t_dict = {}
        if self.alignment_parameter in numpy_params:
            for id, span_data in numpy_params[self.alignment_parameter].iteritems():
                span_size_dict[id] = span_data.get_data().size
                shape_outer_dimmension += span_data.get_data().size
                span_order.append(id)
            span_order.sort()
            t_dict = numpy_params[self.alignment_parameter]
        dt = np.dtype(self.parameter_metadata[self.alignment_parameter].parameter_context.param_type.value_encoding)
        arr = np.empty(shape_outer_dimmension, dtype=dt)
        insert_index = 0
        for span_id in span_order:
            np_data = t_dict[span_id].get_data()
            end_idx = insert_index+np_data.size
            arr[insert_index:end_idx] = np_data
            insert_index += np_data.size
        return_dict[self.alignment_parameter] = arr

        for id, span_data in numpy_params.iteritems():
            if id == self.alignment_parameter:
                continue
            npa = None
            insert_index = 0
            # param_outer_dimmension = 0;
            # param_inner_dimmension = 0;
            # for span_name in span_order:
            #     np_data = span_data[span_name].get_data()
            #     param_outer_dimmension = np_data.shape[0]

            for span_name in span_order:
                if span_name not in span_data:
                    insert_index += span_size_dict[span_name]
                    continue
                np_data = span_data[span_name].get_data()
                if npa is None:
                    npa = self.parameter_metadata[id].parameter_context.param_type.create_filled_array(shape_outer_dimmension)
                end_idx = insert_index + np_data.size
                npa[insert_index:end_idx] = np_data
                insert_index += np_data.size
            return_dict[id] = npa
                # span_order.append((id, np_data.size))
                # arr[insert_index:np_data.size+insert_index] = np_data

        for param_name, param_dict in function_params.iteritems():
            arr = ConstantOverTime.merge_data_as_numpy_array(return_dict[self.alignment_parameter],
                                                             param_dict,
                                                             param_type=self.parameter_metadata[param_name].parameter_context.param_type)
            return_dict[param_name] = arr

        return return_dict

    def _fill_empty_params(self, params, np_dict):
        filled_params = {}
        if params is not None:
            unset_params = set(params) - set(np_dict.keys())
            if len(unset_params) > 0:
                for param in unset_params:
                    filled_params[param] = self.parameter_metadata[param].parameter_context.param_type.create_filled_array(len(np_dict[self.alignment_parameter]))

        np_dict.update(filled_params)
        return np_dict

    def has_data(self):
        return SpanStorageFactory.get_span_storage_obj(self.storage_name).has_data(self.master_manager.guid)

    def num_timesteps(self):
        ts = 0
        span_dict = self._get_span_dict(self.alignment_parameter)
        for span in span_dict:
            ts += span.param_dict[self.alignment_parameter].get_data().size
        return ts

    def has_dirty_values(self):
        """
        Checks if the master file values have been modified

        @return True if master file metadata has been modified
        """
        for v in self.value_list.values():
            if v.has_dirty_values():
                return True

        return False

    def read_parameters_as_dense_array(self, params, time_range=None, time=None, sort_parameter=None):
        return_dict = {}
        arr_size = 0
        function_data = {}
        time_exists = False
        for id, span in self.spans.iteritems():
            for key, data in span.iteritems():
                if key == self.alignment_parameter:
                    time_exists = True
                if key in params:
                    d = self.value_list[key].decompress(data[0])
                    if isinstance(d, ConstantOverTime):
                        if key not in function_data:
                            function_data[key] = list()
                        function_data[key].append(d)

        if not time_exists:
            return None

        for key, d in self.spans.iteritems():
        #     if self.alignment_parameter not in d.keys():
        #         for param, vals in d.iteritems():
        #             if param not in params:
        #                 continue
        #             if

            if self.alignment_parameter not in d:
                continue
            alignment_array = self.value_list[self.alignment_parameter].decompress(d[self.alignment_parameter][0]).get_data()
            new_size = alignment_array.size
            for param, vals in d.iteritems():
                if isinstance(vals[0], np.ndarray) and  new_size != vals[0].size:
                    raise Exception("Span, %s, array is not aligned %s" % (key, param))
                if param in params or param == self.alignment_parameter:
                    if param not in return_dict.keys():
                        dtype = self.value_list[param].fill_value
                        if isinstance(dtype, basestring):
                            dtype = object
                        else:
                            dtype = type(dtype)
                        if arr_size > 0:
                            arr = np.empty(arr_size, dtype)
                            arr.fill(self.value_list[param].fill_value)
                            return_dict[param] = arr
                        else:
                            return_dict[param] = np.empty(0, dtype)
                    param_data = self.value_list[param].decompress(vals[0])
                    if isinstance(param_data, NumpyParameterData):
                        return_dict[param] = np.append(return_dict[param], param_data.get_data())
                    elif isinstance(param_data, ConstantOverTime):
                        return_dict[param] = np.append(return_dict[param], param_data.get_data_as_numpy_array(alignment_array, fill_value=self.value_list[param].fill_value))
                    else:
                        arr = np.empty(arr_size)
                        arr.fill(self.value_list[param].fill_value)
                        return_dict[param] = np.append(return_dict[param], arr)
            if new_size is not None:
                for key in return_dict:
                    if key not in d.keys() and key in self.value_list:
                        dtype = self.value_list[key].fill_value
                        if isinstance(dtype, basestring):
                            dtype = object
                        else:
                            dtype = type(dtype)
                        arr = np.empty(new_size, dtype)
                        arr.fill(self.value_list[key].fill_value)
                        return_dict[key] = np.append(return_dict[key], arr)
                arr_size += new_size

        keys = return_dict.keys()
        vals = [return_dict[key] for key in keys]
        for param, func in function_data.iteritems():
            arr = None
            func.sort()
            for data_obj in func:
                if param in return_dict:
                    arr = return_dict[param]
                arr = data_obj.get_data_as_numpy_array(return_dict[self.alignment_parameter], fill_value=self.value_list[param].fill_value, arr=arr)
            if arr is not None:
                return_dict[param] = arr
        data = NumpyDictParameterData(return_dict)
        if sort_parameter is None or sort_parameter not in return_dict.keys():
            sort_parameter = self.alignment_parameter
        data.get_data().sort(order=sort_parameter)
        return data
        # if sort_parameter is not None:
        #     # sort_parameter = self.alignment_parameter
        #     idx_arr = np.argsort(return_dict[sort_parameter])
        #     for param, arr in return_dict.iteritems():
        #         return_dict[param] = arr[idx_arr]
        # return return_dict

    def flush_values(self):
        if self.mode == 'r':
            log.warn('PersistenceLayer not open for writing: mode=%s', self.mode)
            return

        for k, v in self.value_list.iteritems():
            v.flush_values()

        return self.get_dirty_values_async_result()

    def flush(self):
        if self.mode == 'r':
            log.warn('PersistenceLayer not open for writing: mode=%s', self.mode)
            return

        self.flush_values()
        log.debug('Flushing MasterManager...')
        self.master_manager.flush()
        for pk, pm in self.parameter_metadata.iteritems():
            log.debug('Flushing ParameterManager for \'%s\'...', pk)
            pm.flush()

    def close(self, force=False, timeout=None):
        if not self._closed:
            if self.mode != 'r':
                self.flush()
                if self.brick_dispatcher is not None:
                    self.brick_dispatcher.shutdown(force=force, timeout=timeout)

        self._closed = True

    def validate_span_data(self):
        invalid_spans = []
        valid_spans = []
        associated_spans = self._get_span_dict(self.value_list.keys())
        compressors = self.value_list
        for span in associated_spans:
            span.compressors = compressors
            stored_hash = SpanStorageFactory.get_span_storage_obj(self.storage_name).get_stored_span_hash(span.id)
            recalculated_hash = span.get_hash()
            if stored_hash == recalculated_hash:
                valid_spans.append(span.id)
            else:
                invalid_spans.append(span.id)

        return valid_spans, invalid_spans

import cPickle
def base64encode(np_arr, start=None, stop=None, param_type=None, class_=None):
    if isinstance(np_arr, np.ndarray):
        if np_arr.flags['C_CONTIGUOUS'] is False:
            np_arr = np.copy(np_arr)
        if np_arr.dtype == np.object:
            np_arr = cPickle.dumps(np_arr, cPickle.HIGHEST_PROTOCOL)
            return json.dumps(['_cpickle_', base64.b64encode(np_arr)])
        else:
            return json.dumps([str(np_arr.dtype), base64.b64encode(np_arr), np_arr.shape])
    elif param_type == 'mp':
        return json.dumps([param_type, base64.b64encode(np_arr)])
    else:
        enc = TupleEncoder()
        js = enc.encode([np_arr.__class__.__name__, np_arr, start, stop, class_])
        return js


class TupleEncoder(json.JSONEncoder):
    def encode(self, o):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'_tup_': True, 'v': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            else:
                return item
        return super(TupleEncoder, self).encode(hint_tuples(o))


def hinted_tuple_hook(o):
    if '_tup_' in o:
        return tuple(o['v'])
    else:
        return o


def base64decode(json_str):
    loaded = json.loads(json_str, object_hook=hinted_tuple_hook)
    if isinstance(loaded, list) and len(loaded) == 3:
        data_type = np.dtype(loaded[0])
        arr = np.frombuffer(base64.decodestring(loaded[1]),data_type)
        if len(loaded) > 2:
            return arr.reshape(loaded[2])
        return arr
    elif isinstance(loaded, list) and len(loaded) == 2 and loaded[0] == '_cpickle_':
        val = base64.b64decode(loaded[1])
        val = cPickle.loads(base64.decodestring(loaded[1]))
        return val
    elif isinstance(loaded, list) and len(loaded) == 2 and loaded[0] == 'mp':
        return (base64.b64decode(loaded[1]),)
    elif isinstance(loaded, list) and len(loaded) == 5:
        return (loaded[0], loaded[1], loaded[2], loaded[3], loaded[4])
    else:
        raise TypeError("Cannot decompress type %s" % type(loaded))


def simple_encode(np_arr):
    return json.dumps(np_arr.tolist())


def simple_decode(json_str):
    return np.array(json.loads(json_str))


class PostgresPersistedStorage(AbstractStorage):

    def __init__(self, parameter_manager, metadata_manager, parameter_context, dtype, fill_value, mode=None):
        self.parameter_manager = parameter_manager
        self.metadata_manager = metadata_manager
        self.dtype = dtype
        self.fill_value = fill_value
        self.mode = mode
        self.parameter_context = parameter_context

    def __getitem__(self, slice_):
        return None

    def __setitem__(self, slice_, value):
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

    def create_numpy_object_array(self, array):
        if isinstance(array, np.ndarray):
            array = array.tolist()
        arr = np.empty(len(array), dtype=object)
        arr[:] = array
        return arr

    def compress(self, values):
        if isinstance(values, NumpyParameterData):
            data = values.get_data()
            return base64encode(data)
        elif isinstance(values, ConstantOverTime):
            return base64encode(values.get_data(), start=values.start, stop=values.stop, class_=values.__class__.__name__)
        else:
            raise TypeError("values must implement %s or %s, found %s" % (NumpyParameterData.__name__, ConstantOverTime.__name__, type(values)))

    def _object_unpack(self, value):
        value = unpack(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, tuple):
            return list(value)
        else:
            return value

    def decompress(self, obj):
        vals = base64decode(obj)
        if isinstance(vals, np.ndarray):
            return NumpyParameterData(self.parameter_manager.parameter_name, vals)
        # if isinstance(vals[0], basestring) and vals[0].startswith('_mp_'):
        #
        #     st = vals[0][len('_mp_'):]
        #     if hasattr(vals[0], '__iter__'):
        #         vals = np.array([unpack(x) for x in vals[0]])
        #     else:
        #         vals = np.array([unpack(vals[0])])
        #     return NumpyParameterData(self.parameter_manager.parameter_name, vals)
        elif isinstance(vals, tuple) and len(vals) == 1 and isinstance(vals[0], basestring):
            # if hasattr(vals[0], '__iter__'):
            #     vals = np.array([unpack(x) for x in vals[0]])
            # else:
            vals = np.array([unpack(vals[0])])
            return NumpyParameterData(self.parameter_manager.parameter_name, vals)

        elif isinstance(vals, tuple):
            from coverage_model.parameter_data import RepeatOverTime
            if vals[4] == ConstantOverTime.__name__:
                return ConstantOverTime(self.parameter_manager.parameter_name, vals[1], time_start=vals[2], time_end=vals[3])
            elif vals[4] == RepeatOverTime.__name__:
                return RepeatOverTime(self.parameter_manager.parameter_name, vals[1], time_start=vals[2], time_end=vals[3])
            else:
                raise Exception("Could not rebuild class %s" % vals[4])
        else:
            raise Exception("Could not handle decompressed value %s" % vals)

    def get_statistics(self, v):
        import datetime
        import math
        min_val = None
        max_val = None
        valid_types = [np.bool_, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                       np.uint16, np.uint32, np.uint64, np.float_, np.float16, np.float32, np.float64,
                       np.complex_, np.complex64, np.complex128]
        invalid_fill_values = [None, np.NaN, self.fill_value]
        if self.parameter_manager.parameter_name in self.metadata_manager.param_groups:
            if isinstance(v, NumpyParameterData) and v.get_data().dtype.type is not np.string_ and len(v.get_data().shape) == 1:
                # First try to do it fast
                v = v.get_data()
                try:
                    if issubclass(v.dtype.type, numbers.Number) or v.dtype.type in valid_types:
                        min_val = v.min()
                        max_val = v.max()
                except:
                    min_val = None
                    max_val = None
                # if fast didn't return valid values, do it slow, but right
                if min_val in invalid_fill_values or max_val in invalid_fill_values or math.isnan(min_val) or math.isnan(max_val):
                    ts = datetime.datetime.now()
                    try:
                        mx = [x for x in v.all() if x not in invalid_fill_values and (type(x) in valid_types or issubclass(type(x), numbers.Number))]
                        if len(mx) > 0:
                            min_val = min(mx)
                            max_val = max(mx)
                        time_loss = datetime.datetime.now() - ts
                        log.debug("Repaired numpy statistics inconsistency for parameter/type %s/%s.  Time loss of %s seconds ",
                                  self.parameter_manager.parameter_name, str(v.dtype.type), str(time_loss))
                    except:
                        pass
            elif isinstance(v, ConstantOverTime) and not isinstance(v.get_data(), basestring):
                data = v.get_data()
                if data in valid_types or isinstance(data, numbers.Number):
                    min_val = data
                    max_val = data
        return min_val, max_val

    def expand(self, arrshp, origin, expansion, fill_value=None):
        pass # No op

    def has_dirty_values(self):
        """
        Checks if the master file values have been modified

        @return True if master file metadata has been modified
        """
        if self.metadata_manager.is_dirty():
            return True
        if self.parameter_manager.is_dirty():
            return True

        return False

    def flush_values(self):
        self.parameter_manager.flush()
        self.metadata_manager.flush()

