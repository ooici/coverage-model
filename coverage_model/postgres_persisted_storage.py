#!/usr/bin/env python

"""
@package coverage_model.postgres_persisted_storage
@file coverage_model.postgres_persisted_storage
@author Casey Bryant
@brief Persistence Layer specialized classes for storing persisted data to Postgres
"""

from ooi.logging import log
import os
import numpy as np
import json
import base64
import numbers
from coverage_model.metadata_factory import MetadataManagerFactory
from coverage_model.basic_types import AbstractStorage
from coverage_model.persistence_helpers import ParameterManager
from coverage_model.storage.parameter_data import ParameterData, NumpyParameterData, ConstantOverRange


class PostgresPersistenceLayer(object):
    """
    The PersistenceLayer class manages the disk-level storage (and retrieval) of the Coverage Model using HDF5 files.
    """

    def __init__(self, root, guid, name=None, mode=None, inline_data_writes=True, auto_flush_values=True,
                 bricking_scheme=None, brick_dispatcher=None, value_caching=True, coverage_type=None, **kwargs):
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

        for pname in self.param_groups:
            log.debug('parameter group: %s', pname)
            self.parameter_metadata[pname] = ParameterManager(os.path.join(self.root_dir, self.guid, pname), pname)

        if self.mode != 'r':
            if self.master_manager.is_dirty():
                self.master_manager.flush()

        self._closed = False

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
        if parameter_name in self.parameter_bounds:
            pmin, pmax = self.parameter_bounds[parameter_name]
            dmin = min(dmin, pmin)
            dmax = max(dmax, pmax)
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
                                         dtype=parameter_context.param_type.storage_encoding,
                                         fill_value=parameter_context.param_type.fill_value)
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
            raise TypeError("values must be dict type")
        arr_len = -1
        converted_dict = {}
        for key, arr in values.iteritems():
            if key not in self.value_list:
                raise LookupError("Parameter, %s, has not been initialized" % (key))
            if not isinstance(arr, ParameterData):
                raise TypeError("Value for %s must implement <%s>, found <%s>" % (key, ParameterData.__name__, arr.__class__.__name__))
            if arr_len == -1 and isinstance(arr, NumpyParameterData):
                arr_len = arr.get_data().size
            elif isinstance(arr, NumpyParameterData) and arr.get_data().size != arr_len:
                raise ValueError("Array size for %s is inconsistent.  Expected %s elements, found %s." % (key, str(arr_len), str(arr.get_data().size)))
            compressed_value = self.value_list[key].compress(arr)
            min_val, max_val = self.value_list[key].get_statistics(arr)
            converted_dict[key] = (compressed_value, min_val, max_val)

        if 'time' not in converted_dict:
            raise LookupError("Array must be supplied for parameter, %s, to ensure alignment" % ('time'))

        # print "WRITE ID: ", write_id
        # for param_name, data in converted_dict.iteritems():
        #     print '    ', param_name, ": min/max ", data[1], '/', data[2]
        #     print '        ', data[0]
        self.spans[write_id] = converted_dict

    def read_parameters(self, params, time_range=None, time=None, sort_parameter=None):
        return_dict = {}
        arr_size = 0
        if time_range is None and time is None:
            for key, d in self.spans.iteritems():
                alignment_array = self.value_list['time'].decompress(d['time'][0]).get_data()
                new_size = alignment_array.size
                for param, vals in d.iteritems():
                    if isinstance(vals[0], np.ndarray) and  new_size != vals[0].size:
                        raise Exception("Span, %s, array is not aligned %s" % (key, param))
                    if param in params:
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
                        elif isinstance(param_data, ConstantOverRange):
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
        if sort_parameter is not None:
            # sort_parameter = 'time'
            idx_arr = np.argsort(return_dict[sort_parameter])
            for param, arr in return_dict.iteritems():
                return_dict[param] = arr[idx_arr]
        return return_dict

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


def base64encode(np_arr, start=None, stop=None):
    if isinstance(np_arr, np.ndarray):
        return json.dumps([str(np_arr.dtype), base64.b64encode(np_arr), np_arr.shape])
    else:
        if start is not None and stop is not None:
            return json.dumps([str(type(np_arr)), np_arr, start, stop])
        else:
            raise ValueError("start and stop parameters must be supplied for non-%s types", np.ndarray.__name__)


def base64decode(json_str):
    loaded = json.loads(json_str)
    if isinstance(loaded, list) and len(loaded) == 3:
        data_type = np.dtype(loaded[0])
        arr = np.frombuffer(base64.decodestring(loaded[1]),data_type)
        if len(loaded) > 2:
            return arr.reshape(loaded[2])
        return arr
    elif isinstance(loaded, list) and len(loaded) == 4:
        return (loaded[1], loaded[2], loaded[3])
    else:
        raise TypeError("Cannot decompress type %s" % type(loaded))


def simple_encode(np_arr):
    return json.dumps(np_arr.tolist())


def simple_decode(json_str):
    return np.array(json.loads(json_str))


class PostgresPersistedStorage(AbstractStorage):

    def __init__(self, parameter_manager, metadata_manager, dtype, fill_value):
        self.parameter_manager = parameter_manager
        self.metadata_manager = metadata_manager
        self.dtype = dtype
        self.fill_value = fill_value

    def __setitem__(self, slice_, value):
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

    def compress(self, values):
        if isinstance(values, NumpyParameterData):
            return base64encode(values.get_data())
        elif isinstance(values, ConstantOverRange):
            return base64encode(values.get_data(), start=values.start, stop=values.stop)
        else:
            raise TypeError("values must implement %s or %s, found %s" % (NumpyParameterData.__name__, ConstantOverRange.__name__, type(values)))

    def decompress(self, obj):
        vals = base64decode(obj)
        if isinstance(vals, np.ndarray):
            return NumpyParameterData(self.parameter_manager.parameter_name, vals)
        elif isinstance(vals, tuple):
            return ConstantOverRange(self.parameter_manager.parameter_name, vals[0], range_start=vals[1], range_end=vals[2])
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
            if isinstance(v, NumpyParameterData) and v.get_data().dtype.type is not np.string_:
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
                    mx = [x for x in v if x not in invalid_fill_values and (type(x) in valid_types or issubclass(type(x), numbers.Number))]
                    if len(mx) > 0:
                        min_val = min(mx)
                        max_val = max(mx)
                    time_loss = datetime.datetime.now() - ts
                    log.debug("Repaired numpy statistics inconsistency for parameter/type %s/%s.  Time loss of %s seconds ",
                              self.parameter_manager.parameter_name, str(v.dtype.type), str(time_loss))
            elif isinstance(v, ConstantOverRange) and not isinstance(v.get_data(), basestring):
                data = v.get_data()
                if data in valid_types or isinstance(data, numbers.Number):
                    min_val = data
                    max_val = data
        return min_val, max_val

    def expand(self, arrshp, origin, expansion, fill_value=None):
        pass # No op

