#!/usr/bin/env python

"""
@package coverage_model.coverage
@file coverage_model/coverage.py
@author Christopher Mueller
@brief The core classes comprising the Coverage Model
"""

# http://collabedit.com/kkacp


# Notes:
# cell_map[cell_handle]=(0d_handles,)
# 0d_map[0d_handle]=(cell_handles,)
# ** Done via an association table of 2 arrays - indices are the alignment
#
# parameter values implicitly aligned to 0d handle array
#
# shape applied to 0d handle array
#
# cell is 'top level dimension'
#
# parameters can be implicitly aligned to cell array

#CBM:TODO: Add type checking throughout all classes as determined appropriate, a la:
#@property
#def spatial_domain(self):
#    return self.__spatial_domain
#
#@spatial_domain.setter
#def spatial_domain(self, value):
#    if isinstance(value, AbstractDomain):
#        self.__spatial_domain = value

from copy import deepcopy
import os
import collections
import pickle
from collections import Iterable

import numpy as np

from ooi.logging import log
from pyon.util.async import spawn
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


#=========================
# Coverage Objects
#=========================

class AbstractCoverage(AbstractIdentifiable):
    """
    Core data model, persistence, etc
    TemporalTopology
    SpatialTopology
    """

    VALUE_CACHE_LIMIT = 30
    version = '0.0.3'

    def __init__(self, mode=None):
        AbstractIdentifiable.__init__(self)
        self._closed = False
        self._range_dictionary = ParameterDictionary()
        self._range_value = RangeValues()
        self.value_caching = True
        self._value_cache = collections.deque(maxlen=5)
        self.param_cache = self._value_cache
        self._bricking_scheme = {'brick_size': 100000, 'chunk_size': 100000}

        self.temporal_domain = GridDomain(GridShape('temporal',[0]), CRS.standard_temporal(), MutabilityEnum.EXTENSIBLE)
        self.spatial_domain = None
        self.persistence_name = 'in_memory_storage'

        self._head_coverage_path = None

        if mode is not None and isinstance(mode, basestring) and mode[0] in ['r','w','a','r+']:
            self.mode = mode
        else:
            self.mode = 'r'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure the coverage is closed
        self.close()

    @classmethod
    def pickle_save(cls, cov_obj, file_path, use_ascii=False):
        '''
        Attempts to serialize an instance of an AbstractCoverage using pickle
        '''
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))

        with open(file_path, 'w') as f:
            pickle.dump(cov_obj, f, 0 if use_ascii else 2)

        log.info('Saved to pickle \'%s\'', file_path)
        log.warn('\'pickle_save\' and \'pickle_load\' are not 100% safe, use at your own risk!!')

    @classmethod
    def pickle_load(cls, file_path):
        '''
        Attempts to load a serialized instance of an AbstractCoverage using pickle
        '''
        with open(file_path, 'r') as f:
            obj = pickle.load(f)

        if not isinstance(obj, AbstractCoverage):
            raise StandardError('loaded object must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(obj)))

        log.warn('\'pickle_save\' and \'pickle_load\' are not 100% safe, use at your own risk!!')
        log.info('Loaded from pickle \'%s\'', file_path)
        return obj

    @classmethod
    def load(cls, root_dir, persistence_guid=None, mode=None):
        '''
        Loads a standard persisted instance of a Coverage Model
        root_dir         - The root filesystem directory that hosts the coverage instances
        persistence_guid - Unique identifier for this instance of the coverage
        mode             - I/O mode
        returns an instantiated coverage model instance
        '''
        if not isinstance(root_dir, basestring):
            raise ValueError('\'root_dir\' must be a string')

        if persistence_guid is None:
            if root_dir.endswith(os.path.sep): # Strip trailing separator if it exists
                root_dir = root_dir[:-1]
            root_dir, persistence_guid = os.path.split(root_dir)
        elif not isinstance(persistence_guid, basestring):
            raise ValueError('\'persistence_guid\' must be a string')

        # Otherwise, determine which coverage type to use to open the file
#        ctype = get_coverage_type(os.path.join(root_dir, persistence_guid, '{0}_master.hdf5'.format(persistence_guid)))
        ctype = MetadataManagerFactory.getCoverageType(root_dir, persistence_guid)

        if ctype == 'simplex':
            ccls = SimplexCoverage
        elif ctype == 'view':
            ccls = ViewCoverage
        elif ctype == 'complex':
            ccls = ComplexCoverage
        else:
            raise TypeError('Unknown Coverage type specified in master file : {0}', ctype)

        return ccls(root_dir, persistence_guid, mode=mode)

    @classmethod
    def save(cls, cov_obj, *args, **kwargs):
        '''
        Saves a coverage model instance
        '''
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))
        cov_obj.flush()

    def refresh(self):
        '''
        Refreshes the persistence layer
        '''
        if not getattr(self, '_in_memory_storage', None):
            self.close()
            self.__init__(os.path.split(self.persistence_dir)[0], self.persistence_guid, mode=self.mode)

    @property
    def head_coverage_path(self):
        return self._head_coverage_path

    @property
    def coverage_type(self):
        return getattr(self._persistence_layer, 'coverage_type', 'simplex')

    @property
    def temporal_parameter_name(self):
        return self._range_dictionary.temporal_parameter_name

    @property
    def parameter_dictionary(self):
        return deepcopy(self._range_dictionary)

    @property
    def module_dependencies(self):
        s = set()
        map(s.update, [self._range_dictionary.get_context(p).get_module_dependencies() for p in self.list_parameters()])

        return tuple(s)

    def has_parameter_data(self):
        return self._persistence_layer.has_data()

    def is_empty(self):
        return not self.has_parameter_data()

    def num_timesteps(self):
        return self._persistence_layer.num_timesteps()

    def _assign_domain(self, pcontext):
        no_sdom = self.spatial_domain is None

        ## Determine the correct array shape

        # Get the parameter variability; assign to VariabilityEnum.NONE if None
        pv = pcontext.variability or VariabilityEnum.NONE
        if no_sdom and pv in (VariabilityEnum.SPATIAL, VariabilityEnum.BOTH):
            log.info('ParameterContext \'{0}\' indicates Spatial variability, but coverage has no Spatial Domain'.format(pcontext.name))

        if pv == VariabilityEnum.TEMPORAL: # Only varies in the Temporal Domain
            pcontext.dom = DomainSet(self.temporal_domain, None)
        elif pv == VariabilityEnum.SPATIAL: # Only varies in the Spatial Domain
            pcontext.dom = DomainSet(None, self.spatial_domain)
        elif pv == VariabilityEnum.BOTH: # Varies in both domains
            # If the Spatial Domain is only a single point on a 0d Topology, the parameter's shape is that of the Temporal Domain only
            if no_sdom or (len(self.spatial_domain.shape.extents) == 1 and self.spatial_domain.shape.extents[0] == 0):
                pcontext.dom = DomainSet(self.temporal_domain, None)
            else:
                pcontext.dom = DomainSet(self.temporal_domain, self.spatial_domain)
        elif pv == VariabilityEnum.NONE: # No variance; constant
        # CBM TODO: Not sure we can have this constraint - precludes situations like a TextType with Variablity==None...
        #            # This is a constant - if the ParameterContext is not a ConstantType, make it one with the default 'x' expr
        #            if not isinstance(pcontext.param_type, ConstantType):
        #                pcontext.param_type = ConstantType(pcontext.param_type)

            # The domain is the total domain - same value everywhere!!
            # If the Spatial Domain is only a single point on a 0d Topology, the parameter's shape is that of the Temporal Domain only
            if no_sdom or (len(self.spatial_domain.shape.extents) == 1 and self.spatial_domain.shape.extents[0] == 0):
                pcontext.dom = DomainSet(self.temporal_domain, None)
            else:
                pcontext.dom = DomainSet(self.temporal_domain, self.spatial_domain)
        else:
            # Should never get here...but...
            raise SystemError('Must define the variability of the ParameterContext: a member of VariabilityEnum')

        # Assign the pname to the CRS (if applicable) and select the appropriate domain (default is the spatial_domain)
        dom = self.spatial_domain
        if not pcontext.axis is None and AxisTypeEnum.is_member(pcontext.axis, AxisTypeEnum.TIME):
            dom = self.temporal_domain
            dom.crs.axes[pcontext.axis] = pcontext.name
        elif not no_sdom and (pcontext.axis in self.spatial_domain.crs.axes):
            dom.crs.axes[pcontext.axis] = pcontext.name

    def append_parameter(self, parameter_context):
        """
        Appends a ParameterContext object to the internal set for this coverage.

        A <b>deep copy</b> of the supplied ParameterContext is added to self._range_dictionary.  An AbstractParameterValue of the type
        indicated by ParameterContext.param_type is added to self._range_value.  If the ParameterContext indicates that
        the parameter is a coordinate parameter, it is associated with the indicated axis of the appropriate CRS.

        @param parameter_context    The ParameterContext to append to the coverage <b>as a copy</b>
        @throws StandardError   If the ParameterContext.axis indicates that it is temporal and a temporal parameter
        already exists in the coverage
        """
        if self.closed:
            raise IOError('I/O operation on closed file')

        #if self.mode == 'r':
        #    raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        if not isinstance(parameter_context, ParameterContext):
            raise TypeError('\'parameter_context\' must be an instance of ParameterContext')

        if parameter_context.name in self._range_dictionary:
            raise ValueError('\'paramaeter_context\' with name \'{0}\' already exists'.format(parameter_context.name))

        # Create a deep copy of the ParameterContext
        pcontext = deepcopy(parameter_context)

        pname = pcontext.name

        # Assign the coverage's domain object(s)
        self._assign_domain(pcontext)

        # If this is a ParameterFunctionType parameter, provide a callback to the coverage's _range_value
        if hasattr(pcontext, '_pval_callback'):
            pcontext._pval_callback = self.get_parameter_values
            pcontext._pctxt_callback = self.get_parameter_context

        self._range_dictionary.add_context(pcontext)
        s = self._persistence_layer.init_parameter(pcontext, self._bricking_scheme)
        self._range_value[pname] = get_value_class(param_type=pcontext.param_type, domain_set=pcontext.dom, storage=s)

    def get_parameter(self, param_name):
        """
        Get a Parameter object by name

        The Parameter object contains the ParameterContext and AbstractParameterValue associated with the param_name

        @param param_name  The local name of the parameter to return
        @returns A Parameter object containing the context and value for the specified parameter
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if self.closed:
            raise IOError('I/O operation on closed file')

        if param_name in self._range_dictionary:
            p = Parameter(deepcopy(self._range_dictionary.get_context(param_name)), self._range_value[param_name].shape, self._range_value[param_name])
            return p
        else:
            raise KeyError('Coverage does not contain parameter \'{0}\''.format(param_name))

    def _handle_closed(self):
        if self.closed:
            raise IOError('I/O operation on closed file')

    def set_parameter_function(self, param_name, func):
        self._handle_closed()
        if param_name in self._range_dictionary:
            from coverage_model.parameter_types import ParameterFunctionType
            param_type = self._range_dictionary[param_name].param_type
            if isinstance(param_type, ParameterFunctionType):
                param_type.callback = func
            else:
                raise TypeError('%s is not type %s' % (param_name, ParameterFunctionType.__name__))
        else:
            raise KeyError('Coverage does not contain parameter \'{0}\''.format(param_name))



    def list_parameters(self, coords_only=False, data_only=False):
        """
        List the names of the parameters contained in the coverage

        @param coords_only List only the coordinate parameters
        @param data_only   List only the data parameters (non-coordinate) - superseded by coords_only
        @returns A list of parameter names
        """
        if coords_only:
            lst=[x for x, v in self._range_dictionary.iteritems() if v[1].is_coordinate]
        elif data_only:
            lst=[x for x, v in self._range_dictionary.iteritems() if not v[1].is_coordinate]
        else:
            lst=[x for x in self._range_dictionary]
        lst.sort()
        return lst

    def get_parameter_values(self, param_names=None, time_segment=None, time=None,
                             sort_parameter=None, stride_length=None, return_value=None, fill_empty_params=False,
                             function_params=None, as_record_array=True):
        """
        Retrieve the value for a parameter

        Returns the value from param_name.  Temporal and spatial DomainOfApplication objects can be used to
        constrain the response.  See DomainOfApplication for details.

        @param param_names   A collection of names of parameters to retrieve
        @param time_segment  A tuple of observation start and end time for data to retrieve
        @param time          Retrieve data points with observation time closest to time
        @param return_value  Not valid since we can no longer guarantee the size of the returned data
        @throws KeyError    The coverage does not contain a parameter with name provided in 'param_names'
        """
        if self.closed:
            raise IOError('I/O operation on closed file')

        # Get all parameters if none are specified
        if param_names is None:
            param_names = self._range_dictionary.keys()

        if not isinstance(param_names, Iterable) or isinstance(param_names, basestring):
            param_names = [param_names]
        for param_name in param_names:
            if not param_name in self._range_value:
                raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        if self.value_caching:
            vals = self.get_cached_values(param_names, time_segment, time, sort_parameter, stride_length, fill_empty_params)
            if vals is not None:
                return vals

        vals = self._persistence_layer.read_parameters(param_names, time_segment, time, sort_parameter,
                                                       stride_length=stride_length, fill_empty_params=fill_empty_params,
                                                       function_params=function_params, as_record_array=as_record_array)
        if self.value_caching:
            self.cache_it(vals, param_names, time_segment, time, sort_parameter, stride_length, fill_empty_params)
        return vals

    def cache_it(self, vals, param_names, time_segement, time, sort_parameter, stride_length, fill_empty_params):
        if time_segement is None or time_segement[0] is None or time_segement[1] is None:
            return
        tup = time_segement, time, sort_parameter, stride_length, fill_empty_params
        cache = False
        if len(self._value_cache) !=0:
            for val in self._value_cache:
                if cmp(val[0], tup) == 0:
                    for param_name in param_names:
                        if param_name not in val[1]:
                            cache = True
                            break
                else:
                    cache = True
                    break
        else:
            cache = True
        if cache:
            self._value_cache.append((tup, param_names, vals))

    def get_cached_values(self, params, time_segement, time, sort_parameter, stride_length, fill_empty_params):
        tup = time_segement, time, sort_parameter, stride_length, fill_empty_params
        rv = None
        for val in self._value_cache:
            if val[0] == tup:
                cached = True
                for param in params:
                    if param not in val[1]:
                        cached = False
                if cached:
                    rv = val[2]
                    break
        return rv

    def get_spans(self, spans):
        return self._persistence_layer.get_spans_by_id(spans)

    def get_value_dictionary(self, param_list=None, temporal_slice=None, domain_slice=None):
        '''
        Retrieves a dictionary of parameters and value sets.
        The value set is a cross-section result between the specified temporal_slice
        '''
        if self.closed:
            raise IOError('I/O operation on closed file')
        
        if param_list is None:
            param_list = self.list_parameters()
        else:
            for p in param_list:
                if p not in self._range_value:
                    raise KeyError('Parameter \'{0}\' not found in coverage'.format(p))
        if self.is_empty():
            retval = {}
            for p in param_list:
                retval[p] = np.array([], dtype=self.get_parameter_context(p).param_type.value_encoding)
            return retval


        param_list = param_list or []
        return self.get_parameter_values(param_list, time_segment=temporal_slice, as_record_array=False, fill_empty_params=True).get_data()

    def _get_me_a_slice(self, temporal_slice, domain_slice):
        if temporal_slice is None and domain_slice is None:
            slice_ = slice(None,None)
            return slice_

        elif temporal_slice is None and domain_slice:
            if isinstance(domain_slice, slice):
                return domain_slice
            if len(domain_slice) == 2:
                start, stop = domain_slice
                return slice(start, stop)
            elif len(domain_slice) == 3:
                start, stop, stride = domain_slice
                return slice(start, stop, stride)
            else:
                raise ValueError("Improper domain slice: %s" % domain_slice)



        stride = None
        if isinstance(temporal_slice, slice):
            start, stop, stride = temporal_slice.start, temporal_slice.stop, temporal_slice.step
        elif isinstance(temporal_slice, tuple):
            if len(temporal_slice) == 2:
                start, stop = temporal_slice
            elif len(temporal_slice) == 3:
                start, stop, stride = temporal_slice
            else:
                raise ValueError("Improper temporal slice: %s" % temporal_slice)
        else:
            raise ValueError("Improper temporal slice: %s" % temporal_slice)

        if start is None:
            start = -np.inf
        if stop is None:
            stop = np.inf

        # Make sure the axis is in the value dictionary
        # But remember to remove it on the way out
        time_vector = self._range_value[self.temporal_parameter_name]

        # Dramatically improve performance, no need to retrieve the entire
        # time-array to find out where start and stop are, we just use a
        # binary-search on an "array-like" HDF5 interface

        start_idx = self._searchsorted(time_vector, start)
        stop_idx = self._searchsorted(time_vector, stop)
        slice_ = slice(start_idx, stop_idx, stride)
        return slice_

    def _searchsorted(self, time_vector, val):
        return np.searchsorted(time_vector[:], val)


    def _aggregate_value_dict(self, value_dict, slice_, param_list=None):
        param_list = param_list or self.list_parameters()
        for p in param_list:
            if slice_.start == slice_.stop and slice_.start is not None:
                dtype = self._range_dictionary[p].param_type.value_encoding
                value_set = np.array([], dtype=dtype)
            else:
                try:
                    value_set = self._range_value[p][slice_]
                except ParameterFunctionException:
                    log.exception("Parameter Function Exception")
                    dtype = self._range_dictionary[p].param_type.value_encoding
                    t_range_value = self._range_value[self.temporal_parameter_name]
                    if t_range_value:
                        shape = t_range_value[slice_].shape
                        value_set = np.ones(shape, dtype=dtype) * self._range_dictionary[p].param_type.fill_value
                    else:
                        value_set = None

            if p in value_dict:
                value_set = np.concatenate([value_dict[p], value_set])
            value_dict[p] = value_set


    def _get_cached_values(self, param_name, slice_, total_shape):
        '''
        Gets a value from a properly managed LRU cache
        '''
        # Make slice_ fully expressed such that there are no "None" entries - this lets us ignore domain growth
        slk = utils.express_slice(slice_, total_shape)
        key = (param_name, utils.hash_any(slk))
        try:
            return_value = self._value_cache.pop(key)
        except KeyError:
            return_value = self._range_value[param_name][slice_]
            if len(self._value_cache) >= self.VALUE_CACHE_LIMIT:
                self._value_cache.popitem(0)
        self._value_cache[key] = return_value
        return return_value

    def set_time_values(self, values):
        """
        Convenience method for setting time values

        @param value    The value to set
        @param tdoa The temporal DomainOfApplication; default to full Domain
        """
        if isinstance(values, list):
            values = np.array(values)
        return self._persistence_layer.write_parameters(self.get_write_id(), {self.temporal_parameter_name: values})

    def get_time_values(self, time_segement=None, stride_length=None, return_value=None):
        """
        Convenience method for retrieving time values

        Delegates to get_parameter_values, supplying the temporal parameter name and sdoa == None
        @param tdoa The temporal DomainOfApplication; default to full Domain
        @param return_value If supplied, filled with response value
        """
        return self.get_parameter_values(self.temporal_parameter_name, time_segment=time_segement,
                                         stride_length=stride_length, return_value=return_value).get_data()[self.temporal_parameter_name]

    def _clear_value_cache_for_parameter(self, param_name):
        for k in self._value_cache.keys():
            if k[0] == param_name:
                self._value_cache.pop(k)

    def set_parameter_values(self, values):
        if self.mode == 'r' or self._closed:
            raise IOError("Coverage not open for write")
        self._persistence_layer.write_parameters(self.get_write_id(), values)

    def get_write_id(self):
        return create_guid()

    def clear_value_cache(self):
        if self.value_caching:
            self._value_cache.clear()

    def get_parameter_context(self, param_name):
        """
        Retrieve a deepcopy of the ParameterContext object for the specified parameter

        @param param_name   The name of the parameter for which to retrieve context
        @returns A deepcopy of the specified ParameterContext object
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if not param_name in self._range_dictionary:
            raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        if self.mode == 'r':
            return self._range_dictionary.get_context(param_name)
        
        return deepcopy(self._range_dictionary.get_context(param_name))

    def _axis_arg_to_params(self, axis=None):
        """
        Helper function to compose a list of parameter names based on the <i>axis</i> argument

        If <i>axis</i> is None, all coordinate parameters are included

        @param axis A member of AxisTypeEnum; may be an iterable of such members
        """
        params = []
        if axis is None:
            if self.temporal_domain is not None:
                params.extend(pn for pk, pn in self.temporal_domain.crs.axes.iteritems())
            if self.spatial_domain is not None:
                params.extend(pn for pk, pn in self.spatial_domain.crs.axes.iteritems())
        elif hasattr(axis, '__iter__'):
            for a in axis:
                if self.temporal_domain is not None and a in self.temporal_domain.crs.axes:
                    params.append(self.temporal_domain.crs.axes[a])
                elif self.spatial_domain is not None and a in self.spatial_domain.crs.axes:
                    params.append(self.spatial_domain.crs.axes[a])
                else:
                    raise ValueError('Specified axis ({0}) not found in coverage'.format(a))
        elif self.temporal_domain is not None and axis in self.temporal_domain.crs.axes:
            params.append(self.temporal_domain.crs.axes[axis])
        elif self.spatial_domain is not None and axis in self.spatial_domain.crs.axes:
            params.append(self.spatial_domain.crs.axes[axis])
        else:
            raise ValueError('Specified axis ({0}) not found in coverage'.format(axis))

        return params

    def _parameter_name_arg_to_params(self, parameter_name=None, pdict=None):
        """
        Helper function to compose a list of parameter names based on the <i>parameter_name</i> argument

        If <i>parameter_name</i> is None, all parameters in the coverage are included
        The <i>pdict</i> argument is used to check for validity of <i>parameter_name</i>

        @param parameter_name A string parameter name; may be an iterable of such members
        @pdict  A ParameterDictionary; self._range_dictionary if None
        """
        if pdict is None:
            pdict = self._range_dictionary
        params = []
        if parameter_name is None:
            params.extend(pdict.keys())
        elif hasattr(parameter_name, '__iter__'):
            params.extend(pn for pn in parameter_name)
        else:
            params.append(parameter_name)

        # Verify
        invalid_params = set(params).difference(set(pdict.keys()))

        if None in invalid_params:
            invalid_params = ''

        if len(invalid_params) != 0:
            raise ValueError('Coverage does not have parameters: \'{0}\''.format(invalid_params))

        if None in params:
            params = ''

        return params

    def get_data_bounds(self, parameter_name=None):
        """
        Returns the bounds (min, max) for the parameter(s) indicated by <i>parameter_name</i>

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        If more than one parameter is indicated by <i>parameter_name</i>, a dict of {key:(min,max)} is returned;
        otherwise, only the (min, max) tuple is returned

        @param parameter_name   A string parameter name; may be an iterable of such members
        """

        if not self.has_parameter_data():
            raise ValueError('The coverage has no data!')

        ret = {}
        for pn in self._parameter_name_arg_to_params(parameter_name):
            if pn in self._persistence_layer.parameter_bounds:
                ret[pn] = self._persistence_layer.parameter_bounds[pn]

        if len(ret) == 1:
            ret = ret.values()[0]
        return ret

    def get_data_bounds_by_axis(self, axis=None):
        """
        Returns the bounds (min, max) for the coordinate parameter(s) indicated by <i>axis</i>

        If <i>axis</i> is None, all coordinate parameters are included

        If more than one parameter is indicated by <i>axis</i>, a dict of {key:(min,max)} is returned;
        otherwise, only the (min, max) tuple is returned

        @param axis   A member of AxisTypeEnum; may be an iterable of such members
        """
        return self.get_data_bounds(self._axis_arg_to_params(axis))

    def get_data_extents(self, parameter_name=None):
        """
        Returns the extents (dim_0,dim_1,...,dim_n) for the parameter(s) indicated by <i>parameter_name</i>

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        If more than one parameter is indicated by <i>parameter_name</i>, a dict of {key:(dim_0,dim_1,...,dim_n)} is returned;
        otherwise, only the (dim_0,dim_1,...,dim_n) tuple is returned

        @param parameter_name   A string parameter name; may be an iterable of such members
        """
        ret = {}
        for pn in self._parameter_name_arg_to_params(parameter_name):
            p = self._range_dictionary.get_context(pn)
            ret[pn] = (self._persistence_layer.num_timesteps(),) #TODO - get total extents from value type

        if len(ret) == 1:
            ret = ret.values()[0]
        return ret

    def get_data_extents_by_axis(self, axis=None):
        """
        Returns the extents (dim_0,dim_1,...,dim_n) for the coordinate parameter(s) indicated by <i>axis</i>

        If <i>axis</i> is None, all coordinate parameters are included

        If more than one parameter is indicated by <i>axis</i>, a dict of {key:(dim_0,dim_1,...,dim_n)} is returned;
        otherwise, only the (dim_0,dim_1,...,dim_n) tuple is returned

        @param axis   A member of AxisTypeEnum; may be an iterable of such members
        """
        return self.get_data_extents(self._axis_arg_to_params(axis))

    def get_data_size(self, parameter_name=None, slice_=None, in_bytes=False):
        """
        Returns the size of the <b>data values</b> for the parameter(s) indicated by <i>parameter_name</i>.
        ParameterContext and Coverage metadata is <b>NOT</b> included in the returned size.

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        If more than one parameter is indicated by <i>parameter_name</i>, the sum of the indicated parameters is returned

        If <i>slice_</i> is not None, it is applied to each parameter (after being run through utils.fix_slice) before
        calculation of size

        Sizes are calculated as:
            size = itemsize * total_extent_size

        where:
            itemsize == the per-item size based on the data type of the parameter
            total_extent_size == the total number of elements after slicing is applied (if applicable)

        Sizes are in MB unless <i>in_bytes</i> == True

        @param parameter_name   A string parameter name; may be an iterable of such members
        @param slice_   If not None, applied to each parameter before calculation of size
        @param in_bytes If True, returns the size in bytes; otherwise, returns the size in MB (default)
        """
        size = 0
        if parameter_name is None:
            for pn in self._range_dictionary.keys():
                size += self.get_data_size(pn, in_bytes=in_bytes)

        for pn in self._parameter_name_arg_to_params(parameter_name):
            p = self._range_dictionary.get_context(pn)
            te=(self.num_timesteps(),) #TODO - get extents from ParameterType
            dt = np.dtype(p.param_type.value_encoding)

            if slice_ is not None:
                slice_ = utils.fix_slice(slice_, te)
                a=np.empty(te, dtype=dt)[slice_]
                size += a.nbytes
            else:
                size += dt.itemsize * utils.prod(te)

        if not in_bytes:
            size *= 9.53674e-7

        return size

    def repair_temporal_geometry(self):
        if self.closed:
            raise IOError('I/O operation on closed file')

        if self.mode == 'r':
            raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        if isinstance(self, (ViewCoverage, ComplexCoverage)):
            raise TypeError('This action is only available for SimplexCoverages: {0}'.format(type(self)))

        ont = self.get_time_values()
        nt, inds = np.unique(ont, return_index=True)
        sl = (np.s_[inds.tolist()],)

        # Check if there are any duplicates
        num_dups = len(ont) - len(nt)
        if num_dups == 0:  # No duplicates!!
            log.info('The coverage does not have duplicate timesteps')
        else:
            log.debug('Coverage contains %s duplicate timesteps', num_dups)

        # Overwrite all the values, padding num_dups at the end with fill_value
        # for p in self.list_parameters():
        #     vc = self._range_dictionary[p].param_type._value_class
        #     if vc in ('ParameterFunctionValue', 'FunctionValue', 'ConstantValue', 'ConstantRangeValue'):
        #         continue
        #
        #     elif vc == 'SparseConstantValue':
        #         s = Span(None, max(inds))
        #         try:
        #             spns = self._range_value[p].content
        #             if not hasattr(spns, '__iter__'):
        #                 # spns is the fill_value for the parameter, no span assignment has happened yet, so just move on
        #                 continue
        #         except ValueError, ve:
        #             # Raised when there are no bricks (no timesteps) - this should never happen...
        #             continue
        #         nspns = []
        #         for spn in spns:
        #             if spn in s:
        #                 nspns.append(spn)
        #             elif spn.lower_bound in s:
        #                 nspns.append(Span(spn.lower_bound, None, offset=spn.offset, value=spn.value))
        #             elif spn.upper_bound is not None and spn.upper_bound in s:
        #                 nspns.append(Span(s.lower_bound, spn.upper_bound, offset=spn.offset, value=spn.value))
        #
        #         self._range_value[p]._storage[0] = nspns
        #
        #     else:
        #         if p == self.temporal_parameter_name:
        #             nv = nt
        #         else:
        #             nv = self.get_parameter_values(p)[sl]
        #
        #         self.set_parameter_values({p: np.append(nv, self._range_value[p].get_fill_array(num_dups))})
        #         del nv
        #
        # self.temporal_domain.shape.extents = (len(nt),) + self.temporal_domain.shape.extents[1:]
        # self._persistence_layer.shrink_domain(self.temporal_domain.shape.extents)
        #
        # self.flush()
        #
        # self.clear_value_cache()

    @property
    def persistence_guid(self):
        if isinstance(self._persistence_layer, InMemoryPersistenceLayer):
            return None
        else:
            return self._persistence_layer.guid

    @property
    def persistence_dir(self):
        if isinstance(self._persistence_layer, InMemoryPersistenceLayer):
            return None
        else:
            return self._persistence_layer.master_manager.root_dir

    def has_dirty_values(self):
        return self._persistence_layer.has_dirty_values()

    def get_dirty_values_async_result(self):
        if self.mode == 'r':
            log.warn('Coverage not open for writing: mode=%s', self.mode)
            from gevent.event import AsyncResult
            ret = AsyncResult()
            ret.set(True)
            return ret

        return self._persistence_layer.get_dirty_values_async_result()

    def flush_values(self):
        if self.mode == 'r':
            log.warn('Coverage not open for writing: mode=%s', self.mode)
            return

        return self._persistence_layer.flush_values()

    def flush(self):
        if self.mode == 'r':
            log.warn('Coverage not open for writing: mode=%s', self.mode)
            return

        self._persistence_layer.flush()

    def close(self, force=False, timeout=None):
        if not hasattr(self, '_closed'):
            # _closed is the first attribute added to the coverage object (in AbstractCoverage)
            # If it's not there, a TypeError has likely occurred while instantiating the coverage
            # nothing else exists and we can just return
            return
        if not self._closed:

            if self.mode != 'r':
                log.debug('Ensuring dirty values have been flushed...')
                if not force:
                    self.get_dirty_values_async_result().get(timeout=timeout)

            # If the _persistence_layer attribute is present, call it's close function
            if hasattr(self, '_persistence_layer'):
                self._persistence_layer.close(force=force, timeout=timeout) # Calls flush() on the persistence layer

            # Not much else to do here at this point....but can add other things down the road

        self._closed = True

    @property
    def closed(self):
        return self._closed

    @classmethod
    def copy(cls, cov_obj, *args):
        raise NotImplementedError('Coverages cannot yet be copied. You can load multiple \'independent\' copies of the same coverage, but be sure to save them to different names.')
#        if not isinstance(cov_obj, AbstractCoverage):
#            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))
#
#        # NTK:
#        # Args need to have 1-n (ParameterContext, DomainOfApplication, DomainOfApplication,) tuples
#        # Need to pull the parameter_dictionary, spatial_domain and temporal_domain from cov_obj (TODO: copies!!)
#        # DOA's and PC's used to copy data - TODO: Need way of reshaping PC's?
#        ccov = SimplexCoverage(name='', _range_dictionary=None, spatial_domain=None, temporal_domain=None)
#
#        return ccov

    @property
    def info(self):
        """
        Returns a detailed string representation of the coverage contents
        @returns    string of coverage contents
        """
        lst = []
        indent = ' '
        lst.append('ID: {0}'.format(self._id))
        lst.append('Name: {0}'.format(self.name))
        lst.append('Temporal Domain:\n{0}'.format(self.temporal_domain.__str__(indent*2) if self.temporal_domain is not None else 'None'))
        lst.append('Spatial Domain:\n{0}'.format(self.spatial_domain.__str__(indent*2) if self.spatial_domain is not None else 'None'))

        lst.append('Parameters:')
        for x in self._range_value:
            lst.append('{0}{1} {2}\n{3}'.format(indent*2,x,self._range_value[x].shape,self._range_dictionary.get_context(x).__str__(indent*4)))

        return '\n'.join(lst)

    def __str__(self):
        lst = []
        indent = ' '
        lst.append('ID: {0}'.format(self._id))
        lst.append('Name: {0}'.format(self.name))
        if self.temporal_domain is not None:
            lst.append('TemporalDomain: Shape=>{0} Axes=>{1}'.format(self.temporal_domain.shape.extents, self.temporal_domain.crs.axes))
        else:
            lst.append('TemporalDomain: None')
        if self.spatial_domain is not None:
            lst.append('SpatialDomain: Shape=>{0} Axes=>{1}'.format(self.spatial_domain.shape.extents, self.spatial_domain.crs.axes))
        else:
            lst.append('SpatialDomain: None')
        lst.append('Coordinate Parameters: {0}'.format(self.list_parameters(coords_only=True)))
        lst.append('Data Parameters: {0}'.format(self.list_parameters(coords_only=False, data_only=True)))

        return '\n'.join(lst)


class ViewCoverage(AbstractCoverage):
    # TODO: Implement ViewCoverage
    """
    References 1 AbstractCoverage and applies a Filter
    """
    def __init__(self, root_dir, persistence_guid, name=None, reference_coverage_location=None,
                 parameter_dictionary=None, mode=None):
        AbstractCoverage.__init__(self, mode=mode)

        try:
            # Make sure root_dir and persistence_guid are both not None and are strings
            if not isinstance(root_dir, basestring) or not isinstance(persistence_guid, basestring):
                raise TypeError('\'root_dir\' and \'persistence_guid\' must be instances of basestring')

            root_dir = root_dir if not root_dir.endswith(persistence_guid) else os.path.split(root_dir)[0]

            pth = os.path.join(root_dir, persistence_guid)

            def _doload(self):
                # Make sure the coverage directory exists
#cjb              if not os.path.exists(pth):
                if not is_persisted(root_dir, persistence_guid):
                    raise SystemError('Cannot find specified coverage: {0}'.format(pth))

                self._persistence_layer = PostgresPersistenceLayer(root_dir, persistence_guid, mode=self.mode)
                if self._persistence_layer.version != self.version:
                    raise IOError('Coverage Model Version Mismatch: %s != %s' %(self.version, self._persistence_layer.version))

                self.name = self._persistence_layer.name

                try:
                    self.reference_coverage = AbstractCoverage.load(self._persistence_layer.rcov_loc, mode='r')
                except IOError, ex:
                    if 'unable to create file' in ex.message:
                        raise IOError('Unable to open reference coverage: \'{0}\''.format(self._persistence_layer.rcov_loc))
                    if 'unable to open file' in ex.message:
                        raise IOError('Unable to open reference coverage: \'{0}\''.format(self._persistence_layer.rcov_loc))
                    raise

                self.__setup(self._persistence_layer.param_dict)

#cjb            if os.path.exists(pth):
            if is_persisted(root_dir, persistence_guid):
            # if reference_coverage_location is None or name is None or parameter_dictionary is None:
                # This appears to be a load
                _doload(self)
            else:
                # This appears to be a new coverage
                # Make sure name and parameter_dictionary are not None
                if reference_coverage_location is None or name is None:
                    raise SystemError('\'reference_coverage_location\' and \'name\' cannot be None')

                # If the coverage directory exists, load it instead!!
                if is_persisted(root_dir, persistence_guid):
#cjb                if os.path.exists(pth):
                    log.warn('The specified coverage already exists - performing load of \'{0}\''.format(pth))
                    _doload(self)
                    return

                if not isinstance(reference_coverage_location, basestring):
                    raise TypeError('\'reference_coverage_location\' must be of type basestring')

                if not os.path.exists(reference_coverage_location):
                    raise IOError('\'reference_coverage_location\' cannot be found: \'{0}\''.format(reference_coverage_location))

                # We've checked everything we can - this is a new coverage!!!

                # Must be in 'a' for a new coverage
                self.mode = 'a'

                if not isinstance(name, basestring):
                    raise TypeError('\'name\' must be of type basestring')
                self.name = name

                # Open the reference coverage - ALWAYS in read-only mode (default)
                try:
                    self.reference_coverage = AbstractCoverage.load(reference_coverage_location)
                except IOError, ex:
                    if 'unable to create file' in ex.message:
                        raise IOError('Unable to open reference coverage: \'{0}\''.format(self._persistence_layer.rcov_loc))
                    raise

                parameter_dictionary = self.__normalize_param_dict(parameter_dictionary)

                self._persistence_layer = PostgresPersistenceLayer(root_dir,
                                                                 persistence_guid,
                                                                 name=self.name,
                                                                 param_dict=parameter_dictionary,
                                                                 mode=self.mode,
                                                                 rcov_loc=reference_coverage_location,
                                                                 coverage_type='view',
                                                                 version=self.version)

                self.__setup(parameter_dictionary)

        except:
            self._closed = True
            raise

        # Avoid duplicating cache entries by assigning the ViewCoverage's _value_cache to that of the reference_coverage
        self._value_cache = self.reference_coverage._value_cache

    def close(self, force=False, timeout=None):
        if not hasattr(self, '_closed'):
            # _closed is the first attribute added to the coverage object (in AbstractCoverage)
            # If it's not there, a TypeError has likely occurred while instantiating the coverage
            # nothing else exists and we can just return
            return
        if not self._closed:
            self.reference_coverage.close(force, timeout)

        AbstractCoverage.close(self, force, timeout)

    def replace_reference_coverage(self, path=None, use_current_param_dict=True, parameter_dictionary=None):
        #if self.mode == 'r':
        #    raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        if path is not None:
            ncov = AbstractCoverage.load(path)
            # Loading the coverage worked - go ahead and replace things!
            self.reference_coverage = ncov
        else:
            path = self.reference_coverage.persistence_dir

        self._range_dictionary = ParameterDictionary()
        self._range_value = RangeValues()
        self._persistence_layer.rcov_loc = path

        self.clear_value_cache()

        if use_current_param_dict:
            parameter_dictionary = self._persistence_layer.param_dict
        else:
            parameter_dictionary = self.__normalize_param_dict(parameter_dictionary)

        self._persistence_layer.param_dict = parameter_dictionary

        self.__setup(parameter_dictionary)

        self._persistence_layer.flush()

    def __setup(self, parameter_dictionary):
        for p in parameter_dictionary:
            if p in self.reference_coverage._range_dictionary:
                # Add the context from the reference coverage
                self._range_dictionary.add_context(self.reference_coverage._range_dictionary.get_context(p))
                # Add the value class from the reference coverage
                self._range_value[p] = self.reference_coverage._range_value[p]
            else:
                log.info('Parameter \'%s\' skipped; not in \'reference_coverage\'', p)

        self.temporal_domain = self.reference_coverage.temporal_domain
        self.spatial_domain = self.reference_coverage.spatial_domain

        self._head_coverage_path = self.reference_coverage.head_coverage_path

    def __normalize_param_dict(self, parameter_dictionary):
        if parameter_dictionary is None:
            parameter_dictionary = self.reference_coverage.parameter_dictionary.keys()
        elif hasattr(parameter_dictionary, 'keys'):
            parameter_dictionary = parameter_dictionary.keys()
        elif not hasattr(parameter_dictionary, '__iter__'):
            parameter_dictionary = [parameter_dictionary,]

        return parameter_dictionary

    def set_time_values(self, value, tdoa=None):
        raise TypeError('Cannot set time values against a ViewCoverage')

    def set_parameter_values(self, param_name, value, tdoa=None, sdoa=None):
        raise TypeError('Cannot set parameter values against a ViewCoverage')

    def get_parameter_values(self, param_names, time_segment=None, time=None, sort_parameter=None, stride_length=None,
                             fill_empty_params=False, return_value=None):
        return self.reference_coverage.get_parameter_values(param_names, time_segment, time, sort_parameter, stride_length, fill_empty_params, return_value)

    def get_value_dictionary(self, *args, **kwargs):

        return self.reference_coverage.get_value_dictionary(*args, **kwargs)

    def is_empty(self):
        return self.reference_coverage.is_empty()

    def has_parameter_data(self):
        return self.reference_coverage.has_parameter_data()

    def num_timesteps(self):
        return self.reference_coverage.num_timesteps()

    def get_data_bounds(self, parameter_name=None):
        return self.reference_coverage.get_data_bounds(parameter_name)

    def get_data_extents(self, parameter_name=None):
        return self.reference_coverage.get_data_extents(parameter_name)

    def get_data_extents_by_axis(self, axis=None):
        return self.reference_coverage.get_data_extents_by_axis(axis)

    def get_data_size(self, parameter_name=None, slice_=None, in_bytes=False):
        return self.reference_coverage.get_data_size(parameter_name, in_bytes=in_bytes)


from coverage_model.basic_types import BaseEnum
class ComplexCoverageType(BaseEnum):

    # Complex coverage that combines parameters from multiple coverages
    # must have coincident temporal and spatial geometry
    PARAMETRIC_STRICT = 'PARAMETRIC_STRICT'

    # Complex coverage that combines multiple coverages temporally
    # may have disparate temporal geometry, but coincident spatial geometry
    TEMPORAL_INTERLEAVED = 'TEMPORAL_INTERLEAVED'

    # Complex coverage that aggregates coverages along their temporal axis
    TEMPORAL_AGGREGATION = 'TEMPORAL_AGGREGATION'

    # Complex coverage that broadcasts other coverages onto the temporal axis of the "primary" covreage
    TEMPORAL_BROADCAST = 'TEMPORAL_BROADCAST'

    # Placeholder for spatial complex - not sure what this will look like yet...
    SPATIAL_JOIN = 'SPATIAL_JOIN'

    # TODO: Catchy description
    TIMESERIES = 'TIMESERIES'


class ComplexCoverage(AbstractCoverage):
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
        AbstractCoverage.__init__(self, mode=mode)

        try:
            # Make sure root_dir and persistence_guid are both not None and are strings
            if not isinstance(root_dir, basestring) or not isinstance(persistence_guid, basestring):
                raise TypeError('\'root_dir\' and \'persistence_guid\' must be instances of basestring')

            root_dir = root_dir if not root_dir.endswith(persistence_guid) else os.path.split(root_dir)[0]


            pth = os.path.join(root_dir, persistence_guid)
            if is_persisted(root_dir, persistence_guid):
#cjb            if os.path.exists(pth):
                self._existing_coverage(root_dir, persistence_guid)
            else:
                self._new_coverage(root_dir, persistence_guid, name, reference_coverage_locs, parameter_dictionary, complex_type)

            self._dobuild()

        except:
            self._closed = True
            raise

    def _existing_coverage(self, root_dir, persistence_guid):
        # If the path does exist, initialize the simple persistence layer
        pth = os.path.join(root_dir, persistence_guid)
        if not is_persisted(root_dir, persistence_guid):
#cjb        if not os.path.exists(pth):
            raise SystemError('Cannot find specified coverage: {0}'.format(pth))
        self._persistence_layer = PostgresPersistenceLayer(root_dir, persistence_guid, mode=self.mode)
        if self._persistence_layer.version != self.version:
            raise IOError('Coverage Model Version Mismatch: %s != %s' %(self.version, self._persistence_layer.version))
        self.name = self._persistence_layer.name
        self.mode = self.mode
        self._reference_covs = collections.OrderedDict()

    def _new_coverage(self, root_dir, persistence_guid, name, reference_coverage_locs, parameter_dictionary, complex_type):
        # Coverage doesn't exist, make a new one
        if reference_coverage_locs is None or name is None:
            raise SystemError('\'reference_coverages\' and \'name\' cannot be None')
        if not isinstance(name, basestring):
            raise TypeError('\'name\' must be of type basestring')
        self.name = name
        if parameter_dictionary is None:
            parameter_dictionary = ParameterDictionary()
        elif complex_type != ComplexCoverageType.TIMESERIES:
            from coverage_model import ParameterFunctionType
            for pn, pc in parameter_dictionary.iteritems():
                if not isinstance(pc[1].param_type, ParameterFunctionType):
                    log.warn('Parameters stored in a ComplexCoverage must be ParameterFunctionType parameters: discarding \'%s\'', pn)
                    parameter_dictionary._map.pop(pn)

        # Must be in 'a' for a new coverage
        self.mode = 'a'

        self._reference_covs = collections.OrderedDict()

        if not hasattr(reference_coverage_locs, '__iter__'):
            reference_coverage_locs = [reference_coverage_locs]

        self._persistence_layer = PostgresPersistenceLayer(root_dir,
                                                         persistence_guid,
                                                         name=self.name,
                                                         mode=self.mode,
                                                         param_dict=parameter_dictionary,
                                                         rcov_locs=reference_coverage_locs,
                                                         complex_type=complex_type,
                                                         coverage_type='complex',
                                                         version=self.version)


    def _dobuild(self):
        complex_type = self._persistence_layer.complex_type
        reference_coverages = self._persistence_layer.rcov_locs
        parameter_dictionary = self._persistence_layer.param_dict


        if complex_type == ComplexCoverageType.PARAMETRIC_STRICT:
            # PARAMETRIC_STRICT - combine parameters from multiple coverages - MUST HAVE IDENTICAL TIME VALUES
            self._build_parametric(reference_coverages, parameter_dictionary)
        elif complex_type == ComplexCoverageType.TEMPORAL_INTERLEAVED:
            # TEMPORAL_INTERLEAVED - combine parameters from multiple coverages - may have differing time values
            self._build_temporal_interleaved(reference_coverages, parameter_dictionary)
        elif complex_type == ComplexCoverageType.TEMPORAL_AGGREGATION:
            # TEMPORAL_AGGREGATION - combine coverages temporally
            self._build_temporal_aggregation(reference_coverages, parameter_dictionary)
        elif complex_type == ComplexCoverageType.TEMPORAL_BROADCAST:
            # TEMPORAL_BROADCAST - combine coverages temporally, broadcasting non-primary coverages
            self._build_temporal_broadcast(reference_coverages, parameter_dictionary)
        elif complex_type == ComplexCoverageType.SPATIAL_JOIN:
            # Complex spatial - combine coverages across a higher-order topology
            raise NotImplementedError('Not yet implemented')
        elif complex_type == ComplexCoverageType.TIMESERIES:
            self._build_timeseries(reference_coverages, parameter_dictionary)

    def close(self, force=False, timeout=None):
        if not hasattr(self, '_closed'):
            # _closed is the first attribute added to the coverage object (in AbstractCoverage)
            # If it's not there, a TypeError has likely occurred while instantiating the coverage
            # nothing else exists and we can just return
            return
        if not self._closed:
            for cov_pth, cov in self._reference_covs.iteritems():
                log.info('Closing reference coverage \'%s\'', cov.name if hasattr(cov,'name') else 'unnamed')
                cov.close(force, timeout)


        AbstractCoverage.close(self, force, timeout)

    def append_parameter(self, parameter_context):
        if not isinstance(parameter_context, ParameterContext):
            raise TypeError('\'parameter_context\' must be an instance of ParameterContext, not {0}'.format(type(parameter_context)))
        from coverage_model import ParameterFunctionType
        if not isinstance(parameter_context.param_type, ParameterFunctionType):
            raise ValueError('Parameters stored in a ComplexCoverage must be ParameterFunctionType parameters: cannot append parameter \'{0}\''.format(parameter_context.name))

        AbstractCoverage.append_parameter(self, parameter_context)

    def append_reference_coverage(self, path):
        ncov = AbstractCoverage.load(path)
        ncov.close()

        # Loading the coverage worked - proceed...
        # Get the current set of reference coverages
        if path in self._persistence_layer.rcov_locs:
            # Already there, note it and just return
            log.info('Coverage already referenced: \'%s\'', path)
            return

        self._persistence_layer.rcov_locs.append(path)

        # Reset things to ensure we don't munge everything
        self._reference_covs = collections.OrderedDict()
        self._range_dictionary = ParameterDictionary()
        self._range_value = RangeValues()
        self.temporal_domain = GridDomain(GridShape('temporal',[0]), CRS.standard_temporal(), MutabilityEnum.EXTENSIBLE)
        self.spatial_domain = None
        self._head_coverage_path = None

        # Then, rebuild this badboy!
        self._dobuild()

    def interval_map(self):
        '''
        Builds a reference structure and returns the bounds and the associated reference coverages
        note: only works for 1-d right now
        '''
        interval_map = []
        for scov in self._reference_covs.itervalues():
            interval = scov.get_data_bounds(scov.temporal_parameter_name)
            interval = Interval(interval[0], interval[1], None, None)
            interval_map.append((interval, scov))
        self._interval_qsort(interval_map)
        return interval_map

    @classmethod
    def _interval_swap(cls, arr, x0, x1):
        if x0 != x1:
            t = arr[x0]
            arr[x0] = arr[x1]
            arr[x1] = t
    @classmethod
    def _interval_pivot(cls, arr, left, right, pivot):
        val = arr[pivot][0]
        cls._interval_swap(arr, pivot, right)
        store_index = left
        for i in xrange(left, right):
            if arr[i][0] < val:
                cls._interval_swap(arr, i, store_index)
                store_index += 1
        cls._interval_swap(arr, store_index, right)
        return store_index

    @classmethod
    def _interval_qsort(cls, arr, left=None, right=None):
        '''
        Quicksort for the interval map
        '''
        if left is None:
            left = 0
        if right is None:
            right = len(arr) - 1
        if left < right:
            pivot = (right - left) / 2 + left
            pivot = cls._interval_pivot(arr, left, right, pivot)
            cls._interval_qsort(arr, left, pivot-1)
            cls._interval_qsort(arr, pivot+1, right)
    
    def get_parameter_values(self, param_names, time_segment=None, time=None, sort_parameter=None, stride_length=None, return_value=None):
        '''
        Obtain the value set for a given parameter over a specified domain
        '''
        complex_type = self._persistence_layer.complex_type
        if complex_type == ComplexCoverageType.TIMESERIES:
            raise TypeError("A Timeseries ComplexCoverage doesn't support "
                            "get_parameter_values, please use "
                            "get_value_dictionary")
        return AbstractCoverage.get_parameter_values(self, param_names, time_segment, time, sort_parameter, stride_length=stride_length, return_value=return_value)

    # def get_value_dictionary(self, param_list=None, temporal_slice=None, domain_slice=None):
    #     if temporal_slice and domain_slice:
    #         raise ValueError("Can not specify both a temporal_slice and domain_slice")
    #     if temporal_slice:
    #         return self._get_value_dictionary_temporal(param_list, temporal_slice)
    #
    #     else:
    #         return self._get_value_dictionary_domain(param_list, domain_slice)
    #

    # def _get_value_dictionary_domain(self, param_list=None, domain_slice=None, unsorted=False, raw=False):
    #
    #     if domain_slice is None:
    #         slice_ = slice(None, None)
    #
    #     elif isinstance(domain_slice, slice):
    #         slice_ = domain_slice
    #
    #     elif isinstance(domain_slice, (tuple,list)):
    #         slice_ = slice(*domain_slice)
    #     else:
    #         raise ValueError("Unspported domain slice: %s" % domain_slice)
    #
    #     # Make sure the axis is in the value dictionary
    #     # But remember to remove it on the way out
    #     no_time = param_list and self.temporal_parameter_name not in param_list
    #     if no_time:
    #         param_list.append(self.temporal_parameter_name)
    #
    #     value_interval = Interval(slice_.start, slice_.stop)
    #     interval_map = self.interval_map()
    #
    #     value_dict = {}
    #     total_timesteps = 0
    #     overlaps = False
    #     for i in xrange(len(interval_map)):
    #
    #         interval, cov = interval_map[i]
    #         if i > 0 and interval_map[i-1][0].intersects(interval):
    #             overlaps = True
    #
    #         d = Interval(total_timesteps, total_timesteps+cov.num_timesteps-1)
    #         if d.intersects(value_interval):
    #             if slice_.start is None:
    #                 start = 0
    #             else:
    #                 start = max(0, slice_.start - total_timesteps)
    #             if slice_.stop is None:
    #                 stop = None
    #             else:
    #                 stop = slice_.stop - total_timesteps
    #             s = slice(start, stop, slice_.step)
    #             if start == stop:
    #                 continue
    #
    #             self._aggregate_value_dict(value_dict, cov, param_list, s)
    #         total_timesteps += cov.num_timesteps
    #
    #     if not value_dict:
    #         return value_dict
    #     if overlaps and not unsorted:
    #         self._value_dict_qsort(value_dict, self.temporal_parameter_name)
    #     if not raw:
    #         value_dict = self._value_dict_unique(value_dict, self.temporal_parameter_name)
    #     # Remove the axis from the value_dict if it wasn't asked for
    #     if no_time:
    #         del value_dict[self.temporal_parameter_name]
    #     return value_dict




    def _get_value_dictionary_temporal(self, param_list=None, temporal_slice=None, unsorted=False, raw=False):
        '''
        Retrieves a value_dictionary which is an aggregation of the child
        coverage cross-sections of the temporal domain (temporal_slice)

        temporal_slice - A slice or tuple of start and stop times
        '''
        # (None, None) is (-inf, inf)
        if temporal_slice is None:
            temporal_slice = (None, None)

        param_list = param_list or []

        # Handle two simplistic input types, we may need better support here
        stride = None
        if isinstance(temporal_slice, slice):
            value_interval = Interval(temporal_slice.start, temporal_slice.stop)
            stride = temporal_slice.step
        elif isinstance(temporal_slice, tuple):
            if len(temporal_slice) == 2:
                start, stop = temporal_slice
            elif len(temporal_slice) == 3:
                start, stop, stride = temporal_slice
            else:
                raise ValueError("Improper domain value: %s" % temporal_slice)
            value_interval = Interval(start, stop)
        else:
            raise ValueError("Improper domain value: %s" % temporal_slice)


        # Make sure the axis is in the value dictionary
        # But remember to remove it on the way out
        no_time = param_list and self.temporal_parameter_name not in param_list
        if no_time:
            param_list.append(self.temporal_parameter_name)

        interval_map = self.interval_map()
        value_dict = {}
        overlaps = False
        for i in xrange(len(interval_map)):
            interval, cov = interval_map[i]
            if i > 0 and interval_map[i-1][0].intersects(interval):
                overlaps = True
            if interval.intersects(value_interval):
                slice_ = cov._get_me_a_slice((value_interval.x0, value_interval.x1), None)
                self._aggregate_value_dict(value_dict, cov, param_list, slice_)

        if not value_dict:
            for p in param_list or self.list_parameters():
                dtype = self._range_dictionary[p].param_type.value_encoding
                value_dict[p] = np.array([], dtype=dtype)
            return value_dict

        if overlaps and not unsorted:
            self._value_dict_qsort(value_dict, self.temporal_parameter_name)
        if not raw:
            value_dict = self._value_dict_unique(value_dict, self.temporal_parameter_name)
        if stride:
            self._stride_value_dict(value_dict, stride)
        # Remove the axis from the value_dict if it wasn't asked for
        if no_time:
            del value_dict[self.temporal_parameter_name]
        return value_dict

    def _aggregate_value_dict(self, value_dict, cov, param_list, slice_):
        '''
        Appends the coverage's answer cross-section to the value dictionary
        '''
        param_list = param_list or self.list_parameters()
        tname = self.temporal_parameter_name
        # Potential issue here if the s-cov doesn't use the same temporal 
        # parameter name
        if slice_.start == slice_.stop and slice_.start is not None:
            dtype = self._range_dictionary[tname].param_type.value_encoding
            t_values = np.array([], dtype=dtype)
        else:
            t_values = cov._range_value[tname][slice_]
        if tname not in value_dict:
            value_dict[tname] = t_values
        else:
            value_dict[tname] = np.concatenate([value_dict[tname], t_values])
        for p in param_list:
            if p == tname:
                continue
            if p not in cov.list_parameters():
                value_set = np.ones(len(t_values))
                fv = self.get_parameter_context(p).fill_value
                value_set *= fv
            elif p not in value_dict:
                if slice_.start == slice_.stop and slice_.start is not None:
                    dtype = self._range_dictionary[p].param_type.value_encoding
                    value_set = np.array([], dtype=dtype)
                else:
                    value_set = cov._range_value[p][slice_]
            else:
                if slice_.start == slice_.stop and slice_.start is not None:
                    dtype = self._range_dictionary[p].param_type.value_encoding
                    value_set = np.array([], dtype=dtype)
                else:
                    value_set = cov._range_value[p][slice_]
                value_set = np.concatenate([value_dict[p], value_set])
            value_dict[p] = value_set

    @classmethod
    def _stride_value_dict(self, value_dict, stride):
        '''
        Strides a value dictionary
        '''
        s = slice(None,None,stride)
        for k,v in value_dict.iteritems():
            value_dict[k] = v[s]
        

    def insert_value_set(self, value_dictionary):
        '''
        Inserts a set of values from the value dictionary into the coverage
        system
        '''
        complex_type = self._persistence_layer.complex_type
        if complex_type != ComplexCoverageType.TIMESERIES:
            raise TypeError("Complex Coverage must be a time series to use this method")

        if self.temporal_parameter_name not in value_dictionary:
            raise ValueError("Temporal domain must be specified in value set")

        shape = value_dictionary[self.temporal_parameter_name].shape[0]
        for k,v in value_dictionary.iteritems():
            if v.shape[0] != shape:
                raise ValueError("Improperly shaped value dictionary")

        self._value_dict_qsort(value_dictionary, self.temporal_parameter_name)


        self.append_value_set(value_dictionary)

        return None

    def append_value_set(self, value_dictionary):
        # Get the time value set
        time_array = value_dictionary[self.temporal_parameter_name]
        t0, t1 = time_array[0], time_array[-1]
        value_interval = Interval(t0, t1, None, None)
        # Get the intervals
        interval_map = self.interval_map()

        full_interval = None
        intersecting_insert = False
        for interval, cov in interval_map:
            if full_interval is None:
                full_interval = interval
            else:
                full_interval = full_interval.union(interval)

            # If any new value sets intersect an existing coverage's interval,
            # create a new simplex coverage
            if interval.intersects(value_interval):
                intersecting_insert = True

        if intersecting_insert:
            # Find a disjoint coverage with the least union area
            min_delta = None
            min_interval = None
            min_coverage = None
            for interval,cov in interval_map:
                delta = value_interval.x0 - interval.x1
                if delta > 0:
                    if min_delta is None or delta < min_delta:
                        min_delta = delta
                        min_interval = interval
                        min_coverage = cov
            if min_coverage is not None:
                self._append_to_coverage(min_coverage, value_dictionary)
            else:
                cov = self.new_simplex()
                self._append_to_coverage(cov, value_dictionary)
                return


        if full_interval is None:
            # Empty coverage system
            cov = self.new_simplex()
            self._append_to_coverage(cov, value_dictionary)
            return

        # In an ideal case and simple input the value sets arrive in a
        # monotonically increasing order, the solution is to append the value
        # set to the last coverage in the interval_map. 
        if value_interval.x0 > full_interval.x1:
            interval, cov = interval_map[-1]
            self._append_to_coverage(cov, value_dictionary)
        # The value interval is less than the coverage system's interval
        elif value_interval.x1 < full_interval.x0:
            cov = self.new_simplex()
            self._append_to_coverage(cov, value_dictionary)
            return
        else:
            min_dx = 0
            min_interval, min_cov = interval_map[0]
            for interval, cov in interval_map:
                dx = value_interval.x0 - interval.x1
                if dx > 0 and dx < min_dx:
                    min_dx = dx
                    min_interval, min_cov = interval, cov



    def _append_to_coverage(self, cov, value_dictionary):
        cov = AbstractCoverage.load(cov.persistence_dir, mode='r+')
        cov.set_parameter_values(value_dictionary)
        cov.close()
        self.refresh()

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
        


    @classmethod
    def _value_dict_swap(cls, value_dict, x0, x1):
        '''
        Value dictionary array swap
        '''
        if x0 != x1:
            for name,arr in value_dict.iteritems():
                t = arr[x0]
                arr[x0] = arr[x1]
                arr[x1] = t

    @classmethod
    def _value_dict_pivot(cls, value_dict, axis, left, right, pivot):
        '''
        Pivot algorithm, part of quicksort
        '''
        axis_arr = value_dict[axis]
        idx_arr = value_dict['__idx__']
        val = axis_arr[pivot]
        cls._value_dict_swap(value_dict, pivot, right)
        store_index = left
        for i in xrange(left, right):
            if axis_arr[i] < val:
                cls._value_dict_swap(value_dict, i, store_index)
                store_index += 1
            # This part is critical to maintaining the precedence :)
            if axis_arr[i] == val and idx_arr[i] < idx_arr[right]:
                cls._value_dict_swap(value_dict, i, store_index)
                store_index += 1
                
        cls._value_dict_swap(value_dict, store_index, right)
        return store_index

    @classmethod
    def _value_dict_qsort(cls, value_dict, axis, left=None, right=None):
        '''
        Quicksort, value dictionary edition
        modifications are in-place for a stable search
        '''
        top_call = left is None and right is None
        if top_call:
            value_dict['__idx__'] = np.arange(len(value_dict[axis]))
        if left is None:
            left = 0
        if right is None:
            right = len(value_dict[axis]) - 1
        if left < right:
            pivot = (right - left) / 2 + left
            pivot = cls._value_dict_pivot(value_dict, axis, left, right, pivot)
            cls._value_dict_qsort(value_dict, axis, left, pivot-1)
            cls._value_dict_qsort(value_dict, axis, pivot+1, right)
        if top_call:
            del value_dict['__idx__']

    @classmethod
    def _value_dict_unique(cls, value_dict, axis):
        '''
        A naive unique copy algorithm

        Notes:
        - Last unique axis value has precedence
        '''
        tarray = value_dict[axis]
        truth_array = np.ones(tarray.shape, dtype=np.bool)
        for i in xrange(1, len(tarray)):
            if tarray[i-1] == tarray[i]:
                truth_array[i-1] = False

        vd_copy = {}
        for k,v in value_dict.iteritems():
            vd_copy[k] = v[truth_array]
        return vd_copy

    def _verify_rcovs(self, rcovs):
        for cpth in rcovs:
            if not os.path.exists(cpth):
                log.warn('Cannot find coverage \'%s\'; ignoring', cpth)
                continue

            if cpth in self._reference_covs:
                log.info('Coverage \'%s\' already present; ignoring', cpth)
                continue

            try:
                cov = AbstractCoverage.load(cpth)
            except Exception as ex:
                log.warn('Exception loading coverage \'%s\'; ignoring: ', cpth, ex.message)
                continue

            if cov.temporal_parameter_name is None:
                log.warn('Coverage \'%s\' does not have a temporal_parameter; ignoring', cpth)
                continue

            yield cpth, cov

    def _build_parametric(self, rcovs, parameter_dictionary):
        ntimes = None
        times = None

        for cpth, cov in self._verify_rcovs(rcovs):

            if ntimes is None:
                times = cov.get_time_values()
                self.temporal_domain = cov.temporal_domain
                self.spatial_domain = cov.spatial_domain
                self._reference_covs[cpth] = cov
            else:
                if np.allclose(times, cov.get_time_values()):
                    self._reference_covs[cpth] = cov
                else:
                    log.warn('Coverage timestamps do not match; cannot include: %s', cpth)
                    continue

            # Add parameters from the coverage if not already present
            for p in cov.list_parameters():
                if p not in parameter_dictionary:
                    if p not in self._range_dictionary:
                        # Add the context from the reference coverage
                        self._range_dictionary.add_context(self._reference_covs[cpth]._range_dictionary.get_context(p))
                        # Add the value class from the reference coverage
                        self._range_value[p] = self._reference_covs[cpth]._range_value[p]
                    else:
                        log.info('Parameter \'%s\' from coverage \'%s\' already present, skipping...', p, cpth)

        # Add the parameters for this coverage
        for pc in parameter_dictionary.itervalues():
            self.append_parameter(pc[1])

        self._head_coverage_path = None

    def _build_temporal_broadcast(self, rcovs, parameter_dictionary):
        primary_times = None
        for cpth, cov in self._verify_rcovs(rcovs):
            if primary_times is None:
                # The primary coverage provides the temporal (and spatial) topology & geometry
                primary_times = cov.get_time_values()
                self.temporal_domain = cov.temporal_domain
                self.spatial_domain = cov.spatial_domain
                self._reference_covs[cpth] = cov

                # Add the parameters from this coverage
                for p in cov.list_parameters():
                    if p not in parameter_dictionary:
                        # Add the context from the reference coverage
                        self._range_dictionary.add_context(cov._range_dictionary.get_context(p))
                        # Add teh value class from the reference coverage
                        self._range_value[p] = cov._range_value[p]

                # Add the parameters for this coverage
                for pc in parameter_dictionary.itervalues():
                    self.append_parameter(pc[1])

                self._head_coverage_path = self._reference_covs[cpth].head_coverage_path
            else:
                # Add parameters from this coverage that are not already present
                covpd = cov.parameter_dictionary  # Provides a copy
                params = []
                for p, pc in covpd.iteritems():
                    if p not in self._range_dictionary:
                        pc = pc[1]
                        self._assign_domain(pc)
                        self._range_dictionary.add_context(pc)
                        # Add the sparse value class
                        from coverage_model.parameter_types import SparseConstantType
                        ppt = self._range_dictionary.get_context(p).param_type
                        self._range_value[p] = get_value_class(
                            SparseConstantType(value_encoding=ppt.value_encoding,
                                               fill_value=ppt.fill_value),
                            DomainSet(self.temporal_domain))
                        params.append(p)
                    else:
                        log.info('Parameter \'%s\' from coverage \'%s\' already present, skipping...', p, cpth)

                # Sort out the spans
                spns = {p: [] for p in params}

                def _add_span(start, end, t=None):
                    for p in params:
                        if t is not None:
                            v = cov._range_value[p][t]
                        else:
                            v = cov._range_value[p].fill_value

                        spns[p].append(Span(start, end, value=v))

                cov_times = cov.get_time_values()
                end = None
                lend = None
                for t in xrange(len(cov_times) - 1):
                    if cov_times[t] > primary_times[-1]:  # We've gone past the end of the primary cov, bail
                        break

                    start = utils.find_nearest_index(primary_times, cov_times[t])
                    end = utils.find_nearest_index(primary_times, cov_times[t + 1])
                    if end == start:
                        end += 1

                    if start != 0 and t == 0:
                        _add_span(0, start)
                    elif lend is not None and start != lend:
                        _add_span(lend, start)

                    _add_span(start, end, t)
                    lend = end

                if cov_times[-1] > primary_times[-1]:
                    for p in params:
                        spns[p][-1].upper_bound = len(primary_times)
                else:
                    start = end if end is not None else utils.find_nearest_index(primary_times, cov_times[-1])
                    end = len(primary_times)
                    _add_span(start, end, -1)

                for p, s in spns.iteritems():
                    # Set the spans against the SparseConstantValue manually
                    self._range_value[p]._storage[0] = s
                    # Direct assignment of the spans bypasses min/max updating, perform manually
                    for sp in s:
                        self._range_value[p]._update_min_max(sp.value)

    def _build_temporal_interleaved(self, rcovs, parameter_dictionary):

        # First need to iterate the time arrays and merge them while maintaining a link to the "supplying" coverage...
        # THIS IS EXPENSIVE
        import itertools

        merged = None
        dt = None
        for cpth, cov in self._verify_rcovs(rcovs):
            if merged is None:
                dt = [('v', cov.get_parameter_context('time').param_type.value_encoding), ('k', object)]
                merged = np.empty(0, dtype=dt)

            a = np.array([x for x in itertools.izip_longest(cov.get_time_values(), [], fillvalue=cpth)], dtype=dt)
            merged = np.append(merged, a)
            self._reference_covs[cpth] = cov

            # Add parameters from the coverage if not already present
            covpd = cov.parameter_dictionary # Provides a copy
            for p, pc in covpd.iteritems():
                if p not in parameter_dictionary:
                    if p not in self._range_dictionary:
                        # Add the context from the reference coverage
                        pc = pc[1]  # pc is a tuple (ordinal, ParameterContext)
                        self._assign_domain(pc)
                        self._range_dictionary.add_context(pc)
                        # Add the sparse value class
                        from coverage_model.parameter_types import SparseConstantType
                        ppt=self._range_dictionary.get_context(p).param_type
                        self._range_value[p] = get_value_class(
                            SparseConstantType(value_encoding=ppt.value_encoding,
                                               fill_value=ppt.fill_value),
                            DomainSet(self.temporal_domain))
                    else:
                        log.info('Parameter \'%s\' from coverage \'%s\' already present, skipping...', p, cpth)

        # Sort the merged temporal array by time value...
        merged.sort()

        # Now we can determine the spans
        s = merged[0]['k']
        rcov_domain_spans = []
        curr = []
        counter = {}
        key = None
        for i, v in enumerate(merged):
            key = v['k']
            if key not in counter:
                counter[key] = 0

            if key == s:
                curr.append(i)
            else:
                low = min(curr)
                high = max(curr) + 1
                rcov_domain_spans.append(Span(low, high, offset=counter[s] - low, value=s))
                counter[s] += high-low
                curr = []
                s = key
                curr.append(i)

        # Don't forget the last one!
        low = min(curr)
        high = max(curr) + 1
        rcov_domain_spans.append(Span(low, high, offset=counter[key] - low, value=s))

        self.rcov_domain_spans = rcov_domain_spans

        # Add data for all spans
        for s in self.rcov_domain_spans:
            cov = self._reference_covs[s.value]
            for p in self.list_parameters():
                if p in cov._range_dictionary:
                    self._range_value[p][s] = cov._range_value[p]
                else:
                    self._range_value[p][s] = self._range_dictionary.get_context(p).fill_value

        self._head_coverage_path = None

    # def _build_temporal_aggregation(self, rcovs, parameter_dictionary):
    #     # First open all the coverages and sort them temporally
    #     time_bounds = []
    #     for cpth, cov in self._verify_rcovs(rcovs):
    #
    #         # Get the time bounds for the coverage
    #         if cov.num_timesteps == 0:
    #             tbnds = (None, None)
    #         else:
    #             tbnds = cov.get_data_bounds(cov.temporal_parameter_name)
    #             if tbnds[0] == tbnds[1]:
    #                 tbnds = (tbnds[0], None)
    #
    #         spn = Span(tbnds[0], tbnds[1], value=cpth)
    #
    #         if spn in time_bounds:
    #             log.warn('Coverage with time bounds \'%s\' already present; ignoring', spn.tuplize())
    #             continue
    #
    #         time_bounds.append(spn)
    #
    #         self._reference_covs[cpth] = cov
    #
    #         # Add parameters from the coverage if not already present
    #         covpd = cov.parameter_dictionary # Provides a copy
    #         for p, pc in covpd.iteritems():
    #             if p not in parameter_dictionary:
    #                 if p not in self._range_dictionary:
    #                     # Add the context from the reference coverage
    #                     pc = pc[1]  # pc is a tuple (ordinal, ParameterContext)
    #                     self._assign_domain(pc)
    #                     self._range_dictionary.add_context(pc)
    #                     # Add the sparse value class
    #                     from coverage_model.parameter_types import SparseConstantType
    #                     ppt=self._range_dictionary.get_context(p).param_type
    #                     self._range_value[p] = get_value_class(
    #                         SparseConstantType(value_encoding=ppt.value_encoding,
    #                                            fill_value=ppt.fill_value),
    #                         DomainSet(self.temporal_domain))
    #                 else:
    #                     log.info('Parameter \'%s\' from coverage \'%s\' already present, skipping...', p, cpth)
    #
    #     time_bounds.sort()
    #
    #     # Next build domain spans for each coverage
    #     rcov_domain_spans = []
    #     start = 0
    #     for i, tb in enumerate(time_bounds):
    #         cov = self._reference_covs[tb.value]
    #         end = start + cov.num_timesteps
    #         rcov_domain_spans.append(Span(start, end, value=tb.value))
    #         start = end
    #
    #     rcov_domain_spans.sort()
    #
    #     self.rcov_domain_spans = rcov_domain_spans
    #
    #     # Add data for all spans
    #     for s in self.rcov_domain_spans:
    #         cov = self._reference_covs[s.value]
    #         for p in self.list_parameters():
    #             if p in cov._range_dictionary:
    #                 self._range_value[p][:] = cov._range_value[p]
    #             else:
    #                 self._range_value[p][:] = self._range_dictionary.get_context(p).fill_value
    #         self.insert_timesteps(len(s))
    #
    #     self._head_coverage_path = self._reference_covs[self.rcov_domain_spans[-1].value].head_coverage_path

    def _build_timeseries(self, rcovs, parameter_dictionary):
        
        for p, pc in parameter_dictionary.iteritems():
            if p not in self._range_dictionary:
                pc = pc[1]
                self._range_dictionary.add_context(pc)
        for cpth, cov in self._verify_rcovs(rcovs):
            self._reference_covs[cpth] = cov
            pdict = cov.parameter_dictionary
            for p, pc in pdict.iteritems():
                if p not in parameter_dictionary:
                    if p not in self._range_dictionary:
                        pc = pc[1]
                        # Timeseries coverage doesn't support instance domains just yet
                        self._range_dictionary.add_context(pc)

    def get_complex_type(self):
        return self._persistence_layer.complex_type


class SimplexCoverage(AbstractCoverage):
    """
    A concrete implementation of AbstractCoverage consisting of 2 domains (temporal and spatial)
    and a collection of parameters associated with one or both of the domains.  Each parameter is defined by a
    ParameterContext object (provided via the ParameterDictionary) and has content represented by a concrete implementation
    of the AbstractParameterValue class.

    """

    def __init__(self, root_dir, persistence_guid, name=None, parameter_dictionary=None,
                 temporal_domain=None, spatial_domain=None, mode=None, in_memory_storage=False,
                 bricking_scheme=None, inline_data_writes=True, auto_flush_values=True, value_caching=True,
                 persistence_name=None):
        """
        Constructor for SimplexCoverage

        @param root_dir The root directory for storage of this coverage
        @param persistence_guid The persistence uuid for this coverage
        @param name The name of the coverage
        @param parameter_dictionary    a ParameterDictionary object expected to contain one or more valid ParameterContext objects
        @param spatial_domain  a concrete instance of AbstractDomain for the spatial domain component
        @param temporal_domain a concrete instance of AbstractDomain for the temporal domain component
        @param mode the file mode for the coverage; one of 'r', 'a', 'r+', or 'w'; defaults to 'r'
        @param in_memory_storage    if False (default), HDF5 persistence is used; otherwise, nothing is written to disk and all data is held in memory only
        @param bricking_scheme  the bricking scheme for the coverage; a dict of the form {'brick_size': #, 'chunk_size': #}
        @param inline_data_writes   if True (default), brick data is written as it is set; otherwise it is written out-of-band by worker processes or threads
        @param auto_flush_values    if True (default), brick data is flushed immediately; otherwise it is buffered until SimplexCoverage.flush_values() is called
        @param value_caching  if True (default), up to 30 value requests are cached for rapid duplicate retrieval
        """
        AbstractCoverage.__init__(self, mode=mode)
        try:
            # Make sure root_dir and persistence_guid are both not None and are strings
            if not isinstance(root_dir, basestring) or not isinstance(persistence_guid, basestring):
                raise TypeError('\'root_dir\' and \'persistence_guid\' must be instances of basestring')

            root_dir = root_dir if not root_dir.endswith(persistence_guid) else os.path.split(root_dir)[0]

            pth = os.path.join(root_dir, persistence_guid)

            def _doload(self):
                # Make sure the coverage directory exists
                if not is_persisted(root_dir, persistence_guid):
#cjb                if not os.path.exists(pth):
                    raise SystemError('Cannot find specified coverage: {0}'.format(pth))

                # All appears well - load it up!
                self._persistence_layer = PostgresPersistenceLayer(root_dir, persistence_guid, mode=self.mode)
                if self._persistence_layer.version != self.version:
                    raise IOError('Coverage Model Version Mismatch: %s != %s' %(self.version, self._persistence_layer.version))

                self.name = self._persistence_layer.name
                self._in_memory_storage = False

                auto_flush_values = self._persistence_layer.auto_flush_values
                inline_data_writes = self._persistence_layer.inline_data_writes
                self.value_caching = self._persistence_layer.value_caching

                from coverage_model.persistence import PersistedStorage, SparsePersistedStorage
                from coverage_model.storage.parameter_persisted_storage import PostgresPersistedStorage
                for parameter_name in self._persistence_layer.parameter_metadata:
                    md = self._persistence_layer.parameter_metadata[parameter_name]
                    mm = self._persistence_layer.master_manager
                    pc = md.parameter_context

                    # Assign the coverage's domain object(s)
                    self._assign_domain(pc)

                    # Get the callbacks for ParameterFunctionType parameters
                    if hasattr(pc, '_pval_callback'):
                        pc._pval_callback = self.get_parameter_values
                        pc._pctxt_callback = self.get_parameter_context
                    self._range_dictionary.add_context(pc)
                    # if pc.param_type._value_class == 'SparseConstantValue':
                    #     s = SparsePersistedStorage(md, mm, self._persistence_layer.brick_dispatcher, dtype=pc.param_type.storage_encoding, fill_value=pc.param_type.fill_value, mode=self.mode, inline_data_writes=inline_data_writes, auto_flush=auto_flush_values)
                    # else:
                    s = PostgresPersistedStorage(md, metadata_manager=mm, parameter_context=pc, dtype=pc.param_type.storage_encoding, fill_value=pc.param_type.fill_value, mode=self._persistence_layer.mode)
                    self._persistence_layer.value_list[parameter_name] = s
                    self._range_value[parameter_name] = get_value_class(param_type=pc.param_type, domain_set=pc.dom, storage=s)
                    if parameter_name in self._persistence_layer.parameter_bounds:
                        bmin, bmax = self._persistence_layer.parameter_bounds[parameter_name]
                        self._range_value[parameter_name]._min = bmin
                        self._range_value[parameter_name]._max = bmax

            # TODO: Why do this, just see if the directory is there no?
            # if name is None or parameter_dictionary is None:
            if is_persisted(root_dir, persistence_guid):
#cjb            if os.path.exists(pth):
                # This appears to be a load
                _doload(self)

            else:
                # This appears to be a new coverage
                # Make sure name and parameter_dictionary are not None
                if name is None or parameter_dictionary is None:
                    raise SystemError('\'name\' and \'parameter_dictionary\' cannot be None')

                # Make sure the specified root_dir exists
                if not in_memory_storage and not os.path.exists(root_dir):
                    raise SystemError('Cannot find specified \'root_dir\': {0}'.format(root_dir))

                # If the coverage directory exists, load it instead!!
                if is_persisted(root_dir, persistence_guid):
#cjb                if os.path.exists(pth):
                    log.warn('The specified coverage already exists - performing load of \'{0}\''.format(pth))
                    _doload(self)
                    return

                # We've checked everything we can - this is a new coverage!!!

                # Must be in 'a' for a new coverage
                self.mode = 'a'

                if not isinstance(name, basestring):
                    raise TypeError('\'name\' must be of type basestring')
                self.name = name
                if temporal_domain is None:
                    self.temporal_domain = GridDomain(GridShape('temporal',[0]), CRS.standard_temporal(), MutabilityEnum.EXTENSIBLE)
                elif isinstance(temporal_domain, AbstractDomain):
                    self.temporal_domain = deepcopy(temporal_domain)
                    self.temporal_domain.shape.extents = (0,) + tuple(self.temporal_domain.shape.extents[1:])
                else:
                    raise TypeError('\'temporal_domain\' must be an instance of AbstractDomain')

                if spatial_domain is None or isinstance(spatial_domain, AbstractDomain):
                    self.spatial_domain = deepcopy(spatial_domain)
                else:
                    raise TypeError('\'spatial_domain\' must be an instance of AbstractDomain')

                if not isinstance(parameter_dictionary, ParameterDictionary):
                    raise TypeError('\'parameter_dictionary\' must be of type ParameterDictionary')

                if bricking_scheme is not None:
                    self._bricking_scheme = bricking_scheme

                self.value_caching = value_caching

                # LOCK inline_data_writes to True
                inline_data_writes = True

                self._in_memory_storage = in_memory_storage
                if self._in_memory_storage:
                    persistence_name = 'in_memory_storage'
                self._persistence_layer = PostgresPersistenceLayer(root_dir,
                                                           persistence_guid,
                                                           name=name,
                                                           tdom=self.temporal_domain,
                                                           sdom=self.spatial_domain,
                                                           mode=self.mode,
                                                           bricking_scheme=self._bricking_scheme,
                                                           inline_data_writes=inline_data_writes,
                                                           auto_flush_values=auto_flush_values,
                                                           value_caching=value_caching,
                                                           coverage_type='simplex',
                                                           version=self.version,
                                                           storage_name=persistence_name)

                for o, pc in parameter_dictionary.itervalues():
                    self.append_parameter(pc)

            self._head_coverage_path = self.persistence_dir
        except:
            self._closed = True
            raise

    def calculate_statistics(self, params=None, time_range=None):
        """time_range is currently not implemented"""
        from coverage_model.parameter_values import NumericValue
        if isinstance(params, basestring):
            tmp = set()
            tmp.add(params)
            params = tmp
        stats = {}
        params_to_calculate = []
        for key in self._range_dictionary.keys():
            if isinstance(self._range_value[key], NumericValue):
                if params is not None:
                    if key in params:
                        params_to_calculate.append(key)
                else:
                    params_to_calculate.append(key)
        pv = self.get_value_dictionary(params_to_calculate)
        for key, values in pv.iteritems():
            ma = np.ma.masked_array(values, np.isnan(values))
            stats[key] = (np.min(ma), np.max(ma), np.mean(ma), np.median(ma))
        not_evaluated = []
        if params is not None:
            not_evaluated = [i for i in params if i not in params_to_calculate]

        return stats, not_evaluated

    def validate_span_data(self):
        return self._persistence_layer.validate_span_data()

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        return AbstractCoverage._fromdict(cmdict, {'parameter_dictionary': '_range_dictionary'})


#=========================
# Range Objects
#=========================

class RangeValues(Dictable):
    """
    A simple storage object for the range value objects in the coverage

    Inherits from Dictable.
    """

    def __init__(self):
        Dictable.__init__(self)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        if not isinstance(value, AbstractParameterValue):
            raise TypeError('Can only assign objects inheriting from AbstractParameterValue')

        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __contains__(self, item):
        return hasattr(self, item)

    def __iter__(self):
        return self.__dict__.__iter__()

    def __dir__(self):
        return self.__dict__.keys()

class RangeDictionary(AbstractIdentifiable):
    """
    Currently known as Taxonomy with respect to the Granule & RecordDictionary
    May be synonymous with RangeType

    @deprecated DO NOT USE - Subsumed by ParameterDictionary
    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)

#=========================
# CRS Objects
#=========================
class CRS(AbstractIdentifiable):
    """

    """
    def __init__(self, axis_types=None, epsg_code=None, temporal_code=None):
        AbstractIdentifiable.__init__(self)
        self.axes = {}

        if axis_types is not None:
            for l in axis_types:
                self.add_axis(l)

        if epsg_code is not None:
            self.epsg_code = epsg_code

        if temporal_code is not None:
            self.temporal_code = temporal_code

    @property
    def has_epsg_code(self):
        return hasattr(self, 'epsg_code')

    @property
    def has_temporal_code(self):
        return hasattr(self, 'temporal_code')

    def add_axis(self, axis_type, axis_name=None):
        if not AxisTypeEnum.has_member(axis_type):
            raise KeyError('Invalid \'axis_type\', must be a member of AxisTypeEnum')

        self.axes[axis_type] = axis_name

    @classmethod
    def standard_temporal(cls, temporal_code=None):
        return CRS([AxisTypeEnum.TIME], temporal_code=temporal_code)

    @classmethod
    def lat_lon_height(cls, epsg_code=None):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT], epsg_code=epsg_code)

    @classmethod
    def lat_lon(cls, epsg_code=None):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT], epsg_code=epsg_code)

    @classmethod
    def x_y_z(cls, epsg_code=None):
        return CRS([AxisTypeEnum.GEO_X, AxisTypeEnum.GEO_Y, AxisTypeEnum.GEO_Z], epsg_code=epsg_code)

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Axes: {1}'.format(indent, self.axes))

        return '\n'.join(lst)

#=========================
# Domain Objects
#=========================

class AbstractDomain(AbstractIdentifiable):
    """

    """
    def __init__(self, shape, crs, mutability=None):
        AbstractIdentifiable.__init__(self)
        self.shape = shape
        self.crs = crs
        self.mutability = mutability or MutabilityEnum.IMMUTABLE

#    def get_mutability(self):
#        return self.mutability

    def get_max_dimension(self):
        pass

    def get_num_elements(self, dim_index):
        pass

#    def get_shape(self):
#        return self.shape

    def insert_elements(self, dim_index, count, doa):
        pass

#    def dump(self):
#        return self._todict()
##        ret={'cm_type':self.__class__.__name__}
##        for k,v in self.__dict__.iteritems():
##            if hasattr(v, '_dump'):
##                ret[k]=v._dump()
##            else:
##                ret[k]=v
##
##        return ret

#    @classmethod
#    def load(cls, ddict):
#        """
#        Create a concrete AbstractDomain from a dict representation
#
#        @param cls  An AbstractDomain instance
#        @param ddict   A dict containing information for a concrete AbstractDomain (requires a 'cm_type' key)
#        """
#        return cls._fromdict(ddict)
##        dd=ddict.copy()
##        if isinstance(dd, dict) and 'cm_type' in dd:
##            dd.pop('cm_type')
##            #TODO: If additional required constructor parameters are added, make sure they're dealt with here
##            crs = CRS._load(dd.pop('crs'))
##            shp = AbstractShape._load(dd.pop('shape'))
##
##            pc=AbstractDomain(shp,crs)
##            for k, v in dd.iteritems():
##                setattr(pc,k,v)
##
##            return pc
##        else:
##            raise TypeError('ddict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(ddict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst=[]
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Shape:\n{1}'.format(indent, self.shape.__str__(indent*2)))
        lst.append('{0}CRS:\n{1}'.format(indent, self.crs.__str__(indent*2)))
        lst.append('{0}Mutability: {1}'.format(indent, self.mutability))

        return '\n'.join(lst)

class GridDomain(AbstractDomain):
    """

    """
    def __init__(self, shape, crs, mutability=None):
        AbstractDomain.__init__(self, shape, crs, mutability)

class TopologicalDomain(AbstractDomain):
    """

    """
    def __init__(self):
        AbstractDomain.__init__(self)

class DomainSet(AbstractIdentifiable):

    def __init__(self, tdom=None, sdom=None, **kwargs):
        kwc = kwargs.copy()
        AbstractIdentifiable.__init__(self, **kwc)
        self.tdom = tdom
        self.sdom = sdom

    @property
    def total_extents(self):
        ret = []
        if self.tdom is not None:
            ret += self.tdom.shape.extents
        if self.sdom is not None:
            ret += self.sdom.shape.extents

        return tuple(ret)

class SimpleDomainSet(AbstractIdentifiable):
    def __init__(self, shape, **kwargs):
        kwc=kwargs.copy()
        AbstractIdentifiable.__init__(self, **kwc)
        self.shape = shape

    @property
    def total_extents(self):
        return self.shape

#=========================
# Shape Objects
#=========================

class AbstractShape(AbstractIdentifiable):
    """

    """
    def __init__(self, name, extents=None):
        AbstractIdentifiable.__init__(self)
        self.name = name
        self.extents = extents or [0]
        if not isinstance(self.extents, tuple):
            self.extents = tuple(self.extents)

    @property
    def rank(self):
        return len(self.extents)

#    def rank(self):
#        return len(self.extents)

#    def _dump(self):
#        return self._todict()
##        ret = dict((k,v) for k,v in self.__dict__.iteritems())
##        ret['cm_type'] = self.__class__.__name__
##        return ret

#    @classmethod
#    def _load(cls, sdict):
#        return cls._fromdict(sdict)
##        if isinstance(sdict, dict) and 'cm_type' in sdict and sdict['cm_type']:
##            import inspect
##            mod = inspect.getmodule(cls)
##            ptcls=getattr(mod, sdict['cm_type'])
##
##            args = inspect.getargspec(ptcls.__init__).args
##            del args[0] # get rid of 'self'
##            kwa={}
##            for a in args:
##                kwa[a] = sdict[a] if a in sdict else None
##
##            ret = ptcls(**kwa)
##            for k,v in sdict.iteritems():
##                if not k in kwa.keys() and not k == 'cm_type':
##                    setattr(ret,k,v)
##
##            return ret
##        else:
##            raise TypeError('sdict is not properly formed, must be of type dict and contain a \'cm_type\' key: {0}'.format(sdict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}Extents: {1}'.format(indent, self.extents))

        return '\n'.join(lst)

class GridShape(AbstractShape):
    """

    """
    def __init__(self, name, extents=None):
        AbstractShape.__init__(self, name, extents)

    #CBM: Make extents type-safe


#=========================
# Filter Objects
#=========================

class AbstractFilter(AbstractIdentifiable):
    """

    """
    def __init__(self):
        AbstractIdentifiable.__init__(self)


class StructureFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)

class ParameterFilter(AbstractFilter):
    """

    """
    def __init__(self):
        AbstractFilter.__init__(self)



#=========================
# Possibly OBE ?
#=========================

#class Topology():
#    """
#    Sets of topological entities
#    Mappings between topological entities
#    """
#    pass
#
#class TopologicalEntity():
#    """
#
#    """
#    pass
#
#class ConstructionType():
#    """
#    Lattice or not (i.e. 'unstructured')
#    """
#    pass
