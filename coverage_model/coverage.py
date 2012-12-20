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

from ooi.logging import log

from coverage_model.basic_types import AbstractIdentifiable, AxisTypeEnum, MutabilityEnum, VariabilityEnum, get_valid_DomainOfApplication, Dictable, InMemoryStorage
from coverage_model.parameter import Parameter, ParameterDictionary, ParameterContext
from coverage_model.parameter_values import get_value_class, AbstractParameterValue
from coverage_model.persistence import PersistenceLayer, InMemoryPersistenceLayer
import coverage_model.utils as utils
from copy import deepcopy
import numpy as np
import pickle
import os

#=========================
# Coverage Objects
#=========================
from pyon.util.async import spawn

class AbstractCoverage(AbstractIdentifiable):
    """
    Core data model, persistence, etc
    TemporalTopology
    SpatialTopology
    """
    def __init__(self, mode=None):
        AbstractIdentifiable.__init__(self)
        self._closed = False

        if mode is not None and isinstance(mode, str) and mode[0] in ['r','w','a']:
            self.mode = mode
        else:
            self.mode = 'r+'

    @classmethod
    def pickle_save(cls, cov_obj, file_path, use_ascii=False):
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))

        if not isinstance(cov_obj._persistence_layer, InMemoryPersistenceLayer):
            raise StandardError('cov_obj must be constructed with the \'in_memory_storage\' flag == True')

        with open(file_path, 'w') as f:
            pickle.dump(cov_obj, f, 0 if use_ascii else 2)

        log.info('Saved to pickle \'%s\'', file_path)
        log.warn('\'pickle_save\' and \'pickle_load\' are not 100% safe, use at your own risk!!')

    @classmethod
    def pickle_load(cls, file_path):
        with open(file_path, 'r') as f:
            obj = pickle.load(f)

        if not isinstance(obj, AbstractCoverage):
            raise StandardError('loaded object must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(obj)))

        log.warn('\'pickle_save\' and \'pickle_load\' are not 100% safe, use at your own risk!!')
        log.info('Loaded from pickle \'%s\'', file_path)
        return obj

    @classmethod
    def load(cls, root_dir, persistence_guid=None, mode=None):
        if persistence_guid is None:
            root_dir, persistence_guid = os.path.split(root_dir)

        return SimplexCoverage(root_dir, persistence_guid, mode=mode)

    @classmethod
    def save(cls, cov_obj, *args, **kwargs):
        if not isinstance(cov_obj, AbstractCoverage):
            raise StandardError('cov_obj must be an instance or subclass of AbstractCoverage: object is {0}'.format(type(cov_obj)))

        cov_obj.flush()

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
            log.info('Closing coverage \'%s\'', self.name if hasattr(self,'name') else 'unnamed')

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

    def __del__(self):
        self.close()

class ViewCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1 AbstractCoverage and applies a Filter
    """
    def __init__(self):
        AbstractCoverage.__init__(self)

        self.reference_coverage = ''
        self.structure_filter = StructureFilter()
        self.parameter_filter = ParameterFilter()

class ComplexCoverage(AbstractCoverage):
    # TODO: Implement
    """
    References 1-n coverages
    """
    def __init__(self):
        AbstractCoverage.__init__(self)
        self.coverages = []

class SimplexCoverage(AbstractCoverage):
    """
    A concrete implementation of AbstractCoverage consisting of 2 domains (temporal and spatial)
    and a collection of parameters associated with one or both of the domains.  Each parameter is defined by a
    ParameterContext object (provided via the ParameterDictionary) and has content represented by a concrete implementation
    of the AbstractParameterValue class.

    """
    def __init__(self, root_dir, persistence_guid, name=None, parameter_dictionary=None, temporal_domain=None, spatial_domain=None, mode=None, in_memory_storage=False, bricking_scheme=None, inline_data_writes=True, auto_flush_values=True):
        """
        Constructor for SimplexCoverage

        @param root_dir The root directory for storage of this coverage
        @param persistence_guid The persistence uuid for this coverage
        @param name The name of the coverage
        @param parameter_dictionary    a ParameterDictionary object expected to contain one or more valid ParameterContext objects
        @param spatial_domain  a concrete instance of AbstractDomain for the spatial domain component
        @param temporal_domain a concrete instance of AbstractDomain for the temporal domain component
        """
        AbstractCoverage.__init__(self, mode=mode)
        try:
            # Make sure root_dir and persistence_guid are both not None and are strings
            if not isinstance(root_dir, str) or not isinstance(persistence_guid, str):
                raise TypeError('\'root_dir\' and \'persistence_guid\' must be instances of str')

            root_dir = root_dir if not root_dir.endswith(persistence_guid) else os.path.split(root_dir)[0]

            pth=os.path.join(root_dir, persistence_guid)

            def _doload(self):
                # Make sure the coverage directory exists
                if not os.path.exists(pth):
                    raise SystemError('Cannot find specified coverage: {0}'.format(pth))

                # All appears well - load it up!
                self._persistence_layer = PersistenceLayer(root_dir, persistence_guid, mode=self.mode)

                self.name = self._persistence_layer.name
                self.spatial_domain = self._persistence_layer.sdom
                self.temporal_domain = self._persistence_layer.tdom

                self._range_dictionary = ParameterDictionary()
                self._range_value = RangeValues()

                self._bricking_scheme = self._persistence_layer.global_bricking_scheme

                self._in_memory_storage = False

                auto_flush_values = self._persistence_layer.auto_flush_values
                inline_data_writes = self._persistence_layer.inline_data_writes

                from coverage_model.persistence import PersistedStorage
                for parameter_name in self._persistence_layer.parameter_metadata.keys():
                    md = self._persistence_layer.parameter_metadata[parameter_name]
                    pc = md.parameter_context
                    self._range_dictionary.add_context(pc)
                    s = PersistedStorage(md, self._persistence_layer.brick_dispatcher, dtype=pc.param_type.storage_encoding, fill_value=pc.param_type.fill_value, mode=self.mode, inline_data_writes=inline_data_writes, auto_flush=auto_flush_values)
                    self._range_value[parameter_name] = get_value_class(param_type=pc.param_type, domain_set=pc.dom, storage=s)

            if name is None or parameter_dictionary is None:
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
                if os.path.exists(pth):
                    log.warn('The specified coverage already exists - performing load of \'{0}\''.format(pth))
                    _doload(self)
                    return

                # Check the mode - can't make a new coverage in 'r' mode!!
                if self.mode == 'r':
                    raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

                self.name = name
                if temporal_domain is None:
                    self.temporal_domain = GridDomain(GridShape('temporal',[0]), CRS.standard_temporal(), MutabilityEnum.EXTENSIBLE)
                elif isinstance(temporal_domain, AbstractDomain):
                    self.temporal_domain = deepcopy(temporal_domain)
                else:
                    raise TypeError('\'temporal_domain\' must be an instance of AbstractDomain')

                if spatial_domain is None or isinstance(spatial_domain, AbstractDomain):
                    self.spatial_domain = deepcopy(spatial_domain)
                else:
                    raise TypeError('\'spatial_domain\' must be an instance of AbstractDomain')

                if not isinstance(parameter_dictionary, ParameterDictionary):
                    raise TypeError('\'parameter_dictionary\' must be of type ParameterDictionary')
                self._range_dictionary = ParameterDictionary()
                self._range_value = RangeValues()

                self._bricking_scheme = bricking_scheme or {'brick_size':1000,'chunk_size':500}

                self._in_memory_storage = in_memory_storage
                if self._in_memory_storage:
                    self._persistence_layer = InMemoryPersistenceLayer()
                else:
                    self._persistence_layer = PersistenceLayer(root_dir, persistence_guid, name=name, tdom=temporal_domain, sdom=spatial_domain, mode=self.mode, bricking_scheme=self._bricking_scheme, inline_data_writes=inline_data_writes, auto_flush_values=auto_flush_values)

                for o, pc in parameter_dictionary.itervalues():
                    self._append_parameter(pc)
        except:
            self._closed = True
            raise

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        return super(SimplexCoverage, cls)._fromdict(cmdict, {'parameter_dictionary':'_range_dictionary'})

    @property
    def temporal_parameter_name(self):
        return self._range_dictionary.temporal_parameter_name

    @property
    def parameter_dictionary(self):
        return deepcopy(self._range_dictionary)

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

    def append_parameter(self, parameter_context):
        """
        Append a ParameterContext to the coverage

        @deprecated use a ParameterDictionary during construction of the coverage
        """
        log.warn('SimplexCoverage.append_parameter() is deprecated: use a ParameterDictionary during construction of the coverage')
        self._append_parameter(parameter_context)

    def _append_parameter(self, parameter_context):
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

        if self.mode == 'r':
            raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        if not isinstance(parameter_context, ParameterContext):
            raise TypeError('\'parameter_context\' must be an instance of ParameterContext')

        # Create a deep copy of the ParameterContext
        pcontext = deepcopy(parameter_context)

        pname = pcontext.name

        no_sdom = self.spatial_domain is None

        ## Determine the correct array shape

        # Get the parameter variability; assign to VariabilityEnum.NONE if None
        pv=pcontext.variability or VariabilityEnum.NONE
        if no_sdom and pv in (VariabilityEnum.SPATIAL, VariabilityEnum.BOTH):
            log.warn('Provided \'parameter_context\' indicates Spatial variability, but coverage has no Spatial Domain')

        if pv == VariabilityEnum.TEMPORAL: # Only varies in the Temporal Domain
            pcontext.dom = DomainSet(self.temporal_domain.shape.extents, None)
        elif pv == VariabilityEnum.SPATIAL: # Only varies in the Spatial Domain
            pcontext.dom = DomainSet(None, self.spatial_domain.shape.extents)
        elif pv == VariabilityEnum.BOTH: # Varies in both domains
            # If the Spatial Domain is only a single point on a 0d Topology, the parameter's shape is that of the Temporal Domain only
            if no_sdom or (len(self.spatial_domain.shape.extents) == 1 and self.spatial_domain.shape.extents[0] == 0):
                pcontext.dom = DomainSet(self.temporal_domain.shape.extents, None)
            else:
                pcontext.dom = DomainSet(self.temporal_domain.shape.extents, self.spatial_domain.shape.extents)
        elif pv == VariabilityEnum.NONE: # No variance; constant
            # CBM TODO: Not sure we can have this constraint - precludes situations like a TextType with Variablity==None...
#            # This is a constant - if the ParameterContext is not a ConstantType, make it one with the default 'x' expr
#            if not isinstance(pcontext.param_type, ConstantType):
#                pcontext.param_type = ConstantType(pcontext.param_type)

            # The domain is the total domain - same value everywhere!!
            # If the Spatial Domain is only a single point on a 0d Topology, the parameter's shape is that of the Temporal Domain only
            if no_sdom or (len(self.spatial_domain.shape.extents) == 1 and self.spatial_domain.shape.extents[0] == 0):
                pcontext.dom = DomainSet(self.temporal_domain.shape.extents, None)
            else:
                pcontext.dom = DomainSet(self.temporal_domain.shape.extents, self.spatial_domain.shape.extents)
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
            raise ValueError('I/O operation on closed file')

        if param_name in self._range_dictionary:
            p = Parameter(deepcopy(self._range_dictionary.get_context(param_name)), self._range_value[param_name].shape, self._range_value[param_name])
            return p
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

    def insert_timesteps(self, count, origin=None):
        """
        Insert count # of timesteps beginning at the origin

        The specified # of timesteps are inserted into the temporal value array at the indicated origin.  This also
        expands the temporal dimension of the AbstractParameterValue for each parameters

        @param count    The number of timesteps to insert
        @param origin   The starting location, from which to begin the insertion
        """
        if self.closed:
            raise IOError('I/O operation on closed file')

        if self.mode == 'r':
            raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        # Get the current shape of the temporal_dimension
        shp = self.temporal_domain.shape

        # If not provided, set the origin to the end of the array
        if origin is None or not isinstance(origin, int):
            origin = shp.extents[0]

        # Expand the shape of the temporal_domain - following works if extents is a list or tuple
        shp.extents = (shp.extents[0]+count,)+tuple(shp.extents[1:])

        # Expand the temporal dimension of each of the parameters - the parameter determines how to apply the change
        for n in self._range_dictionary:
            pc = self._range_dictionary.get_context(n)
            # Update the dom of the parameter_context
            if pc.dom.tdom is not None:
                pc.dom.tdom = self.temporal_domain.shape.extents

            self._persistence_layer.expand_domain(pc)
            self._range_value[n].expand_content(VariabilityEnum.TEMPORAL, origin, count)

        # Update the temporal_domain in the master_manager, do NOT flush!!
        self._persistence_layer.update_domain(tdom=self.temporal_domain, do_flush=False)
        # Flush the master_manager & parameter_managers in a separate greenlet
        spawn(self._persistence_layer.flush)

    def set_time_values(self, value, tdoa=None):
        """
        Convenience method for setting time values

        @param value    The value to set
        @param tdoa The temporal DomainOfApplication; default to full Domain
        """
        return self.set_parameter_values(self.temporal_parameter_name, value, tdoa, None)

    def get_time_values(self, tdoa=None, return_value=None):
        """
        Convenience method for retrieving time values

        Delegates to get_parameter_values, supplying the temporal parameter name and sdoa == None
        @param tdoa The temporal DomainOfApplication; default to full Domain
        @param return_value If supplied, filled with response value
        """
        return self.get_parameter_values(self.temporal_parameter_name, tdoa, None, return_value)

    @property
    def num_timesteps(self):
        """
        The current number of timesteps
        """
        return self.temporal_domain.shape.extents[0]

    def set_parameter_values(self, param_name, value, tdoa=None, sdoa=None):
        """
        Assign value to the specified parameter

        Assigns the value to param_name within the coverage.  Temporal and spatial DomainOfApplication objects can be
        applied to constrain the assignment.  See DomainOfApplication for details

        @param param_name   The name of the parameter
        @param value    The value to set
        @param tdoa The temporal DomainOfApplication
        @param sdoa The spatial DomainOfApplication
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if self.closed:
            raise IOError('I/O operation on closed file')

        if self.mode == 'r':
            raise IOError('Coverage not open for writing: mode == \'{0}\''.format(self.mode))

        if not param_name in self._range_value:
            raise KeyError('Parameter \'{0}\' not found in coverage_model'.format(param_name))

        slice_ = []

        tdoa = get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        log.debug('Temporal doa: %s', tdoa.slices)
        slice_.extend(tdoa.slices)

        if self.spatial_domain is not None:
            sdoa = get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)
            log.debug('Spatial doa: %s', sdoa.slices)
            slice_.extend(sdoa.slices)

        log.debug('Setting slice: %s', slice_)

        self._range_value[param_name][slice_] = value

    def get_parameter_values(self, param_name, tdoa=None, sdoa=None, return_value=None):
        """
        Retrieve the value for a parameter

        Returns the value from param_name.  Temporal and spatial DomainOfApplication objects can be used to
        constrain the response.  See DomainOfApplication for details.

        @param param_name   The name of the parameter
        @param tdoa The temporal DomainOfApplication
        @param sdoa The spatial DomainOfApplication
        @param return_value If supplied, filled with response value - currently via OVERWRITE
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if self.closed:
            raise ValueError('I/O operation on closed file')

        if not param_name in self._range_value:
            raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        if return_value is not None:
            log.warn('Provided \'return_value\' will be OVERWRITTEN')

        slice_ = []

        tdoa = get_valid_DomainOfApplication(tdoa, self.temporal_domain.shape.extents)
        log.debug('Temporal doa: %s', tdoa.slices)
        slice_.extend(tdoa.slices)

        if self.spatial_domain is not None:
            sdoa = get_valid_DomainOfApplication(sdoa, self.spatial_domain.shape.extents)
            log.debug('Spatial doa: %s', sdoa.slices)
            slice_.extend(sdoa.slices)

        log.debug('Getting slice: %s', slice_)

        return_value = self._range_value[param_name][slice_]
        return return_value

    def get_parameter_context(self, param_name):
        """
        Retrieve a deepcopy of the ParameterContext object for the specified parameter

        @param param_name   The name of the parameter for which to retrieve context
        @returns A deepcopy of the specified ParameterContext object
        @throws KeyError    The coverage does not contain a parameter with name 'param_name'
        """
        if not param_name in self._range_dictionary:
            raise KeyError('Parameter \'{0}\' not found in coverage'.format(param_name))

        return deepcopy(self._range_dictionary.get_context(param_name))

    def __axis_arg_to_params(self, axis=None):
        """
        Helper function to compose a list of parameter names based on the <i>axis</i> argument

        If <i>axis</i> is None, all coordinate parameters are included

        @param axis A member of AxisTypeEnum; may be an iterable of such members
        """
        params = []
        if axis is None:
            params.extend(pn for pk, pn in self.temporal_domain.crs.axes.iteritems())
            params.extend(pn for pk, pn in self.spatial_domain.crs.axes.iteritems())
        elif hasattr(axis, '__iter__'):
            for a in axis:
                if a in self.temporal_domain.crs.axes:
                    params.append(self.temporal_domain.crs.axes[a])
                elif a in self.spatial_domain.crs.axes:
                    params.append(self.spatial_domain.crs.axes[a])
                else:
                    raise ValueError('Specified axis ({0}) not found in coverage'.format(a))
        elif axis in self.temporal_domain.crs.axes:
            params.append(self.temporal_domain.crs.axes[axis])
        elif axis in self.spatial_domain.crs.axes:
            params.append(self.spatial_domain.crs.axes[axis])
        else:
            raise ValueError('Specified axis ({0}) not found in coverage'.format(axis))

        return params

    def __parameter_name_arg_to_params(self, parameter_name=None):
        """
        Helper function to compose a list of parameter names based on the <i>parameter_name</i> argument

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        @param parameter_name A string parameter name; may be an iterable of such members
        """
        params = []
        if parameter_name is None:
            params.extend(self._range_dictionary.keys())
        elif hasattr(parameter_name, '__iter__'):
            params.extend(pn for pn in parameter_name if pn in self._range_dictionary.keys())
        else:
            params.append(parameter_name)

        return params

    def get_data_bounds(self, parameter_name=None):
        """
        Returns the bounds (min, max) for the parameter(s) indicated by <i>parameter_name</i>

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        If more than one parameter is indicated by <i>parameter_name</i>, a dict of {key:(min,max)} is returned;
        otherwise, only the (min, max) tuple is returned

        @param parameter_name   A string parameter name; may be an iterable of such members
        """
        from coverage_model import QuantityType, ConstantType
        ret = {}
        for pn in self.__parameter_name_arg_to_params(parameter_name):
            ctxt = self._range_dictionary.get_context(pn)
            if isinstance(ctxt.param_type, QuantityType) or isinstance(ctxt.param_type, ConstantType):
                fv = ctxt.fill_value
                varr = np.ma.masked_equal(self._range_value[pn][:], fv, copy=False)
                r = (varr.min(), varr.max())
                ret[pn] = tuple([fv if isinstance(x, np.ma.core.MaskedConstant) else x for x in r])
            else:
                # CBM TODO: Sort out if this is an appropriate way to deal with non-numeric types
                ret[pn] = (fv, fv)

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
        return self.get_data_bounds(self.__axis_arg_to_params(axis))

    def get_data_extents(self, parameter_name=None):
        """
        Returns the extents (dim_0,dim_1,...,dim_n) for the parameter(s) indicated by <i>parameter_name</i>

        If <i>parameter_name</i> is None, all parameters in the coverage are included

        If more than one parameter is indicated by <i>parameter_name</i>, a dict of {key:(dim_0,dim_1,...,dim_n)} is returned;
        otherwise, only the (dim_0,dim_1,...,dim_n) tuple is returned

        @param parameter_name   A string parameter name; may be an iterable of such members
        """
        ret = {}
        for pn in self.__parameter_name_arg_to_params(parameter_name):
            p = self._range_dictionary.get_context(pn)
            ret[pn] = p.dom.total_extents

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
        return self.get_data_extents(self.__axis_arg_to_params(axis))

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

        for pn in self.__parameter_name_arg_to_params(parameter_name):
            p = self._range_dictionary.get_context(pn)
            te=p.dom.total_extents
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
        lst.append('Temporal Domain:\n{0}'.format(self.temporal_domain.__str__(indent*2)))
        lst.append('Spatial Domain:\n{0}'.format(self.spatial_domain.__str__(indent*2)))

        lst.append('Parameters:')
        for x in self._range_value:
            lst.append('{0}{1} {2}\n{3}'.format(indent*2,x,self._range_value[x].shape,self._range_dictionary.get_context(x).__str__(indent*4)))

        return '\n'.join(lst)

    def __str__(self):
        lst = []
        indent = ' '
        lst.append('ID: {0}'.format(self._id))
        lst.append('Name: {0}'.format(self.name))
        lst.append('TemporalDomain: Shape=>{0} Axes=>{1}'.format(self.temporal_domain.shape.extents, self.temporal_domain.crs.axes))
        lst.append('SpatialDomain: Shape=>{0} Axes=>{1}'.format(self.spatial_domain.shape.extents, self.spatial_domain.crs.axes))
        lst.append('Coordinate Parameters: {0}'.format(self.list_parameters(coords_only=True)))
        lst.append('Data Parameters: {0}'.format(self.list_parameters(coords_only=False, data_only=True)))

        return '\n'.join(lst)


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
    def __init__(self, axis_types=None):
        AbstractIdentifiable.__init__(self)
        self.axes={}
        if axis_types is not None:
            for l in axis_types:
                self.add_axis(l)

    def add_axis(self, axis_type, axis_name=None):
        if not AxisTypeEnum.has_member(axis_type):
            raise KeyError('Invalid \'axis_type\', must be a member of AxisTypeEnum')

        self.axes[axis_type] = axis_name

    @classmethod
    def standard_temporal(cls):
        return CRS([AxisTypeEnum.TIME])

    @classmethod
    def lat_lon_height(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    @classmethod
    def lat_lon(cls):
        return CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    @classmethod
    def x_y_z(cls):
        return CRS([AxisTypeEnum.GEO_X, AxisTypeEnum.GEO_Y, AxisTypeEnum.GEO_Z])

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
        kwc=kwargs.copy()
        AbstractIdentifiable.__init__(self, **kwc)
        self.tdom = tdom
        self.sdom = sdom

    @property
    def total_extents(self):
        ret=[]
        if self.tdom:
            ret += self.tdom
        if self.sdom:
            ret += self.sdom

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