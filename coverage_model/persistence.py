#!/usr/bin/env python

"""
@package coverage_model.persistence
@file coverage_model/persistence.py
@author James Case
@brief The core classes comprising the Persistence Layer
"""

from coverage_model.brick_dispatch import BrickWriterDispatcher
from ooi.logging import log
from coverage_model.basic_types import create_guid, AbstractStorage, InMemoryStorage
from coverage_model.persistence_helpers import MasterManager, ParameterManager, pack, unpack
import numpy as np
import h5py
from coverage_model.hdf_utils import HDFLockingFile
import os
import itertools
from copy import deepcopy
from coverage_model.metadata_factory import MetadataManagerFactory


def is_persisted(file_dir, guid):
    return MetadataManagerFactory.isPersisted(file_dir,guid)


def dir_exists(file_dir):
    return MetadataManagerFactory.dirExists(file_dir)


def get_coverage_type(file_dir, guid):
    return MetadataManagerFactory.getCoverageType(file_dir,guid)


# TODO: Make persistence-specific error classes
class PersistenceError(Exception):
    pass


class SimplePersistenceLayer(object):

    def __init__(self, root, guid, name=None, param_dict=None, mode=None, coverage_type=None, **kwargs):
        root = '.' if root is ('' or None) else root

#        self.master_manager = MasterManager(root_dir=root, guid=guid, name=name, param_dict=param_dict,
        self.master_manager = MetadataManagerFactory.buildMetadataManager(root, guid, name=name, param_dict=param_dict,
                                            parameter_bounds=None, tree_rank=2, coverage_type=coverage_type, **kwargs)

        if not hasattr(self.master_manager, 'coverage_type'):
            self.master_manager.coverage_type = coverage_type

        self.mode = mode

        if self.mode != 'r':
            self.master_manager.flush()

        self._closed = False

    def __getattr__(self, key):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            return getattr(self.master_manager, key)
        else:
            return getattr(super(SimplePersistenceLayer, self), key)

    def __setattr__(self, key, value):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            setattr(self.master_manager, key, value)
        else:
            super(SimplePersistenceLayer, self).__setattr__(key, value)

    def update_parameter_bounds(self, parameter_name, bounds):
        # No-op - would be called by parameters stored in a ComplexCoverage, which can only be ParameterFunctions
        pass

    def has_dirty_values(self):
        # Never has dirty values
        return False

    def get_dirty_values_async_result(self):
        from gevent.event import AsyncResult
        ret = AsyncResult()
        ret.set(True)
        return ret

    def flush_values(self):
        return self.get_dirty_values_async_result()

    def flush(self):
        if self.mode == 'r':
            log.warn('SimplePersistenceLayer not open for writing: mode=%s', self.mode)
            return

        log.debug('Flushing MasterManager...')
        self.master_manager.flush()

    def close(self, force=False, timeout=None):
        if not self._closed:
            if self.mode != 'r':
                self.flush()

        self._closed = True

    def expand_domain(self, *args, **kwargs):
        # No Op - storage expanded by *Value classes
        pass

    def shrink_domain(self, total_domain, do_flush=True):
        pass

    def init_parameter(self, parameter_context, *args, **kwargs):
        return InMemoryStorage(dtype=parameter_context.param_type.value_encoding, fill_value=parameter_context.param_type.fill_value)

    def update_domain(self, tdom=None, sdom=None, do_flush=True):
        # No Op
        pass


class PersistenceLayer(object):
    """
    The PersistenceLayer class manages the disk-level storage (and retrieval) of the Coverage Model using HDF5 files.
    """

    def __init__(self, root, guid, name=None, tdom=None, sdom=None, mode=None, bricking_scheme=None, inline_data_writes=True, auto_flush_values=True, value_caching=True, coverage_type=None, **kwargs):
        """
        Constructor for PersistenceLayer

        @param root The <root> component of the filesystem path for the coverage (/<root>/<guid>)
        @param guid The <guid> component of the filesystem path for the coverage (/<root>/<guid>)
        @param name CoverageModel's name persisted to the metadata attribute in the master HDF5 file
        @param tdom Concrete instance of AbstractDomain for the temporal domain component
        @param sdom Concrete instance of AbstractDomain for the spatial domain component
        @param bricking_scheme  A dictionary containing the brick and chunk sizes
        @param auto_flush_values    True = Values flushed to HDF5 files automatically, False = Manual
        @param value_caching  if True (default), value requests should be cached for rapid duplicate retrieval
        @param kwargs
        @return None
        """

        log.debug('Persistence GUID: %s', guid)
        root = '.' if root is ('' or None) else root

#        self.master_manager = MasterManager(root, guid, name=name, tdom=tdom, sdom=sdom, global_bricking_scheme=bricking_scheme, parameter_bounds=None, coverage_type=coverage_type, **kwargs)
        self.master_manager = MetadataManagerFactory.buildMetadataManager(root, guid, name=name, tdom=tdom, sdom=sdom, global_bricking_scheme=bricking_scheme, parameter_bounds=None, coverage_type=coverage_type, **kwargs)

        self.mode = mode
        if not hasattr(self.master_manager, 'auto_flush_values'):
            self.master_manager.auto_flush_values = auto_flush_values
        if not hasattr(self.master_manager, 'inline_data_writes'):
            self.master_manager.inline_data_writes = inline_data_writes
        if not hasattr(self.master_manager, 'value_caching'):
            self.master_manager.value_caching = value_caching
        if not hasattr(self.master_manager, 'coverage_type'):
            self.master_manager.coverage_type = coverage_type

        # TODO: This is not done correctly
        if tdom != None:
            self._init_master(tdom.shape.extents, bricking_scheme)

        self.value_list = {}

        self.parameter_metadata = {} # {parameter_name: [brick_list, parameter_domains, rtree]}

        for pname in self.param_groups:
            log.debug('parameter group: %s', pname)
            self.parameter_metadata[pname] = ParameterManager(os.path.join(self.root_dir, self.guid, pname), pname)

        if self.mode != 'r':
            if self.master_manager.is_dirty():
                self.master_manager.flush()

        if self.mode == 'r' or self.inline_data_writes:
            self.brick_dispatcher = None
        else:
            self.brick_dispatcher = BrickWriterDispatcher(self.write_failure_callback)
            self.brick_dispatcher.run()

        self._closed = False

        log.debug('Persistence Layer Successfully Initialized')

    def __getattr__(self, key):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            return getattr(self.master_manager, key)
        else:
            return getattr(super(PersistenceLayer, self), key)

    def __setattr__(self, key, value):
        if 'master_manager' in self.__dict__ and hasattr(self.master_manager, key):
            setattr(self.master_manager, key, value)
        else:
            super(PersistenceLayer, self).__setattr__(key, value)

    def update_parameter_bounds(self, parameter_name, bounds):
        dmin, dmax = bounds
        if parameter_name in self.parameter_bounds:
            pmin, pmax = self.parameter_bounds[parameter_name]
            dmin = min(dmin, pmin)
            dmax = max(dmax, pmax)
        self.parameter_bounds[parameter_name] = (dmin, dmax)
        self.master_manager.flush()

    def _init_master(self, tD, bricking_scheme):
        log.debug('Performing Rtree dict setup')
        # tD = parameter_context.dom.total_extents
        bD,cD = self.calculate_brick_size(tD, bricking_scheme) #remains same for each parameter

        self.master_manager._init_rtree(bD)

        self.master_manager.brick_list = {}
        self.master_manager.brick_domains = [tD, bD, cD, bricking_scheme]

    # CBM TODO: This needs to be improved greatly - should callback all the way to the Application layer as a "failure handler"
    def write_failure_callback(self, message, work):
        log.error('WORK DISCARDED!!!; %s: %s', message, work)

    def calculate_brick_size(self, tD, bricking_scheme):
        """
        Calculates and returns the brick and chunk size for each dimension
        in the total domain based on the bricking scheme

        @param tD   Total domain
        @param bricking_scheme  A dictionary containing the brick and chunk sizes
        @return Brick and Chunk sizes based on the total domain
        """

        log.debug('Calculating the size of a brick...')
        log.debug('Bricking scheme: %s', bricking_scheme)
        log.debug('tD: %s', tD)
        bD = [bricking_scheme['brick_size'] for x in tD]
        cD = [bricking_scheme['chunk_size'] for x in tD]
        log.debug('bD: %s', bD)
        log.debug('cD: %s', cD)
        return bD,tuple(cD)

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

        self.global_bricking_scheme = bricking_scheme

        pm = ParameterManager(os.path.join(self.root_dir, self.guid, parameter_name), parameter_name, read_only=False)
        self.parameter_metadata[parameter_name] = pm

        pm.parameter_context = parameter_context

        log.debug('Initialize %s', parameter_name)

        self.master_manager.create_group(parameter_name)

        if parameter_context.param_type._value_class == 'SparseConstantValue':
            v = SparsePersistedStorage(pm, self.master_manager, self.brick_dispatcher, dtype=parameter_context.param_type.storage_encoding, fill_value=parameter_context.param_type.fill_value, mode=self.mode, inline_data_writes=self.inline_data_writes, auto_flush=self.auto_flush_values)
        else:
            v = PersistedStorage(pm, self.master_manager, self.brick_dispatcher, dtype=parameter_context.param_type.storage_encoding, fill_value=parameter_context.param_type.fill_value, mode=self.mode, inline_data_writes=self.inline_data_writes, auto_flush=self.auto_flush_values)
        self.value_list[parameter_name] = v

        # CBM TODO: Consider making this optional and bulk-flushing from the coverage after all parameters have been initialized
        # No need to check if they're dirty, we know they are!
        pm.flush()

        # Put the pm into read_only mode
        pm.read_only = True

        # If there are already bricks, ensure there are appropriate links for this new parameter
        for brick_guid in self.master_manager.brick_list:
            brick_file_name = '{0}.hdf5'.format(brick_guid)
            self._add_brick_link(parameter_name, brick_guid, brick_file_name)

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

    def _add_brick_link(self, parameter_name, brick_guid, brick_file_name):
        brick_rel_path = os.path.join(self.parameter_metadata[parameter_name].root_dir.replace(self.root_dir,'.'), brick_file_name)
        link_path = '/{0}/{1}'.format(parameter_name, brick_guid)

        # Add brick to Master HDF file
        self.master_manager.add_external_link(link_path, brick_rel_path, brick_guid)

    # Write empty HDF5 brick to the filesystem
    def _write_brick(self, rtree_extents, brick_extents, brick_active_size, origin, bD):
        """
        Creates a virtual brick in the PersistenceLayer by updating the HDF5 master file's
        brick list, rtree and ExternalLink to where the HDF5 file will be saved in the future (lazy create)

        @param rtree_extents    Total extents of brick's domain in rtree format
        @param brick_extents    Size of brick
        @param brick_active_size    Size of brick (same rank as parameter)
        @param origin   Domain origin offset
        @param bD   Slice-friendly size of brick's domain
        @return N/A
        """
        log.debug('Writing virtual brick...')

        # Set HDF5 file and group
        # Create a GUID for the brick
        brick_guid = create_guid()
        brick_file_name = '{0}.hdf5'.format(brick_guid)

        #TODO: Inclusion of external links only used for external viewing of master file, remove if non-performant
        for parameter_name in self.parameter_metadata.keys():
            self._add_brick_link(parameter_name, brick_guid, brick_file_name)

        # Update the brick listing
        log.debug('Updating brick list[%s] with (%s, %s, %s, %s)', brick_guid, brick_extents, origin, tuple(bD), brick_active_size)
        brick_count = len(self.master_manager.brick_list)
        self.master_manager.brick_list[brick_guid] = [brick_extents, origin, tuple(bD), brick_active_size]
        log.debug('Brick count is %s', brick_count)

        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count, rtree_extents, brick_guid)
        self.master_manager.update_rtree(brick_count, rtree_extents, obj=brick_guid)

    # Expand the domain
    def expand_domain(self, total_extents, do_flush=False):
        """
        Expands a parameter's total domain based on the requested new temporal and/or spatial domains.
        Temporal domain expansion is most typical.
        Number of dimensions may not change for the parameter.

        @param total_extents    The total extents of the domain
        @return N/A
        """
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

        if self.master_manager.brick_domains[0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(total_extents) != len(self.master_manager.brick_domains[0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = self.master_manager.brick_domains[0]
                bD = self.master_manager.brick_domains[1]
                cD = self.master_manager.brick_domains[2]

                delta_domain = [(x - y) for x, y in zip(total_extents, tD)]
                log.debug('delta domain: %s', delta_domain)

                tD = [(x + y) for x, y in zip(tD, delta_domain)]
                self.master_manager.brick_domains[0] = tD
        else:
            tD = total_extents
            bricking_scheme = self.master_manager.brick_domains[3]
            bD,cD = self.calculate_brick_size(tD, bricking_scheme)
            self.master_manager.brick_domains = [tD, bD, cD, bricking_scheme]

        try:
            # Gather block list
            log.trace('tD, bD, cD: %s, %s, %s', tD, bD, cD)
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick origins
            need_origins = set(itertools.product(*lst))
            log.trace('need_origins: %s', need_origins)
            have_origins = set([v[1] for k,v in self.master_manager.brick_list.iteritems() if (v[2] == v[3])])
            log.trace('have_origins: %s', have_origins)
            need_origins.difference_update(have_origins)
            log.trace('need_origins: %s', need_origins)

            need_origins = list(need_origins)
            need_origins.sort()

            if len(need_origins)>0:
                log.debug('Number of Bricks to Create: %s', len(need_origins))

                # Write virtual HDF5 brick file
                for origin in need_origins:
                    rtree_extents, brick_extents, brick_active_size = self.calculate_extents(origin, bD, total_extents)

                    do_write, bguid = self._brick_exists_master(brick_extents)
                    if not do_write:
                        log.debug('Brick already exists!  Updating brick metadata...')
                        self.master_manager.brick_list[bguid] = [brick_extents, origin, tuple(bD), brick_active_size]
                    else:
                        self._write_brick(rtree_extents, brick_extents, brick_active_size, origin, bD)

            else:
                log.debug('No bricks to create to satisfy the domain expansion...')
        except Exception:
            raise

        ## .flush() is called by insert_timesteps - no need to call these here
        self.master_manager.flush()
        if do_flush:
            # If necessary (i.e. write_brick has been called), flush the master_manager461
            if self.master_manager.is_dirty():
                self.master_manager.flush()

    def shrink_domain(self, total_domain, do_flush=True):
        from coverage_model import bricking_utils
        # Find the last brick needed to contain the domain
        brick = bricking_utils.get_bricks_from_slice(total_domain, self.master_manager.brick_tree)

        bid, bguid = brick[0]

        # Get the brick_guids for all the bricks after the one we need
        rm_bricks = [s.value for s in self.master_manager.brick_tree._spans[bid+1:]]
        # Remove everything that comes after the brick we still need from the RTree
        self.master_manager.brick_tree._spans = self.master_manager.brick_tree._spans[:bid+1]

        # Remove the unnecessary bricks from the brick list
        for r in rm_bricks:
            del self.master_manager.brick_list[r]
            # and the file system...



        # Reset the first member of brick_domains
        self.master_manager.brick_domains[0] = list(total_domain)
        # And the appropriate entry in brick_list
        self.master_manager.brick_list[bguid] = tuple(self.master_manager.brick_list[bguid][:-1]) + ((total_domain[0] - self.master_manager.brick_list[bguid][1][0],),)

        if do_flush:
            if self.master_manager.is_dirty():
                self.master_manager.flush()

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

    def update_domain(self, tdom=None, sdom=None, do_flush=True):
        """
        Updates the temporal and/or spatial domain in the MasterManager.

        If do_flush is unspecified or True, the MasterManager is flushed within this call

        @param tdom     the value to update the Temporal Domain to
        @param sdom     the value to update the Spatial Domain to
        @param do_flush    Flush the MasterManager after updating the value(s); Default is True
        """
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

        # Update the global tdom & sdom as necessary
        if tdom is not None:
            self.master_manager.tdom = tdom
        if sdom is not None:
            self.master_manager.sdom = sdom

        if do_flush:
            self.master_manager.flush()

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


class PersistedStorage(AbstractStorage):
    """
    A concrete implementation of AbstractStorage utilizing the ParameterManager and brick dispatcher
    """

    def __init__(self, parameter_manager, master_manager, brick_dispatcher, dtype=None, fill_value=None, mode=None, inline_data_writes=True, auto_flush=True, **kwargs):
        """
        Constructor for PersistedStorage

        @param parameter_manager    ParameterManager object for the coverage
        @param master_manager   MasterManager object for the coverage
        @param brick_dispatcher BrickDispatcher object for the coverage
        @param dtype    Data type (HDF5/numpy) of parameter
        @param fill_value   HDF5/numpy compatible value based on dtype, returned if no value set within valid extent request
        @param auto_flush   Saves/flushes data to HDF5 files on every assignment
        @param kwargs   Additional keyword arguments
        @return N/A
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, dtype=dtype, fill_value=fill_value, **kwc)

        # Filesystem path to HDF brick file(s)
        self.brick_path = parameter_manager.root_dir

        from coverage_model.coverage import DomainSet
        self.total_domain = DomainSet(master_manager.tdom, master_manager.sdom)

        self.brick_tree = master_manager.brick_tree
        self.brick_list = master_manager.brick_list
        self.brick_domains = master_manager.brick_domains

        self._pending_values = {}
        self.brick_dispatcher = brick_dispatcher
        self.mode = mode
        self.inline_data_writes = inline_data_writes
        self.auto_flush = auto_flush

    def has_dirty_values(self):
        return len(self._pending_values) > 0

    def flush_values(self):
        if self.has_dirty_values():
            for k, v in self._pending_values.iteritems():
                wk, wm = k
                for vi in v:
                    self.brick_dispatcher.put_work(wk, wm, vi)

            self._pending_values = {}

    def _queue_work(self, work_key, work_metrics, work):
        wk = (work_key, work_metrics)
        if wk not in self._pending_values:
            self._pending_values[wk] = []

        self._pending_values[wk].append(work)

    def __getitem__(self, slice_):
        """
        Called to implement evaluation of self[slice_].

        Not implemented by the abstract class

        @param slice_   A set of valid constraints - int, [int,], (int,), or slice
        @return The value contained by the storage at location slice
        @raise  ValueError when brick contains no values for specified slice
        """
        from coverage_model import bricking_utils, utils

        extents = tuple([s for s in self.total_domain.total_extents if s != 0])
        if extents == ():  # Empty domain(s) - no data, return empty array
            return np.empty(0, dtype=self.dtype)

        # bricks is a list of tuples [(b_ord, b_guid), ...]
        slice_ = utils.fix_slice(deepcopy(slice_), extents)
        log.trace('slice_=%s', slice_)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.brick_tree, extents)
        log.trace('Found bricks: %s', bricks)

        ret_shp = utils.slice_shape(slice_, extents)
        log.trace('Return array shape: %s', ret_shp)
        ret_arr = np.empty(ret_shp, dtype=self.dtype)
        ret_arr.fill(self.fill_value)

        for b in bricks:
            # b is (brick_ordinal, brick_guid)
            _, bid = b
            # brick_list[brick_guid] contains: [brick_extents, origin, tuple(bD), brick_active_size]
            _, bori, _, bact = self.brick_list[bid]
            bbnds = []
            for i, bnd in enumerate(bori):
                bbnds.append((bori[i], bori[i] + bact[i] - 1))
            bbnds = tuple(bbnds)

            brick_slice, brick_mm = bricking_utils.get_brick_slice_nd(slice_, bbnds)
            log.trace('brick_slice=%s\tbrick_mm=%s', brick_slice, brick_mm)

            if None in brick_slice:
                log.debug('Brick does not contain any of the requested indices: Move to next brick')
                continue

            ret_slice = bricking_utils.get_value_slice_nd(slice_, ret_shp, bbnds, brick_slice, brick_mm)

            brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, bid)
            if not os.path.exists(brick_file_path):
                log.trace('Found virtual brick file: %s', brick_file_path)
            else:
                log.trace('Found real brick file: %s', brick_file_path)

                with HDFLockingFile(brick_file_path) as brick_file:
                    ret_vals = brick_file[bid][brick_slice]

                # Check if object type
                if self.dtype == '|O8':
                    if hasattr(ret_vals, '__iter__'):
                        ret_vals = [self._object_unpack_hook(x) for x in ret_vals]
                    else:
                        ret_vals = self._object_unpack_hook(ret_vals)

                ret_arr[ret_slice] = ret_vals

        # ret_arr = np.atleast_1d(ret_arr.squeeze())
        # ret_arr = np.atleast_1d(ret_arr)
        #
        # # If the array is size 1 AND a slice object was NOT part of the query
        # if ret_arr.size == 1 and not np.atleast_1d([isinstance(s, slice) for s in slice_]).all():
        #     ret_arr = ret_arr[0]

        return ret_arr

    def _object_unpack_hook(self, value):
        value = unpack(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        elif isinstance(value, tuple):
            return list(value)
        else:
            return value

    def __setitem__(self, slice_, value):
        """
        Called to implement assignment of self[slice_, value].

        Not implemented by the abstract class

        @param slice    A set of valid constraints - int, [int,], (int,), or slice
        @param value    The value to assign to the storage at location slice_
        @raise  ValueError when brick contains no values for specified slice
        """
        if self.mode == 'r':
            raise IOError('PersistenceLayer not open for writing: mode == \'{0}\''.format(self.mode))

        from coverage_model import bricking_utils, utils

        extents = tuple([s for s in self.total_domain.total_extents if s != 0])
        # bricks is a list of tuples [(b_ord, b_guid), ...]
        slice_ = utils.fix_slice(deepcopy(slice_), extents)
        log.trace('slice_=%s', slice_)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.brick_tree, extents)
        log.trace('Found bricks: %s', bricks)

        values = np.asanyarray(value)
        v_shp = values.shape
        log.trace('value_shape: %s', v_shp)
        s_shp = utils.slice_shape(slice_, extents)
        log.trace('slice_shape: %s', s_shp)
        is_broadcast = False
        if v_shp == ():
            log.trace('Broadcast!!')
            is_broadcast = True
            value_slice = ()
        elif v_shp != s_shp:
            if v_shp == tuple([i for i in s_shp if i != 1]): # Missing dimensions are singleton, just reshape to fit
                values = values.reshape(s_shp)
                v_shp = values.shape
            else:
                raise IndexError(
                    'Shape of \'value\' is not compatible with \'slice_\': slice_ shp == {0}\tvalue shp == {1}'.format(
                        s_shp, v_shp))
        else:
            value_slice = None

        log.trace('value_shape: %s', v_shp)

        for b in bricks:
            # b is (brick_ordinal, brick_guid)
            _, bid = b
            # brick_list[brick_guid] contains: [brick_extents, origin, tuple(bD), brick_active_size]
            _, bori, _, bact = self.brick_list[bid]
            bbnds = []
            bexts = []
            for i, bnd in enumerate(bori):
                bbnds.append((bori[i], bori[i] + bact[i] - 1))
                bexts.append(bori[i] + bact[i])
            bbnds = tuple(bbnds)
            bexts = tuple(bexts)
            log.trace('bid=%s, bbnds=%s, bexts=%s', bid, bbnds, bexts)

            log.trace('Determining slice for brick: %s', b)

            brick_slice, brick_mm = bricking_utils.get_brick_slice_nd(slice_, bbnds)
            log.trace('brick_slice=%s\tbrick_mm=%s', brick_slice, brick_mm)

            if None in brick_slice: # Brick does not contain any of the requested indices
                log.debug('Brick does not contain any of the requested indices: Move to next brick')
                continue

            try:
                brick_slice = utils.fix_slice(brick_slice, bexts)
            except IndexError:
                log.debug('Malformed brick_slice: move to next brick')
                continue

            if not is_broadcast:
                value_slice = bricking_utils.get_value_slice_nd(slice_, v_shp, bbnds, brick_slice, brick_mm)

                try:
                    value_slice = utils.fix_slice(value_slice, v_shp)
                except IndexError:
                    log.debug('Malformed value_slice: move to next brick')
                    continue

            log.trace('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s', b,
                      utils.slice_shape(brick_slice, bexts), brick_slice, brick_mm,
                      utils.slice_shape(value_slice, v_shp), value_slice)
            v = values[value_slice]

            self._set_values_to_brick(bid, brick_slice, v)

    def _set_values_to_brick(self, brick_guid, brick_slice, values, value_slice=None):
        brick_file_path = os.path.join(self.brick_path, '{0}.hdf5'.format(brick_guid))
        log.trace('Brick slice to fill: %s', brick_slice)
        log.trace('Value slice to extract: %s', value_slice)

        # Create the HDF5 dataset that represents one brick
        bD = tuple(self.brick_domains[1])
        cD = self.brick_domains[2]
        if value_slice is not None:
            vals = values[value_slice]
        else:
            vals = values

        if values.ndim == 0 and len(values.shape) == 0 and np.iterable(vals): # Prevent single value strings from being iterated
            vals = [vals]

        # Check for object type
        data_type = self.dtype
        fv = self.fill_value

        # Check for object type
        if data_type == '|O8':
            if np.iterable(vals):
                vals = [pack(x) for x in vals]
            else:
                vals = pack(vals)

        if self.inline_data_writes:
            if data_type == '|O8':
                data_type = h5py.special_dtype(vlen=str)
            if 0 in cD or 1 in cD:
                cD = True
            with HDFLockingFile(brick_file_path, 'a') as f:
                # TODO: Due to usage concerns, currently locking chunking to "auto"
                f.require_dataset(brick_guid, shape=bD, dtype=data_type, chunks=None, fillvalue=fv)
                f[brick_guid][brick_slice] = vals
        else:
            work_key = brick_guid
            work = (brick_slice, vals)
            work_metrics = (brick_file_path, bD, cD, data_type, fv)
            log.trace('Work key: %s', work_key)
            log.trace('Work metrics: %s', work_metrics)
            log.trace('Work[0]: %s', work[0])

            # If the brick file doesn't exist, 'touch' it to make sure it's immediately available
            if not os.path.exists(brick_file_path):
                if data_type == '|O8':
                    data_type = h5py.special_dtype(vlen=str)
                if 0 in cD or 1 in cD:
                    cD = True
                with HDFLockingFile(brick_file_path, 'a') as f:
                    # TODO: Due to usage concerns, currently locking chunking to "auto"
                    f.require_dataset(brick_guid, shape=bD, dtype=data_type, chunks=None, fillvalue=fv)

            if self.auto_flush:
                # Immediately submit work to the dispatcher
                self.brick_dispatcher.put_work(work_key, work_metrics, work)
            else:
                # Queue the work for later flushing
                self._queue_work(work_key, work_metrics, work)

    def expand(self, arrshp, origin, expansion, fill_value=None):
        pass # No op

    def fill(self, value):
        pass # No op

    def __len__(self):
        # TODO: THIS IS NOT CORRECT
        return 1

    def __iter__(self):
        # TODO: THIS IS NOT CORRECT
        return [1,1].__iter__()


class SparsePersistedStorage(AbstractStorage):

    def __init__(self, parameter_manager, master_manager, brick_dispatcher, dtype=None, fill_value=None, mode=None, inline_data_writes=True, auto_flush=True, **kwargs):
        """
        Constructor for PersistedStorage

        @param parameter_manager    ParameterManager object for the coverage
        @param master_manager   MasterManager object for the coverage
        @param brick_dispatcher BrickDispatcher object for the coverage
        @param dtype    Data type (HDF5/numpy) of parameter
        @param fill_value   HDF5/numpy compatible value based on dtype, returned if no value set within valid extent request
        @param auto_flush   Saves/flushes data to HDF5 files on every assignment
        @param kwargs   Additional keyword arguments
        @return N/A
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, dtype=dtype, fill_value=fill_value, **kwc)

        # Filesystem path to HDF brick file(s)
        self.brick_path = parameter_manager.root_dir

        from coverage_model.coverage import DomainSet
        self.total_domain = DomainSet(master_manager.tdom, master_manager.sdom)

        self.brick_tree = master_manager.brick_tree
        self.brick_list = master_manager.brick_list
        self.brick_domains = master_manager.brick_domains

        self._pending_values = {}
        self.brick_dispatcher = brick_dispatcher
        self.mode = mode
        self.inline_data_writes = inline_data_writes
        self.auto_flush = auto_flush

    def has_dirty_values(self):
        return len(self._pending_values) > 0

    def flush_values(self):
        if self.has_dirty_values():
            for k, v in self._pending_values.iteritems():
                wk, wm = k
                for vi in v:
                    self.brick_dispatcher.put_work(wk, wm, vi)

            self._pending_values = {}

    def _queue_work(self, work_key, work_metrics, work):
        wk = (work_key, work_metrics)
        if wk not in self._pending_values:
            self._pending_values[wk] = []

        self._pending_values[wk].append(work)

    def __getitem__(self, slice_):
        # Always storing in first slot - ignore slice
        if len(self.brick_list) == 0:
            raise ValueError('No Bricks!')

        bid = 'sparse_value_brick'

        brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, bid)

        if os.path.exists(brick_file_path):
            with HDFLockingFile(brick_file_path) as f:
                ret_vals = f[bid][0]
        else:
            ret_vals = None

        if ret_vals is None:
            return self.fill_value

        ret_vals = unpack(ret_vals)

        ret = [self.__deserialize(v) for v in ret_vals]

        return ret

    def __setitem__(self, slice_, value):
        # Always storing in first slot - ignore slice
        bid = 'sparse_value_brick'

        bD = (1,)
        cD = None
        brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, bid)

        vals = [self.__serialize(v) for v in value]

        vals = pack(vals)

        set_arr = np.empty(1, dtype=object)
        set_arr[0] = vals

        data_type = h5py.special_dtype(vlen=str)

        if self.inline_data_writes:
            with HDFLockingFile(brick_file_path, 'a') as f:
                f.require_dataset(bid, shape=bD, dtype=data_type, chunks=cD, fillvalue=None)
                f[bid][0] = set_arr
        else:
            work_key = bid
            work = ((0,), set_arr)
            work_metrics = (brick_file_path, bD, cD, data_type, None)

            # If the brick file doesn't exist, 'touch' it to make sure it's immediately available
            if not os.path.exists(brick_file_path):
                with HDFLockingFile(brick_file_path, 'a') as f:
                    # TODO: Due to usage concerns, currently locking chunking to "auto"
                    f.require_dataset(bid, shape=bD, dtype=data_type, chunks=cD, fillvalue=None)

            if self.auto_flush:
                # Immediately submit work to the dispatcher
                self.brick_dispatcher.put_work(work_key, work_metrics, work)
            else:
                # Queue the work for later flushing
                self._queue_work(work_key, work_metrics, work)

    def __deserialize(self, payload):
        if isinstance(payload, basestring) and payload.startswith('DICTABLE'):
            i = payload.index('|', 9)
            smod, sclass = payload[9:i].split(':')
            value = unpack(payload[i + 1:])
            module = __import__(smod, fromlist=[sclass])
            classobj = getattr(module, sclass)
            payload = classobj._fromdict(value)

        return payload

    def __serialize(self, payload):
        from coverage_model.basic_types import Dictable
        if isinstance(payload, Dictable):
            prefix = 'DICTABLE|{0}:{1}|'.format(payload.__module__, payload.__class__.__name__)
            payload = prefix + pack(payload.dump())

        return payload

    def expand(self, arrshp, origin, expansion, fill_value=None):
        pass # No op

    def fill(self, value):
        pass # No op

    def __len__(self):
        # TODO: THIS IS NOT CORRECT
        return 1

    def __iter__(self):
        # TODO: THIS IS NOT CORRECT
        return [1,1].__iter__()


class InMemoryPersistenceLayer(object):

    parameter_bounds = {}

    def expand_domain(self, *args, **kwargs):
        # No Op - storage expanded by *Value classes
        pass

    def shrink_domain(self, total_domain, do_flush=True):
        pass

    def init_parameter(self, parameter_context, *args, **kwargs):
        return InMemoryStorage(dtype=parameter_context.param_type.value_encoding, fill_value=parameter_context.param_type.fill_value)

    def has_dirty_values(self):
        # Never has dirty values
        return False

    def update_parameter_bounds(self, parameter_name, bounds):
        dmin, dmax = bounds
        if parameter_name in self.parameter_bounds:
            pmin, pmax = self.parameter_bounds[parameter_name]
            dmin = min(dmin, pmin)
            dmax = max(dmax, pmax)
        self.parameter_bounds[parameter_name] = (dmin, dmax)

    def get_dirty_values_async_result(self):
        from gevent.event import AsyncResult
        ret = AsyncResult()
        ret.set(True)
        return ret

    def update_domain(self, tdom=None, sdom=None, do_flush=True):
        # No Op
        pass

    def flush_values(self):
        # No Op
        pass

    def flush(self):
        # No Op
        pass

    def close(self, force=False, timeout=None):
        # No Op
        pass

