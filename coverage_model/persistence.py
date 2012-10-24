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
from coverage_model.parameter_types import FunctionType, ConstantType
from coverage_model.persistence_helpers import MasterManager, ParameterManager, pack, unpack
import numpy as np
import h5py
import os
import rtree
import itertools
from copy import deepcopy

# TODO: Make persistence-specific error classes
class PersistenceError(Exception):
    pass

class PersistenceLayer(object):
    def __init__(self, root, guid, name=None, tdom=None, sdom=None, bricking_scheme=None, auto_flush_values=True, **kwargs):
        """
        Constructor for Persistence Layer
        @param root: Where to save/look for HDF5 files
        @param guid: CoverageModel GUID
        @param name: CoverageModel Name
        @param tdom: Temporal Domain
        @param sdom: Spatial Domain
        @param kwargs:
        @return:
        """
        log.debug('Persistence GUID: %s', guid)
        root = '.' if root is ('' or None) else root

        self.master_manager = MasterManager(root, guid, name=name, tdom=tdom, sdom=sdom, global_bricking_scheme=bricking_scheme)

        self.auto_flush_values = auto_flush_values
        self.value_list = {}

        self.parameter_metadata = {} # {parameter_name: [brick_list, parameter_domains, rtree]}

        for pname in self.param_groups:
            log.debug('parameter group: %s', pname)
            self.parameter_metadata[pname] = ParameterManager(os.path.join(self.root_dir, self.guid, pname), pname)

        if self.master_manager.is_dirty():
            self.master_manager.flush()

        self.brick_dispatcher = BrickWriterDispatcher(self.write_failure_callback)
        self.brick_dispatcher.run()

        self._closed = False

        log.info('Persistence Layer Successfully Initialized')

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

    # CBM TODO: This needs to be improved greatly - should callback all the way to the Application layer as a "failure handler"
    def write_failure_callback(self, message, work):
        log.error('WORK DISCARDED!!!; %s: %s', message, work)

    def calculate_brick_size(self, tD, bricking_scheme):
        """
        Calculate brick domain size given a target file system brick size (Mbytes) and dtype
        @param tD:
        @param bricking_scheme:
        @return:
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
        parameter_name = parameter_context.name

        self.global_bricking_scheme = bricking_scheme

        pm = ParameterManager(os.path.join(self.root_dir, self.guid, parameter_name), parameter_name)
        self.parameter_metadata[parameter_name] = pm

        pm.parameter_context = parameter_context

        log.debug('Initialize %s', parameter_name)

        self.master_manager.create_group(parameter_name)

        log.debug('Performing Rtree dict setup')
        tD = parameter_context.dom.total_extents
        bD,cD = self.calculate_brick_size(tD, bricking_scheme) #remains same for each parameter
        # Verify domain is Rtree friendly
        tree_rank = len(bD)
        log.debug('tree_rank: %s', tree_rank)
        if tree_rank == 1:
            tree_rank += 1
        log.debug('tree_rank: %s', tree_rank)
        p = rtree.index.Property()
        p.dimension = tree_rank

        brick_tree = rtree.index.Index(properties=p)

        pm.brick_list = {}
        if isinstance(parameter_context.param_type, (FunctionType, ConstantType)):
            # These have constant storage, never expand!!
                pm.brick_domains = [(1,),(1,),(1,),bricking_scheme]
        else:
            pm.brick_domains = [tD, bD, cD, bricking_scheme]

        pm.tree_rank = tree_rank
        pm.brick_tree = brick_tree

        v = PersistedStorage(pm, self.brick_dispatcher, dtype=parameter_context.param_type.value_encoding, fill_value=parameter_context.param_type.fill_value, auto_flush=self.auto_flush_values)
        self.value_list[parameter_name] = v

        self.expand_domain(parameter_context)

        if pm.is_dirty():
            pm.flush()

        if self.master_manager.is_dirty():
            self.master_manager.flush()

        return v

    def calculate_extents(self, origin, bD, parameter_name):
        """
        Calculates the Rtree extents, brick extents and active brick size for the parameter
        @param origin:
        @param bD:
        @param parameter_name:
        @return:
        """
        log.debug('origin: %s', origin)
        log.debug('bD: %s', bD)
        log.debug('parameter_name: %s', parameter_name)

        # Calculate the brick extents
        origin = list(origin)

        pc = self.parameter_metadata[parameter_name].parameter_context
        total_extents = pc.dom.total_extents # index space
        log.debug('Total extents for parameter %s: %s', parameter_name, total_extents)

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

    def _brick_exists(self, parameter_name, brick_extents):
        # Make sure the brick doesn't already exist if we already have some bricks
        do_write = True
        brick_guid = ''
        log.debug('Check bricks for parameter \'%s\'',parameter_name)
        if parameter_name in self.parameter_metadata:
            for x,v in self.parameter_metadata[parameter_name].brick_list.iteritems():
                if brick_extents == v[0]:
                    log.debug('Brick found with matching extents: guid=%s', x)
                    do_write = False
                    brick_guid = x
                    break

        return do_write, brick_guid

    # Write empty HDF5 brick to the filesystem
    def write_brick(self, rtree_extents, brick_extents, brick_active_size, origin, bD, parameter_name):
        pm = self.parameter_metadata[parameter_name]

#        rtree_extents, brick_extents, brick_active_size = self.calculate_extents(origin, bD, parameter_name)
#
#        do_write, bguid = self._brick_exists(parameter_name, brick_extents)
#        if not do_write:
#            log.debug('Brick already exists!  Updating brick metadata...')
#            pm.brick_list[bguid] = [brick_extents, origin, tuple(bD), brick_active_size]
#        else:
        log.debug('Writing virtual brick for parameter %s', parameter_name)

        # Set HDF5 file and group
        # Create a GUID for the brick
        brick_guid = create_guid()
        brick_file_name = '{0}.hdf5'.format(brick_guid)
        brick_rel_path = os.path.join(pm.root_dir.replace(self.root_dir,'.'), brick_file_name)
        link_path = '/{0}/{1}'.format(parameter_name, brick_guid)

        # Add brick to Master HDF file
        self.master_manager.add_external_link(link_path, brick_rel_path, brick_guid)

        # Update the brick listing
        log.debug('Updating brick list[%s] with (%s, %s)', parameter_name, brick_guid, brick_extents)
        brick_count = self.parameter_brick_count(parameter_name)
        pm.brick_list[brick_guid] = [brick_extents, origin, tuple(bD), brick_active_size]
        log.debug('Brick count for %s is %s', parameter_name, brick_count)

        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count, rtree_extents, brick_guid)
        pm.update_rtree(brick_count, rtree_extents, obj=brick_guid)

        # Flush the parameter_metadata
        if pm.is_dirty():
            pm.flush()

        if self.master_manager.is_dirty():
            self.master_manager.flush()

    # Expand the domain
    def expand_domain(self, parameter_context, tdom=None, sdom=None):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)
        pm = self.parameter_metadata[parameter_name]

        if pm.brick_domains[0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(parameter_context.dom.total_extents) != len(pm.brick_domains[0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = pm.brick_domains[0]
                bD = pm.brick_domains[1]
                cD = pm.brick_domains[2]
                if not isinstance(pm.parameter_context.param_type, (FunctionType, ConstantType)): # These have constant storage, never expand!!
                    new_domain = parameter_context.dom.total_extents

                    delta_domain = [(x - y) for x, y in zip(new_domain, tD)]
                    log.debug('delta domain: %s', delta_domain)

                    tD = [(x + y) for x, y in zip(tD, delta_domain)]
                    pm.brick_domains[0] = tD
        else:
            tD = parameter_context.dom.total_extents
            bricking_scheme = pm.brick_domains[3]
            bD,cD = self.calculate_brick_size(tD, bricking_scheme)
            pm.brick_domains = [tD, bD, cD, bricking_scheme]

        try:
            # Gather block list
            log.trace('tD, bD, cD: %s, %s, %s', tD, bD, cD)
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick origins
            need_origins = set(itertools.product(*lst))
            log.trace('need_origins: %s', need_origins)
            have_origins = set([v[1] for k,v in pm.brick_list.iteritems() if v[2] == v[3]])
            log.trace('have_origins: %s', have_origins)
            need_origins.difference_update(have_origins)
            log.trace('need_origins: %s', need_origins)

            need_origins = list(need_origins)
            need_origins.sort()

            if len(need_origins)>0:
                log.debug('Number of Bricks to Create: %s', len(need_origins))

#                # Write brick to HDF5 file
#                map(lambda origin: self.write_brick(origin,bD,parameter_name), need_origins)

                # Write brick to HDF5 file
                for origin in need_origins:
                    rtree_extents, brick_extents, brick_active_size = self.calculate_extents(origin, bD, parameter_name)

                    do_write, bguid = self._brick_exists(parameter_name, brick_extents)
                    if not do_write:
                        log.debug('Brick already exists!  Updating brick metadata...')
                        pm.brick_list[bguid] = [brick_extents, origin, tuple(bD), brick_active_size]
                    else:
                        self.write_brick(rtree_extents, brick_extents, brick_active_size, origin, bD, parameter_name)

            else:
                log.debug('No bricks to create to satisfy the domain expansion...')
        except Exception:
            raise

        # Flush the parameter_metadata
        if pm.is_dirty():
            pm.flush()

        # Update the global tdom & sdom as necessary
        if tdom is not None:
            self.master_manager.tdom = tdom
        if sdom is not None:
            self.master_manager.sdom = sdom

        if self.master_manager.is_dirty():
            self.master_manager.flush()

    # Returns a count of bricks for a parameter
    def parameter_brick_count(self, parameter_name):
        ret = 0
        if parameter_name in self.parameter_metadata:
            ret = len(self.parameter_metadata[parameter_name].brick_list)
        else:
            log.debug('No bricks found for parameter: %s', parameter_name)

        return ret

    def has_dirty_values(self):
        for v in self.value_list.itervalues():
            if v.has_dirty_values():
                return True

        return False

    def get_dirty_values_async_result(self):
        return self.brick_dispatcher.get_dirty_values_async_result()

    def flush_values(self):
        for k, v in self.value_list.iteritems():
            v.flush_values()

        return self.get_dirty_values_async_result()

    def flush(self):
        for pk, pm in self.parameter_metadata.iteritems():
            log.debug('Flushing ParameterManager for \'%s\'...', pk)
            pm.flush()
        log.debug('Flushing MasterManager...')
        self.master_manager.flush()

    def close(self, force=False, timeout=None):
        if not self._closed:
            self.flush()
            self.brick_dispatcher.shutdown(force=force, timeout=timeout)

        self._closed = True

class PersistedStorage(AbstractStorage):

    def __init__(self, parameter_manager, brick_dispatcher, dtype=None, fill_value=None, auto_flush=True, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractStorage; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, dtype=dtype, fill_value=fill_value, **kwc)

        # Rtree of bricks for parameter
        self.brick_tree = parameter_manager.brick_tree

        # Filesystem path to HDF brick file(s)
        self.brick_path = parameter_manager.root_dir

        # Listing of bricks and their metadata for parameter
        self.brick_list = parameter_manager.brick_list

        self.brick_domains = parameter_manager.brick_domains

        self._pending_values = {}
        self.brick_dispatcher = brick_dispatcher
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

    # Calculates the bricks from Rtree (brick_tree) using the slice_
    def _bricks_from_slice(self, slice_):
        # Make sure we don't modify the global slice_ object
        sl = deepcopy(slice_)

        # Ensure the slice_ is iterable
        if not isinstance(sl, (list,tuple)):
            sl = [sl]

        # Check the rank of the slice and pad if == 1 to satisfy Rtree requirement of rank >= 2
        rank = len(sl)
        if rank == 1:
            rank += 1
            sl += (0,)

        if self.brick_tree.properties.dimension != rank:
            raise ValueError('slice_ is of incorrect rank: is {0}, must be {1}'.format(rank, self.brick_tree.properties.dimension))

        bnds = self.brick_tree.bounds

        # Perform the calculations for the slice_ start and stop bounds in Rtree format
        # CBM TODO: INACCURATE BRICK SELECTION --> THIS IS NOT LIKELY TO WORK FOR ANY RANK > 1!!!
        start = []
        end = []
        for x in xrange(rank):
            sx=sl[x]
            if isinstance(sx, slice):
                si=sx.start if sx.start is not None else bnds[x::rank][0]
                start.append(si)
                ei=sx.stop-1 if sx.stop is not None else bnds[x::rank][1]
                end.append(ei)
            elif isinstance(sx, (list, tuple)):
                start.append(min(sx))
                end.append(max(sx))
            elif isinstance(sx, int):
                start.append(sx)
                end.append(sx)

        hits = list(self.brick_tree.intersection(tuple(start+end), objects=True))
        ret = [(h.id,h.object) for h in hits]
        ret.sort()
        return ret

    def __getitem__(self, slice_):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('getitem slice_: %s', slice_)

        arr_shp = self._get_array_shape_from_slice(slice_)

        ret_arr = np.empty(arr_shp, dtype=self.dtype)
        ret_arr.fill(self.fill_value)
        ret_origin = [0 for x in range(ret_arr.ndim)]
        log.trace('Shape of returned array: %s', ret_arr.shape)

        brick_origin_offset = 0

        bricks = self._bricks_from_slice(slice_)
        log.trace('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)

            # Figuring out which part of brick to set values - also appropriately increments the ret_origin
            log.trace('Return array origin: %s', ret_origin)
            try:
                brick_slice, value_slice, brick_origin_offset = self._calc_slices(slice_, brick_guid, ret_arr, ret_origin, brick_origin_offset)
                if brick_slice is None:
                    raise ValueError('Brick contains no values for specified slice')
            except ValueError as ve:
                log.warn(ve.message + '; moving to next brick')
                continue

            log.trace('Brick slice to extract: %s', brick_slice)
            log.trace('Value slice to fill: %s', value_slice)

            if not os.path.exists(brick_file_path):
                log.trace('Found virtual brick file: %s', brick_file_path)
            else:
                log.trace('Found real brick file: %s', brick_file_path)

                with h5py.File(brick_file_path) as brick_file:
                    v = brick_file[brick_guid].__getitem__(*brick_slice)

                # Check if object type
                if self.dtype == '|O8':
                    if not hasattr(v, '__iter__'):
                        v = [v]
                    v = [unpack(x) for x in v]

                ret_arr[value_slice] = v

        if ret_arr.size == 1:
            if ret_arr.ndim==0:
                ret_arr=ret_arr[()]
            else:
                ret_arr=ret_arr[0]
        return ret_arr

    def __setitem__(self, slice_, value):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('setitem slice_: %s', slice_)
        val = np.asanyarray(value)
        val_origin = [0 for x in range(val.ndim)]

        brick_origin_offset = 0

        bricks = self._bricks_from_slice(slice_)
        log.trace('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            # Figuring out which part of brick to set values
            try:
                brick_slice, value_slice, brick_origin_offset = self._calc_slices(slice_, brick_guid, val, val_origin, brick_origin_offset)
                log.trace('brick_slice: %s, value_slice: %s, brick_origin_offset: %s', brick_slice, value_slice, brick_origin_offset)
                if brick_slice is None:
                    raise ValueError('Brick contains no values for specified slice')
            except ValueError as ve:
                log.warn(ve.message + '; moving to next brick')
                continue

            brick_file_path = os.path.join(self.brick_path, '{0}.hdf5'.format(brick_guid))
            log.trace('Brick slice to fill: %s', brick_slice)
            log.trace('Value slice to extract: %s', value_slice)

            # Create the HDF5 dataset that represents one brick
            bD = tuple(self.brick_domains[1])
            cD = self.brick_domains[2]
            v = val[value_slice]
            if val.ndim == 0 and len(val.shape) == 0 and np.iterable(v): # Prevent single value strings from being iterated
                v = [v]

            # Check for object type
            data_type = self.dtype
            fv = self.fill_value

            # Check for object type
            if data_type == '|O8':
                if np.iterable(v):
                    v = [pack(x) for x in v]
                else:
                    v = pack(v)

            work_key = brick_guid
            work = (brick_slice, v)
            work_metrics = (brick_file_path, bD, cD, data_type, fv)
            log.trace('Work key: %s', work_key)
            log.trace('Work metrics: %s', work_metrics)
            log.trace('Work[0]: %s', work[0])

            # If the brick file doesn't exist, 'touch' it to make sure it's immediately available
            if not os.path.exists(brick_file_path):
                if data_type == '|O8':
                    data_type = h5py.special_dtype(vlen=str)
                # TODO: Uncomment this to properly turn 0 & 1 chunking into True
                if 0 in cD or 1 in cD:
                    cD = True
                with h5py.File(brick_file_path, 'a') as f:
                    # TODO: Due to usage concerns, currently locking chunking to "auto"
                    f.require_dataset(brick_guid, shape=bD, dtype=data_type, chunks=True, fillvalue=fv)

            #region FOR TESTING WITHOUT OUT-OF-BAND WRITES - IN-LINE WRITING OF VALUES
#            if data_type == '|O8':
#                data_type = h5py.special_dtype(vlen=str)
#                # TODO: Uncomment this to properly turn 0 & 1 chunking into True
#            if 0 in cD or 1 in cD:
#                cD = True
#            with h5py.File(brick_file_path, 'a') as f:
#                # TODO: Due to usage concerns, currently locking chunking to "auto"
#                f.require_dataset(brick_guid, shape=bD, dtype=data_type, chunks=True, fillvalue=fv)
#            if isinstance(brick_slice, tuple):
#                brick_slice = list(brick_slice)
#
#            f[brick_guid].__setitem__(*brick_slice, val=v)
            #endregion

            if self.auto_flush:
                # Immediately submit work to the dispatcher
                self.brick_dispatcher.put_work(work_key, work_metrics, work)
            else:
                # Queue the work for later flushing
                self._queue_work(work_key, work_metrics, work)

    def _calc_slices(self, slice_, brick_guid, value, val_origin, brick_origin_offset=0):
        brick_origin, _, brick_size = self.brick_list[brick_guid][1:]
        log.debug('Brick %s:  origin=%s, size=%s', brick_guid, brick_origin, brick_size)
        log.debug('Slice set: %s', slice_)

        brick_slice = []
        value_slice = []

        # Get the value into a numpy array - should do all the heavy lifting of sorting out what's what!!!
        val_arr = np.asanyarray(value)
        val_shp = val_arr.shape
        val_rank = val_arr.ndim
        log.trace('Value asanyarray: rank=%s, shape=%s', val_rank, val_shp)

        if val_origin is None or len(val_origin) == 0:
            val_ori = [0 for x in range(len(slice_))]
        else:
            val_ori = val_origin

        for i, sl in enumerate(slice_):
            bo=brick_origin[i]
            bs=brick_size[i]
            bn=bo+bs
            vo=val_ori[i]
            vs=val_shp[i] if len(val_shp) > 0 else None
            log.trace('i=%s, sl=%s, bo=%s, bs=%s, bn=%s, vo=%s, vs=%s',i,sl,bo,bs,bn,vo,vs)
            if isinstance(sl, int):
                if bo <= sl < bn: # The slice is within the bounds of the brick
                    brick_slice.append(sl-bo) # brick_slice is the given index minus the brick origin
                    value_slice.append(0 + vo)
                    val_ori[i] = vo + 1
                else:
                    raise ValueError('Specified index is not within the brick: {0}'.format(sl))
            elif isinstance(sl, (list,tuple)):
                lb = [x - bo for x in sl if bo <= x < bn]
                if len(lb) == 0:
                    raise ValueError('None of the specified indices are within the brick: {0}'.format(sl))
                brick_slice.append(lb)
                value_slice.append(slice(vo, vo+len(lb), None)) # Everything from the appropriate index to the size needed
                val_ori[i] = len(lb) + vo
            elif isinstance(sl, slice):
                if sl.start is None:
                    start = 0
                else:
                    if bo <= sl.start < bn:
                        start = sl.start - bo
                    elif bo > sl.start:
                        start = 0
                    else: #  sl.start > bn
                        raise ValueError('The slice is not contained in this brick (sl.start > bn)')

                if sl.stop is None:
                    stop = bs
                else:
                    if bo < sl.stop <= bn:
                        stop = sl.stop - bo
                    elif sl.stop > bn:
                        stop = bs
                    else: #  bo > sl.stop
                        raise ValueError('The slice is not contained in this brick (bo > sl.stop)')

                log.trace('start=%s, stop=%s', start, stop)
                log.trace('brick_origin_offset=%s', brick_origin_offset)
                start += brick_origin_offset
                log.trace('start=%s, stop=%s', start, stop)
                nbs = slice(start, stop, sl.step)
                nbsi = range(*nbs.indices(stop))
                nbsl = len(nbsi)
                if nbsl == 0: # No values in this brick!!
                    if sl.step is not None:
                        brick_origin_offset = brick_origin_offset - bs
                    return None, None, brick_origin_offset
                last_index = nbsi[-1]
                log.trace('last_index=%s',last_index)
                log.trace('nbsl=%s',nbsl)
                # brick_origin_offset should make this check unnecessary!!
#                if vs is not None and vs != vo:
#                    while nbsl > (vs-vo):
#                        stop -= 1
#                        nbs = slice(start, stop, sl.step)
#                        nbsl = len(range(*nbs.indices(stop)))
#                log.trace('nbsl=%s',nbsl)

                brick_slice.append(nbs)
                vstp = vo+nbsl
                log.trace('vstp=%s',vstp)
                if vs is not None and vstp > vs: # Don't think this will ever happen, should be dealt with by active_brick_size logic...
                    log.warn('Value set not the proper size for setter slice (vstp > vs)!')
#                    vstp = vs

                value_slice.append(slice(vo, vstp, None))
                val_ori[i] = vo + nbsl

                if sl.step is not None:
                    brick_origin_offset = last_index - bs + sl.step
                    log.trace('brick_origin_offset = %s', brick_origin_offset)

        if val_origin is not None and len(val_origin) != 0:
            val_origin = val_ori
        else:
            if val_arr.ndim == 0 and len(val_arr.shape) == 0:
                value_slice = ()
            else:
                value_slice = brick_slice

        return brick_slice, value_slice, brick_origin_offset
    # TODO: Does not support n-dimensional
    def _get_array_shape_from_slice(self, slice_):
        log.debug('Getting array shape for slice_: %s', slice_)

        vals = self.brick_list.values()
        log.trace('vals: %s', vals)
        # Calculate the min and max brick value indices for each dimension
        if len(vals[0][1]) > 1:
            min_len = min([min(*x[0][i])+1 for i,x in enumerate(vals)])
            max_len = max([min(*x[0][i])+min(x[3]) for i,x in enumerate(vals)])
        else:
            min_len = min([min(*x[0])+1 for i,x in enumerate(vals)])
            max_len = max([min(*x[0])+min(x[3]) for i,x in enumerate(vals)])

        maxes = [max_len, min_len]

        # Calculate the shape base on the type of slice_
        shp = []
        for i, s in enumerate(slice_):
            if isinstance(s, int):
                shp.append(1)
            elif isinstance(s, (list,tuple)):
                shp.append(len(s))
            elif isinstance(s, slice):
                shp.append(len(range(*s.indices(maxes[i])))) # TODO: Does not support n-dimensional

        return tuple(shp)

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

    def expand_domain(self, *args, **kwargs):
        # No Op - storage expanded by *Value classes
        pass

    def init_parameter(self, parameter_context, *args, **kwargs):
        return InMemoryStorage(dtype=parameter_context.param_type.value_encoding, fill_value=parameter_context.param_type.fill_value)

    def has_dirty_values(self):
        # Never has dirty values
        return False

    def get_dirty_values_async_result(self):
        from gevent.event import AsyncResult
        ret = AsyncResult()
        ret.set(True)
        return ret

    def flush_values(self):
        # No Op
        pass

    def flush(self):
        # No Op
        pass

    def close(self):
        # No Op
        pass
