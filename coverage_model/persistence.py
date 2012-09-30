#!/usr/bin/env python

"""
@package coverage_model.persistence
@file coverage_model/persistence.py
@author James Case
@brief The core classes comprising the Persistence Layer
"""

from pyon.public import log
from coverage_model.basic_types import create_guid, AbstractStorage, InMemoryStorage
from coverage_model.parameter import ParameterDictionary
import numpy as np
import h5py
import os
import rtree
import itertools
import msgpack
from copy import deepcopy

# TODO: Make persistence-specific error classes
class PersistenceError(Exception):
    pass

class PersistenceLayer():
    def __init__(self, root, guid, **kwargs):
        """
        Constructor for Persistence Layer
        @param root: Where to save/look for HDF5 files
        @param guid: CoverageModel GUID
        @param parameter_dictionary: CoverageModel ParameterDictionary
        @param tdom: Temporal Domain
        @param sdom: Spatial Domain
        @param kwargs:
        @return:
        """
        self.root = '.' if root is ('' or None) else root
        self.guid = guid
        log.debug('Persistence GUID: %s', self.guid)

        self.parameter_metadata = {} # {parameter_name: [brick_list, parameter_domains, rtree]}
        self.parameter_dictionary = ParameterDictionary()

        # Setup Master HDF5 File using the root path and the supplied coverage guid
        self.master_file_path = '{0}/{1}_master.hdf5'.format(self.root,self.guid)

        # Make sure the root path exists, if not, make it
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Check if the master file already exists, if not, create it
        if not os.path.exists(self.master_file_path):
            master_file = h5py.File(self.master_file_path, 'w')
            master_file.close()

        # Peek inside the master file and populate master_groups and master_links
        # TODO:  This is where the brick_list and brick_tree_dict will be populated if the master file exists
        self.master_groups = []
        self.master_links = {}
        self._inspect_master()

        self.value_list = {}

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
        self.parameter_metadata[parameter_name] = {}
        log.debug('Initialize %s', parameter_name)

        self.parameter_dictionary.add_context(parameter_context)

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

        self.parameter_metadata[parameter_name][0] = {} # brick_list {brick_guid: [brick_extents, origin, tuple(bD), brick_active_size]
        self.parameter_metadata[parameter_name][1] = [tD, bD, cD, bricking_scheme] # brick_domain_dict [tD, bD, cD, bricking_scheme]
        self.parameter_metadata[parameter_name][2] = brick_tree # brick_tree

        # TODO: Sort out the path to the bricks for this parameter
        brick_path = '{0}/{1}/{2}'.format(self.root, self.guid, parameter_name)
        v = PersistedStorage(brick_path=brick_path, brick_tree=self.parameter_metadata[parameter_name][2], brick_list=self.parameter_metadata[parameter_name][0], dtype=parameter_context.param_type.value_encoding)
        self.value_list[parameter_name] = v

        self.expand_domain(parameter_context)

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

        pc = self.parameter_dictionary.get_context(parameter_name)
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

        return rtree_extents, brick_extents, brick_active_size

    def _brick_exists(self, parameter_name, brick_extents):
        # Make sure the brick doesn't already exist if we already have some bricks
        do_write = True
        brick_guid = ''
        log.debug('Check bricks for parameter \'%s\'',parameter_name)
        if parameter_name in self.parameter_metadata:
            for x,v in self.parameter_metadata[parameter_name][0].iteritems():
                if brick_extents == v[0]:
                    log.debug('Brick found with matching extents: guid=%s', x)
                    do_write = False
                    brick_guid = x
                    break

        return do_write, brick_guid

    # Write empty HDF5 brick to the filesystem
    def write_brick(self,origin,bD,cD,parameter_name,data_type):
        rtree_extents, brick_extents, brick_active_size = self.calculate_extents(origin, bD, parameter_name)

        do_write, bguid = self._brick_exists(parameter_name, brick_extents)
        if not do_write:
            log.debug('Brick already exists!  Do not write')
            # TODO: We need to update the brick_list's brick_active_size here
            self.parameter_metadata[parameter_name][0][bguid] = [brick_extents, origin, tuple(bD), brick_active_size]
            return

        log.debug('Writing brick for parameter %s', parameter_name)

        root_path = '{0}/{1}/{2}'.format(self.root,self.guid,parameter_name)

        # Create the root path if it does not exist
        # TODO: Eliminate possible race condition
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        # Create a GUID for the brick
        brick_guid = create_guid()

        # Set HDF5 file and group
        brick_file_name = '{0}.hdf5'.format(brick_guid)
        brick_file_path = '{0}/{1}'.format(root_path,brick_file_name)
        brick_file = h5py.File(brick_file_path, 'w')

        # Check for object type
        if data_type == '|O8':
            data_type = h5py.new_vlen(str)

        # Create the HDF5 dataset that represents one brick
        brick_cubes = brick_file.create_dataset('{0}'.format(brick_guid), bD, dtype=data_type, chunks=cD)

        # Close the HDF5 file that represents one brick
        log.debug('Size Before Close: %s', os.path.getsize(brick_file_path))
        brick_file.close()
        log.debug('Size After Close: %s', os.path.getsize(brick_file_path))

        # Add brick to Master HDF file
        log.debug('Adding %s external link to %s.', brick_file_path, self.master_file_path)
        _master_file = h5py.File(self.master_file_path, 'r+')
        _master_file['/{0}/{1}'.format(parameter_name, brick_guid)] = h5py.ExternalLink('./{0}/{1}/{2}'.format(self.guid, parameter_name, brick_file_name), brick_guid)

        # Update the brick listing
        log.debug('Updating brick list[%s] with (%s, %s)', parameter_name, brick_guid, brick_extents)
        brick_count = self.parameter_brick_count(parameter_name)
        self.parameter_metadata[parameter_name][0][brick_guid] = [brick_extents, origin, tuple(bD), brick_active_size]
        log.debug('Brick count for %s is %s', parameter_name, brick_count)

        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count, rtree_extents, brick_guid)
        self.parameter_metadata[parameter_name][2].insert(brick_count, rtree_extents, obj=brick_guid)
        log.debug('Rtree inserted successfully.')

        # Brick metadata
        _master_file['{0}/{1}'.format(parameter_name, brick_guid)].attrs['dirty'] = 0
        _master_file['{0}/{1}'.format(parameter_name, brick_guid)].attrs['brick_origin'] = str(origin)
        _master_file['{0}/{1}'.format(parameter_name, brick_guid)].attrs['brick_size'] = str(tuple(bD))

        log.debug('Brick Size At End: %s', os.path.getsize(brick_file_path))

        # TODO: This is a really ugly way to store the brick_list for a parameter
#        brick_list_attrs_filtered = [(k,v) for k, v in self.brick_list.items() if k[0]==parameter_name]
#        str_size = 'S'+str(len(str(brick_list_attrs_filtered[0])))
#        brick_list_attrs = np.ndarray(tuple([len(brick_list_attrs_filtered)]), str_size)
#        for i,v in enumerate(brick_list_attrs_filtered):
#            brick_list_attrs[i] = str(v)
#        _master_file[brick_group_path].attrs['brick_list'] = brick_list_attrs

        # Close the master file
        _master_file.close()

    # Expand the domain
    # TODO: Verify brick and chunk sizes are still valid????
    def expand_domain(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)

        if self.parameter_metadata[parameter_name][1][0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(parameter_context.dom.total_extents) != len(self.parameter_metadata[parameter_name][1][0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = self.parameter_metadata[parameter_name][1][0]
                bD = self.parameter_metadata[parameter_name][1][1]
                cD = self.parameter_metadata[parameter_name][1][2]
                new_domain = parameter_context.dom.total_extents

                delta_domain = [(x - y) for x, y in zip(new_domain, tD)]
                log.debug('delta domain: %s', delta_domain)

                tD = [(x + y) for x, y in zip(tD, delta_domain)]
                self.parameter_metadata[parameter_name][1][0] = tD
        else:
            tD = parameter_context.dom.total_extents
            bricking_scheme = self.parameter_metadata[parameter_name][1][3]
            bD,cD = self.calculate_brick_size(tD, bricking_scheme)
            self.parameter_metadata[parameter_name][1] = [tD, bD, cD, bricking_scheme]

        try:
            # Gather block list
            log.debug('tD, bD, cD: %s, %s, %s', tD, bD, cD)
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: %s', len(vertices))

                # Write brick to HDF5 file
                map(lambda origin: self.write_brick(origin,bD,cD,parameter_name,parameter_context.param_type.value_encoding), vertices)

                # Refresh master file object data
                self._inspect_master()

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain is empty...')
        except Exception:
            raise

    # Returns a count of bricks for a parameter
    def parameter_brick_count(self, parameter_name):
        ret = 0
        if parameter_name in self.parameter_metadata:
            ret = len(self.parameter_metadata[parameter_name][0])
        else:
            log.debug('No bricks found for parameter: %s', parameter_name)

        return ret

    def _inspect_master(self):
        # Open the master file
        _master_file = h5py.File(self.master_file_path, 'r+')

        # Get all the groups
        self.master_groups = []
        _master_file.visit(self.master_groups.append)

        # Get all parameter's external links to datasets
        self.master_links = {}
        for g in self.master_groups:
            grp = _master_file[g]
            for v in grp.values():
                if isinstance(v,h5py.Dataset):
                    self.master_links[v.name] = v.file.filename
                    v.file.close()

        # TODO: Generate/Update the brick_list

        # TODO: Generate/Update the brick_tree_dict

        # Close the master file
        _master_file.close()

class PersistedStorage(AbstractStorage):

    def __init__(self, brick_path, brick_tree, brick_list, dtype, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractStorage; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, **kwc)

        # Rtree of bricks for parameter
        self.brick_tree = brick_tree

        # Filesystem path to HDF brick file(s)
        self.brick_path = brick_path

        # Listing of bricks and their metadata for parameter
        self.brick_list = brick_list

        # Data type for parameter
        self.dtype = dtype

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
        return [(h.id,h.object) for h in hits]

    def __getitem__(self, slice_):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('getitem slice_: %s', slice_)

        arr_shp = self._get_array_shape_from_slice(slice_)

        ret_arr = np.empty(arr_shp, dtype=self.dtype)
        ret_arr.fill(-1)
        ret_origin = [0 for x in range(ret_arr.ndim)]
        log.debug('Shape of returned array: %s', ret_arr.shape)

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
            if not os.path.exists(brick_file):
                raise IndexError('Expected brick_file \'%s\' not found', brick_file)
            log.debug('Found brick file: %s', brick_file)

            # Figuring out which part of brick to set values
            log.debug('Return array origin: %s', ret_origin)
            brick_slice, value_slice = self._calc_slices(slice_, brick_guid, ret_arr, ret_origin)
            log.debug('Brick slice to extract: %s', brick_slice)
            log.debug('Value slice to fill: %s', value_slice)

            bf = h5py.File(brick_file, 'r+')
            ds_path = '/{0}'.format(brick_guid)

            v = bf[ds_path].__getitem__(*brick_slice)

            # Check if object type
            if self.dtype == '|O8':
                if not hasattr(v, '__iter__'):
                    v = [v]
                v = [msgpack.unpackb(x) for x in v]

            ret_arr[value_slice] = v

            bf.close()

#            log.error(ret_arr)

        return ret_arr

    def __setitem__(self, slice_, value):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('setitem slice_: %s', slice_)
        val = np.asanyarray(value)
        val_origin = [0 for x in range(val.ndim)]

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
            if os.path.exists(brick_file):
                log.debug('Found brick file: %s', brick_file)

                # Figuring out which part of brick to set values
                brick_slice, value_slice = self._calc_slices(slice_, brick_guid, val, val_origin)
                log.debug('Brick slice to fill: %s', brick_slice)
                log.debug('Value slice to extract: %s', value_slice)

                # TODO: Move this to writer function

                bf = h5py.File(brick_file, 'r+')

                ds_path = '/{0}'.format(brick_guid)

#                log.error('BEFORE: %s', bf[ds_path][:])

                v = val if value_slice is None else val[value_slice]

                # Check for object type
                if self.dtype == '|O8':
                    if not hasattr(v, '__iter__'):
                        v = [v]
                    v = [msgpack.packb(x) for x in v]

                bf[ds_path].__setitem__(*brick_slice, val=v)

                bf.flush()

#                log.error('AFTER: %s', bf[ds_path][:])

                bf.close()

            else:
                raise SystemError('Can\'t find brick: %s', brick_file)

    def _calc_slices(self, slice_, brick_guid, value, val_origin):
        brick_origin, _, brick_size = self.brick_list[brick_guid][1:]
        log.debug('Brick %s:  origin=%s, size=%s', brick_guid, brick_origin, brick_size)
        log.debug('Slice set: %s', slice_)

        brick_slice = []
        value_slice = []

        # Get the value into a numpy array - should do all the heavy lifting of sorting out what's what!!!
        val_arr = np.asanyarray(value)
        val_shp = val_arr.shape
        val_rank = val_arr.ndim
        log.debug('Value asanyarray: rank=%s, shape=%s', val_rank, val_shp)

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
            log.debug('i=%s, sl=%s, bo=%s, bs=%s, bn=%s, vo=%s, vs=%s',i,sl,bo,bs,bn,vo,vs)
            if isinstance(sl, int):
                if bo <= sl < bn: # The slice is within the bounds of the brick
                    brick_slice.append(sl-bo) # brick_slice is the given index minus the brick origin
                    value_slice.append(0 + vo)
                    val_ori[i] = vo + 1
                else: # TODO: If you specify a brick boundary this occurs [10] or [6]
                    raise ValueError('Specified index is not within the brick: {0}'.format(sl))
            elif isinstance(sl, (list,tuple)):
                lb = [x - bo for x in sl if bo <= x < bn]
                if len(lb) == 0: # TODO: In a list the last brick boundary seems to break it [1,3,10]
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

                log.debug('start=%s, stop=%s', start, stop)
                nbs = slice(start, stop, sl.step)
                brick_slice.append(nbs)
                nbsl = len(range(*nbs.indices(stop)))
                log.debug('nbsl=%s',nbsl)
                vstp = vo+nbsl
                log.debug('vstp=%s',vstp)
                if vs is not None and vstp > vs: # Don't think this will ever happen, caught by 'if vs is not None:' above
                    log.warn('Value set not the proper size for setter slice (vstp > vs)!')
                    vstp = vs

                value_slice.append(slice(vo, vstp, None))
                val_ori[i] = vo + nbsl

        if val_origin is not None and len(val_origin) != 0:
            val_origin = val_ori
        else:
            value_slice = None

        return brick_slice, value_slice
    # TODO: Does not support n-dimensional
    def _get_array_shape_from_slice(self, slice_):
        log.debug('Getting array shape for slice_: %s', slice_)

        vals = self.brick_list.values()
        log.debug('vals: %s', vals)
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

    def reinit(self, storage):
        pass # No op

    def fill(self, value):
        pass # No op

    def __len__(self):
        # TODO: THIS IS NOT CORRECT
        return 1

    def __iter__(self):
        # TODO: THIS IS NOT CORRECT
        return [1,1].__iter__()

class InMemoryPersistenceLayer():

    def expand_domain(self, *args, **kwargs):
        # No Op - storage expanded by *Value classes
        pass

    def init_parameter(self, *args, **kwargs):
        return InMemoryStorage()

