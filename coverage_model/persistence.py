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
import sys, traceback
import numpy as np
import h5py
import os
import rtree
import itertools
import msgpack

class PersistenceError(Exception):
    pass

# TODO: Make error classes for persistence crap

class PersistenceLayer():
    def __init__(self, root, guid, parameter_dictionary=None, tdom=None, sdom=None, **kwargs):
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
        self.parameter_dictionary = parameter_dictionary
        self.tdom = tdom
        self.sdom = sdom

        self.brick_tree_dict = {}

        self.brick_list = {}

        self.parameter_domain_dict = {}

        # Setup Master HDF5 File
        self.master_file_path = '{0}/{1}_master.hdf5'.format(self.root,self.guid)
        # Make sure the root path exists, if not, make it
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # Check if the master file already exists, if not, create it
        if not os.path.exists(self.master_file_path):
            master_file = h5py.File(self.master_file_path, 'w')
            master_file.close()

        self.master_groups = []
        self.master_links = {}
        # Peek inside the master file and populate master_groups and master_links
        self._inspect_master()

        self.value_list = {}

        # TODO: Loop through parameter_dictionary
        if isinstance(self.parameter_dictionary, ParameterDictionary):
            log.debug('Using a ParameterDictionary!')
            for param in self.parameter_dictionary:
                pc = self.parameter_dictionary.get_context(param)
                tD = pc.dom.total_extents
                bD,cD = self.calculate_brick_size(64) #remains same for each parameter
                self.parameter_domain_dict[pc.name] = [tD,bD,cD,pc.param_type._value_encoding]
                self.init_parameter(pc)
                self._inspect_master()
#                log.debug('Performing Rtree dict setup')
#                # Verify domain is Rtree friendly
#                tree_rank = len(bD)
#                log.debug('tree_rank: %s', tree_rank)
#                if tree_rank == 1:
#                    tree_rank += 1
#                log.debug('tree_rank: %s', tree_rank)
#                p = rtree.index.Property()
#                p.dimension = tree_rank
#                brick_tree = rtree.index.Index(properties=p)
#                self.brick_tree_dict[pc.name] = [brick_tree,bD]
#        elif isinstance(self.parameter_dictionary, list):
#            log.debug('Found a list of parameters, assuming all have the same total domain')
#        elif isinstance(self.parameter_dictionary, dict):
#            log.debug('Found a dictionary of parameters, assuming parameter name is key and has value of total domain,dtype')
#            for pname,tD in self.parameter_dictionary:
#                tD = list(self.tdom+self.sdom) #can increase
#                bD,cD = self.calculate_brick_size(64) #remains same for each parameter
#                self.parameter_domain_dict[pname] = [tD,bD,cD]
#
#                self.init_parameter(tD,bD,cD,pname,'f')
        else:
            pass

#            log.debug('No parameter_dictionary defined.  Running a test script...')
#            if self.sdom is None:
#                tD = list(self.tdom)
#            else:
#                tD = list(self.tdom+self.sdom) #can increase
#            bD,cD = self.calculate_brick_size(64) #remains same for each parameter
#            self.parameterDomainDict['Test Parameter'] = [tD,bD,cD]
#
#            # Verify domain is Rtree friendly
#            if len(bD) > 1:
#                p = rtree.index.Property()
#                p.dimension = len(bD)
#                brickTree = rtree.index.Index(properties=p)
#                self.brickTreeDict['Test Parameter'] = [brickTree,tD]
#            self.init_parameter(tD,bD,cD,'Test Parameter','f')

    # Calculate brick domain size given a target file system brick size (Mbytes) and dtype
    def calculate_brick_size(self, tD, target_fs_size):
        log.debug('Calculating the size of a brick...')

        # TODO: Hardcoded!!!!!!!!!!
        if self.sdom==None:
            bD = [6]
            cD = tuple([2])
        else:
            bD = [6]+list(self.sdom)
            cD = tuple([2]+list(self.sdom))

        return bD,cD

    def init_parameter(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Initialize %s', parameter_name)

        log.debug('Performing Rtree dict setup')
        tD = parameter_context.dom.total_extents
        bD,cD = self.calculate_brick_size(tD, 64) #remains same for each parameter
        # Verify domain is Rtree friendly
        tree_rank = len(bD)
        log.debug('tree_rank: %s', tree_rank)
        if tree_rank == 1:
            tree_rank += 1
        log.debug('tree_rank: %s', tree_rank)
        p = rtree.index.Property()
        p.dimension = tree_rank
        brick_tree = rtree.index.Index(properties=p)
        self.brick_tree_dict[parameter_name] = [brick_tree, bD]

        # TODO: Sort out the path to the bricks for this parameter
        brick_path = '{0}/{1}/{2}'.format(self.root, self.guid, parameter_name)
        self.brick_list[parameter_name] = {}
        v = PersistedStorage(brick_path=brick_path, brick_tree=self.brick_tree_dict[parameter_name][0], brick_list=self.brick_list[parameter_name], dtype=parameter_context.param_type.value_encoding)
        self.value_list[parameter_name] = v
        self.parameter_domain_dict[parameter_name] = [None, None, None]

        self.expand_domain(parameter_context)

        return v

    def calculate_extents(self, origin, bD):
        log.debug('origin: %s', origin)
        # Calculate the brick extents
        brick_max = []
        brick_extents = []

        origin=list(origin)

        rtree_extents = origin + map(lambda o,s: o+s-1,origin,bD)
        brick_extents = zip(origin,map(lambda o,s: o+s-1, origin,bD))

        # Fake out the rtree if rank == 1
        if len(origin) == 1:
            rtree_extents = [e for ext in zip(rtree_extents,[0 for x in rtree_extents]) for e in ext]

        log.debug('Rtree extents: %s', rtree_extents)
        log.debug('Brick extents: %s', brick_extents)

        return rtree_extents, brick_extents

    def _brick_exists(self,parameter_name, brick_extents):
        # Make sure the brick doesn't already exist if we already have some bricks
        do_write = True
        log.error(parameter_name)
        log.error(self.brick_list)
        if parameter_name in self.brick_list:
            for x,v in self.brick_list[parameter_name].iteritems():
                if brick_extents == v[0]:
                    do_write = False
                    break

        return do_write

    # Write empty HDF5 brick to the filesystem
    def write_brick(self,origin,bD,cD,parameter_name,data_type):
        rtree_extents, brick_extents = self.calculate_extents(origin, bD)

        if not self._brick_exists(parameter_name, brick_extents):
            log.debug('Brick already exists!  Do not write')
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

        brick_group_path = '/{0}/{1}'.format(self.guid,parameter_name)
        #brick_group = brick_file.create_group(brick_group_path)
        #brick_group = brick_file.create_group('/')

    # Create the HDF5 dataset that represents one brick
        #brick_cubes = brick_group.create_dataset('{0}'.format(brick_guid), bD, dtype=data_type, chunks=cD)
        brick_cubes = brick_file.create_dataset('{0}'.format(brick_guid), bD, dtype=data_type, chunks=cD)

    # Close the HDF5 file that represents one brick
        log.debug('Size Before Close: %s', os.path.getsize(brick_file_path))
        brick_file.close()
        log.debug('Size After Close: %s', os.path.getsize(brick_file_path))

        # Add brick to Master HDF file
        log.debug('Adding %s external link to %s.', brick_file_path, self.master_file_path)
        _master_file = h5py.File(self.master_file_path, 'r+')
#        _master_file['{0}/{1}'.format(brick_group_path, brick_guid)] = h5py.ExternalLink('./{0}/{1}/{2}'.format(self.guid, parameter_name, brick_file_name), '{0}/{1}'.format(brick_group_path, brick_guid))
        _master_file['/{0}/{1}'.format(parameter_name, brick_guid)] = h5py.ExternalLink('./{0}/{1}/{2}'.format(self.guid, parameter_name, brick_file_name), brick_guid)


    # Update the brick listing
        log.debug('Updating brick list[%s] with (%s, %s)',parameter_name, brick_guid, brick_extents)
        self.brick_list[parameter_name][brick_guid] = [brick_extents, origin, tuple(bD)]
        brick_count = self.parameter_brick_count(parameter_name)
        log.debug('Brick count for %s is %s', parameter_name, brick_count)

        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count - 1, rtree_extents, brick_guid)
        self.brick_tree_dict[parameter_name][0].insert(brick_count - 1, rtree_extents, obj=brick_guid)
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
    # TODO: Only expands in first (time) dimension at the moment
    def expand_domain(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)

        if self.parameter_domain_dict[parameter_name][0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(parameter_context.dom.total_extents) != len(self.parameter_domain_dict[parameter_name][0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = self.parameter_domain_dict[parameter_name][0]
                bD = self.parameter_domain_dict[parameter_name][1]
                cD = self.parameter_domain_dict[parameter_name][2]
                new_domain = parameter_context.dom.total_extents

                delta_domain = [(x - y) for x, y in zip(new_domain, tD)]
                log.debug('delta domain: %s', delta_domain)

                tD = [(x + y) for x, y in zip(tD, delta_domain)]
                self.parameter_domain_dict[parameter_name][0] = tD
        else:
            tD = parameter_context.dom.total_extents
            #bD = (5,) # TODO: Make this calculated based on tD and file system constraints
            #cD = (2,)
            bD,cD = self.calculate_brick_size(tD, 64)
            self.parameter_domain_dict[parameter_name] = [tD, bD, cD]

        try:
            # Gather block list
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: %s', len(vertices))

                # Write brick to HDF5 file
                # TODO: This is where we'll need to deal with objects and unsupported numpy types :(
                map(lambda origin: self.write_brick(origin,bD,cD,parameter_name,parameter_context.param_type.value_encoding), vertices)

                # Refresh master file object data
                self._inspect_master()

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain is empty...')
        except Exception as ex:
            raise ex
#            log.error('Failed to Initialize Persistence Layer: %s', ex)
#            log.debug('Cleaning up bricks (not ;))')
            # TODO: Add brick cleanup routine

    # Retrieve all or subset of data from HDF5 bricks
    def get_values(self, parameterName, minExtents, maxExtents):

        log.debug('Getting value(s) from brick(s)...')

        # Find bricks for given extents
        brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)
        log.debug('Found bricks that may contain data: {0}'.format(brickSearchList))

        # Figure out slices for each brick

        # Get the data (if it exists, jagged?) for each sliced brick

        # Combine the data into one numpy array and set fill value to null values

        # Pass back to coverage layer

    # Write all or subset of Coverage's data to HDF5 brick(s)
    def set_values(self, parameter_name, payload, slice_):
        log.debug('Setting value(s) of payload to brick(s)...')
        self.value_list[parameter_name][slice_] = payload

    # List bricks for a parameter based on domain range
    def list_bricks(self, parameter_name, start, end):
        log.debug('Placeholder for listing bricks based on a domain range...')
        hits = list(self.brick_tree_dict[parameter_name][0].intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

    # Returns a count of bricks for a parameter
    def parameter_brick_count(self, parameter_name):
        ret = 0
        if parameter_name in self.brick_list:
            ret = len(self.brick_list[parameter_name])
        else:
            log.info('No bricks found for parameter: %s', parameter_name)

        return ret

    def total_brick_count(self):
        ret = 0
        for x in self.parameter_dictionary.keys():
            ret += self.parameter_brick_count(x)

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
        #self._storage = np.empty((0,))

        self.brick_tree = brick_tree

        self.brick_path = brick_path

        self.brick_list = brick_list

        self.dtype = dtype

        # We just need to know which brick(s) to save to and their slice(s)
        self.storage_bricks = {}

    def _bricks_from_slice(self, slice_):
        if not isinstance(slice_, tuple):
            raise ValueError('slice_ must be a tuple')

        rank = len(slice_)
        if rank == 1:
            rank += 1
            slice_ += (0,)

        if self.brick_tree.properties.dimension != rank:
            raise ValueError('slice_ is of incorrect rank: is {0}, must be {1}'.format(rank, self.brick_tree.properties.dimension))

        bnds = self.brick_tree.bounds

        start = []
        end = []
        for x in xrange(rank):
            sx=slice_[x]
            if isinstance(sx, slice):
                si=sx.start if sx.start is not None else bnds[x::rank][0]
                start.append(si)
                ei=sx.stop if sx.stop is not None else bnds[x::rank][1]
                end.append(ei)
            elif isinstance(sx, (list, tuple)):
                start.append(min(sx))
                end.append(max(sx))
            elif isinstance(sx, int):
                start.append(sx)
                end.append(sx)

        hits = list(self.brick_tree.intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

    # TODO: After getting slice_'s data remember to fill nulls with fill value for the parameter before passing back
    def __getitem__(self, slice_):
        log.debug('getitem slice_: %s', slice_)

        arr_shp = self._get_array_shape_from_slice(slice_)
        arr_origin = []
        for x in slice_:
            if isinstance(x, int):
                arr_origin.append(x)
            elif isinstance(x, (list, tuple, slice)):
                arr_origin.append(0)

        ret_arr = np.empty(arr_shp, dtype=self.dtype)
        ret_arr.fill(-1)
        log.debug('Shape of returned array: %s', ret_arr.shape)

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
            if not os.path.exists(brick_file):
                raise IndexError('Expected brick_file \'%s\' not found', brick_file)
            log.debug('Found brick file: %s', brick_file)

            # Figuring out which part of brick to set values
            log.error('Return array origin: %s', arr_origin)
            b_slice, v_slice = self._calc_brick_getter_slices(slice_, brick_guid, arr_origin)
            log.debug('Brick slice to extract: %s', b_slice)
            log.debug('Value slice to fill: %s', v_slice)

            bf = h5py.File(brick_file, 'r+')
            ds_path = '/{0}'.format(brick_guid)


            outy = bf[ds_path].__getitem__(*b_slice)
            log.error('OUT: %s',outy)

            ret_arr[v_slice] = outy
            bf.close()

            log.error(ret_arr)

        return ret_arr

    def __setitem__(self, slice_, value):
        log.debug('setitem slice_: %s', slice_)

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
            if os.path.exists(brick_file):
                log.debug('Found brick file: %s', brick_file)

                # Figuring out which part of brick to set values
                brick_slice, value_slice = self._calc_brick_setter_slices(slice_, brick_guid)
                log.debug('Brick slice to fill: %s', brick_slice)
                log.debug('Value slice to extract: %s', value_slice)

                # TODO: Move this to writer function

                bf = h5py.File(brick_file, 'r+')

                ds_path = '/{0}'.format(brick_guid)

                log.error('BEFORE: %s', bf[ds_path][:])

                # Setting payload values to brick
                if hasattr(value, '__iter__'):
                    val = value[value_slice]
                else:
                    val = value
                log.error('value: %s', val)

                bf[ds_path].__setitem__(*brick_slice, val=val)

                bf.flush()

                log.error('AFTER: %s', bf[ds_path][:])

                bf.close()

            else:
                raise SystemError('Can\'t find brick: %s', brick_file)

    def _get_array_shape_from_slice(self, slice_):
        shp = []
        dim = self.brick_tree.properties.dimension
        maxes = self.brick_tree.bounds[dim:]
        maxes = [int(x)+1 for x in maxes]

        for i, s in enumerate(slice_):
            if isinstance(s, int):
                shp.append(1)
            elif isinstance(s, (list,tuple)):
                shp.append(len(s))
            elif isinstance(s, slice):
                shp.append(len(range(*s.indices(maxes[i]))))

        return tuple(shp)

    def _calc_brick_getter_slices(self, slice_, brick_guid, arr_origin):
        brick_origin, brick_size = self.brick_list[brick_guid][1:]
        log.debug('brick %s:  origin=%s, size=%s', brick_guid, brick_origin, brick_size)
        log.debug('slice_: %s', slice_)

        brick_slice = []
        value_slice = []

        for i, s in enumerate(slice_):
            o=brick_origin[i]
            sz=brick_size[i]
            n=sz+o
            ao = arr_origin[i]
            log.debug('o=%s  sz=%s  n=%s, ao=%s',o,sz,n,ao)
            if isinstance(s, int):
                log.debug('is int')
                if o <= s < n:
                    brick_slice.append(s-o)
                    value_slice.append(s-ao)
                    arr_origin[i] = ao+1
                else:
                    raise ValueError('Specified index is not within the brick: %s', s)
            elif isinstance(s, (list, tuple)):
                log.debug('is list/tuple')
                lb = [x-o for x in s if o <= x < n]
                lv = [xi + ao for xi,x in enumerate(lb)]
                if len(lb) == 0:
                    raise ValueError('None of the specified indices are within the brick: %s', s)
                brick_slice.append(lb)
                value_slice.append(lv)
                arr_origin[i] = ao+len(lb)
            elif isinstance(s, slice):
                log.debug('is slice')
                if s.start is None:
                    start = 0
                    astart=ao
                else:
                    start = s.start - o if o <= s.start < n else 0
                    astart = ao
                if s.stop is None:
                    stop = n - o
                    astop = stop - start + ao
                else:
                    stop = s.stop - o if o <= s.stop < n else n - o
                    astop = stop - start + ao
                log.debug('new start/stop: %s %s', start, stop)
                log.debug('astart/astop: %s %s', astart, astop)
                brick_slice.append(slice(start, stop, s.step))
                vs = slice(astart, astop, s.step)
                vsi = range(*vs.indices(astop))
                log.warn('vs: %s', vs)
                log.warn('vsi: %s', vsi)
                inds = [x+ao for x in range(len(vsi))]
                log.warn('inds: %s',inds)
                value_slice.append(inds)
                arr_origin[i] = ao+len(vsi)

        return brick_slice, value_slice

    def _calc_brick_setter_slices(self, slice_, brick_guid):
        brick_origin, brick_size = self.brick_list[brick_guid][1:]
        log.debug('brick %s:  origin=%s, size=%s', brick_guid, brick_origin, brick_size)
        log.debug('slice_: %s', slice_)

        brick_slice = []
        value_slice = []

        for i, s in enumerate(slice_):
            o=brick_origin[i]
            sz=brick_size[i]
            n=sz+o
            log.debug('o=%s  sz=%s  n=%s',o,sz,n)
            if isinstance(s, int):
                log.debug('is int')
                if o <= s < n:
                    brick_slice.append(s-o)
                    value_slice.append(s)
                else:
                    raise ValueError('Specified index is not within the brick: %s', s)
            elif isinstance(s, (list, tuple)):
                log.debug('is list/tuple')
                lb = [x-o for x in s if o <= x < n]
                lv = [x for x in s if o <= x < n]
                if len(lb) == 0:
                    raise ValueError('None of the specified indices are within the brick: %s', s)
                brick_slice.append(lb)
                value_slice.append(lv)
            elif isinstance(s, slice):
                log.debug('is slice')
                if s.start is None:
                    start = o
                else:
                    start = s.start if o <= s.start < n else o
                if s.stop is None:
                    stop = n
                else:
                    stop = s.stop if o <= s.stop < n else n
                log.debug('new start/stop: %s %s', start, stop)
#                brick_slice.append(slice(start-o, stop-o, s.step))
                brick_slice.append(range(*slice(start-o, stop-o, s.step).indices(stop)))
                value_slice.append(slice(start, stop, s.step))

        return brick_slice, value_slice

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

    def calculate_storage_bricks(self, slice_):
        pass

    # List bricks for a parameter
    def list_bricks(self, start, end):
        """

        @param start: rtree min list
        @param end: rtree max list
        @return:
        """
        hits = list(self.brick_tree.intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

class InMemoryPersistenceLayer():

    def expand_domain(self, parameter_context):
        # No Op - storage expanded by *Value classes
        pass

    def init_parameter(self, parameter_context):
        return InMemoryStorage()

