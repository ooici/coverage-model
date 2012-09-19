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
    def calculate_brick_size(self, target_fs_size):
        log.debug('Calculating the size of a brick...')

        # TODO: Hardcoded!!!!!!!!!!
        if self.sdom==None:
            bD = [10]
            cD = tuple([5])
        else:
            bD = [10]+list(self.sdom)
            cD = tuple([5]+list(self.sdom))

        return bD,cD

    def init_parameter(self, parameter_context):
        log.debug('Initialize %s', parameter_context.name)

        # TODO: Check the flag to see which kind of storage this should be
        v = InMemoryStorage()
        self.value_list[parameter_context.name] = v
        self.parameter_domain_dict[parameter_context.name] = [None, None, None]

        log.debug('Performing Rtree dict setup')
        tD = parameter_context.dom.total_extents
        bD,cD = self.calculate_brick_size(64) #remains same for each parameter
        # Verify domain is Rtree friendly
        tree_rank = len(bD)
        log.debug('tree_rank: %s', tree_rank)
        if tree_rank == 1:
            tree_rank += 1
        log.debug('tree_rank: %s', tree_rank)
        p = rtree.index.Property()
        p.dimension = tree_rank
        brick_tree = rtree.index.Index(properties=p)
        self.brick_tree_dict[parameter_context.name] = [brick_tree,bD]

        self.expand_domain(parameter_context)

        return v

    # Write empty HDF5 brick to the filesystem
    # Input: Brick origin , brick dimensions (topological), chunk dimensions, coverage GUID, parameterName
    # TODO: ParameterContext (for HDF attributes)
    def write_brick(self,origin,bD,cD,parameter_name,data_type):
        # Calculate the brick extents
        brick_max = []
        for idx,val in enumerate(origin):
            brick_max.append(bD[idx]+val)

        brick_extents = list(origin)+brick_max
        log.debug('Brick extents (rtree format): %s', brick_extents)

        # Make sure the brick doesn't already exist if we already have some bricks
        if len([(k[0]) for k,v in self.brick_list.items() if parameter_name in k])>0:
            check = [(brick_extents==v[1]) for k,v in self.brick_list.items() if parameter_name in k]
            if True in check:
                log.debug('Brick already exists!')
            else:
                self._write_brick(origin,bD,cD,parameter_name,data_type)
        else:
            self._write_brick(origin,bD,cD,parameter_name,data_type)

    def _write_brick(self,origin,bD,cD,parameter_name,data_type):
        log.debug('origin: %s', origin)
        # Calculate the brick extents
        brick_max = []

        for idx,val in enumerate(origin):
            brick_max.append(bD[idx]+val)
        log.debug('brick_max: %s', brick_max)

        brick_extents = list(origin)+brick_max
#        if len(bD) == 1:
#            brick_extents = [0,0] + brick_max + [0]
        log.debug('brick_extents: %s', brick_extents)

        if len([(k[0]) for k,v in self.brick_list.items() if parameter_name in k])>0:
            brick_count = max(k[1] for k, v in self.brick_list.items() if k[0]==parameter_name)+1
        else:
            brick_count = 1

        log.debug('Writing brick for parameter %s', parameter_name)
        log.debug('Brick origin: %s', origin)

        root_path = '{0}/{1}/{2}'.format(self.root,self.guid,parameter_name)

        # Create the root path if it does not exist
        # TODO: Eliminate possible race condition
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        # Create a GUID for the brick
        brickGUID = create_guid()

        # Set HDF5 file and group
        sugar_file_name = '{0}.hdf5'.format(brickGUID)
        sugar_file_path = '{0}/{1}'.format(root_path,sugar_file_name)
        sugar_file = h5py.File(sugar_file_path, 'w')

        sugar_group_path = '/{0}/{1}'.format(self.guid,parameter_name)
        sugar_group = sugar_file.create_group(sugar_group_path)

        # Create the HDF5 dataset that represents one brick
        sugarCubes = sugar_group.create_dataset('{0}'.format(brickGUID), bD, dtype=data_type, chunks=cD)

        # Close the HDF5 file that represents one brick
        log.debug('Size Before Close: %s', os.path.getsize(sugar_file_path))
        sugar_file.close()
        log.debug('Size After Close: %s', os.path.getsize(sugar_file_path))

        # Add brick to Master HDF file
        log.debug('Adding %s external link to %s.', sugar_file_path, self.master_file_path)
        _master_file = h5py.File(self.master_file_path, 'r+')
        _master_file['{0}/{1}'.format(sugar_group_path, brickGUID)] = h5py.ExternalLink('./{0}/{1}/{2}'.format(self.guid, parameter_name, sugar_file_name), '{0}/{1}'.format(sugar_group_path, brickGUID))

        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count, brick_extents, brickGUID)
        self.brick_tree_dict[parameter_name][0].insert(brick_count, brick_extents, obj=brickGUID)
        log.debug('Rtree inserted successfully.')

        # Update the brick listing
        log.debug('Updating brick list[%s, %s]=[%s, %s]',parameter_name, brick_count, brickGUID, brick_extents)
        self.brick_list[parameter_name,brick_count]=[brickGUID, brick_extents]

        # Brick metadata
        # Sets the initial state of the External Link's "dirty" bit
        _master_file['{0}/{1}'.format(sugar_group_path, brickGUID)].attrs['dirty'] = 0
        _master_file['{0}/{1}'.format(sugar_group_path, brickGUID)].attrs['brick_domain'] = str(tuple(bD))

        # TODO: This is a really ugly way to store the brick_list for a parameter
#        brick_list_attrs_filtered = [(k,v) for k, v in self.brick_list.items() if k[0]==parameter_name]
#        str_size = 'S'+str(len(str(brick_list_attrs_filtered[0])))
#        brick_list_attrs = np.ndarray(tuple([len(brick_list_attrs_filtered)]), str_size)
#        for i,v in enumerate(brick_list_attrs_filtered):
#            brick_list_attrs[i] = str(v)
#        _master_file[sugar_group_path].attrs['brick_list'] = brick_list_attrs

        # Close the master file
        _master_file.close()

    # Expand the domain
    # TODO: Verify brick and chunk sizes are still valid????
    # TODO: Only expands in first (time) dimension at the moment
    def expand_domain(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)

        if self.parameter_domain_dict[parameter_context.name][0] is not None:
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
            bD = (5,) # TODO: Make this calculated based on tD and file system constraints
            cD = (2,)
            self.parameter_domain_dict[parameter_name] = [tD, bD, cD]

        try:
            # Gather block list
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: %s', len(vertices))

                # Write brick to HDF5 file
                # TODO: Loop over self.parameter_dictionary
                # TODO: This is where we'll need to deal with objects and unsupported numpy types :(
                map(lambda origin: self.write_brick(origin,bD,cD,parameter_name,parameter_context.param_type.value_encoding), vertices)

                # Refresh master file object data
                self._inspect_master()

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain is empty...')
        except Exception as ex:
            log.error('Failed to Initialize Persistence Layer: %s', ex)
            log.debug('Cleaning up bricks (not ;))')
            # TODO: Add brick cleanup routine

    # Retrieve all or subset of data from HDF5 bricks
    def get_values(self, parameterName, minExtents, maxExtents):

        log.debug('Getting value(s) from brick(s)...')

        # Find bricks for given extents
        brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)
        log.debug('Found bricks that may contain data: {0}'.format(brickSearchList))

        # Figure out slices for each brick

        # Get the data (if it exists, jagged?) for each sliced brick

        # Combine the data into one numpy array

        # Pass back to coverage layer

    # Write all or subset of Coverage's data to HDF5 brick(s)
    def set_values(self, parameter_name, payload, min_extents, max_extents):
        log.debug('Setting value(s) of payload to brick(s)...')

        # TODO:  Make sure the content's domain has a brick available, otherwise make more bricks (expand)
        brick_search_list = self.list_bricks(parameter_name, min_extents, max_extents)

        if len(brick_search_list)==0:
            log.debug('No existing bricks found, creating now...')
            self.expand_domain(max_extents)
            brickSearchList = self.list_bricks(parameter_name, min_extents, max_extents)

        if len(brick_search_list) > 1:
            log.debug('Splitting data across multiple bricks: %s', brick_search_list)
            # TODO: have to split data across multiple bricks
        else:
            log.debug('Writing all data to one brick: %s', brick_search_list)
            # TODO: all data goes in one brick
            # TODO: open brick and place the data in the dataset

    # List bricks for a parameter based on domain range
    def list_bricks(self, parameter_name, start, end):
        log.debug('Placeholder for listing bricks based on a domain range...')
        hits = list(self.brick_tree_dict[parameter_name][0].intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

    # Returns a count of bricks for a parameter
    def count_bricks(self, parameter_name):
        try:
            return max(k[1] for k, v in self.brick_list.items() if k[0]==parameter_name)
        except:
            return 'No bricks found for parameter: %s', parameter_name

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

        # TODO: Get the brick list for each parameter from the parameter's group brick_list attribute and combine
        # Close the master file
        _master_file.close()

class PersistedStorage(AbstractStorage):

    def __init__(self, master_file, parameter_name, brick_tree, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractStorage; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, **kwc)
        self._storage = np.empty((0,))

        self.master_file = master_file
        self.parameter_name = parameter_name
        self.brick_tree = brick_tree

        # We just need to know which brick(s) to save to and their slice(s)
        self.storage_bricks = {}

    # TODO: After getting slice_'s data remember to fill nulls with fill value for the parameter before passing back
    def __getitem__(self, slice_):
        log.debug('getitem slice_: %s', slice_)
        return self._storage.__getitem__(slice_)

    def __setitem__(self, slice_, value):
        log.debug('setitem slice_: %s', slice_)
        # TODO: Populate storage_bricks dict based on
        self._storage.__setitem__(slice_, value)

    def reinit(self, storage):
        pass # No op

    def fill(self, value):
        pass # No op

    @property
    def shape(self):
        # TODO: Will be removed shortly
        return self._storage.shape

    @property
    def dtype(self):
        # TODO: Will be removed shortly
        return self._storage.dtype

    def calculate_storage_bricks(self, slice_):
        pass
