#!/usr/bin/env python

"""
@package coverage_model.persistence
@file coverage_model/persistence.py
@author James Case
@brief The core classes comprising the Persistence Layer
"""

from pyon.public import log
from coverage_model.basic_types import create_guid
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
        Constructor for PersistenceLayer
        """
        self.root = '.' if root is ('' or None) else root
        self.guid = guid
        log.debug('Persistence GUID: %s', self.guid)
        self.parameter_dictionary = parameter_dictionary
        self.tdom = tdom
        self.sdom = sdom

        brickTree = rtree.Rtree()
        self.brickTreeDict = {}
        self.brickList = {}
        self.parameterDomainDict = {}

        self.value_list = {}

        #self.param_dtype = np.empty((1,), dtype='f').dtype

        # TODO: Loop through parameter_dictionary
        if isinstance(self.parameter_dictionary, ParameterDictionary):
            log.debug('Using a ParameterDictionary!')
            for param in self.parameter_dictionary:
                pc = self.parameter_dictionary.get_context(param)

                #if self.sdom is None:
                #    tD = list(self.tdom)
                #else:
                #    tD = list(self.tdom+self.sdom) #can increase
                tD = pc.dom.total_extents
                bD,cD = self.calculate_brick_size(64) #remains same for each parameter
                self.parameterDomainDict[pc.name] = [tD,bD,cD,pc.param_type._value_encoding]
                self.init_parameter(pc)
        elif isinstance(self.parameter_dictionary, list):
            log.debug('Found a list of parameters, assuming all have the same total domain')
        elif isinstance(self.parameter_dictionary, dict):
            log.debug('Found a dictionary of parameters, assuming parameter name is key and has value of total domain,dtype')
            for pname,tD in self.parameter_dictionary:
                tD = list(self.tdom+self.sdom) #can increase
                bD,cD = self.calculate_brick_size(64) #remains same for each parameter
                self.parameterDomainDict[pname] = [tD,bD,cD]
                # Verify domain is Rtree friendly
                if len(bD) > 1:
                    p = rtree.index.Property()
                    p.dimension = len(bD)
                    brickTree = rtree.index.Index(properties=p)
                    self.brickTreeDict[pname] = [brickTree,tD]
                self.init_parameter(tD,bD,cD,pname,'f')
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

        v = PLValue()
        self.value_list[parameter_context.name] = v
        self.parameterDomainDict[parameter_context.name] = [None, None, None]

        self.expand_domain(parameter_context)

        return v

    # Generate empty bricks
    # Input: totalDomain, brickDomain, chunkDomain, parameterName, parameter data type
    def init_parameter_o(self, tD, bD, cD, parameterName, dataType):
        log.debug('Total Domain: {0}'.format(tD))
        log.debug('Brick Domain: {0}'.format(bD))
        log.debug('Chunk Domain: {0}'.format(cD))

        try:
            # Gather block list
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: {0}'.format(len(vertices)))

                # Write brick to HDF5 file
                # TODO: Loop over self.parameter_dictionary
                map(lambda origin: self.write_brick(origin,bD,cD,parameterName,dataType), vertices)

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain in empty...')
        except:
            log.error('Failed to Initialize Persistence Layer')
            exc_type, exc_value, exc_traceback = sys.exc_info()
            log.error('{0}'.format(repr(traceback.format_exception(exc_type, exc_value, exc_traceback))))

            log.debug('Cleaning up bricks.')
            # TODO: Add brick cleanup routine

    # Write empty HDF5 brick to the filesystem
    # Input: Brick origin , brick dimensions (topological), chunk dimensions, coverage GUID, parameterName
    # TODO: ParameterContext (for HDF attributes)
    def write_brick(self,origin,bD,cD,parameterName,dataType):
        # Calculate the brick extents
        brickMax = []
        for idx,val in enumerate(origin):
            brickMax.append(bD[idx]+val)

        brickExtents = list(origin)+brickMax
        log.debug('Brick extents (rtree format): {0}'.format(brickExtents))

        # Make sure the brick doesn't already exist if we already have some bricks
        if len([(k[0]) for k,v in self.brickList.items() if parameterName in k])>0:
            check = [(brickExtents==v[1]) for k,v in self.brickList.items() if parameterName in k]
            if True in check:
                log.debug('Brick already exists!')
            else:
                self._write_brick(origin,bD,cD,parameterName,dataType)
        else:
            self._write_brick(origin,bD,cD,parameterName,dataType)

    def _write_brick(self,origin,bD,cD,parameterName,dataType):
        # Calculate the brick extents
        brickMax = []
        for idx,val in enumerate(origin):
            brickMax.append(bD[idx]+val)

        brickExtents = list(origin)+brickMax
        log.debug('Brick extents (rtree format): {0}'.format(brickExtents))

        if len([(k[0]) for k,v in self.brickList.items() if parameterName in k])>0:
            brickCount = max(k[1] for k, v in self.brickList.items() if k[0]==parameterName)+1
        else:
            brickCount = 1

        log.debug('Writing brick for parameter {0}'.format(parameterName))
        log.debug('Brick origin: {0}'.format(origin))

        rootPath = '{0}/{1}/{2}'.format(self.root,self.guid,parameterName)

        # Create the root path if it does not exist
        # TODO: Eliminate possible race condition
        if not os.path.exists(rootPath):
            os.makedirs(rootPath)

        # Create a GUID for the brick
        brickGUID = create_guid()

        # Set HDF5 file and group
        sugarFileName = '{0}.hdf5'.format(brickGUID)
        sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
        sugarFile = h5py.File(sugarFilePath, 'w')

        sugarGroupPath = '/{0}/{1}'.format(self.guid,parameterName)
        sugarGroup = sugarFile.create_group(sugarGroupPath)

        # Create the HDF5 dataset that represents one brick
        sugarCubes = sugarGroup.create_dataset('{0}'.format(brickGUID), bD, dtype=dataType, chunks=cD)

        # Close the HDF5 file that represents one brick
        log.debug('Size Before Close: {0}'.format(os.path.getsize(sugarFilePath)))
        sugarFile.close()
        log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

        # Verify domain is Rtree friendly
        if len(bD) > 1:
            log.debug('Inserting into Rtree {0}:{1}:{2}'.format(brickCount,brickExtents,brickGUID))
            self.brickTreeDict[parameterName][0].insert(brickCount,brickExtents,obj=brickGUID)

        # Update the brick listing
        self.brickList[parameterName,brickCount]=[brickGUID, brickExtents]

    # Expand the domain
    # TODO: Verify brick and chunk sizes are still valid????
    # TODO: Only expands in first (time) dimension at the moment
    def expand_domain(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)

        if self.parameterDomainDict[parameter_context.name][0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(parameter_context.dom.total_extents) != len(self.parameterDomainDict[parameter_name][0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = self.parameterDomainDict[parameter_name][0]
                bD = self.parameterDomainDict[parameter_name][1]
                cD = self.parameterDomainDict[parameter_name][2]
                newDomain = parameter_context.dom.total_extents

                deltaDomain = [(x - y) for x, y in zip(newDomain, tD)]
                log.debug('delta domain: {0}'.format(deltaDomain))

                tD = [(x + y) for x, y in zip(tD, deltaDomain)]
                self.parameterDomainDict[parameter_name][0] = tD
        else:
            tD = parameter_context.dom.total_extents
            bD = (5,) # TODO: Make this calculated based on tD and file system constraints
            cD = (2,)
            self.parameterDomainDict[parameter_name] = [tD, bD, cD]

        try:
            # Gather block list
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: {0}'.format(len(vertices)))

                # Write brick to HDF5 file
                # TODO: Loop over self.parameter_dictionary
                # TODO: This is where we'll need to deal with objects and unsupported numpy types :(
                map(lambda origin: self.write_brick(origin,bD,cD,parameter_name,parameter_context.param_type.value_encoding), vertices)

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain in empty...')
        except Exception as ex:
            log.error('Failed to Initialize Persistence Layer: %s', ex)
            log.debug('Cleaning up bricks (not ;))')
            # TODO: Add brick cleanup routine

        # TODO: Setup your bricking for the first time based on the parameter_context.dom - the logic from init_parameter
        # TODO: Call something to actually write the bricks - a.k.a what used to be in init_parameter

    def expand_domain_o(self, parameterName, new_tdom, new_sdom=None):
        log.debug('Expanding domain (n-dimension)')

        # Check if the number of dimensions of the total domain has changed
        # TODO: Will this ever happen???  If so, how to handle?
        if len(new_tdom+new_sdom)!=len(self.parameterDomainDict[parameterName][0]):
            log.error('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
        else:
            tD = self.parameterDomainDict[parameterName][0]
            bD = self.parameterDomainDict[parameterName][1]
            cD = self.parameterDomainDict[parameterName][2]
            newDomain = new_tdom+new_sdom

            deltaDomain = [(x - y) for x, y in zip(newDomain, tD)]
            log.debug('delta domain: {0}'.format(deltaDomain))

            tD = [(x + y) for x, y in zip(tD, deltaDomain)]
            self.parameterDomainDict[parameterName][0] = tD
            self.tdom = new_tdom
            self.sdom = new_sdom

            # TODO: Handle parameter dtype based
            self.init_parameter(tD, bD, cD, parameterName, self.parameterDomainDict[parameterName][3])

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
    def set_values(self, parameterName, payload, minExtents, maxExtents):
        log.debug('Setting value(s) of payload to brick(s)...')

        # TODO:  Make sure the content's domain has a brick available, otherwise make more bricks (expand)
        brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)

        if len(brickSearchList)==0:
            log.debug('No existing bricks found, creating now...')
            self.expand_domain(maxExtents)
            brickSearchList = self.list_bricks(parameterName, minExtents, maxExtents)

        if len(brickSearchList) > 1:
            log.debug('Splitting data across multiple bricks: {0}'.format(brickSearchList))
            # TODO: have to split data across multiple bricks
        else:
            log.debug('Writing all data to one brick: {0}'.format(brickSearchList))
            # TODO: all data goes in one brick
            # TODO: open brick and place the data in the dataset

    # List bricks for a parameter based on domain range
    def list_bricks(self, parameterName, start, end):
        log.debug('Placeholder for listing bricks based on a domain range...')
        hits = list(self.brickTreeDict[parameterName][0].intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

    # Returns a count of bricks for a parameter
    def count_bricks(self, parameterName):
        try:
            return max(k[1] for k, v in self.brickList.items() if k[0]==parameterName)
        except:
            return 'No bricks found for parameter: {0}'.format(parameterName)

class PLValue():

    def __init__(self):
        self._storage = np.empty((0,))

    def __getitem__(self, item):
        return self._storage.__getitem__(item)

    def __setitem__(self, key, value):
        self._storage.__setitem__(key, value)

    def reinit(self, storage):
        self._storage = storage.copy()

    def fill(self, value):
        self._storage.fill(value)

    @property
    def shape(self):
        return self._storage.shape

    @property
    def dtype(self):
        return self._storage.dtype