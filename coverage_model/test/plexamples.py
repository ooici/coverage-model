#!/usr/bin/env python

"""
@package coverage_model.test.plexamples
@file coverage_model/test/plexamples.py
@author James Case
@brief Exemplar functions for creation and manipulation of the persistence layer between Coverages (memory) and HDF5 (disk)
"""

from pyon.public import log
from netCDF4 import Dataset
from coverage_model.basic_types import *
from coverage_model.coverage import *
from coverage_model.parameter_types import *
from coverage_model.persistence import PersistenceLayer, AbstractStorage, PersistedStorage
import numpy as np
import h5py
from coverage_model.hdf_utils import HDFLockingFile
import os
import math
import itertools

# Globals
brickTree = rtree.Rtree()

class A(object):
    def __getitem__(self, slice_):
        print slice_

def run_test():
    pdict = standup_pdict()
    scov = SimplexCoverage('writer4',pdict,in_memory_storage=False)
    pl = scov._persistence_layer
    values = { 'salinity': np.arange(5),
               scov.temporal_parameter_name: np.arange(10, 10+5)}
    pl.write_parameters('1', values)
    return pl

# Stand up an example parameter dictionary
def standup_pdict():
    # Open the netcdf dataset
    ds = Dataset('test_data/ncom.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','depth','water_u','water_v','salinity','water_temp',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionary
    for v in var_names:
        var = ds.variables[v]

        pcontext = ParameterContext(v, param_type=QuantityType(value_encoding=ds.variables[v].dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the axis for the coordinate parameters
        if v == 'time':
            pcontext.axis = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.axis = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.axis = AxisTypeEnum.LON
        elif v == 'depth':
            pcontext.axis = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)
    return pdict

# Generate empty bricks
# Input: totalDomain, brickDomain, chunkDomain, parameterName, 0=do not write brick, 1=write brick
def create_bricks(tD,bD,cD,parameterName,wBrick):
    log.debug('Total Domain: {0}'.format(tD))
    log.debug('Brick Domain: {0}'.format(bD))
    log.debug('Chunk Domain: {0}'.format(cD))

    p = rtree.index.Property()
    p.dimension = len(bD)
    brickTree = rtree.index.Index(properties=p)

    # Gather block list
    lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

    # Gather brick vertices
    vertices = list(itertools.product(*lst))

    # Write brick to HDF5 file
    if wBrick==1:
        # TODO: Need the coverage GUID or we won't know how to associate the bricks to the coverage later
        coverageGUID = 'COV'
        map(lambda x: write_brick(x,bD,cD,coverageGUID,parameterName), vertices)

    log.debug('Number of Bricks: {0}'.format(len(vertices)))

# Write empty HDF5 brick to the filesystem
# Input: Brick origin , brick dimensions (topological), TODO: ParameterContext (for HDF attributes)
def write_brick(origin,bD,cD,coverageGUID,parameterName):
    #global brickTree


    log.debug('Writing brick for parameter {0}'.format(parameterName))
    log.debug('Brick origin: {0}'.format(origin))

    rootPath = 'test_data/{0}/{1}'.format(coverageGUID,parameterName)
    # Create the root path if it does not exist
    # TODO: Eliminate possible race condition
    if not os.path.exists(rootPath):
        os.makedirs(rootPath)

    # Create a GUID for the brick
    brickGUID = create_guid()

    # Set HDF5 file and group
    sugarFileName = '{0}.hdf5'.format(brickGUID)
    sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
    sugarFile = HDFLockingFile(sugarFilePath, 'w')

    sugarGroupPath = '/{0}/{1}'.format(coverageGUID,parameterName)
    sugarGroup = sugarFile.create_group(sugarGroupPath)

    # TODO: Remove after the ParameterContext is passed to the function
    parameterData = np.empty(bD, dtype='f')

    # Create the HDF5 dataset that represents one brick
    sugarCubes = sugarGroup.create_dataset('{0}'.format(brickGUID), bD, dtype=parameterData.dtype, chunks=cD)

    # Close the HDF5 file that represents one brick
    log.debug('Size Before Close: {0}'.format(os.path.getsize(sugarFilePath)))
    sugarFile.close()
    log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

    # TODO: Keep track of brick GUID and index in an R-tree somewhere
    # Calculate the brick extents for the Rtree
    # TODO: May not need this
    brickMax = []
    #brickExtents = []
    for idx,val in enumerate(origin):
        brickMax.append(bD[idx]+val)
    #for val in itertools.izip(origin,brickMax):
    #    brickExtents.append(val)

    #bE = np.array(brickExtents).flatten().tolist()
    bE = list(origin)+brickMax
    log.debug('Brick extents (rtree format): {0}'.format(bE))

    # Insert into the Rtree
    # TODO: This is not working yet.  Need an id system and the location of the brick
    brickTree.insert(1,bE,obj=brickGUID)

# Generate generic bricks of (1,100,100) using dummy coverage parameters and attributes
# Input: Percentage to fill, Value to fill with
def plGenericBricks(pctFill,fillValue):
    log.debug('Creating sample HDF5 persistence objects...')
    rootPath = 'test_data'
    masterFileName = 'sugarMaster.hdf5'
    masterFilePath = '{0}/{1}'.format(rootPath,masterFileName)
    coverageName = 'COV'
    
    # Open HDf5 file for writing out the SimplexCoverage
    sugarMaster = HDFLockingFile(masterFilePath, 'w')
    sugarMasterGroup = sugarMaster.create_group('/{0}'.format(coverageName))
    
    # TODO: Create coverage object from netCDF input file
    
    #TODO: Get parameter names, context and data from the coverage
    parameterNames = ['param01','param02','param03','param04']
    
    
    # Create persistence objects in HDF5
    # Loop through each parameter in the coverage
    for parameterName in parameterNames:
        
        # TODO: Brick and chunking size is based on the parameter data and is configurable
        sugarBrickSize = (1,100,100)
        sugarCubeSize = (1,10,10)
        
        # TODO: Calculate the number of bricks required based on coverage extents and target brick size
        sugarBricks = np.arange(10)
        
        # TODO: Split up all parameter data into brick-sized arrays
        for sugarBrick in sugarBricks:
            
            sugarFileName = '{0}_{1}_Brick{2}.hdf5'.format(coverageName,parameterName,sugarBrick)
            sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
            sugarFile = HDFLockingFile(sugarFilePath, 'w')
            
            sugarGroupPath = '/{0}/{1}'.format(coverageName,parameterName)
            sugarGroup = sugarFile.create_group(sugarGroupPath)
        
            #parameterData = np.empty(sugarBrickSize, dtype='f')
            
            #if pctFill > 0:
            #    fillData = parameterData[:,0:pctFill,0:pctFill]
            #    fillData[...] = fillValue
            
            # Create the HDf5 dataset that represents one brick
            sugarCubes = sugarGroup.create_dataset('Brick{0}'.format(sugarBrick), sugarBrickSize, dtype=parameterData.dtype, chunks=sugarCubeSize)
            sugarCubes[:,0:10,0:10] = np.ones((1,10,10))
            
            log.debug('Size Before Flush: {0}'.format(os.path.getsize(sugarFilePath)))
            
            # Add temporal and spatial metadata as attributes
            temporalDomainStart = '20120815060030102030' #CCYYMMDDHHMMSSAABBCC
            sugarCubes.attrs["temporal_domain_start"] = np.array(['{0}'.format(temporalDomainStart)])
            temporalDomainEnd = '20120815063000102030'
            sugarCubes.attrs["temporal_domain_end"] = np.array(['{0}'.format(temporalDomainEnd)])

            spatialDomainMinX = '-73' # Longitude
            spatialDomainMinY = '40' # Latitude
            spatialDomainMinZ = '-10000' # Depth below MLLW in meters
            sugarCubes.attrs["spatial_domain_min"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
            spatialDomainMaxX = '-70' # Longitude
            spatialDomainMaxY = '32' # Latitude
            spatialDomainMaxZ = '500' # Depth below MLLW in meters
            sugarCubes.attrs["spatial_domain_max"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])

            # Close the HDF5 file that represents one brick
            sugarFile.flush()
            log.debug('Size After Flush: {0}'.format(os.path.getsize(sugarFilePath)))
            sugarFile.close()
            log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

            
            sugarMaster['{0}/Brick{1}'.format(sugarGroupPath,sugarBrick)] = h5py.ExternalLink(sugarFileName, '{0}/Brick{1}'.format(sugarGroupPath,sugarBrick))
            sugarMasterGroup.attrs["temporal_domain_start"] = np.array(['{0}'.format(temporalDomainStart)])
            sugarMasterGroup.attrs["temporal_domain_end"] = np.array(['{0}'.format(temporalDomainEnd)])
            sugarMasterGroup.attrs["spatial_domain_min"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
            sugarMasterGroup.attrs["spatial_domain_max"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
    # Close the master HDF5 file
    log.debug('Master File Size Before Close: {0}'.format(os.path.getsize(masterFilePath)))
    sugarMaster.close()
    log.debug('Master File Size After Close: {0}'.format(os.path.getsize(masterFilePath)))
    
    log.debug('Finished creating many sugar cubes and bricks!')
    return sugarCubes

# Create a dataset that has deltas linked back to the master.
def DataProductDeltas():
    log.debug('Example to create delta datasets based on a master...')
    rootPath = 'test_data'
    masterFileName = 'master0.hdf5'
    deltaFileName = 'delta0.hdf5'
    masterFilePath = '{0}/{1}'.format(rootPath,masterFileName)
    dpFilePath = '{0}/{1}'.format(rootPath,deltaFileName)
    coverageName = 'COV'
    
    # Open HDf5 file for writing out the SimplexCoverage
    sugarMaster = HDFLockingFile(masterFilePath, 'w')
    sugarMasterGroup = sugarMaster.create_group('/{0}'.format(coverageName))
    
    dpMaster = HDFLockingFile(dpFilePath, 'w')
    dpMasterGroup = dpMaster.create_group('/{0}'.format(coverageName))
    dpDeltaGroup = dpMaster.create_group('/{0}'.format('DELTAS'))
    
    parameterNames = ['param01']
    
    # Create persistence objects in HDF5
    # Loop through each parameter in the coverage
    for parameterName in parameterNames:
        
        # TODO: Brick and chunking size is based on the parameter data and is configurable
        sugarBrickSize = (1,100,100)
        sugarCubeSize = (1,10,10)
        
        # TODO: Calculate the number of bricks required based on coverage extents and target brick size
        sugarBricks = np.arange(10)
        
        for sugarBrick in sugarBricks:
            
            sugarFileName = '{0}_{1}_Brick{2}.hdf5'.format(coverageName,parameterName,sugarBrick)
            sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
            sugarFile = HDFLockingFile(sugarFilePath, 'w')
            
            sugarGroupPath = '/{0}/{1}'.format(coverageName,parameterName)
            sugarGroup = sugarFile.create_group(sugarGroupPath)
            
            # Create the HDf5 dataset that represents one brick
            sugarCubes = sugarGroup.create_dataset('Brick{0}'.format(sugarBrick), sugarBrickSize, dtype='f', chunks=sugarCubeSize)
            sugarCubes[:] = np.ones((1,100,100))
            
            log.debug('Size Before Flush: {0}'.format(os.path.getsize(sugarFilePath)))

            # Close the HDF5 file that represents one brick
            sugarFile.flush()
            log.debug('Size After Flush: {0}'.format(os.path.getsize(sugarFilePath)))
            sugarFile.close()
            log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

            
            sugarMaster['{0}/Brick{1}'.format(sugarGroupPath,sugarBrick)] = h5py.ExternalLink(sugarFileName, '{0}/Brick{1}'.format(sugarGroupPath,sugarBrick))
            dpMaster['{0}/Brick{1}'.format(sugarGroupPath,sugarBrick)] = h5py.ExternalLink(sugarFileName, '{0}/Brick{1}'.format(sugarGroupPath,sugarBrick))
    
    
    deltaFilePath = '{0}/{1}'.format(rootPath,'delta.hdf5')
    deltaFile = HDFLockingFile(deltaFilePath, 'w')
    deltaGroupPath = '/{0}/{1}'.format('DELTAS',parameterName)
    deltaGroup = deltaFile.create_group(deltaGroupPath)
    
    # Create the HDf5 dataset that represents one brick
    sugarGrains = deltaGroup.create_dataset('Brick{0}'.format(sugarBrick), sugarBrickSize, dtype='f', chunks=sugarCubeSize)
    sugarGrains[0,0,0] = 2
    
    dpMaster['DELTAS/param01/Brick9'] = h5py.ExternalLink('delta.hdf5', 'DELTAS/param01/Brick9')
    
    log.debug('Size Before Flush: {0}'.format(os.path.getsize(deltaFilePath)))

    # Close the HDF5 file that represents one brick
    deltaFile.flush()
    log.debug('Size After Flush: {0}'.format(os.path.getsize(deltaFilePath)))
    deltaFile.close()
    log.debug('Size After Close: {0}'.format(os.path.getsize(deltaFilePath)))
    
    m = dpMaster[('/COV/param01/Brick9')]
    d = dpMaster[('/DELTAS/param01/Brick9')]
    
    dpMaster.close()
    
    # Close the master HDF5 file
    sugarMaster.close()
    log.debug('Finished creating many sugar cubes and bricks!')
    return (m,d)

# Covert a SimplexCoverage to HDF5 objects
# Input: SimplexCoverage
# Output: Success
def cov2hdf(scov):
    log.debug('Converting SimplexCoverage to HDF5 objects...')
    rootPath = 'test_data'
    masterFileName = '{0}_{1}.hdf5'.format('sugarMaster',scov.label)
    masterFilePath = '{0}/{1}'.format(rootPath,masterFileName)
    coverageName = scov.label
    
    # Open HDf5 file for writing out the SimplexCoverage
    sugarMaster = HDFLockingFile(masterFilePath, 'w')
    sugarMasterGroup = sugarMaster.create_group('/{0}'.format(coverageName))
    
    #  Get parameter names, context and data from the coverage
    parameterNames = scov.list_parameters()
    
    # Create persistence objects in HDF5
    # Loop through each parameter in the coverage
    for parameterName in parameterNames:
        
        # Brick and chunking size is based on the parameter data and is configurable
        parameterData = scov.range_value[parameterName]
        
        # TODO: Get the brick and cube size from a pre-calculated and assigned coverage parameter
        sugarBrickSize = parameterData.shape
        sugarCubeSizeList = list()
        sugarCubeSize = tuple
        
        # Calculate the chunking size
        for idx, val in enumerate(parameterData.shape):
            if val <= 10:
                sugarCubeSizeList.append(1)
            else:
                sugarCubeSizeList.append(math.trunc(val/10))
        
        # Convert list to tuple
        sugarCubeSize = tuple(sugarCubeSizeList)
        
        # TODO: Calculate the number of bricks required based on coverage extents and target brick size
        sugarBricks = np.arange(10)
        
        # TODO: Split up all parameter data into brick-sized arrays
        for sugarBrick in sugarBricks:
            
            sugarFileName = '{0}_{1}_Brick{2}.hdf5'.format(coverageName,parameterName,sugarBrick)
            sugarFilePath = '{0}/{1}'.format(rootPath,sugarFileName)
            sugarFile = HDFLockingFile(sugarFilePath, 'w')
            
            sugarGroupPath = '/{0}/{1}'.format(coverageName,parameterName)
            sugarGroup = sugarFile.create_group(sugarGroupPath)
            
            # Create the HDf5 dataset that represents one brick
            sugarCubes = sugarGroup.create_dataset('Brick{0}'.format(sugarBrick), sugarBrickSize, dtype=parameterData.content.dtype, chunks=sugarCubeSize)
            sugarCubes[:] = np.ones(sugarBrickSize)
            #sugarCubes[:] = parameterData
            
            log.debug('Size Before Close: {0}'.format(os.path.getsize(sugarFilePath)))
            
            # Add temporal and spatial metadata as attributes
            temporalDomainStart = '20120815060030102030' #CCYYMMDDHHMMSSAABBCC
            sugarCubes.attrs["temporal_domain_start"] = np.array(['{0}'.format(temporalDomainStart)])
            temporalDomainEnd = '20120815063000102030'
            sugarCubes.attrs["temporal_domain_end"] = np.array(['{0}'.format(temporalDomainEnd)])

            spatialDomainMinX = '-73' # Longitude
            spatialDomainMinY = '40' # Latitude
            spatialDomainMinZ = '-10000' # Depth below MLLW in meters
            sugarCubes.attrs["spatial_domain_min"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
            spatialDomainMaxX = '-70' # Longitude
            spatialDomainMaxY = '32' # Latitude
            spatialDomainMaxZ = '500' # Depth below MLLW in meters
            sugarCubes.attrs["spatial_domain_max"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])

            # Close the HDF5 file that represents one brick
            sugarFile.flush()
            log.debug('Size After Flush: {0}'.format(os.path.getsize(sugarFilePath)))
            sugarFile.close()
            log.debug('Size After Close: {0}'.format(os.path.getsize(sugarFilePath)))

            
            sugarMaster['{0}/Brick{1}'.format(sugarGroupPath,sugarBrick)] = h5py.ExternalLink(sugarFileName, '{0}/Brick{1}'.format(sugarGroupPath,sugarBrick))
            sugarMasterGroup.attrs["temporal_domain_start"] = np.array(['{0}'.format(temporalDomainStart)])
            sugarMasterGroup.attrs["temporal_domain_end"] = np.array(['{0}'.format(temporalDomainEnd)])
            sugarMasterGroup.attrs["spatial_domain_min"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
            sugarMasterGroup.attrs["spatial_domain_max"] = np.array(['{0},{1},{2}'.format(spatialDomainMinX, spatialDomainMinY, spatialDomainMinZ)])
    # Close the master HDF5 file
    log.debug('Master File Size Before Close: {0}'.format(os.path.getsize(masterFilePath)))
    sugarMaster.close()
    log.debug('Master File Size After Close: {0}'.format(os.path.getsize(masterFilePath)))
    
    log.debug('Finished creating many sugar cubes and bricks!')
    return True

# Produce bricks from a netCDF file that has been converted to a SimplexCoverage
def nc2bricks():
    log.debug('Create bricks from a coverage model object')
    # Open the netcdf dataset
    ds = Dataset('test_data/ncom.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','depth','water_u','water_v','salinity','water_temp',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionary
    for v in var_names:
        var = ds.variables[v]

        pcontext = ParameterContext(v, param_type=QuantityType(value_encoding=ds.variables[v].dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the axis for the coordinate parameters
        if v == 'time':
            pcontext.axis = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.axis = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.axis = AxisTypeEnum.LON
        elif v == 'depth':
            pcontext.axis = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal'), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [34,57,89]), scrs, MutabilityEnum.IMMUTABLE) # 3d spatial topology (grid)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('sample grid coverage_model', pdict, sdom, tdom)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
        # TODO: Sort out how to leave these sparse internally and only broadcast during read
        if v == 'depth':
            z,_,_ = my_meshgrid(arr,np.zeros([57]),np.zeros([89]),indexing='ij',sparse=True)
            scov.range_value[v][:] = z
        elif v == 'lat':
            _,y,_ = my_meshgrid(np.zeros([34]),arr,np.zeros([89]),indexing='ij',sparse=True)
            scov.range_value[v][:] = y
        elif v == 'lon':
            _,_,x = my_meshgrid(np.zeros([34]),np.zeros([57]),arr,indexing='ij',sparse=True)
            scov.range_value[v][:] = x
        else:
            scov.range_value[v][:] = var[:]
    
    scov.label = 'ncom'
    
    # Start processing Coverage to HDF
    cov2hdf(scov)
    
    return scov

# Based on scitools meshgrid
def my_meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from two or more coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
         If True a sparse grid is returned in order to conserve memory.
         Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in
        order to conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous arrays.
        Furthermore, more than one element of a broadcast array may refer to
        a single memory location.  If you need to write to the arrays, make
        copies first.

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing keyword
    argument.  Giving the string 'ij' returns a meshgrid with matrix indexing,
    while 'xy' returns a meshgrid with Cartesian indexing.  In the 2-D case
    with inputs of length M and N, the outputs are of shape (N, M) for 'xy'
    indexing and (M, N) for 'ij' indexing.  In the 3-D case with inputs of
    length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M,
    N, P) for 'ij' indexing.  The difference is illustrated by the following
    code snippet::

        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)

    copy_ = kwargs.get('copy', True)
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    if not indexing in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    if indexing == 'xy':
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


if __name__ == "__main__":
#    scov, _ = ncstation2cov()
#    log.debug(scov)
#
#    log.debug('\n=======\n')
#
#    gcov, _ = ncgrid2cov()
#    log.debug(gcov)

#    direct_read_write()
    methodized_read()

#    from coverage_model.coverage_model import AxisTypeEnum
#    axis = 'TIME'
#    log.debug(axis == AxisTypeEnum.TIME)

    pass

"""

from coverage_model.test.simple_cov import *
scov, ds = ncstation2cov()


"""
