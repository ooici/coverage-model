#!/usr/bin/env python

"""
@package 
@file simple_cov
@author Christopher Mueller
@brief 
"""

from netCDF4 import Dataset
from coverage_model.coverage import *
from coverage_model.parameter import *
import numpy as np

def ncstation2cov():
    ds = Dataset('test_data/usgs.nc')

    rdict = RangeDictionary()
    rdict.items = {
        'coords':['time','lat','lon','z',],
        'vars':['streamflow','water_temperature',],
    }

    tcrs = CRS(['t'])
    scrs = CRS(['x','y','z'])

    tdom = GridDomain(GridShape('temporal', [1]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    #CBM: the leading 0 is for time - need to refine append_parameter to deal with the addition of time so that this is just space
    sdom = GridDomain(GridShape('spatial', [1,1]), scrs, MutabilityEnum.IMMUTABLE) # 1d spatial topology (station/trajectory)
#    sdom = GridDomain(GridShape('spatial', [1,3,5]), scrs, None) # 2d spatial topology (grid)

    scov = SimplexCoverage(rdict, sdom, tdom)

    # Collect all var names (for convenience)
    all = []
    for x in rdict.items:
        all.extend(rdict.items[x])

    # Iterate over the vars and add them to the coverage
    for v in all:
        is_time = v is 'time'
        is_coord = v in rdict.items['coords']

        pcontext = ParameterContext(v, is_coord=is_coord)
        if is_coord:
            if v is 'lat':
                pcontext.axis = 'y'
            elif v is 'lon':
                pcontext.axis = 'x'
            elif v is 'z':
                pcontext.axis = 'z'

        pcontext.param_type = ds.variables[v].dtype.type
        scov.append_parameter(pcontext, is_time)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size - 1)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in all:
        is_time = v is 'time'
        is_coord = v in rdict.items['coords']

        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
        print scov.range_[v].shape, arr.shape
        if is_coord:
            scov.range_[v][:] = var[:]
#            print scov.range_[v][:]
        else:
            scov.range_[v][:] = var[:].reshape(scov.range_[v].shape)

    SimplexCoverage.save(scov, 'test_data/usgs.cov')

    return scov, ds


if __name__ == "__main__":
    scov, ds = ncstation2cov()
    shp = scov.range_.streamflow.shape

    print '<========= Query =========>'
    slice_ = 0
    print 'sflow <shape {0}> sliced with: {1}'.format(shp,slice_)
    print scov.range_['streamflow'][slice_]

    slice_ = (slice(0,10,2),slice(None))
    print 'sflow <shape {0}> sliced with: {1}'.format(shp,slice_)
    print scov.range_['streamflow'][slice_]

    slice_ = (slice(0,10,2),slice(None),0)
    print 'sflow <shape {0}> sliced with: {1}'.format(shp,slice_)
    print scov.range_['streamflow'][slice_]

    slice_ = (0,slice(None),slice(None))
    print 'sflow <shape {0}> sliced with: {1}'.format(shp,slice_)
    print scov.range_['streamflow'][slice_]

    slice_ = ((1,5,7,),0,0)
    print 'sflow <shape {0}> sliced with: {1}'.format(shp,slice_)
    print scov.range_['streamflow'][slice_]

    print '<========= Assignment =========>'

    slice_ = (0,slice(None),slice(None))
    value = [[ 22,  22,  22,  22,  22,], [ 33,  33,  33,  33,  33], [ 44,  44,  44,  44,  44]]
    print 'sflow <shape {0}> assigned with slice: {1} and value: {2}'.format(shp,slice_,value)
    scov.range_['streamflow'][slice_] = value
    print scov.range_['streamflow'][slice_]

    slice_ = ((1,5,7,),0,0)
    value = [10, 20, 30]
    print 'sflow <shape {0}> assigned with slice: {1} and value: {2}'.format(shp,slice_,value)
    scov.range_['streamflow'][slice_] = value
    print scov.range_['streamflow'][slice_]

"""

from coverage_model.test.simple_cov import *
scov, ds = ncstation2cov()


"""

