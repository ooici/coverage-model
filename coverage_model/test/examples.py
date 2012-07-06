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

def ncgrid2cov(save_coverage=True):
    ds = Dataset('test_data/ncom.nc')

    rdict = RangeDictionary()
    rdict.items = {
        'coords':['time','lat','lon','depth',],
        'vars':['water_u','water_v','salinity','water_temp',],
        }

    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    tdom = GridDomain(GridShape('temporal', [1]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [34,57,89]), scrs, MutabilityEnum.IMMUTABLE) # 3d spatial topology (grid)

    scov = SimplexCoverage('sample grid coverage_model', rdict, sdom, tdom)

    # Collect all var names (for convenience)
    all = []
    for x in rdict.items:
        all.extend(rdict.items[x])

    # Iterate over the vars and add them to the coverage_model
    for v in all:
        is_time = v is 'time'
        is_coord = v in rdict.items['coords']

        pcontext = ParameterContext(v, is_coord=is_coord, param_type=ds.variables[v].dtype.type)
        if is_coord:
            if v is 'time':
                pcontext.axis = AxisTypeEnum.TIME
            elif v is 'lat':
                pcontext.axis = AxisTypeEnum.LAT
            elif v is 'lon':
                pcontext.axis = AxisTypeEnum.LON
            elif v is 'depth':
                pcontext.axis = AxisTypeEnum.HEIGHT

        scov.append_parameter(pcontext)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size - 1)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in all:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
#        print 'variable = \'{2}\' coverage_model range shape: {0}  array shape: {1}'.format(scov.range_[v].shape, arr.shape, v)

        # TODO: Sort out how to leave these sparse internally and only broadcast during read
        if v is 'depth':
            z,_,_ = my_meshgrid(arr,np.zeros([57]),np.zeros([89]),indexing='ij',sparse=True)
            scov.range_[v][:] = z
        elif v is 'lat':
            _,y,_ = my_meshgrid(np.zeros([34]),arr,np.zeros([89]),indexing='ij',sparse=True)
            scov.range_[v][:] = y
        elif v is 'lon':
            _,_,x = my_meshgrid(np.zeros([34]),np.zeros([57]),arr,indexing='ij',sparse=True)
            scov.range_[v][:] = x
        else:
            scov.range_[v][:] = var[:]

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/ncom.cov')

    return scov, ds

def ncstation2cov(save_coverage=True):
    ds = Dataset('test_data/usgs.nc')

    rdict = RangeDictionary()
    rdict.items = {
        'coords':['time','lat','lon','z',],
        'vars':['streamflow','water_temperature',],
    }

    tcrs = CRS.standard_temporal()
    scrs = CRS.lat_lon_height()

    tdom = GridDomain(GridShape('temporal', [1]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    #CBM: the leading 0 is for time - need to refine append_parameter to deal with the addition of time so that this is just space
    sdom = GridDomain(GridShape('spatial', [1]), scrs, MutabilityEnum.IMMUTABLE) # 1d spatial topology (station/trajectory)

    scov = SimplexCoverage('sample station coverage_model', rdict, sdom, tdom)

    # Collect all var names (for convenience)
    all = []
    for x in rdict.items:
        all.extend(rdict.items[x])

    # Iterate over the vars and add them to the coverage_model
    for v in all:
        is_time = v is 'time'
        is_coord = v in rdict.items['coords']

        pcontext = ParameterContext(v, is_coord=is_coord, param_type=ds.variables[v].dtype.type)
        if is_coord:
            if v is 'time':
                pcontext.axis = AxisTypeEnum.TIME
            elif v is 'lat':
                pcontext.axis = AxisTypeEnum.LAT
            elif v is 'lon':
                pcontext.axis = AxisTypeEnum.LON
            elif v is 'z':
                pcontext.axis = AxisTypeEnum.HEIGHT

        scov.append_parameter(pcontext)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size - 1)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in all:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
#        print 'variable = \'{2}\' coverage_model range shape: {0}  array shape: {1}'.format(scov.range_[v].shape, arr.shape, v)

        # TODO: Sort out how to leave the coordinates sparse internally and only broadcast during read
        scov.range_[v][:] = var[:]

    if save_coverage:
        SimplexCoverage.save(scov, 'test_data/usgs.cov')

    return scov, ds


def direct_read_write():
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

def methodized_read_write():
    from coverage_model.test.examples import SimplexCoverage
    import numpy as np

    cov=SimplexCoverage.load('test_data/usgs.cov')
    ra=np.zeros([0])
    cov.get_parameter_values('water_temperature',slice(None),slice(None),ra)
    cov.get_parameter_values('water_temperature',None,None,None)
    cov.get_parameter_values('water_temperature',[1,4,5],None,None)
    cov.get_parameter_values('water_temperature',slice(0,None,5),None,None)
    tdoa = DomainOfApplication(0,slice(0,10))
    sdoa = DomainOfApplication(0,slice(None))
    cov.get_parameter_values('water_temperature',tdoa,sdoa,None)


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
    scov, _ = ncstation2cov(False)
    print scov

    print '\n=======\n'

    gcov, _ = ncgrid2cov(False)
    print gcov

    #    direct_read_write()
    #    methodized_read_write()

#    from coverage_model.coverage_model import AxisTypeEnum
#    axis = 'TIME'
#    print axis == AxisTypeEnum.TIME

    pass

"""

from coverage_model.test.simple_cov import *
scov, ds = ncstation2cov()


"""