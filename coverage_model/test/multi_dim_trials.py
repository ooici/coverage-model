#!/usr/bin/env python

"""
@package 
@file brick_split
@author Christopher Mueller
@brief

from coverage_model.test.multi_dim_trials import *
md=MultiDim()
val_arr = np.arange(100).reshape(10,10)
#sl = (slice(None),slice(None))
#sl = ([1,2,5,8],2)
#sl = (2,[1,2,5,8])
sl = (slice(6,9,2),)
md.put_values_to_bricks(sl, val_arr[sl])
md.reset_bricks()



from coverage_model.test.multi_dim_trials import *
md=MultiDim()
val_arr = np.arange(100).reshape(10,10)

sl_list = []
# Single value slices
sl_list.append((0,1))
sl_list.append((9,3))
sl_list.append((3,8))
sl_list.append((8,7))

# List slices
sl_list.append(([1,2],2))
sl_list.append(([1,4],6))
sl_list.append(([6,9],3))
sl_list.append(([1,2,5,8],5))
sl_list.append((2,[2,5,6,9]))

# Slice slices
sl_list.append((slice(6,9,2),))
sl_list.append((slice(2,7),slice(3,8)))
sl_list.append((slice(1,None),slice(4,8)))
sl_list.append((slice(None),slice(None)))
sl_list.append((slice(2,8),slice(None)))
sl_list.append((slice(2,8),slice(3,6)))
sl_list.append((slice(None,None,3),slice(None,None,2)))
sl_list.append((slice(1,8,3),slice(3,None,2)))
sl_list.append((slice(3,None),slice(3,9,2)))

for sl in sl_list:
    tstr = '*** Slice: {0} ***'.format(sl)
    print tstr
    md.reset_bricks()
    vals = val_arr[sl]
    md.put_values_to_bricks(sl, vals)
    vo=md.get_values_from_bricks(sl)
    eq = np.array_equal(vals, vo)
    print "Equal" if eq else "Not Equal!!"
    if not eq:
        print 'vals in:\n%s' % (vals,)
        print 'vals out:\n%s' % (vo,)
#    print 'bricks:'
#    for b in md.bricks:
#        print '{0}\n{1}'.format(b,md.bricks[b])
    print '*'*len(tstr)


"""
from ooi.logging import log
from coverage_model import fix_slice
from coverage_model import bricking_utils

from copy import deepcopy
from rtree import index
import numpy as np

class MultiDim(object):

    def __init__(self, total_domain=(10,10), brick_size=5):
        self.total_domain = total_domain
        self.brick_sizes = tuple(brick_size for x in total_domain)
        self.bricks = {}
        p = index.Property()
        p.dimension=len(self.total_domain)
        self.rtree = index.Index(properties=p)

        self.brick_origins = bricking_utils.calc_brick_origins(self.total_domain, self.brick_sizes)
        self.brick_extents, self.rtree_extents = bricking_utils.calc_brick_and_rtree_extents(self.brick_origins, self.brick_sizes)
        self.build_bricks()
        bricking_utils.populate_rtree(self.rtree, self.rtree_extents, self.brick_extents)

    def build_bricks(self):
        for x in xrange(len(self.brick_origins)):
            self.bricks[x] = np.empty(self.brick_sizes, dtype='int16')
            self.bricks[x].fill(-1)

    def reset_bricks(self):
        for arr in self.bricks.itervalues():
            arr.fill(-1)

    def put_values_to_bricks(self, slice_, values):
        slice_ = fix_slice(slice_, self.total_domain)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.rtree, self.total_domain) # this is a list of tuples [(b_id, (bounds...),), ...]

        values = np.asanyarray(values)
        v_shp = values.shape
        log.debug('value_shape: %s', v_shp)
        s_shp = bricking_utils.get_shape_from_slice(slice_, self.total_domain)
        log.debug('slice_shape: %s', s_shp)
        is_broadcast = False
        if v_shp == ():
            log.debug('Broadcast!!')
            is_broadcast = True
            value_slice = ()
        elif v_shp != s_shp:
            if v_shp == tuple([i for i in s_shp if i != 1]): # Missing dimensions are singleton, just reshape to fit
                values = values.reshape(s_shp)
                v_shp = values.shape
            else:
                raise IndexError('Shape of \'value\' is not compatible with \'slice_\': slice_ shp == {0}\tvalue shp == {1}'.format(s_shp, v_shp))
        else:
            value_slice = None

        log.debug('value_shape: %s', v_shp)

        for b in bricks:
            # b is (brick_id, (brick_bounds per dim...),)
            bid, bbnds = b
            log.debug('Determining slice for brick: %s', b)
            log.debug('bid=%s, bbnds=%s', bid, bbnds)
            brick_slice = []
            brick_mm = []
            for x, sl in enumerate(slice_): # Dimensionality
                log.debug('x=%s  sl=%s', x, sl)
                log.debug('bbnds[%s]: %s', x, bbnds[x])
                bsl, mm = bricking_utils.calc_brick_slice(sl, bbnds[x])
                brick_slice.append(bsl)
                brick_mm.append(mm)

            if None in brick_slice: # Brick does not contain any of the requested indices
                continue

            brick_slice = tuple(brick_slice)
            brick_mm = tuple(brick_mm)

            if not is_broadcast:
                value_slice = []
                for x, sl in enumerate(slice_): # Dimensionality
                    vm=v_shp[x] if x < len(v_shp) else 1
                    vs = bricking_utils.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=vm)
                    value_slice.append(vs)

                value_slice = tuple(value_slice)

            bss = bricking_utils.get_shape_from_slice(brick_slice, self.brick_extents[bid])
            vss = bricking_utils.get_shape_from_slice(value_slice, v_shp)
            log.debug('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s', b, bss, brick_slice, brick_mm, vss, value_slice)
            v = values[value_slice]
            log.debug('\nvalues %s=\n%s', v.shape, v)
            self.bricks[bid][brick_slice] = v

    def get_values_from_bricks(self, slice_):
        slice_ = fix_slice(slice_, self.total_domain)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.rtree, self.total_domain) # this is a list of tuples [(b_id, (bounds...),), ...]

        ret_shp = bricking_utils.get_shape_from_slice(slice_, self.total_domain)
        ret_arr = np.empty(ret_shp, dtype='int16')

        for b in bricks:
            bid, bbnds = b
            brick_slice = []
            brick_mm = []
            for x, sl in enumerate(slice_):
                bsl, mm = bricking_utils.calc_brick_slice(sl, bbnds[x])
                brick_slice.append(bsl)
                brick_mm.append(mm)

            if None in brick_slice:
                continue

            brick_slice = tuple(brick_slice)
            brick_mm = tuple(brick_mm)

            ret_slice = []
            for x, sl in enumerate(slice_):
                rm = ret_shp[x] if x < len(ret_shp) else 1
                rs = bricking_utils.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=rm)
                ret_slice.append(rs)

            ret_slice = tuple(ret_slice)

            ret_vals = self.bricks[bid][brick_slice]
            ret_arr[ret_slice] = ret_vals

        ret_arr = ret_arr.squeeze()

        if ret_arr.size == 1:
            if ret_arr.ndim==0:
                ret_arr=ret_arr[()]
            else:
                ret_arr=ret_arr[0]

        return ret_arr


