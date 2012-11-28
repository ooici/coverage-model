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
sl = ([1,2,5,8],2)
#sl = (2,[1,2,5,8])
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
    print "\tInnie == Outtie:  %s" % (eq,)
    if not eq:
        print 'vals in:\n%s' % (vals,)
        print 'vals out:\n%s' % (vo,)
#    print 'bricks:'
#    for b in md.bricks:
#        print '{0}\n{1}'.format(b,md.bricks[b])
    print '*'*len(tstr)


"""
from ooi.logging import log
from coverage_model import fix_slice, utils

from copy import deepcopy
import itertools
from rtree import index
import numpy as np

class MultiDim(object):

    def __init__(self, total_domain=(10,10), brick_size=5):
        self.total_domain = total_domain
        self.brick_size = tuple(brick_size for x in total_domain)
        self.bricks = {}
        p = index.Property()
        p.dimension=len(self.total_domain)
        self.rtree = index.Index(properties=p)

        self.calc_brick_origins()
        self.calc_brick_extents()
        self.build_bricks()
        self.populate_rtree()

    def calc_brick_origins(self):
        bo = list(set(itertools.product(*[range(d)[::self.brick_size[i]] for i,d in enumerate(self.total_domain)])))
        bo.sort()
        self.brick_origins = tuple(bo)

    def calc_brick_extents(self):
        be=[]
        rte=[]
        for ori in self.brick_origins:
            be.append(tuple(zip(ori,map(lambda o,s: o+s-1, ori, self.brick_size))))
            r = ori+tuple(map(lambda o,s: o+s-1, ori, self.brick_size))
            if len(ori) == 1:
                r = tuple([e for ext in zip(r,[0 for x in r]) for e in ext])
            rte.append(r)

        self.brick_extents = tuple(be)
        self.rtree_extents = tuple(rte)

    def build_bricks(self):
        for x in xrange(len(self.brick_origins)):
            self.bricks[x] = np.empty(self.brick_size, dtype='int16')
            self.bricks[x].fill(-1)

    def reset_bricks(self):
        for arr in self.bricks.itervalues():
            arr.fill(-1)

    def populate_rtree(self):
        for i, e in enumerate(self.rtree_extents):
            self.rtree.insert(i, e, obj=self.brick_extents[i])

    def get_bricks_from_slice(self, slice_):
        sl = deepcopy(slice_)
        sl = fix_slice(sl, self.total_domain)

        rank = len(sl)
        if rank == 1:
            rank += 1
            sl += (slice(None),)

        bnds = self.rtree.bounds
        log.debug('slice_ ==> %s', sl)
        log.debug('rtree bounds ==> %s', bnds)

        start=[]
        end=[]
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

        bricks = list(self.rtree.intersection(tuple(start+end), objects=True))
        bricks = [(b.id, b.object) for b in bricks]
        log.debug('bricks found ==> %s', bricks)
        return bricks

    def put_values_to_bricks(self, slice_, values):
        slice_ = fix_slice(slice_, self.total_domain)
        bricks = self.get_bricks_from_slice(slice_) # this is a list of tuples [(b_id, (bounds...),), ...]

        values = np.asanyarray(values)
        v_shp = values.shape
        log.debug('value_shape: %s', v_shp)
        s_shp = self.get_shape_from_slice(slice_)
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
                bsl, mm = self.calc_brick_slice(sl, bbnds[x])
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
                    vs = self.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=vm)
                    value_slice.append(vs)

                value_slice = tuple(value_slice)

            bss = self.get_shape_from_slice(brick_slice, self.brick_extents[bid])
            vss = self.get_shape_from_slice(value_slice, v_shp)
            log.debug('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s', b, bss, brick_slice, brick_mm, vss, value_slice)
            v = values[value_slice]
            log.debug('\nvalues %s=\n%s', v.shape, v)
            self.bricks[bid][brick_slice] = v

    def get_values_from_bricks(self, slice_):
        slice_ = fix_slice(slice_, self.total_domain)
        bricks = self.get_bricks_from_slice(slice_) # this is a list of tuples [(b_id, (bounds...),), ...]

        ret_shp = self.get_shape_from_slice(slice_, self.total_domain)
        ret_arr = np.empty(ret_shp, dtype='int16')

        for b in bricks:
            bid, bbnds = b
            brick_slice = []
            brick_mm = []
            for x, sl in enumerate(slice_):
                bsl, mm = self.calc_brick_slice(sl, bbnds[x])
                brick_slice.append(bsl)
                brick_mm.append(mm)

            if None in brick_slice:
                continue

            brick_slice = tuple(brick_slice)
            brick_mm = tuple(brick_mm)

            ret_slice = []
            for x, sl in enumerate(slice_):
                rm = ret_shp[x] if x < len(ret_shp) else 1
                rs = self.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=rm)
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

    def calc_value_slice(self, slice_, brick_ext, brick_slice, brick_sl, val_shp_max):
        log.debug('slice_==%s\tbrick_ext==%s\tbrick_slice==%s\tbrick_sl==%s\tval_shp_max==%s', slice_, brick_ext, brick_slice, brick_sl, val_shp_max)

        sl = deepcopy(slice_)
        brick_ext_min = brick_ext[0]
        brick_ext_max = brick_ext[1] + 1
        brick_shp = brick_ext_max - brick_ext[0]

        if isinstance(sl, int):
            ts = sl
            # Value Slice in Total Domain Notation
            val_sl_tn_min = max(ts, brick_ext_min)
            value_slice = val_sl_tn_min - ts

            log.debug('ts=%s\tbrick_ext_min=%s\tval_sl_tn_min=%s\tvalue_slice=%s', ts, brick_ext_min, val_sl_tn_min, value_slice)
        elif isinstance(sl, (list,tuple)):
            si = utils.find_nearest_index(sl, brick_sl[0] + brick_ext_min)
            ei = utils.find_nearest_index(sl, brick_sl[1] + brick_ext_min) + 1 # Slices use exclusive upper!!

            value_slice = slice(si, ei, None)
            log.debug('si=%s\tei=%s\tvalue_slice=%s', si, ei, value_slice)
        elif isinstance(sl, slice):
            ts = sl.start if sl.start is not None else 0
            if sl.step is not None and sl.step != 1:
                brick_ext_min = len(xrange(*sl.indices(brick_ext_min))) + ts
                brick_ext_max = len(xrange(*sl.indices(brick_ext_max))) + ts
                log.debug('Correct for step: step=%s\tbrick_ext_min=%s\tbrick_ext_max=%s', sl.step, brick_ext_min, brick_ext_max)

            # Value Slice in Total Domain Notation
            val_sl_tn_min = max(ts, brick_ext_min)
            val_sl_tn_max = brick_sl[1] + brick_ext_max - brick_shp


            val_sl_min = val_sl_tn_min - ts
            val_sl_max = val_sl_tn_max - ts

            value_slice = slice(val_sl_min, val_sl_max, None)
            log.debug('ts=%s\tbrick_ext_min=%s\tbrick_ext_max=%s\tval_sl_tn_min=%s\tval_sl_tn_max=%s\tval_sl_min=%s\tval_sl_max=%s\tvalue_slice=%s', ts, brick_ext_min, brick_ext_max, val_sl_tn_min, val_sl_tn_max, val_sl_min, val_sl_max, value_slice)
        else:
            value_slice = ()
            log.debug('value_slice=%s', value_slice)

        return value_slice

    def get_shape_from_slice(self, slice_, max_shp=None):
        log.debug('Getting array shape for slice_: %s', slice_)

        if len(self.brick_extents) == 0:
            return 0

        if max_shp is None:
            max_shp = self.total_domain

        shp=[]
        for i, s in enumerate(slice_):
            if isinstance(s, int):
                shp.append(1)
            elif isinstance(s, (list,tuple)):
                shp.append(len(s))
            elif isinstance(s, slice):
                st=min(s.stop, max_shp[i]) if s.stop is not None else max_shp[i]
                shp.append(len(range(*s.indices(st))))

        return tuple(shp)


    def calc_brick_slice(self, slice_, bounds):
        log.debug('slice_=%s\tbounds=%s', slice_, bounds)
        sl = deepcopy(slice_)
        bo = bounds[0]
        bn = bounds[1] + 1
        bs = bn - bo
        if isinstance(sl, int):
            if bo <= sl < bn:
                brick_slice = sl-bo
                log.debug('slice_ is int: bo=%s\tbn=%s\tbrick_slice=%s',bo, bn, brick_slice)
                return brick_slice, (sl, sl)
            else:
                raise ValueError('Outside brick bounds: %s <= %s < %s', bo, sl, bn)
        elif isinstance(sl, (list,tuple)):
            filt_slice = [x - bo for x in sl if bo <= x < bn]
            if len(filt_slice) > 0:
                log.debug('slice_ is list: bo=%s\tbn=%s\tfilt_slice=%s',bo, bn, filt_slice)
                return filt_slice, (min(filt_slice), max(filt_slice))
            else:
                raise ValueError('No values within brick bounds: %s <= %s < %s', bo, sl, bn)
        elif isinstance(sl, slice):
            if sl.start is None:
                start = 0
            else:
                if bo <= sl.start < bn:
                    start = sl.start - bo
                elif bo > sl.start:
                    start = 0
                else:
                    raise ValueError('Slice not in brick: %s > %s', sl.start, bn)

            if sl.stop is None:
                stop = bs
            else:
                if bo < sl.stop <= bn:
                    stop = sl.stop - bo
                elif sl.stop > bn:
                    stop = bs
                else: #  bo > sl.stop
                    raise ValueError('Slice not in brick: %s > %s', bo, sl.stop)

            if bo != 0 and sl.step is not None and sl.step != 1:
                log.debug('pre-step-adjustment: start=%s\tstop=%s', start, stop)
                try:
                    ss = 0 if sl.start is None else sl.start
                    brick_origin_offset = xrange(*slice(ss,bn,sl.step).indices(bo+sl.step))[-1] - bo
                except:
                    brick_origin_offset = 0
                log.debug('brick_origin_offset=%s',brick_origin_offset)
                start += brick_origin_offset
                log.debug('post-step-adjustment: start=%s\tstop=%s', start, stop)
            brick_slice = slice(start, stop, sl.step)
            if start >= stop: # Brick does not contain any of the requested indices
                return None, None

            log.debug('slice_ is slice: bo=%s\tbn=%s\tsl=%s\tbrick_slice=%s',bo, bn, sl, brick_slice)
            return brick_slice, (brick_slice.start, bs)

    def do_recurse(self, slice_, depth=None):
        if depth is None:
            depth = len(self.total_domain)-1

        if depth != 0:
            self.do_recurse(slice_, depth-1)

        sl=slice_[depth]

        log.debug('depth %s: %s', depth, sl)
        log.debug(self.total_domain[depth])




