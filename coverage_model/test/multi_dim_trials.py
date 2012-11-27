#!/usr/bin/env python

"""
@package 
@file brick_split
@author Christopher Mueller
@brief

from coverage_model.test.multi_dim_trials import *
md=MultiDim()
val_arr = np.arange(100).reshape(10,10)
sl = (slice(None),slice(None))
md.put_values_to_bricks(sl, val_arr[sl])
md.reset_bricks()



from coverage_model.test.multi_dim_trials import *
md=MultiDim()
val_arr = np.arange(100).reshape(10,10)

sl_list = []
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
    md.put_values_to_bricks(sl, val_arr[sl])
    print 'bricks:'
    for b in md.bricks:
        print '{0}\n{1}'.format(b,md.bricks[b])
    print '*'*len(tstr)


"""
from ooi.logging import log
from coverage_model import fix_slice

from copy import deepcopy
import itertools
from rtree import index
import numpy as np

class MultiDim(object):

    def __init__(self, total_domain=(10,10), brick_size=5):
        self.total_domain = total_domain
        self.brick_size = tuple(brick_size for x in total_domain)
        self.bricks = {}
        self.rtree = index.Index()

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
            # CBM TODO: May be able to leverage np.broadcast() here - raises ValueError if the two arrays can't be broadcast
            # Must account for missing 1's!!
            if v_shp != tuple([i for i in s_shp if i != 1]):
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

            brick_slice = tuple(brick_slice)
            brick_mm = tuple(brick_mm)

            if not is_broadcast:
                value_slice = []
                for x, sl in enumerate(slice_): # Dimensionality
                    vm=v_shp[x] if x < len(v_shp) else 1
                    vs = self.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=vm)
                    value_slice.append(vs)

                value_slice = tuple(value_slice)

            v = values[value_slice]
            bss = self.get_shape_from_slice(brick_slice, self.brick_extents[bid])
            vss = self.get_shape_from_slice(value_slice, v_shp)
            log.debug('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s\n\tvalues %s=\n%s', b, bss, brick_slice, brick_mm, vss, value_slice, v.shape, v)
            self.bricks[bid][brick_slice] = v

    def get_values_from_bricks(self, slice_):
        pass

    def calc_value_slice(self, slice_, brick_ext, brick_slice, brick_sl, val_shp_max):
        log.debug('slice_==%s\tbrick_ext==%s\tbrick_slice==%s\tbrick_sl==%s\tval_shp_max==%s', slice_, brick_ext, brick_slice, brick_sl, val_shp_max)

        brick_ext_min = brick_ext[0]
        brick_ext_max = brick_ext[1] + 1
        brick_shp = brick_ext_max - brick_ext[0]

        # Value Slice in Total Domain Notation:
        if isinstance(slice_, int):
            ts = slice_
        elif isinstance(slice, (list,tuple)):
            ts = slice_[0]
        elif isinstance(slice_, slice):
            ts = slice_.start if slice_.start is not None else 0
            if slice_.step is not None and slice_.step != 1:
                brick_ext_min = len(xrange(*slice_.indices(brick_ext_min))) + ts
                brick_ext_max = len(xrange(*slice_.indices(brick_ext_max))) + ts
                log.debug('STEP ADJUSTMENT: brick_ext_min=%s\tbrick_ext_max=%s', brick_ext_min, brick_ext_max)
        else:
            ts = 0

        log.debug('ts = %s', ts)


        val_sl_tn_min = max(ts, brick_ext_min)
        val_sl_tn_max = brick_sl[1] + brick_ext_max - brick_shp

        log.debug('val_sl_tn_min/max = %s/%s', val_sl_tn_min, val_sl_tn_max)

        val_sl_min = val_sl_tn_min - ts
        val_sl_max = val_sl_tn_max - ts

        value_slice = slice(val_sl_min, val_sl_max, None)
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
        log.debug('%s  %s', slice_, bounds)
        sl = deepcopy(slice_)
        bo = bounds[0]
        bn = bounds[1] + 1
        bs = bn - bo
        if isinstance(sl, int):
            if bo <= sl < bn:
                return sl-bo, (sl, sl)
            else:
                raise ValueError('Outside brick bounds: %s <= %s < %s', bo, sl, bn)
        elif isinstance(sl, (list,tuple)):
            filt_slice = [x - bo for x in sl if bo <= x < bn]
            if len(filt_slice) > 0:
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

            log.debug('start=%s, stop=%s', start, stop)
            if bo != 0 and sl.step is not None and sl.step != 1:
                try:
                    ss = 0 if sl.start is None else sl.start
                    calc_boo = xrange(*slice(ss,bn,sl.step).indices(bo+sl.step))[-1] - bo
                except:
                    calc_boo = 0
                log.debug('calc_boo=%s',calc_boo)
                start += calc_boo
            log.debug('start=%s, stop=%s', start, stop)
            nbs = slice(start, stop, sl.step)
            nbsi = range(*nbs.indices(stop))
            nbsl = len(nbsi)
            if nbsl == 0: # No values in this brick!!
#                if sl.step is not None:
#                    brick_origin_offset -= bs
#                return None, brick_origin_offset
                return None
            log.debug('nbsl=%s',nbsl)

            bsl_min = nbs.start
            bsl_max = bs

            return nbs, (bsl_min, bsl_max)

    def do_recurse(self, slice_, depth=None):
        if depth is None:
            depth = len(self.total_domain)-1

        if depth != 0:
            self.do_recurse(slice_, depth-1)

        sl=slice_[depth]

        log.debug('depth %s: %s', depth, sl)
        log.debug(self.total_domain[depth])




