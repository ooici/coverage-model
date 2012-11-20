#!/usr/bin/env python

"""
@package 
@file brick_split
@author Christopher Mueller
@brief

from coverage_model.test.multi_dim_trials import *
md=MultiDim()
val_arr = np.arange(100).reshape(10,10)
sl = (0,0)
md.put_values_to_bricks(sl, val_arr[sl])
md.reset_bricks()
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
        log.error('value_shape: %s', v_shp)
        s_shp = self.get_shape_from_slice(slice_)
        log.error('slice_shape: %s', s_shp)
        is_broadcast = False
        if v_shp == ():
            log.info('Broadcast!!')
            is_broadcast = True
            value_slice = ()
        elif v_shp != s_shp:
            # CBM TODO: May be able to leverage np.broadcast() here - raises ValueError if the two arrays can't be broadcast
            # Must account for missing 1's!!
            if v_shp != tuple([i for i in s_shp if i != 1]):
                raise IndexError('Shape of \'value\' is not compatible with \'slice_\': slice_ shp == {0}\tvalue shp == {1}'.format(s_shp, v_shp))
        else:
            value_slice = None

        log.error('value_shape: %s', v_shp)

        boo = [0 for x in slice_]
#        v_ori = [0 for x in v_shp]
        for b in bricks:
            # b is (brick_id, (brick_bounds per dim...),)
            bid, bbnds = b
            log.warn('Determining slice for brick: %s', b)
            log.info('bid=%s, bbnds=%s', bid, bbnds)
            brick_slice = []
            brick_mm = []
            for x, sl in enumerate(slice_): # Dimensionality
                log.info('x=%s  sl=%s', x, sl)
                log.warn('bbnds[%s]: %s', x, bbnds[x])
                log.warn('boo[%s]: %s', x, boo[x])
                bsl, nboo, mm = self.calc_brick_slice(sl, bbnds[x], boo[x])
                brick_slice.append(bsl)
                brick_mm.append(mm)
                boo[x] = nboo

            brick_slice = tuple(brick_slice)
            brick_mm = tuple(brick_mm)

            if not is_broadcast:
                value_slice = []
                td_value_slice = []
                for x, sl in enumerate(slice_): # Dimensionality
                    vm=v_shp[x] if x < len(v_shp) else 1
                    vs = self.calc_value_slice(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_max=vm)
                    value_slice.append(vs)

                value_slice = tuple(value_slice)

            v = values[value_slice]
            bss = self.get_shape_from_slice(brick_slice, self.brick_extents[bid])
            vss = self.get_shape_from_slice(value_slice, v_shp)
            log.warn('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s\n\tvalues %s=\n%s', b, bss, brick_slice, brick_mm, vss, value_slice, v.shape, v)
            self.bricks[bid][brick_slice] = v

    def get_values_from_bricks(self, slice_):
        pass

    def calc_value_slice(self, slice_, brick_ext, brick_slice, brick_sl, val_max):
        log.error('slice_==%s\tbrick_ext==%s\tbrick_slice==%s\tbrick_sl==%s', slice_, brick_ext, brick_slice, brick_sl)

        # Value Slice in Total Domain Notation:
        if isinstance(slice_, int):
            ts = slice_
            es = slice_
        elif isinstance(slice, (list,tuple)):
            ts = slice_[0]
            es = slice_[-1]
        elif isinstance(slice_, slice):
            ts = slice_.start if slice_.start is not None else brick_ext[0]
            es = slice_.stop if slice_.stop is not None else brick_ext[0]
        else:
            ts = 0

        #  xmin = max(total_sl.xmin, brick_ext.xmin)
        td_start = max(ts, brick_ext[0])
        # xmax = sum(brick_sl.xmax, brick_ext.xmax, -brick_ext.xshp)
        td_stop = brick_sl[1] + brick_ext[1] + 1 - (brick_ext[1] + 1 - brick_ext[0])

        log.warn('td_start==%s\ttd_stop==%s', td_start, td_stop)

        # IF(brick_ext.xmin < brick_sl.xmax, brick_ext.xmin, brick_ext.xmin - brick_sl.xmax)
        if brick_ext[0] < brick_sl[1]:
            start = brick_ext[0]
        else:
            start = brick_ext[0] - brick_sl[1]

        # sum(brick_ext.xmax, -brick_ext.xmin, -brick_sl.xmin)
        stop = (brick_ext[1]+1) - brick_ext[0] - brick_sl[0]

        if start >= val_max:
            offset = start - 0
            start -= offset
            stop -= offset

        value_slice = slice(start, stop, None)
        total_domain_value_slice = slice(td_start, td_stop, None)

        log.warn('\n\ttd_value_slice=%s\n\tvalue_slice=%s', total_domain_value_slice, value_slice)

        if isinstance(slice_, slice) and slice_.start is None:
            log.error('slice_ starts with None, using total_doman_value_slice')
            value_slice = total_domain_value_slice

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


    def calc_brick_slice(self, slice_, bounds, brick_origin_offset=0):
        log.error('%s  %s', slice_, bounds)
        sl = deepcopy(slice_)
        bo = bounds[0]
        bn = bounds[1] + 1
        bs = bn - bo
        if isinstance(sl, int):
            if bo <= sl < bn:
                return sl-bo, brick_origin_offset, (sl, sl)
            else:
                raise ValueError('Outside brick bounds: %s <= %s < %s', bo, sl, bn)
        elif isinstance(sl, (list,tuple)):
            filt_slice = [x - bo for x in sl if bo <= x < bn]
            if len(filt_slice) > 0:
                return filt_slice, brick_origin_offset, (min(filt_slice), max(filt_slice))
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
            log.debug('brick_origin_offset=%s', brick_origin_offset)
            start += brick_origin_offset
            log.debug('start=%s, stop=%s', start, stop)
            nbs = slice(start, stop, sl.step)
            nbsi = range(*nbs.indices(stop))
            nbsl = len(nbsi)
            if nbsl == 0: # No values in this brick!!
                if sl.step is not None:
                    brick_origin_offset -= bs
                return None, brick_origin_offset
            last_index = nbsi[-1]
            log.debug('last_index=%s',last_index)
            log.debug('nbsl=%s',nbsl)

            if sl.step is not None:
                brick_origin_offset = last_index - bs + sl.step
                log.debug('brick_origin_offset = %s', brick_origin_offset)

            bsl_min = nbs.start
            bsl_max = last_index + 1 # ?? Is this right ??

            return nbs, brick_origin_offset, (bsl_min, bsl_max)

    def do_recurse(self, slice_, depth=None):
        if depth is None:
            depth = len(self.total_domain)-1

        if depth != 0:
            self.do_recurse(slice_, depth-1)

        sl=slice_[depth]

        log.warn('depth %s: %s', depth, sl)
        log.error(self.total_domain[depth])




