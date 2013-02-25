#!/usr/bin/env python

"""
@package coverage_model.bricking_utils
@file coverage_model/bricking_utils.py
@author Christopher Mueller
@brief Utility methods for slicing across bricks and value arrays
"""

from ooi.logging import log
from coverage_model import fix_slice, utils
from copy import deepcopy
import itertools

def calc_brick_origins(total_domain, brick_sizes):
    bo = list(set(itertools.product(*[range(d)[::brick_sizes[i]] for i,d in enumerate(total_domain)])))
    bo.sort()
    return tuple(bo)

def calc_brick_and_rtree_extents(brick_origins, brick_sizes):
    be=[]
    rte=[]
    for ori in brick_origins:
        be.append(tuple(zip(ori,map(lambda o,s: o+s-1, ori, brick_sizes))))
        r = ori+tuple(map(lambda o,s: o+s-1, ori, brick_sizes))
        if len(ori) == 1:
            r = tuple([e for ext in zip(r,[0 for x in r]) for e in ext])
        rte.append(r)

    return tuple(be), tuple(rte)

def populate_rtree(rtree, rtree_extents, brick_extents):
    for i, e in enumerate(rtree_extents):
        rtree.insert(i, e, obj=brick_extents[i])


def get_bricks_from_slice(slice_, rtree, total_domain):
    sl = deepcopy(slice_)
    sl = fix_slice(sl, total_domain)

    rank = len(sl)
    if rank == 1:
        rank += 1
        sl += (slice(None),)

    bnds = rtree.bounds
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

    bricks = list(rtree.intersection(tuple(start+end), objects=True))
    bricks = [(b.id, b.object) for b in bricks]
    log.debug('bricks found ==> %s', bricks)

    return bricks


def get_brick_slice_nd(slice_, bounds):
    if len(slice_) != len(bounds):
        raise ValueError('\'slice_\' and \'bounds\' must be of equal length')

    brick_slice = []
    brick_mm = []
    for x, sl in enumerate(slice_):  # Dimensionality
        log.debug('x=%s  sl=%s', x, sl)
        log.debug('bbnds[%s]: %s', x, bounds[x])
        try:
            bsl, mm = calc_brick_slice_1d(sl, bounds[x])
            brick_slice.append(bsl)
            brick_mm.append(mm)
        except ValueError:
            continue

    return tuple(brick_slice), tuple(brick_mm)


def calc_brick_slice_1d(slice_, bounds):
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
        else:# Brick does not contain any of the requested indices
            log.debug('Outside brick bounds: %s <= %s < %s', bo, sl, bn)
            return None, None
    elif isinstance(sl, (list,tuple)):
        filt_slice = [x - bo for x in sl if bo <= x < bn]
        if len(filt_slice) > 0:
            log.debug('slice_ is list: bo=%s\tbn=%s\tfilt_slice=%s',bo, bn, filt_slice)
            return filt_slice, (min(filt_slice), max(filt_slice))
        else:# Brick does not contain any of the requested indices
            log.debug('No values within brick bounds: %s <= %s < %s', bo, sl, bn)
            return None, None
    elif isinstance(sl, slice):
        if sl.start is None:
            start = 0
        else:
            if bo <= sl.start < bn:
                start = sl.start - bo
            elif bo > sl.start:
                start = 0
            else:# Brick does not contain any of the requested indices
                log.debug('Slice not in brick: %s > %s', sl.start, bn)
                return None, None

        if sl.stop is None:
            stop = bs
        else:
            if bo < sl.stop <= bn:
                stop = sl.stop - bo
            elif sl.stop > bn:
                stop = bs
            else: #  bo > sl.stop
                # Brick does not contain any of the requested indices
                log.debug('Slice not in brick: %s > %s', bo, sl.stop)
                return None, None

        if bo != 0 and sl.step is not None and sl.step != 1:
            log.debug('pre-step-adjustment: start=%s\tstop=%s', start, stop)
            try:
                ss = 0 if sl.start is None else sl.start
                sli = xrange(*slice(ss,bn,sl.step).indices(bo+sl.step))
                if len(sli) > 1:
                    brick_origin_offset = sli[-1] - bo
                else:
                    brick_origin_offset = 0
            except:
                brick_origin_offset = 0
            log.debug('brick_origin_offset=%s',brick_origin_offset)
            start += brick_origin_offset
            log.debug('post-step-adjustment: start=%s\tstop=%s', start, stop)
        brick_slice = slice(start, stop, sl.step)
        if start >= stop: # Brick does not contain any of the requested indices
            log.debug('Slice does not contain any of the requested indices: %s', brick_slice)
            return None, None

        log.debug('slice_ is slice: bo=%s\tbn=%s\tsl=%s\tbrick_slice=%s',bo, bn, sl, brick_slice)
        return brick_slice, (brick_slice.start, bs)


def get_value_slice_nd(slice_, v_shp, bbnds, brick_slice, brick_mm):
    if len(slice_) != len(v_shp) != len(bbnds) != len(brick_slice) != brick_mm:
        raise ValueError('All arguments must be of equal length')

    value_slice = []
    for x, sl in enumerate(slice_): # Dimensionality
        vm = v_shp[x] if x < len(v_shp) else 1
        vs = calc_value_slice_1d(sl, bbnds[x], brick_slice=brick_slice[x], brick_sl=brick_mm[x], val_shp_max=vm)
        value_slice.append(vs)

    return tuple(value_slice)


def calc_value_slice_1d(slice_, brick_ext, brick_slice, brick_sl, val_shp_max):
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