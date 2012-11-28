#!/usr/bin/env python

"""
@package coverage_model.bricking_utils
@file coverage_model/bricking_utils.py
@author Christopher Mueller
@brief Utility methods for slicing across bricks and value arrays
"""

from ooi.logging import log
from coverage_model import utils
from copy import deepcopy

def get_shape_from_slice(slice_, max_shp):
    log.debug('Getting array shape for slice_: %s', slice_)

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

def calc_brick_slice(slice_, bounds):
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

def calc_value_slice(slice_, brick_ext, brick_slice, brick_sl, val_shp_max):
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