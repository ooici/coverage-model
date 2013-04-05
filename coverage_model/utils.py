#!/usr/bin/env python

"""
@package 
@file utils
@author Christopher Mueller
@brief 
"""

import numpy as np
import uuid
import re
import math
from putil.hash import hash_any


def create_guid():
    """
    @retval Return global unique id string
    """
    # guids seem to be more readable if they are UPPERCASE
    return str(uuid.uuid4()).upper()


def is_guid(str_val):
    guid_match = r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}'
    return re.match(guid_match, str_val) is not None


def prod(lst):
    import operator
    return reduce(operator.mul, lst, 1)


def is_valid_constraint(v):
    from coverage_model.basic_types import Span
    ret = False
    if isinstance(v, (slice, int)) or isinstance(v, Span) or\
       (isinstance(v, (list,tuple)) and np.array([is_valid_constraint(e) for e in v]).all()):
        ret = True

    return ret


def find_nearest_index(seq, val):
    """
    Returns the index of the value in the array-like object <i>arr</i> nearest to <i>val</i>.

    <i>seq</i> is viewed as a numpy array using numpy.asanyarray()

    If <i>seq</i> contains duplicates, the first match is returned

    @param seq  The array-like object to search
    @param val  The value to search for
    @return     The index nearest to <i>val</i>
    """
    idx = np.abs(np.asanyarray(seq)-val).argmin()
    return idx


def find_nearest_value(seq, val):
    """
    Returns the value in the array-like object <i>arr</i> nearest to but not greater than <i>val</i>.

    <i>seq</i> is viewed as a numpy array using numpy.asanyarray()

    If <i>seq</i> contains duplicates, the first match is returned

    @param seq  The array-like object to search
    @param val  The value to search for
    @return     The value nearest to but not greater than <i>val</i>
    """
    a = np.asanyarray(seq)
    idx = find_nearest_index(a, val)
    fv = a.flat[idx]
    if val < fv:
        return a.flat[idx-1]
    else:
        return fv


def _raise_index_error_int(slice_, size, dim):
    if slice_ < 0:
        raise IndexError('On dimension {0}; index cannot be < 0: index => {0}'.format(dim, slice_))
    if slice_ >= size:
        raise IndexError('On dimension {0}; index cannot be >= the size: index => {1}, size => {2}'.format(dim, slice_, size))


def _raise_index_error_list(slice_, size, dim):
    last = -1
    for i in xrange(len(slice_)):
        c=slice_[i]
        if last >= c:
            raise IndexError('On dimension {0}; indices must be in increasing order and cannot be duplicated: list => {1}'.format(dim, slice_))
        last = c
        if slice_[i] >= size:
            raise IndexError('On dimension {0}; index {1} of list cannot be >= the size: list => {2}, size => {3}'.format(dim, i, slice_, size))


def _raise_index_error_slice(slice_, size, dim):
    if slice_.start is not None:
        if slice_.start < 0:
            raise IndexError('On dimension {0}; start index of slice cannot be < 0: slice => {1}, size => {2}'.format(dim, slice_, size))
        if slice_.start >= size:
            raise IndexError('On dimension {0}; start index of slice cannot be >= size: slice => {1}, size => {2}'.format(dim, slice_, size))
    if slice_.stop is not None:
        if slice_.start is None and slice_.stop == 0:
            raise IndexError('On dimension {0}; stop index of slice cannot be == 0 when the start index is None: slice => {1}, size => {2}'.format(dim, slice_, size))
        if slice_.stop < 0:
            raise IndexError('On dimension {0}; stop index of slice cannot be < 0: slice => {1}, size => {2}'.format(dim, slice_, size))
        # This case seems to be counter to how numpy functions - if a slice is longer than the shape, it's simply truncated to the appropriate size...
#        if slice_.stop > size:
#            raise IndexError('On dimension {0}; stop index of slice cannot be > size: slice => {1}, size => {2}'.format(dim, slice_, size))
    if slice_.start is not None and slice_.stop is not None:
        if slice_.start >= slice_.stop:
            raise IndexError('On dimension {0}; start index of slice cannot be >= stop index: slice => {1}'.format(dim, slice_))


def get_shape_from_slice(slice_, max_shp):
    """
    @deprecated Use slice_shape instead
    """
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


def fix_slice(slice_, shape):
    from coverage_model.basic_types import Span
    # CBM: First swack - see this for more possible checks: http://code.google.com/p/netcdf4-python/source/browse/trunk/netCDF4_utils.py
    if not is_valid_constraint(slice_):
        raise SystemError('invalid constraint supplied: {0}'.format(slice_))

    # First, ensure we're working with a list so we can make edits...
    if not np.iterable(slice_):
        slice_ = [slice_,]
    elif not isinstance(slice_,list):
        slice_ = list(slice_)

    # If shape is an integer, tuplize it
    if isinstance(shape, int):
        shape = (shape,)

    # Then make sure it's the correct rank
    rank = len(shape)
    slen = len(slice_)
    if not slen == rank:
        if slen > rank:
            slice_ = slice_[:rank]
        else:
            for n in xrange(slen, rank):
                slice_.append(slice(None,None,None))

    # Next, deal with negative indices and check for IndexErrors
    # Logic for handling negative indices from: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
    for x in xrange(rank):
        sl = slice_[x]
        if isinstance(sl, int):
            if sl < 0:
                sl = shape[x] + slice_[x]

            _raise_index_error_int(sl, shape[x], x)
            slice_[x] = sl
        elif isinstance(sl, list):
            for i in xrange(len(sl)):
                if sl[i] < 0:
                    sl[i] = shape[x] + sl[i]

            _raise_index_error_list(sl, shape[x], x)
            slice_[x] = sl
        elif isinstance(sl, slice):
            sl_ = [sl.start, sl.stop, sl.step]
            if sl_[0] is not None and sl_[0] < 0:
                sl_[0] = shape[x] + sl_[0]

            if sl_[1] is not None and sl_[1] < 0:
                sl_[1] = shape[x] + sl_[1]

            if sl_[2] is not None:
                if sl_[2] < 0:
                    sl_[0], sl_[1], sl_[2] = sl_[1], sl_[0], -sl_[2]

#                # Fix the stop value based on the step
#                s=sl_[0] if sl_[0] is not None else 0
#                e=sl_[1] if sl_[1] is not None and sl_[1] <= shape[x] else shape[x]
#                m=xrange(s, e, sl_[2])[-1]
#                m += 1 # Stop is non-inclusive
#                sl_[1] = m

            # Reassign the slice
            slice_[x] = slice(*sl_)

            _raise_index_error_slice(slice_[x], shape[x], x)
        elif isinstance(sl, Span):

            if sl.lower_bound is not None and sl.lower_bound < 0:
                sl.lower_bound = shape[x] + sl.lower_bound

            if sl.upper_bound is not None and sl.upper_bound < 0:
                sl.upper_bound = shape[x] + sl.upper_bound

            slice_[x] = sl

            # Spans can be assigned BEFORE domain expansion, so checking for IndexErrors is tricky (if even possible)...

    # Finally, make it a tuple
    return tuple(slice_)


def slice_shape(slice_, shape):
    """
    Returns a tuple containing the length of each dimension
    @param slice_       A slice, integer or list/tuple of indices
    @param shape        The shape of the data
    """
    from coverage_model.basic_types import Span
    # If shape is an integer, tuplize it
    if isinstance(shape, int):
        shape = (shape,)

    fixed_slice = fix_slice(slice_, shape)

    dim_lengths = []

    for s,shape in zip(fixed_slice, shape):
        if isinstance(s, (slice, Span)):
            if isinstance(s, Span):
                s = s.to_slice()
            start, stop, stride = s.indices(shape)
            dim_len = int(math.ceil((stop - start) / float(stride)))
        elif isinstance(s, (list,tuple)):
            dim_len = len(s)
        elif isinstance(s, int):
            dim_len = 1
        else:
            raise TypeError('Unsupported slice method') # TODO: Better error message
        
        dim_lengths.append(dim_len)

    return tuple(dim_lengths)


def express_slice(slice_, total_shape):
    from coverage_model.basic_types import Span
    fsl = fix_slice(slice_, total_shape)
    ret = []
    for i, x in enumerate(fsl):
        if isinstance(x, slice):
            start = x.start if not x.start is None else 0
            stop = x.stop if not x.stop is None else total_shape[i]
            step = x.step if not x.step is None else 1
            ret.append(slice(start, stop, step))
        elif isinstance(x, Span):
            low = x.lower_bound if not x.lower_bound is None else 0
            up = x.upper_bound if not x.upper_bound is None else total_shape[i]
            ret.append(Span(low, up, x.offset, x.value))
        else:
            ret.append(x)

    return tuple(ret)

