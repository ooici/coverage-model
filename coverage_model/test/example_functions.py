#!/usr/bin/env python

import numpy as np

def secondary_interpolation(x, range0, range1, starts, ends):
    """
    Description:

        OOI 1-D Interpolation quality control algorithm used to perform a
        one-dimensional interpolation. 

    Implemented by:

        2014-05-28: Luke Campbell. Initial Code

    Usage:

    References:

    """
    result = np.empty_like(x, dtype=np.float32)
    starts = np.unique(starts)
    ends = np.unique(ends)
    windows = np.column_stack([starts,ends])

    for i, x_i in enumerate(x):
        window = _intersects(x_i, windows)
        if window is None:
            result[i] = np.nan
            continue

        # Project the x_i onto the closed interval [0,1]
        # t=0, where x_i == window[0]
        # t=1, where x_i == window[1]
        t = (x_i - window[0]) / ( window[1] - window[0])
        result[i] = (1-t) * range0[i] + t * range1[i]
    return result



def _intersects(x, windows):
    '''
    Returns the first interval/window where x intersects, or None
    '''
    for window in windows:
        if x>= window[0] and x <= window[1]:
            return window
    return None

def identity(x):
    '''
    Identify Function
    '''
    return np.copy(x)

def polyval_calibration(coefficients, x):
    retval = np.empty_like(x)
    print coefficients[0:2]
    print type(coefficients[0])
    # Strip away the record array
    coefficients = coefficients.view('float32').reshape(coefficients.shape + (-1,))
    for i in xrange(x.shape[0]):
        c_i = coefficients[i]
        r = np.polyval(c_i, [x[i]])
        retval[i] = r
    return retval


