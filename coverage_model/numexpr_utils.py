#!/usr/bin/env python

"""
@package 
@file numexpr_try
@author Christopher Mueller
@brief 
"""

import re
import numpy as np
import numexpr as ne

digit_match = r'[-+]?[0-9]*\.?[0-9]+?([eE][-+]?[0-9]+)?'
def nest_wheres(*args):
    """
    Generates a 'nested' numexpr expression that is the sum of the provided where expressions.

    Expressions in the *args list are combined together in entry order such that the 'else' portion of the previous
    expression is replaced with the current expression.

    <b>NOTE: Expressions are evaluated from left to right:  BE CAREFUL WITH YOUR ORDERING</b>

    Example:
    \code{.py}
        In [1]: from scratch.numexpr_utils import nest_wheres
        In [2]: expr1 = 'where(x <= 10, 10, nan)'
        In [3]: expr2 = 'where(x <= 30, 100, nan)'
        In [4]: expr3 = 'where(x < 50, 150, nan)'
        In [5]: nest_wheres(expr1, expr2, expr3)
        Out[5]: 'where(x <= 10, 10, where(x <= 30, 100, where(x < 50, 150, nan)))'
    \endcode

    @param *args    One or more str or unicode expressions matching the form: 'where(.*,.*,.*)'
    """
    where_list = [w for w in args if isinstance(w, (str, unicode)) and re.match('where(.*,.*,.*)', w)]
    if not where_list:
        raise IndexError('There are no appropriate arguments; each argument must be a str/unicode matching \'where(.*,.*,.*)\'')

    ret = where_list[0]
    for w in where_list[1:]:
        ret = ret[:ret.rindex(', ')+2] + w

    ret += (')' * (len(where_list)-1))

    return ret

def make_range_expr(val, min=None, max=None, min_incl=False, max_incl=True, else_val=None):
    """
    Generate a range expression for use in numexpr via numexpr.evaluate(expr).  Can be used to generate constant and bounded expressions.

    If neither 'min' or 'max' is supplied, a constant expression with value 'val' is returned

    Examples:
    \code{.py}
        In [1]: from scratch.numexpr_utils import make_range_expr

        In [2]: make_range_expr(10)
        Out[2]: 'c*10'

        In [3]: make_range_expr(8, min=99, else_val=-999)
        Out[3]: 'where(x > 99, 8, -999)'

        In [4]: make_range_expr(8, min=99, min_incl=True)
        Out[4]: 'where(x >= 99, 8, nan)'

        In [5]: make_range_expr(100, max=10, max_incl=False, else_val=-999)
        Out[5]: 'where(x < 10, 100, -999)'

        In [6]: make_range_expr(100, max=10, else_val=-999)
        Out[6]: 'where(x <= 10, 100, -999)'

        In [7]: make_range_expr(55, min=0, max=100, else_val=100)
        Out[7]: 'where((x > 0) & (x <= 100), 55, 100)'

        In [8]: make_range_expr(55, min=0, max=100, min_incl=True, else_val=100)
        Out[8]: 'where((x >= 0) & (x <= 100), 55, 100)'

        In [9]: make_range_expr(55, min=0, max=100, min_incl=True, max_incl=False, else_val=100)
        Out[9]: 'where((x >= 0) & (x < 100), 55, 100)'

        In [10]: make_range_expr(55, min=0, max=100, min_incl=False, max_incl=False, else_val=100)
        Out[10]: 'where((x > 0) & (x < 100), 55, 100)'
    \endcode


    @param val  The value to return if the expression is satisfied
    @param min  The minimum bound
    @param max  The maximum bound
    @param min_incl If True, the minimum bound is included (i.e. 'x >= min'); otherwise it is not (i.e. 'x > min'); default is False
    @param max_incl If True, the maximum bound is included (i.e. 'x <= max'); otherwise it is not (i.e. 'x < max'); default is True
    @param else_val The value to return if the expression is NOT satisfied; np.nan if not provided
    """
    else_val = else_val or np.nan

    # Neither minimum or maximum provided - CONSTANT
    if min is None and max is None:
        return 'c*{0}'.format(val)

    # Only max provided
    if min is None:
        return 'where(x {0} {1}, {2}, {3})'.format(('<=' if max_incl else '<'), max, val, else_val)

    # Only min provided
    if max is None:
        return 'where(x {0} {1}, {2}, {3})'.format(('>=' if min_incl else '>'), min, val, else_val)

    # Both min and max provided
    return 'where((x {0} {1}) & (x {2} {3}), {4}, {5})'.format(('>=' if min_incl else '>'), min, ('<=' if max_incl else '<'), max, val, else_val)

def test_mkrng():

    print 'c*10'
    expr = make_range_expr(10)
    print '>> {0}\n'.format(expr)

    print 'where(x > 99, 8, -999)'
    expr = make_range_expr(8, min=99, else_val=-999)
    print '>> {0}\n'.format(expr)

    print 'where(x >= 99, 8, -999)'
    expr = make_range_expr(8, min=99, min_incl=True, else_val=-999)
    print '>> {0}\n'.format(expr)

    print 'where(x < 10, 100, -999)'
    expr = make_range_expr(100, max=10, max_incl=False, else_val=-999)
    print '>> {0}\n'.format(expr)

    print 'where(x <= 10, 100, -999)'
    expr = make_range_expr(100, max=10, else_val=-999)
    print '>> {0}\n'.format(expr)

    print 'where((x > 0) & (x <= 100), 55, 100)'
    expr = make_range_expr(55, min=0, max=100, else_val=100)
    print '>> {0}\n'.format(expr)

    print 'where((x >= 0) & (x <= 100), 55, 100)'
    expr = make_range_expr(55, min=0, max=100, min_incl=True, else_val=100)
    print '>> {0}\n'.format(expr)

    print 'where((x >= 0) & (x < 100), 55, 100)'
    expr = make_range_expr(55, min=0, max=100, min_incl=True, max_incl=False, else_val=100)
    print '>> {0}\n'.format(expr)

    print 'where((x > 0) & (x < 100), 55, 100)'
    expr = make_range_expr(55, min=0, max=100, min_incl=False, max_incl=False, else_val=100)
    print '>> {0}\n'.format(expr)

def test_nest_wheres():
    nanval=np.nan
    expr1 = make_range_expr(val=np.nan, max=0, max_incl=False)
    expr2 = make_range_expr(val=111, min=0, min_incl=True, max=10, max_incl=False)
    expr3 = make_range_expr(val=222, max=20, max_incl=False, else_val=-999)

    nexpr = nest_wheres(expr1, expr2, expr3, 'should not be included', 'where(also not included)')

    print expr1
    print expr2
    print expr3
    print '\nwhere(x < 0, {0}, where((x >= 0) & (x < 10), 111, where(x < 20, 222, -999)))'.format(nanval)
    print '>> {0}'.format(nexpr)
