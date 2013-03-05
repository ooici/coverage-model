#!/usr/bin/env python

"""
@package coverage_model.numexpr_utils
@file coverage_model/numexpr_utils.py
@author Christopher Mueller
@brief Utility functions for building and verifying 'where' expressions for use with numexpr
"""

from ooi.logging import log
import re
import numpy as np
import numexpr as ne

digit_match = r'[-+]?[0-9]*\.?[0-9]+?([eE][-+]?[0-9]+)?'
digit_or_nan = r'(?:{0}|(?:nan))'.format(digit_match)
single_where_match = r'^where\(\(?x ?[><=]+ ?{0}\)?( ?&? ?\(?x ?[><=]+ ?{0}\)?)?, ?{0}, ?{0}\)$'.format(digit_or_nan)
where_match_noelse = r'where\(\(?x ?[><=]+ ?{0}\)?( ?&? ?\(?x ?[><=]+ ?{0}?\)?)?, ?{0}, ?'.format(digit_or_nan)

def is_well_formed_where(value):
    ret = False
    if is_nested_where(value):
        ret = is_well_formed_nested(value)
    elif re.match(single_where_match, value) is not None:
        ret = True

    return ret

def is_nested_where(value):
    return len(re.findall('where', value)) > 1

def is_well_formed_nested(value):
    ret = False
    where_count = len(re.findall('where', value))
    if where_count > 1:
        if where_count == len(re.findall(where_match_noelse, value)):
            expr = r'.*{0}{1}$'.format(digit_or_nan, ('\)'*where_count))
            if re.match(expr, value) is not None:
                ret = True
            else:
                log.warn('\'value\' does not end as expected: \'%s\' does not match \'%s\'', value, expr)
        else:
            log.warn('\'value\' appears to have malformed where clauses')

    return ret

def denest_wheres(where_expr, else_val=-9999):
    if not is_well_formed_where(where_expr):
        raise ValueError('\'where_expr\' does not pass is_well_formed_where: {0}'.format(where_expr))

    expr=where_expr
    where_count = len(re.findall('where', expr))
    ret=[]
    for x in range(where_count):
        m=re.match(where_match_noelse, expr)
        if m is not None:
            ret.append('{0}{1})'.format(expr[m.start():m.end()], else_val))
            expr=expr[m.end():]

    return ret

def nest_wheres(*args):
    """
    Generates a 'nested' numexpr expression that is the sum of the provided where expressions.

    Expressions in the *args list are combined together in entry order such that the 'else' portion of the previous
    expression is replaced with the current expression.

    <b>NOTE: Expressions are evaluated from left to right:  BE CAREFUL WITH YOUR ORDERING</b>

    Example:
    \code{.py}
        In [1]: from coverage_model.numexpr_utils import nest_wheres
        In [2]: expr1 = 'where(x <= 10, 10, nan)'
        In [3]: expr2 = 'where(x <= 30, 100, nan)'
        In [4]: expr3 = 'where(x < 50, 150, nan)'
        In [5]: nest_wheres(expr1, expr2, expr3)
        Out[5]: 'where(x <= 10, 10, where(x <= 30, 100, where(x < 50, 150, nan)))'
    \endcode

    @param *args    One or more str or unicode expressions that pass the is_well_formed_where function
    """
    where_list = [w for w in args if isinstance(w, basestring) and is_well_formed_where(w)]
    if not where_list:
        raise IndexError('There are no appropriate arguments; each argument must be a basestring matching \'where(.*,.*,.*)\'')

    ret = where_list[0]
    for w in where_list[1:]:
        ret = ret[:ret.rindex(', ')+2] + w

    ret += (')' * (len(where_list)-1))

    return ret

def make_range_expr(val, min=None, max=None, min_incl=False, max_incl=True, else_val=-9999):
    """
    Generate a range expression for use in numexpr via numexpr.evaluate(expr).  Can be used to generate constant and bounded expressions.

    If neither 'min' or 'max' is supplied, a constant expression with value 'val' is returned

    Examples:
    \code{.py}
        In [1]: from coverage_model.numexpr_utils import make_range_expr

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
    @param else_val The value to return if the expression is NOT satisfied; -9999 if not provided
    """
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
