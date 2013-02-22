#!/usr/bin/env python

"""
@package coverage_model.parameter_expressions
@file coverage_model/parameter_expressions.py
@author Christopher Mueller
@brief Classes for holding expressions evaluated against parameters
"""

from ooi.logging import log

import numpy as np
import numexpr as ne
from numbers import Number
from collections import OrderedDict
from coverage_model.basic_types import AbstractBase


class ParameterFunctionException(Exception):
    def __init__(self, message, original_type=None):
        self.original_type = original_type
        if self.original_type is not None:
            message = '{0} :: original_type = {1}'.format(message, str(original_type))
        Exception.__init__(self, message)


class AbstractFunction(AbstractBase):
    def __init__(self, name, arg_list, param_map):
        AbstractBase.__init__(self)
        self.name = name
        self.arg_list = arg_list
        self.param_map = param_map

    def _apply_mapping(self):
        if self.param_map is not None:
            keyset = set(self.param_map.keys())
            argset = set(self.arg_list)
            if not keyset.issubset(argset):
                log.warn('\'param_map\' does not contain keys for all items in \'arg_list\'; '
                         'arg will be used for missing keys = %s', keyset.difference(argset))

            args = self.arg_list
            vals = [self.param_map[a] if a in self.param_map else a for a in self.arg_list]
        else:
            args = vals = self.arg_list

        return OrderedDict(zip(args, vals))

    def _get_map_name(self, a, n):
        if a is None or a == '':
            return n
        else:
            return '{0} :|: {1}'.format(a, n)

    def evaluate(self, *args):
        raise NotImplementedError('Not implemented in abstract class')

    def get_function_map(self, pctxt_callback, parent_arg_name=None):
        arg_map = self._apply_mapping()

        ret = {}
        arg_count = 0
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                ret['arg_{0}'.format(arg_count)] = a.get_function_map(pctxt_callback, k)
            else:
                if isinstance(a, Number) or hasattr(a, '__iter__') and np.array([isinstance(ai, Number) for ai in a]).all():
                    # Treat numerical arguments as independents
                    a = '<{0}>'.format(self._get_map_name(k, a))
                else:
                    # Check to see if the argument is a ParameterFunctionType
                    try:
                        spc = pctxt_callback(a)
                        if hasattr(spc.param_type, 'get_function_map'):
                            a = spc.param_type.get_function_map(parent_arg_name=k)
                        else:
                            # An independent parameter argument
                            a = '<{0}>'.format(self._get_map_name(k, a))
                    except KeyError:
                        a = '!{0}!'.format(self._get_map_name(k, a))

                ret['arg_{0}'.format(arg_count)] = a

            arg_count += 1

        # Check to see if this expression represents a parameter
        try:
            pctxt_callback(self.name)
            n = self._get_map_name(parent_arg_name, self.name)
        except KeyError:
            # It is an intermediate expression
            n = '[{0}]'.format(self._get_map_name(parent_arg_name, self.name))

        return {n: ret}


class PythonFunction(AbstractFunction):
    def __init__(self, name, owner, func_name, arg_list, kwarg_map=None, param_map=None):
        AbstractFunction.__init__(self, name, arg_list, param_map)
        self.owner = owner
        self.func_name = func_name
        self.kwarg_map = kwarg_map

    def _import_func(self):
        import importlib

        module = importlib.import_module(self.owner)
        self._callable = getattr(module, self.func_name)

    def evaluate(self, pval_callback, slice_, fill_value=-9999):
        self._import_func()

        arg_map = self._apply_mapping()

        args = []
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                args.append(a.evaluate(pval_callback, slice_, fill_value))
            elif isinstance(a, Number) or hasattr(a, '__iter__') and np.array(
                    [isinstance(ai, Number) for ai in a]).all():
                args.append(a)
            else:
                v = pval_callback(a, slice_)
                args.append(v)

        if self.kwarg_map is None:
            return self._callable(*args)
        else:
            raise NotImplementedError('Handling for kwargs not yet implemented')
            # TODO: Add handling for kwargs
            # return self._callable(*args, **kwargs)

    def _todict(self, exclude=None):
        return super(PythonFunction, self)._todict(exclude=['_callable'])

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(PythonFunction, cls)._fromdict(cmdict, arg_masks=arg_masks)
        return ret


class NumexprFunction(AbstractFunction):
    def __init__(self, name, expression, arg_list, param_map=None):
        AbstractFunction.__init__(self, name, arg_list, param_map)
        self.expression = expression

    def evaluate(self, pval_callback, slice_, fill_value=-9999):
        arg_map = self._apply_mapping()

        ld = {}
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                ld[k] = a.evaluate(pval_callback, slice_, fill_value)
            elif isinstance(a, Number) or hasattr(a, '__iter__') and np.array(
                    [isinstance(ai, Number) for ai in a]).all():
                ld[k] = a
            else:
                ld[k] = pval_callback(a, slice_)

        return ne.evaluate(self.expression, local_dict=ld)
