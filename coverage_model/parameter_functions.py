
#!/usr/bin/env python

"""
@package coverage_model.parameter_expressions
@file coverage_model/parameter_expressions.py
@author Christopher Mueller
@brief Classes for holding expressions evaluated against parameters
"""

import numpy as np
import numexpr as ne
from coverage_model.basic_types import AbstractBase

class AbstractFunction(AbstractBase):
    def __init__(self, name):
        AbstractBase.__init__(self)
        self.name = name

    def evaluate(self, *args):
        raise NotImplementedError('Not implemented in abstract class')

    def get_function_map(self, pctxt_callback):
        raise NotImplementedError('Not implemented in abstract class')

class PythonFunction(AbstractFunction):
    def __init__(self, name, owner, callable, arg_list, kwarg_map=None):
        AbstractFunction.__init__(self, name)
        self.owner = owner
        self.callable = callable
        self.arg_list = arg_list
        self.kwarg_map = kwarg_map

    def _import_func(self):
        module = __import__(self.owner)
        self._callable = getattr(module, self.callable)

    def evaluate(self, pval_callback, ptype, slice_):
        self._import_func()

        args = []
        for a in self.arg_list:
            if isinstance(a, AbstractFunction):
                args.append(a.evaluate(pval_callback, ptype, slice_))
            else:
                v = pval_callback(a, slice_)
                np.putmask(v, v == ptype.fill_value, np.nan)
                args.append(v)

        if self.kwarg_map is None:
            return self._callable(*args)
        else:
            raise NotImplementedError('Handling for kwargs not yet implemented')
            # TODO: Add handling for kwargs
#            return self._callable(*args, **kwargs)

    def get_function_map(self, pctxt_callback):
        ret={}
        for i, a in enumerate(self.arg_list):
            if isinstance(a, AbstractFunction):
                ret['arg_{0}'.format(i)] = a.get_function_map(pctxt_callback)
            else:
                # Check to see if the argument is a ParameterFunctionType
                try:
                    spc = pctxt_callback(a)
                    if hasattr(spc.param_type, 'get_function_map'):
                        a = spc.param_type.get_function_map()
                    else:
                        # An independent parameter argument
                        a = '<{0}>'.format(a)
                except KeyError:
                    a = 'MISSING:!{0}!'.format(a)

                ret['arg_{0}'.format(i)] = a

        # Check to see if this expression represents a parameter
        try:
            pctxt_callback(self.name)
            n = self.name
        except KeyError:
            # It is an intermediate expression
            n = '[{0}]'.format(self.name)

        return {n:ret}

    def _todict(self, exclude=None):
        return super(PythonFunction, self)._todict(exclude=['_callable'])

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(PythonFunction, cls)._fromdict(cmdict, arg_masks=arg_masks)
        ret._setup()
        return ret

class NumexprFunction(AbstractFunction):
    def __init__(self, name, expression, param_map):
        AbstractFunction.__init__(self, name)
        self._expr = expression
        self._param_map = param_map

    def evaluate(self, pval_callback, ptype, slice_):
        ld={v:pval_callback(p, slice_) for v, p in self._param_map.iteritems()}

        return ne.evaluate(self._expr, local_dict=ld)

    def get_function_map(self, pctxt_callback):
        ret = {}
        for i, a in enumerate(self._param_map.values()):
            # Check to see if the argument is a ParameterFunctionType
            try:
                spc = pctxt_callback(a)
                if hasattr(spc.param_type, 'get_function_map'):
                    a = spc.param_type.get_function_map()
                else:
                    # An independent parameter argument
                    a = '<{0}>'.format(a)
            except KeyError:
                a = 'MISSING:!{0}!'.format(a)

            ret['arg_{0}'.format(i)] = a

        # Check to see if this expression represents a parameter
        try:
            pctxt_callback(self.name)
            n = self.name
        except KeyError:
            # It is an intermediate expression
            n = '[{0}]'.format(self.name)

        return {n:ret}
