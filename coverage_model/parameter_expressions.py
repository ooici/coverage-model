
#!/usr/bin/env python

"""
@package coverage_model.parameter_expressions
@file coverage_model/parameter_expressions.py
@author Christopher Mueller
@brief Classes for holding expressions evaluated against parameters
"""

import numexpr as ne
from coverage_model.basic_types import AbstractBase

class AbstractExpression(AbstractBase):
    def __init__(self):
        AbstractBase.__init__(self)

    def evaluate(self, *args):
        raise NotImplementedError('Not implemented in abstract class')


class PythonExpression(AbstractExpression):
    def __init__(self, owner, callable, arg_list, kwarg_map=None):
        AbstractExpression.__init__(self)
        self.owner = owner
        self.callable = callable
        self.arg_list = arg_list
        self.kwarg_map = kwarg_map

        self._setup()

    def _setup(self):
        module = __import__(self.owner)
        self._callable = getattr(module, self.callable)

    def evaluate(self, pval_callback, ptype, slice_):
        args = []
        for a in self.arg_list:
            if isinstance(a, AbstractExpression):
                args.append(a.evaluate(pval_callback, ptype, slice_))
            else:
                args.append(pval_callback(a, slice_))

        # TODO: Add similar handling for kwargs
        if self.kwarg_map is None:
            return self._callable(*args)

    def _todict(self, exclude=None):
        return super(PythonExpression, self)._todict(exclude=['_callable'])

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(PythonExpression, cls)._fromdict(cmdict, arg_masks=arg_masks)
        ret._setup()
        return ret


class NumexprExpression(AbstractExpression):
    def __init__(self, expression, param_map):
        AbstractExpression.__init__(self)
        self._expr = expression
        self._param_map = param_map

    def evaluate(self, pval_callback, ptype, slice_):
        ld={v:pval_callback(p, slice_) for v, p in self._param_map.iteritems()}

        return ne.evaluate(self._expr, local_dict=ld)