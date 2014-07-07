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
import os
from numbers import Number
from collections import OrderedDict
from coverage_model.basic_types import AbstractBase
from coverage_model.parameter_data import NumpyDictParameterData


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

    @classmethod
    def _get_map_name(cls, a, n):
        if a is None or a == '':
            return n
        else:
            return '{0} :|: {1}'.format(a, n)

    @classmethod
    def _parse_map_name(cls, name):
        try:
            a, n = name.split(':|:')
            a = a.strip()
            n = n.strip()
        except ValueError:
            return '', name

        return a, n

    def evaluate(self, *args):
        raise NotImplementedError('Not implemented in abstract class')

    def get_module_dependencies(self):
        deps = set()

        if hasattr(self, 'expression'):  # NumexprFunction
            deps.add('numexpr')
        elif hasattr(self, 'owner'):  # PythonFunction
            deps.add(self.owner)

        arg_map = self._apply_mapping()
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                deps.update(a.get_module_dependencies())

        return tuple(deps)

    def get_function_map(self, pctxt_callback=None, parent_arg_name=None):
        if pctxt_callback is None:
            log.warn('\'_pctxt_callback\' is None; using placeholder callback')

            def raise_keyerror(*args):
                raise KeyError()
            pctxt_callback = raise_keyerror

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

    def __eq__(self, other):
        ret = False
        if isinstance(other, AbstractFunction):
            sfm = self.get_function_map()
            ofm = other.get_function_map()
            ret = sfm == ofm

        return ret

    def __ne__(self, other):
        return not self == other


class PythonFunction(AbstractFunction):
    def __init__(self, name, owner, func_name, arg_list, kwarg_map=None, param_map=None, egg_uri='', remove_fills=True):
        AbstractFunction.__init__(self, name, arg_list, param_map)
        self.owner = owner
        self.func_name = func_name
        self.kwarg_map = kwarg_map
        self.egg_uri = egg_uri

    def _import_func(self):
        try:
            import importlib

            module = importlib.import_module(self.owner)
            self._callable = getattr(module, self.func_name)
        except ImportError:
            if self.egg_uri:
                self.download_and_load_egg(self.egg_uri)
                module = importlib.import_module(self.owner)
                self._callable = getattr(module, self.func_name)
            else:
                raise

    def evaluate(self, pval_callback, time_segment, fill_value=-9999, stride_length=None):
        self._import_func()

        arg_map = self._apply_mapping()

        args = []
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                args.append(a.evaluate(pval_callback, time_segment, fill_value))
            elif isinstance(a, Number) or hasattr(a, '__iter__') and np.array(
                    [isinstance(ai, Number) for ai in a]).all():
                args.append(a)
            else:
                if k == 'pv_callback':
                    args.append(lambda arg: pval_callback(arg, time_segment))
                else:
                    v = pval_callback(a, time_segment)
                    if isinstance(v, NumpyDictParameterData):
                        v = v.get_data()[a]
                    if k.endswith('*'):
                        v = v[-1]
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

    def __eq__(self, other):
        ret = False
        if super(PythonFunction, self).__eq__(other):
            ret = self.owner == other.owner and self.func_name == other.func_name

        return ret

    @classmethod
    def download_and_load_egg(cls, url):
        '''
        Downloads an egg from the URL specified into the cache directory
        Returns the full path to the egg
        '''
        from tempfile import gettempdir
        import os
        import requests
        import pkg_resources
        # Get the filename based on the URL
        filename = url.split('/')[-1]
        # Store it in the $TMPDIR
        egg_cache = gettempdir()
        path = os.path.join(egg_cache, filename)
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            # Download the file using requests stream
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            # Add it to the working set of eggs
            pkg_resources.working_set.add_entry(path)
            return

        raise IOError("Couldn't download the file at %s" % url)


class NumexprFunction(AbstractFunction):
    def __init__(self, name, expression, arg_list, param_map=None):
        AbstractFunction.__init__(self, name, arg_list, param_map)
        self.expression = expression

    def evaluate(self, pval_callback, time_segment, fill_value=-9999, stride_length=None):
        arg_map = self._apply_mapping()

        ld = {}
        for k in self.arg_list:
            a = arg_map[k]
            if isinstance(a, AbstractFunction):
                ld[k] = a.evaluate(pval_callback, time_segment, fill_value, stride_length=stride_length)
            elif isinstance(a, Number) or hasattr(a, '__iter__') and np.array(
                    [isinstance(ai, Number) for ai in a]).all():
                ld[k] = a
            else:
                if k.endswith('*'):
                    vals = pval_callback(a, time_segment, stride_length)
                    if isinstance(vals, NumpyDictParameterData):
                        vals = vals.get_data()[a]
                    ld[k[:-1]] = vals[-1]
                else:
                    vals = pval_callback(a, time_segment, stride_length=stride_length)
                    if isinstance(vals, NumpyDictParameterData):
                        vals = vals.get_data()[a]
                    ld[k] = vals

        return ne.evaluate(self.expression, local_dict=ld)

    def __eq__(self, other):
        ret = False
        if super(NumexprFunction, self).__eq__(other):
            ret = self.expression == other.expression

        return ret


class ExternalFunction(AbstractFunction):
    def __init__(self, name, external_guid, external_name):
        self.external_name = external_name
        param_map = {external_name : external_guid}
        AbstractFunction.__init__(self, name, [], param_map)

    def load_coverage(self, pdir):
        from coverage_model.coverage import AbstractCoverage
        external_guid = self.param_map[self.external_name]
        cov = AbstractCoverage.resurrect(external_guid, mode='r')
        return cov

    def evaluate(self, pval_callback, pdir, time_segment, fill_value=-9999):
        return self.linear_map(pval_callback, pdir, time_segment)

    def linear_map(self, pval_callback, pdir, time_segment):
        cov = self.load_coverage(pdir)
        # TODO: Might not want to hard-code time
        x = pval_callback('time', time_segment).get_data()['time']
        x_i = cov.get_parameter_values('time', time_segment=time_segment).get_data()['time']
        y_i = cov.get_parameter_values(self.external_name, time_segment=time_segment).get_data()[self.external_name]


        # Where in x_i does x fit in?
        upper = np.searchsorted(x_i, x)
        # Clip values not in [1, N-1]
        upper = upper.clip(1, len(x_i)-1).astype(int)
        lower = upper - 1

        # Linear interpolation
        w = (x - x_i[lower]) / (x_i[upper] - x_i[lower])
        y = y_i[lower] * (1-w) + y_i[upper] * w
        return y


