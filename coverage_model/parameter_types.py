#!/usr/bin/env python

"""
@package coverage_model.parameter_types
@file coverage_model/parameter_types.py
@author Christopher Mueller
@brief Abstract and concrete typing classes for parameters
"""


#CBM: TODO: Add type checking throughout all classes as determined appropriate, ala:
#@property
#def is_coordinate(self):
#    return self.__is_coordinate
#
#@is_coordinate.setter
#def is_coordinate(self, value):
#    if isinstance(value, bool):
#        self.__is_coordinate = value

from ooi.logging import log
from coverage_model.basic_types import AbstractIdentifiable
from coverage_model.parameter_values import ConstantValue
from coverage_model.parameter_functions import AbstractFunction
from coverage_model.numexpr_utils import digit_match, is_well_formed_where, single_where_match
from coverage_model.persistence import system_type
from coverage_model.util.numpy_utils import create_numpy_object_array
import numpy as np
import networkx as nx
import re

UNSUPPORTED_DTYPES = {np.dtype('float16'), np.dtype('complex'), np.dtype('complex64'), np.dtype('complex128')}
import platform
if platform.uname()[-2] != 'armv7l' and system_type() > 1:
    UNSUPPORTED_DTYPES.add(np.dtype('complex256'))

#==================
# Abstract Parameter Type Objects
#==================


def verify_encoding(value_encoding):
    if value_encoding is None:
        value_encoding = np.dtype('float32').str
    else:
        try:
            dt = np.dtype(value_encoding)
            if dt.isbuiltin not in (0,1):
                raise TypeError('\'value_encoding\' must be a valid numpy dtype: {0}'.format(value_encoding))
            if dt in UNSUPPORTED_DTYPES:
                raise TypeError('\'value_encoding\' {0} is not supported by H5py: UNSUPPORTED types ==> {1}'.format(value_encoding, UNSUPPORTED_DTYPES))

            value_encoding = dt.str

        except TypeError:
            raise

    return value_encoding


def verify_fill_value(value_encoding, value, is_object_type):
    if value_encoding is not None and not isinstance(value_encoding, tuple):
        dt = np.dtype(value_encoding)
        dtk = dt.kind

        if dtk == 'u': # Unsigned integer's must be positive
            if value is not None:
                return abs(value)
            else:
                return np.iinfo(dt).max

        elif dtk == 'S': # must be a string value
            return str(value)

        else:
            if value is not None:
                return value
            else:
                if dtk == 'i':
                    return np.iinfo(dt).max
                elif dtk == 'f':
                    return np.asscalar(np.finfo(dt).max)

    else:
        return value


class AbstractParameterType(AbstractIdentifiable):
    """
    Base class for parameter typing

    Provides
    """
    def __init__(self, value_module=None, value_class=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractIdentifiable; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractIdentifiable.__init__(self, **kwc)
        self._template_attrs = {}
        self._value_module = value_module or 'coverage_model.parameter_values'
        self._value_class = value_class or 'NumericValue'
        self.name = None

    def is_valid_value(self, value):
        raise NotImplementedError('Function not implemented by abstract class')

    @property
    def fill_value(self):
        if hasattr(self, '_fill_value'):
            return self._fill_value
        else:
            return None

    @fill_value.setter
    def fill_value(self, value):
        self._fill_value = verify_fill_value(self.value_encoding, value, False)

    @property
    def value_encoding(self):
        return self._value_encoding

    @value_encoding.setter
    def value_encoding(self, value):
        self._value_encoding = value

    @property
    def storage_encoding(self):
        return self._value_encoding

    def _add_graph_node(self, graph, name):
        if name.startswith('<') and name.endswith('>'):
            n = name[1:-1]
            c = 'forestgreen'
        elif name.startswith('[') and name.endswith(']'):
            n = name[1:-1]
            c = 'blue'
        elif name.startswith('!') and name.endswith('!'):
            n = name[1:-1]
            c = 'red'
        else:
            n = name
            c = 'black'

        if ':|:' in n:
            a, n = AbstractFunction._parse_map_name(n)
        else:
            a = ''

        graph.add_node(n, color=c, fontcolor=c)

        return a, n

    def _calc_param_sets(self):

        def walk(fmap, ipset, dpset):
            for k, v in fmap.iteritems():
                # if not an 'arg_#' or intermediate 'non-parameter' - add to dpset
                if 'arg' not in k:
                    if k.startswith('<') and k.endswith('>'):
                        ipset.add(k[1:-1])
                    elif k.startswith('[') and k.endswith(']'):  # Intermediate 'non parameter' - continue
                        pass
                    else:
                        # dependent parameter
                        dpset.add(AbstractFunction._parse_map_name(k)[1])

                if v is None:
                    continue
                elif isinstance(v, dict):
                    walk(v, ipset, dpset)
                elif v.startswith('<') and v.endswith('>'):
                    # independent parameter
                    ipset.add(AbstractFunction._parse_map_name(v[1:-1])[1])
                elif k.startswith('[') and k.endswith(']'):  # Intermediate 'non parameter' - continue
                    continue
                else:
                    # dependent parameter
                    dpset.add(AbstractFunction._parse_map_name(v)[1])

        ipset = set()
        dpset = set()
        fmap = self.get_function_map()
        walk(fmap, ipset, dpset)
        return tuple(ipset), tuple(dpset)

    def get_dependency_graph(self, name=None):
        graph = nx.DiGraph()

        def fmap_to_graph(fmap, graph, pnode=None):
            for k, v in fmap.iteritems():
                if 'arg' not in k:
                    a, n = self._add_graph_node(graph, k)

                    if pnode is not None:
                        graph.add_edge(pnode, n, {'label': a})
                else:
                    n = pnode

                if v is None:  # Singleton
                    pass
                elif isinstance(v, dict):
                    fmap_to_graph(v, graph, n)
                else:
                    a, n = self._add_graph_node(graph, v)

                    graph.add_edge(pnode, n, {'label': a})

        fmap = self.get_function_map()
        fmap_to_graph(fmap, graph)

        return graph

    def write_dependency_graph(self, outpath, graph=None):
        if graph is None:
            graph = self.get_dependency_graph()

        return nx.write_dot(graph, outpath)

    def get_function_map(self, parent_arg_name=None):
        return {'<{0}>'.format(self.name): None}

    def get_module_dependencies(self):
        # Return empty tuple
        return ()

    def get_independent_parameters(self):
        iparams, dparams = self._calc_param_sets()

        return iparams

    def get_dependent_parameters(self):
        iparams, dparams = self._calc_param_sets()

        return dparams

    def create_filled_array(self, size):
        arr = np.empty(size, dtype=np.dtype(self.value_encoding))
        arr[:] = self.fill_value
        return arr

    def create_data_array(self, data=None, size=None):
        if data is not None:
            return np.array(data, dtype=np.dtype(self.value_encoding))
        elif size is not None:
            return self.create_filled_array(size)
        else:
            raise RuntimeError('Not enough information to create array')

    def create_value_array(self, data=None, size=None):
        if data is not None:
            return np.array(data, dtype=np.dtype(self.value_encoding))
        elif size is not None:
            return np.zeros(size, dtype=np.dtype(self.value_encoding))
        else:
            raise RuntimeError('Not enough information to create array')

    def validate_value_set(self, value_set):
        pass

    def _gen_template_attrs(self):
        for k, v in self._template_attrs.iteritems():
            setattr(self,k,v)
            self._template_attrs[k] = None # Leave the key, but replace the value - avoid replicates

    def __eq__(self, other):
        return self.__class__.__instancecheck__(other)

    def __ne__(self, other):
        """
        Return the negative of __eq__(), implemented by concrete classes
        See http://docs.python.org/reference/datamodel.html
            "... when defining __eq__(), one should also define __ne__() so that the operators will behave as expected ..."
        """
        return not self == other

    def __hash__(self):
        """
        Designate object as explicitly unhashable - See http://docs.python.org/reference/datamodel.html
            "... If a class defines mutable objects and implements a __cmp__() or __eq__() method, it should not implement __hash__() ..."
        """
        return None



class AbstractSimplexParameterType(AbstractParameterType):
    """

    """
    def __init__(self, quality=None, nilValues=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractParameterType.__init__(self, **kwc)
        self._template_attrs['quality'] = quality
        self._template_attrs['nilValues'] = nilValues
        self._template_attrs['fill_value'] = -9999

class AbstractComplexParameterType(AbstractParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractParameterType.__init__(self, **kwc)

        self.value_encoding = np.dtype(object).str
        self._template_attrs['fill_value'] = None

    @property
    def value_encoding(self):
        if hasattr(self, 'base_type'):
            t = self.base_type
        else:
            t = self

        return t._value_encoding

    @value_encoding.setter
    def value_encoding(self, value):
        if hasattr(self, 'base_type'):
            t = self.base_type
        else:
            t = self

        t._value_encoding = value

    @property
    def storage_encoding(self):
        return self._value_encoding

#==================
# Parameter Type Objects
#==================

class ReferenceType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class BooleanType(AbstractSimplexParameterType):
    """
    BooleanType object.  The only valid values are True or False
    """
    def __init__(self, **kwargs):
        """
        Constructor for BooleanType

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, value_class='BooleanValue', **kwc)

        self._template_attrs['fill_value'] = False
        self.value_encoding = 'bool'

        self._gen_template_attrs()

    def is_valid_value(self, value):
        return np.asanyarray(value, 'bool').dtype.kind == 'b'

class CategoryType(AbstractComplexParameterType):
    """

    """

    SUPPORTED_CATETEGORY_KEY_KINDS = ({np.dtype(int).kind, np.dtype(float).kind, np.dtype(str).kind})

    def __init__(self, categories, key_value_encoding=None, key_fill_value=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='CategoryValue', **kwc)

        if not isinstance(categories, dict) or len(categories.keys()) == 0:
            raise TypeError('\'categories\' must be of type dict and cannot be empty: {0}'.format(categories))

        if key_value_encoding is None:
            # Get the type of the first key
            key_value_encoding = np.asanyarray(categories.keys()[0]).dtype.str
        else:
            key_value_encoding = np.dtype(key_value_encoding).str

        self._key_dtype = np.dtype(key_value_encoding).str
        want_kind=np.dtype(key_value_encoding).kind
        if want_kind not in self.SUPPORTED_CATETEGORY_KEY_KINDS:
            raise TypeError('\'key_value_encoding\' is not supported; supported np.dtype.kinds: {0}'.format(self.SUPPORTED_CATETEGORY_KEY_KINDS))

        for k in categories.keys():
            if np.asanyarray(k).dtype.kind != want_kind:
                raise ValueError('A key in \'categories\' ({0}) does not match the specified \'key_value_encoding\' ({1})'.format(k, key_value_encoding))

        if want_kind == 'S':
            self.base_type = ArrayType()
        else:
            self.base_type = QuantityType(value_encoding=key_value_encoding)
            self._value_encoding = key_value_encoding

        if key_fill_value is None or key_fill_value not in categories:
            key_fill_value = categories.keys()[0]

        self._template_attrs['categories'] = categories
        self._template_attrs['fill_value'] = key_fill_value
        self._gen_template_attrs()

    def is_valid_value(self, value):
        if not isinstance(value, basestring) and np.iterable(value):
            for v in value:
                self.is_valid_value(v)
        else:
            return value in self.categories.keys() or value in self.categories.values()

    def _todict(self):
        ret = super(CategoryType, self)._todict()
        ret['categories'] = {str(k):v for k, v in ret['categories'].iteritems()}

        return ret

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(CategoryType, cls)._fromdict(cmdict, arg_masks=arg_masks)
        dt = np.dtype(ret._key_dtype)
        ret.categories = {dt.type(k):v for k, v in ret.categories.iteritems()}

        return ret


class CountType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class QuantityType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, value_encoding=None, uom=None, constraint=None, **kwargs):
        """
        ParameterType for Quantities (float, int, etc)

        @param value_encoding   The intrinsic type of the Quantity
        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, value_class='NumericValue', **kwc)
        if value_encoding is None:
            self._value_encoding = 'float32'
        else:
            if isinstance(value_encoding, np.dtype):
                value_encoding = value_encoding.str
            try:
                dt = np.dtype(value_encoding)
                if dt.isbuiltin not in (0,1):
                    raise TypeError('\'value_encoding\' must be a valid numpy dtype: {0}'.format(value_encoding))
                if dt in UNSUPPORTED_DTYPES:
                    raise TypeError('\'value_encoding\' {0} is not supported by H5py: UNSUPPORTED types ==> {1}'.format(value_encoding, UNSUPPORTED_DTYPES))

                self._value_encoding = value_encoding

            except TypeError:
                raise

        self._template_attrs['uom'] = uom or 'unspecified'
        self._template_attrs['constraint'] = constraint
        self._gen_template_attrs()

    @property
    def value_encoding(self):
        return self._value_encoding

    def is_valid_value(self, value):
        # CBM TODO: This may be too restrictive - for example: wouldn't allow assignment of ints to a float array
        # Could do something like np.issubdtype, but this also wouldn't allow the above!!
        return np.dtype(self._value_encoding) == np.asanyarray(value).dtype

    def __eq__(self, other):
        if super(QuantityType, self).__eq__(other):
            #CBM TODO: Need to validate that UOM's are compatible, not just equal
            if self.uom.lower() == other.uom.lower():
                return True

        return False

class TextType(AbstractSimplexParameterType):
    """
    Text ParameterType.  Allows "string" values

    Currently supports python str or unicode; other encodings can be added as necessary
    """
    def __init__(self, **kwargs):
        """
        Constructor for TextType

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

        self._value_encoding = str

        self._template_attrs['fill_value'] = ''

        self._gen_template_attrs()

class TimeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class CategoryRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class CountRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class QuantityRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class TimeRangeType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

class ParameterFunctionType(AbstractSimplexParameterType):

    def __init__(self, function, value_encoding=None, callback=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, value_class='ParameterFunctionValue', **kwc)
        if not isinstance(function, AbstractFunction):
            raise TypeError('\'function\' must be a subclass of AbstractFunction')

        if value_encoding is None:
            self._value_encoding = np.dtype('float32').str
        else:
            try:
                dt = np.dtype(value_encoding)
                if dt.isbuiltin not in (0,1):
                    raise TypeError('\'value_encoding\' must be a valid numpy dtype: {0}'.format(value_encoding))
                if dt in UNSUPPORTED_DTYPES:
                    raise TypeError('\'value_encoding\' {0} is not supported by H5py: UNSUPPORTED types ==> {1}'.format(value_encoding, UNSUPPORTED_DTYPES))

                self._value_encoding = dt.str

            except TypeError:
                raise

        self._template_attrs['function'] = function

        self._template_attrs['_pval_callback'] = None
        self._template_attrs['_pctxt_callback'] = None

        self._gen_template_attrs()

        # TODO: Find a way to allow a parameter to NOT be stored at all....basically, storage == None
        # For now, just use whatever the _value_encoding and _fill_value say it should be...

    def get_module_dependencies(self):
        return self.function.get_module_dependencies()

    def get_function_map(self, parent_arg_name=None):
        self._fmap = self.function.get_function_map(self._pctxt_callback, parent_arg_name=parent_arg_name)

        return self._fmap

    def _todict(self, exclude=None):
        # Must exclude _cov_range_value from persistence
        return super(ParameterFunctionType, self)._todict(exclude=['_pval_callback', '_pctxt_callback', '_fmap', '_iparams', '_dparams', '_callback'])

    @property
    def callback(self):
        return self._pval_callback

    @callback.setter
    def callback(self, value):
        self._pval_callback = value

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(ParameterFunctionType, cls)._fromdict(cmdict, arg_masks=arg_masks)
        # Add the _pval_callback attribute, initialized to None
        ret._pval_callback = None
        ret.callback = None
        return ret

    def __eq__(self, other):
        ret = False
        if super(ParameterFunctionType, self).__eq__(other):  # Performs instance check
            ret = self.value_encoding == other.value_encoding and self.function == other.function

        return ret


class SparseConstantType(AbstractComplexParameterType):
    """

    """
    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        fv = None
        if 'fill_value' in kwc:
            fv = kwc.pop('fill_value')
        ve = None
        if 'value_encoding' in kwc:
            ve = kwc.pop('value_encoding')
        AbstractComplexParameterType.__init__(self, value_class='SparseConstantValue', **kwc)
        if base_type is not None and not isinstance(base_type, (ConstantType, ArrayType)):
            raise TypeError('\'base_type\' must be an instance of ConstantType or ArrayType')

        self.base_type = base_type or ConstantType(value_encoding=ve)

        self._template_attrs.update(self.base_type._template_attrs)

        self._gen_template_attrs()

        if fv is not None:
            self.fill_value = fv
        else:
            self.fill_value = np.NaN


class FunctionType(AbstractComplexParameterType):
    """

    """
    # CBM TODO: There are 2 'classes' of Function - those that operate against an INDEX, and those that operate against a VALUE
    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='FunctionValue', **kwc)
        if base_type is not None and not isinstance(base_type, QuantityType):
            raise TypeError('\'base_type\' must be an instance of QuantityType')

        self.base_type = base_type or QuantityType()

        self._template_attrs.update(self.base_type._template_attrs)

#        self._template_attrs['value_encoding'] = '|O8'
        self._template_attrs['fill_value'] = None

        self._gen_template_attrs()

    def is_valid_value(self, value):
        if not is_well_formed_where(value):
            raise ValueError('\'value\' must be a string matching the form (may be nested): \'{0}\' ; for example, \'where(x > 99, 8, -999)\', \'where((x > 0) & (x <= 100), 55, 100)\', or \'where(x <= 10, 10, where(x <= 30, 100, where(x < 50, 150, nan)))\''.format(single_where_match))

    def __eq__(self, other):
        if super(FunctionType, self).__eq__(other):
            if self.base_type == other.base_type:
                return True

        return False

class ConstantType(AbstractComplexParameterType):

#    _rematch='^(c\*)?{0}$'.format(digit_match)

    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        ve = None
        if 'value_encoding' in kwc:
            ve = kwc.pop('value_encoding')
        AbstractComplexParameterType.__init__(self, value_class='ConstantValue', **kwc)
        if base_type is not None and not isinstance(base_type, QuantityType):
            raise TypeError('\'base_type\' must be an instance of QuantityType')

        self.base_type = base_type or QuantityType(value_encoding=ve)

        self._template_attrs.update(self.base_type._template_attrs)
#        self._template_attrs['fill_value'] = None

        self._gen_template_attrs()

        # Override the _value_encoding - this does NOT need to store objects (vlen-str)!!
        self._value_encoding = self.base_type.value_encoding

    def is_valid_value(self, value):
        dt=np.dtype(self.value_encoding)
        if dt.kind == 'S':
            if isinstance(value, ConstantValue):
                if np.dtype(value.parameter_type.value_encoding).kind != 'S':
                    raise ValueError('\'value\' is a ConstantValue, with an invalid value_encoding; must be of kind=\'S\', is kind={0}'.format(np.dtype(value.parameter_type.value_encoding).kind))
            elif np.atleast_1d(value).dtype.kind != 'S':
                raise ValueError('\'value\' is a numpy array, with an invalid dtype; must be kind=\'S\', is kind={0}'.format(value.dtype.kind))
        else:
            # TODO: Check numeric??
            pass

#        if re.match(self._rematch, value) is None:
#            raise ValueError('\'value\' must be a string matching the form: \'{0}\' ; for example, \'43.2\', \'c*12\', or \'-12.2e4\''.format(self._rematch))

        return True

    def __eq__(self, other):
        if super(ConstantType, self).__eq__(other):
            if self.base_type == other.base_type:
                return True

        return False


class ConstantRangeType(AbstractComplexParameterType):
    """

    """
    def __init__(self, base_type=None, fill_value=("", ""), **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        ve = None
        if 'value_encoding' in kwc:
            ve = kwc.pop('value_encoding')
        AbstractComplexParameterType.__init__(self, value_class='ConstantRangeValue', **kwc)
        if base_type is not None and not isinstance(base_type, QuantityType):
            raise TypeError('\'base_type\' must be an instance of QuantityType')

        self.base_type = base_type or QuantityType(value_encoding=ve)
        self._template_attrs.update(self.base_type._template_attrs)

        self._gen_template_attrs()
        if base_type is not None:
            self.value_encoding = base_type.value_encoding
        else:
            self.value_encoding = ve
        self.value_encoding = "%s, %s" % (self.value_encoding, self.value_encoding)

        self.fill_value = fill_value

    @property
    def fill_value(self):
        if hasattr(self, '_fill_value'):
            return self._fill_value
        else:
            return None

    @fill_value.setter
    def fill_value(self, value):
        if self.value_encoding.find('None') != -1:
            self._fill_value = value
        else:
            self._fill_value = verify_fill_value(self.value_encoding, value, False)

    def is_valid_value(self, value):
        # my_kind = np.dtype(self.value_encoding).kind
        # varr = np.atleast_1d(value).flatten()
        # v_kind = varr.dtype.kind
        # if v_kind == np.dtype(object):
        #     if len(varr) < 1:
        #         raise ValueError('\'value\' must be an iterable of size >= 2 and kind={0}; value={1}'.format(my_kind, value))
        #     else:
        #         varr = np.atleast_1d(varr[0]).flatten()
        #         v_kind = varr.dtype.kind
        #
        # if len(varr) < 2 or v_kind != my_kind:
        #     raise ValueError('\'value\' must be an iterable of size >= 2 and kind={0}; value={1}'.format(my_kind, value))

        return True

class RecordType(AbstractComplexParameterType):
    """
    Heterogeneous set of named things (dict)
    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='RecordValue', **kwc)

        self._gen_template_attrs()

class VectorType(AbstractComplexParameterType):
    """
    Heterogeneous set of unnamed things (tuple)
    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='VectorValue', **kwc)

        self._gen_template_attrs()

class OldArrayType(AbstractComplexParameterType):
    """
    Homogeneous set of unnamed things (array)
    """
    def __init__(self, inner_encoding=None, inner_fill_value=None, inner_length=1, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='ArrayValue', **kwc)

        if inner_encoding is None or np.dtype(inner_encoding).kind in ['S', 'O']:
            self.inner_encoding = 'object'
        else:
            self.inner_encoding = verify_encoding(inner_encoding)

        self.inner_fill_value = verify_fill_value(self.inner_encoding, inner_fill_value, False)
        self._gen_template_attrs()

        self._fill_value = tuple([self.inner_fill_value for x in range(inner_length)])
        self.value_encoding = ', '.join([self.inner_encoding for x in range(inner_length)])


class ArrayType(AbstractComplexParameterType):
    """
    Homogeneous set of unnamed things (array)
    """
    def __init__(self, inner_encoding=None, inner_fill_value=None, inner_length=1, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='ArrayValue', **kwc)

        if inner_encoding is None or np.dtype(inner_encoding).kind in ['S', 'O']:
            self.inner_encoding = 'object'
        else:
            self.inner_encoding = verify_encoding(inner_encoding)

        self.inner_fill_value = verify_fill_value(self.inner_encoding, inner_fill_value, False)
        self._gen_template_attrs()

        self.inner_length=inner_length
        self._fill_value = list([self.inner_fill_value for x in range(inner_length)])
        self.value_encoding = self.inner_encoding

    def create_filled_array(self, size):
        if self.inner_length == 1:
            arr = np.empty(size, dtype=np.dtype(self.value_encoding))
            arr[:] = self.fill_value[0]
        else:
            arr = np.empty((size,self.inner_length), dtype=np.dtype(self.value_encoding))
            arr[:] = self.fill_value
        return arr

    def create_data_array(self, data=None, size=None):
        if data is not None:
            arr = np.array(data, dtype=np.dtype(self.value_encoding))
            if len(arr.shape) == 1 or arr.shape[1] != self.inner_length:
                arr = arr.flatten()
                if self.inner_length == 1:
                    return arr
                outer_dim, rem = divmod(arr.size, 3)
                if rem != 0:
                    raise IndexError('Shape of data, %s, does not fit array shape (n,%i)' % (len(data), self.inner_length))
                arr = arr.reshape((outer_dim, self.inner_length))
            return arr
        elif size is not None:
            return self.create_filled_array(size)
        else:
            raise RuntimeError('Not enough information to create array')

    def create_value_array(self, data=None, size=None):
        return self.create_data_array(data, size)

    def validate_value_set(self, value_set):
        if not isinstance(value_set, np.ndarray):
            raise TypeError('Value set must implement type: %s' % np.ndarray.__name__)
        throw_shape_error = False
        shape_len = value_set.shape
        if shape_len == 2:
            if value_set[1] != self.inner_length:
                throw_shape_error = True
        elif shape_len == 1 and self.inner_length != 1:
            throw_shape_error = True
        else:
            throw_shape_error = True
        if throw_shape_error:
            raise ValueError('Array shape must be 2D with second dimension size %i' % self.inner_length)

        if value_set.dtype != np.dtype(self.inner_encoding):
            raise TypeError('Expected array dtype %s, found %s' % (np.dtype(self.inner_encoding), value_set.dtype))


class RaggedArrayType(AbstractComplexParameterType):
    """
    Non-Homogeneous set of unnamed things array of tuples)
    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='RaggedArrayValue', **kwc)

        self.inner_encoding = 'object'

        self._gen_template_attrs()

        self.value_encoding = 'O'

    def create_filled_array(self, size):
        arr = np.empty(size, dtype=np.dtype(self.value_encoding))
        arr[:] = self.fill_value
        return arr

    def create_data_array(self, data=None, size=None):
        if data is not None:
            arr = np.array(data, dtype=np.dtype(self.value_encoding))
            return arr
        elif size is not None:
            return self.create_filled_array(size)
        else:
            raise RuntimeError('Not enough information to create array')

    def create_value_array(self, data=None, size=None):
        return self.create_data_array(data, size)

    @classmethod
    def create_ragged_array(cls, data):
        return create_numpy_object_array(data)

    def validate_value_set(self, value_set):
        if not isinstance(value_set, np.ndarray):
            raise TypeError('Value set must implement type: %s' % np.ndarray.__name__)
        if len(value_set.shape) != 1:
            raise ValueError('Array must be 1D of type object.  Found type (%s) with shape %s' % (str(value_set.dtype), value_set.shape))

        if value_set.dtype != np.dtype(self.inner_encoding):
            raise TypeError('Expected array dtype %s, found %s' % (np.dtype(self.inner_encoding), value_set.dtype))
