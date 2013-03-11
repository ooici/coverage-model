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
import numpy as np
import networkx as nx
import re

UNSUPPORTED_DTYPES = {np.dtype('float16'), np.dtype('complex'), np.dtype('complex64'), np.dtype('complex128')}
import platform
if platform.uname()[-2] != 'armv7l':
    UNSUPPORTED_DTYPES.add(np.dtype('complex256'))

#==================
# Abstract Parameter Type Objects
#==================

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
        if hasattr(self, 'value_encoding'):
            dt = np.dtype(self.value_encoding)
            dtk = dt.kind
            if dtk == 'O' or isinstance(self, ConstantRangeType): # object & CategoryRangeType must be None for now...
                self._fill_value = None

            elif dtk == 'u': # Unsigned integer's must be positive
                if value is not None:
                    self._fill_value = abs(value)
                else:
                    self._fill_value = np.iinfo(dt).max

            elif dtk == 'S': # must be a string value
                self._fill_value = str(value)

            else:
                if value is not None:
                    self._fill_value = value
                else:
                    if dtk == 'i':
                        self._fill_value = np.iinfo(dt).max
                    elif dtk == 'f':
                        self._fill_value = np.finfo(dt).max

        else:
            self._fill_value = value

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

    SUPPORTED_CATETEGORY_KEY_KINDS = set([np.dtype(int).kind, np.dtype(float).kind, np.dtype(str).kind])

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

    def __init__(self, function, value_encoding=None, **kwargs):
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
        return super(ParameterFunctionType, self)._todict(exclude=['_pval_callback', '_pctxt_callback', '_fmap', '_iparams', '_dparams'])

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        ret = super(ParameterFunctionType, cls)._fromdict(cmdict, arg_masks=arg_masks)
        # Add the _pval_callback attribute, initialized to None
        ret._pval_callback = None
        return ret

    def __eq__(self, other):
        ret = False
        if super(ParameterFunctionType, self).__eq__(other):  # Performs instance check
            ret = self.value_encoding == other.value_encoding and self.function == other.function

        return ret

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
            raise ValueError('\value\' must be a string matching the form (may be nested): \'{0}\' ; for example, \'where(x > 99, 8, -999)\', \'where((x > 0) & (x <= 100), 55, 100)\', or \'where(x <= 10, 10, where(x <= 30, 100, where(x < 50, 150, nan)))\''.format(single_where_match))

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
        AbstractComplexParameterType.__init__(self, value_class='ConstantValue', **kwc)
        if base_type is not None and not isinstance(base_type, QuantityType):
            raise TypeError('\'base_type\' must be an instance of QuantityType')

        self.base_type = base_type or QuantityType()

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
            elif isinstance(value, np.ndarray):
                if value.dtype.kind != 'S':
                    raise ValueError('\'value\' is a numpy array, with an invalid dtype; must be kind=\'S\', is kind={0}'.format(value.dtype.kind))
            elif not isinstance(value, basestring):
                raise ValueError('\'value\' must be a string with a max length of {0}; longer strings will be truncated'.format(dt.str[dt.str.index('S')+1:]))
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
    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='ConstantRangeValue', **kwc)
        if base_type is not None and not isinstance(base_type, QuantityType):
            raise TypeError('\'base_type\' must be an instance of QuantityType')

        self.base_type = base_type or QuantityType()
        self._template_attrs.update(self.base_type._template_attrs)

        self._gen_template_attrs()

    def is_valid_value(self, value):
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            my_kind=np.dtype(self.value_encoding).kind
            for v in value[:2]:
                if np.asanyarray(v).dtype.kind != my_kind:
                    raise ValueError('\'value\' must be a list or tuple of size >= 2 and kind={0}; value={1}'.format(my_kind, value))

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

class ArrayType(AbstractComplexParameterType):
    """
    Homogeneous set of unnamed things (array)
    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='ArrayValue', **kwc)
        self._gen_template_attrs()
