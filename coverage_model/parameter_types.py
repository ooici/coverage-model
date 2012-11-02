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
from coverage_model.numexpr_utils import digit_match, is_well_formed_where, single_where_match
import numpy as np
import re

UNSUPPORTED_DTYPES = set([np.dtype('float16'), np.dtype('complex'), np.dtype('complex64'), np.dtype('complex128'), np.dtype('complex256')])

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
            dtk = np.dtype(self.value_encoding).kind
            if dtk == 'u': # Unsigned integer's must be positive
                self._fill_value = abs(value)
            elif dtk == 'O': # object, must be None for now...
                self._fill_value = None
            else:
                self._fill_value = value
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

    def _gen_template_attrs(self):
        for k, v in self._template_attrs.iteritems():
            setattr(self,k,v)

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
        AbstractSimplexParameterType.__init__(self, **kwc)

        self._template_attrs['fill_value'] = False

        self._gen_template_attrs()

    def is_valid_value(self, value):
        return isinstance(value, bool)

class CategoryType(AbstractSimplexParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterType.__init__(self, **kwc)

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

    _rematch='^(c\*)?{0}$'.format(digit_match)

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
#        self._template_attrs['value_encoding'] = '|O8'
        self._template_attrs['fill_value'] = None

        self._gen_template_attrs()

    def is_valid_value(self, value):
        if re.match(self._rematch, value) is None:
            raise ValueError('\'value\' must be a string matching the form: \'{0}\' ; for example, \'43.2\', \'c*12\', or \'-12.2e4\''.format(self._rematch))

        return True

    def __eq__(self, other):
        if super(ConstantType, self).__eq__(other):
            if self.base_type == other.base_type:
                return True

        return False

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