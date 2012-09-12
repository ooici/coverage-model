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

from pyon.public import log
from coverage_model.basic_types import AbstractIdentifiable
import numpy as np

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

class AbstractComplexParameterType(AbstractParameterType):
    """

    """
    def __init__(self, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractParameterType.__init__(self, **kwc)

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
            self._value_encoding = '<f4'
        else:
            try:
                dt = np.dtype(value_encoding)
                if dt.isbuiltin in (0,1):
                    self._value_encoding = dt.str
                else:
                    raise TypeError()
            except TypeError:
                raise TypeError('\'value_encoding\' must be a valid numpy dtype: {0}'.format(value_encoding))

        self._template_attrs['uom'] = uom or 'unspecified'
        self._template_attrs['constraint'] = constraint
        self._template_attrs['fill_value'] = -9999
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

    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='FunctionValue', **kwc)
        if not base_type is None and not isinstance(base_type, AbstractParameterType):
            raise TypeError('\'base_type\' must be a subclass of AbstractParameterType')

        self.base_type = base_type or QuantityType()

        self._template_attrs['fill_value'] = 'c*1'

        self._template_attrs.update(self.base_type._template_attrs)
        self._gen_template_attrs()

    def is_valid_value(self, value):
        raise NotImplementedError('not yet')

    def __eq__(self, other):
        if super(FunctionType, self).__eq__(other):
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
    def __init__(self, base_type=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterType; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterType.__init__(self, value_class='ArrayValue', **kwc)
#        if not base_type is None and not isinstance(base_type, AbstractParameterType):
#            raise TypeError('\'base_type\' must be a subclass of AbstractParameterType')

        self.base_type = base_type or object

        self._template_attrs['fill_value'] = None

        if hasattr(self.base_type, '_template_attrs'):
            self._template_attrs.update(self.base_type._template_attrs)

        self._gen_template_attrs()

    def __eq__(self, other):
        if super(ArrayType, self).__eq__(other):
            if self.base_type == other.base_type:
                return True

        return False