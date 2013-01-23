#!/usr/bin/env python

"""
@package coverage_model.parameter_values
@file coverage_model/parameter_values.py
@author Christopher Mueller
@brief Abstract and concrete value objects for parameters
"""
from ooi.logging import log
from coverage_model.basic_types import AbstractBase, InMemoryStorage, VariabilityEnum
from coverage_model.numexpr_utils import is_well_formed_where, nest_wheres
from coverage_model import utils
import numpy as np
import numexpr as ne

def get_value_class(param_type, domain_set, **kwargs):
    module = __import__(param_type._value_module, fromlist=[param_type._value_class])
    classobj = getattr(module, param_type._value_class)
    return classobj(parameter_type=param_type, domain_set=domain_set, **kwargs)

#=========================
# Abstract Parameter Value Objects
#=========================

class AbstractParameterValue(AbstractBase):
    """

    """
    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractBase; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractBase.__init__(self, **kwc)
        self.parameter_type = parameter_type
        self.domain_set = domain_set
        self._storage = storage or InMemoryStorage(dtype=self.parameter_type.storage_encoding, fill_value=self.parameter_type.fill_value)

    @property
    def shape(self):
        return self.domain_set.total_extents

    @property
    def storage(self):
        return self._storage

    @property
    def content(self):
        return self._storage[:]

    @property
    def value_encoding(self):
        return self.parameter_type.value_encoding

    def expand_content(self, domain, origin, expansion):
        if domain == VariabilityEnum.TEMPORAL: # Temporal
            self._storage.expand(self.shape[1:], origin, expansion)
        elif domain == VariabilityEnum.SPATIAL: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret = self._storage[slice_]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)

        self._storage[slice_] = value

    def __len__(self):
        # I don't think this is correct - should be the length of the total available set of values, not the length of storage...
#        return len(self._storage)
        return utils.prod(self.shape)

    def __str__(self):
        return str(self.content)

class AbstractSimplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)


class AbstractComplexParameterValue(AbstractParameterValue):
    """

    """
    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

#=========================
# Concrete Parameter Value Objects
#=========================

class NumericValue(AbstractSimplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])

class FunctionValue(AbstractComplexParameterValue):
    # CBM TODO: There are 2 'classes' of Function - those that operate against an INDEX, and those that operate against a VALUE
    # CBM TODO: Does this actually end up as a subclass of VectorValue?  basically a 2 member tuple?

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand((1,), 0, 1)
        self._storage[0] = []

    @property
    def content(self):
        if len(self._storage[0]) > 1:
            return nest_wheres(*[x for x in self._storage[0]])
        else:
            return self._storage[0][0]

    def expand_content(self, domain, origin, expansion):
        # No op storage is always 1 - appropriate domain applied during retrieval of data
        pass

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        total_indices = utils.prod(self.shape)
        x = np.arange(total_indices).reshape(self.shape)[slice_] # CBM TODO: This is INDEX based evaluation!!!

        return ne.evaluate(self.content).astype(self.value_encoding)

    def __setitem__(self, slice_, value):
        if is_well_formed_where(value):
            slice_ = utils.fix_slice(slice_, self.shape)
            if len(slice_) == 1 and isinstance(slice_[0], int) and slice_ < len(self._storage[0]):
                self._storage[0][slice_[0]] = value
            else:
                self._storage[0].append(value)

class ConstantValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand((1,), 0, 1)

    @property
    def content(self):
        return self._storage[0]

    def expand_content(self, domain, origin, expansion):
        # No op storage is always 1 - appropriate domain applied during retrieval of data
        pass

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret_shape = utils.get_shape_from_slice(slice_, self.shape)
        ret = np.empty(ret_shape, dtype=np.dtype(self.value_encoding))
        ret.fill(self.content)

        if ret.size==1:
            if ret.ndim==0:
                ret=ret[()]
            else:
                ret=ret[0]
        return ret

    def __setitem__(self, slice_, value):
        if self.parameter_type.is_valid_value(value):
            if np.iterable(value) and not isinstance(value, basestring):
                value = value[0]
            if np.dtype(self.value_encoding).kind == 'S': # If we're dealing with a str
                value = str(value) # Ensure the value is a str!!
            self._storage[0] = value

class ConstantRangeValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param parameter_type:
        @param domain_set:
        @param storage:
        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        @return:
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand((1,), 0, 1)

    @property
    def content(self):
        import ast
        return ast.literal_eval(self._storage[0])

    def expand_content(self, domain, origin, expansion):
        # No op storage is always 1 - appropriate domain applied during retrieval of data
        pass

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret_shape = utils.get_shape_from_slice(slice_, self.shape)
        ret = np.empty(ret_shape, dtype=np.dtype(object)) # Always object type because it's 2 values / element!!
        ret.fill(self.content)

        if ret.size==1:
            if ret.ndim==0:
                ret=ret[()]
            else:
                ret=ret[0]
        return ret

    def __setitem__(self, slice_, value):
        if self.parameter_type.is_valid_value(value):
            # We already know it's either a list or tuple, that it's length is >= 2, and that both of
            # the first two values are of the correct type...so...
            self._storage[0] = str(tuple(value[:2]))

class CategoryValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret = self._storage[slice_]
        cats=self.parameter_type.categories
        if np.iterable(ret):
            ret = np.array([cats[x] if x is not None else self.parameter_type.fill_value for x in ret], dtype=object)
        elif isinstance(ret, np.ndarray):
            ret = cats[ret.item()]
        else:
            ret = cats[ret]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)
        self.parameter_type.is_valid_value(value)
        if np.asanyarray(value).dtype.kind == np.dtype(self.parameter_type.value_encoding).kind:
            # Set as ordinals
            self._storage[slice_] = value
        else:
            rcats={v:k for k,v in self.parameter_type.categories.iteritems()}
            try:
                vals=[rcats[v] for v in value]
                self._storage[slice_] = vals
            except KeyError, ke:
                raise ValueError('Invalid category specified: \'{0}\''.format(ke.message))

class RecordValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])

class VectorValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])

class ArrayValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])
