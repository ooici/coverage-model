#!/usr/bin/env python

"""
@package coverage_model.parameter_values
@file coverage_model/parameter_values.py
@author Christopher Mueller
@brief Abstract and concrete value objects for parameters
"""
from pyon.public import log
from coverage_model.basic_types import AbstractBase, is_valid_constraint, InMemoryStorage
from coverage_model.numexpr_utils import is_well_formed_where, nest_wheres
import numpy as np
import numexpr as ne

def get_value_class(param_type, **kwargs):
    module = __import__(param_type._value_module, fromlist=[param_type._value_class])
    classobj = getattr(module, param_type._value_class)
    return classobj(parameter_type=param_type, **kwargs)

def prod(lst):
    import operator
    return reduce(operator.mul, lst, 1)

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
        self._storage = storage or InMemoryStorage()

    def _fix_slice(self, slice_, rank):
        # CBM: First swack - see this for more possible checks: http://code.google.com/p/netcdf4-python/source/browse/trunk/netCDF4_utils.py
        if not is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a tuple
        if not np.iterable(slice_):
            slice_ = (slice_,)
        elif not isinstance(slice_,tuple):
            slice_ = tuple(slice_)

        # Then make it's the correct rank
        slen = len(slice_)
        if not slen == rank:
            if slen > rank:
                slice_ = slice_[:rank]
            else:
                for n in range(slen, rank):
                    slice_ += (slice(None,None,None),)

        # Remove the tuple if it's len==1
        if len(slice_):
            slice_ = slice_[0]
        return slice_

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
        if hasattr(self.parameter_type, 'base_type'):
            t = self.parameter_type.base_type
        else:
            t = self.parameter_type

        return t.value_encoding

    def expand_content(self, domain, origin, expansion):
        raise NotImplementedError('Not implemented by abstract class')

    def __len__(self):
        return len(self._storage)

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
        self._storage.reinit(np.empty(self.shape, dtype=self.value_encoding))
        self._storage.fill(self.parameter_type.fill_value)

    def expand_content(self, domain, origin, expansion):
        if domain == self.domain_set.tdom: # Temporal
            narr = np.empty(self.shape[1:], dtype=self.value_encoding)
            narr.fill(self.parameter_type.fill_value)
            loc=[origin for x in xrange(expansion)]
            self._storage.reinit(np.insert(self._storage[:], loc, narr, axis=0))
        elif domain == self.domain_set.sdom: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_, len(self.shape))

        return self._storage[slice_]

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_, len(self.shape))

        self._storage[slice_] = value

class FunctionValue(AbstractComplexParameterValue):
    # CBM TODO: There are 2 'classes' of Function - those that operate against an INDEX, and those that operate against a VALUE
    # CBM TODO: Does this actually end up as a subclass of VectorValue?  basically a 2 member tuple?

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

        self._storage.reinit(np.empty([0], dtype=object))

    @property
    def content(self):
        if len(self._storage) > 1:
            return nest_wheres(*[x for x in self._storage[:]])
        else:
            return self._storage[0]

    def expand_content(self, domain, origin, expansion):
        # No op - appropriate domain applied during retrieval of data
        pass

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_, len(self.shape))

        total_indices = prod(self.shape)
        x = np.arange(total_indices).reshape(self.shape)[slice_] # CBM TODO: This is INDEX based evaluation!!!

        return ne.evaluate(self.content).astype(self.value_encoding)

    def __setitem__(self, slice_, value):
        if is_well_formed_where(value):
            slice_ = self._fix_slice(slice_, len(self.shape))
            if isinstance(slice_, int) and slice_ < len(self._storage):
                self._storage[slice_] = value
            else:
                self._storage.reinit(np.append(self._storage[:], np.array([value],dtype=object), axis=0))

class ConstantValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

        self._storage.reinit(np.empty([1], dtype=object))

    @property
    def content(self):
        return self._storage[0]

    def expand_content(self, domain, origin, expansion):
        # No op - constant over any domain
        pass

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_, len(self.shape))

        c = np.ones(self.shape)[slice_] # Make an array of ones of the appropriate shape and slice it as desired

        return ne.evaluate(self.content).astype(self.value_encoding)

    def __setitem__(self, slice_, value):
        value = str(value) # Ensure we're dealing with a str
        if self.parameter_type.is_valid_value(value):
            if not value.startswith('c*'):
                value = 'c*'+str(value)
            self._storage[0] = value

class RecordValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

        self._storage.reinit(np.empty(self.shape, dtype=object))

    def expand_content(self, domain, origin, expansion):
        if domain == self.domain_set.tdom: # Temporal
            narr = np.empty(self.shape[1:], dtype=self.value_encoding)
            narr.fill(self.parameter_type.fill_value)
            loc=[origin for x in xrange(expansion)]
            self._storage.reinit(np.insert(self._storage[:], loc, narr, axis=0))
        elif domain == self.domain_set.sdom: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_, len(self.shape))

        ret = self._storage[slice_]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_, len(self.shape))

        self._storage[slice_] = value

class VectorValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

class ArrayValue(AbstractComplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractComplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)

        self._storage.reinit(np.empty(self.shape, dtype=object))

    def expand_content(self, domain, origin, expansion):
        if domain == self.domain_set.tdom: # Temporal
            narr = np.empty(self.shape[1:], dtype=self.value_encoding)
            narr.fill(self.parameter_type.fill_value)
            loc=[origin for x in xrange(expansion)]
            self._storage.reinit(np.insert(self._storage[:], loc, narr, axis=0))
        elif domain == self.domain_set.sdom: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_, len(self.shape))

        ret = self._storage[slice_]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_, len(self.shape))

        self._storage[slice_] = value