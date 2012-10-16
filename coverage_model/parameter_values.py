#!/usr/bin/env python

"""
@package coverage_model.parameter_values
@file coverage_model/parameter_values.py
@author Christopher Mueller
@brief Abstract and concrete value objects for parameters
"""
from ooi.logging import log
from coverage_model.basic_types import AbstractBase, is_valid_constraint, InMemoryStorage, VariabilityEnum
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
        self._storage = storage or InMemoryStorage(dtype=self.storage_encoding, fill_value=self.parameter_type.fill_value)

    def _fix_slice(self, slice_):
        # CBM: First swack - see this for more possible checks: http://code.google.com/p/netcdf4-python/source/browse/trunk/netCDF4_utils.py
        if not is_valid_constraint(slice_):
            raise SystemError('invalid constraint supplied: {0}'.format(slice_))

        # First, ensure we're working with a list so we can make edits...
        if not np.iterable(slice_):
            slice_ = [slice_,]
        elif not isinstance(slice_,list):
            slice_ = list(slice_)

        # Then make sure it's the correct rank
        rank = len(self.shape)
        slen = len(slice_)
        if not slen == rank:
            if slen > rank:
                slice_ = slice_[:rank]
            else:
                for n in xrange(slen, rank):
                    slice_.append(slice(None,None,None))

        # Next, deal with negative indices
        # Logic for handling negative indices from: http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html
        sshp = self.shape
        for x in xrange(rank):
            sl=slice_[x]
            if isinstance(sl, int):
                if sl < 0:
                    slice_[x] = sshp[x] + slice_[x]
            elif isinstance(sl, list):
                for i in xrange(len(sl)):
                    if sl[i] < 0:
                        sl[i] = sshp[x] + sl[i]
                slice_[x] = sl
            elif isinstance(sl, slice):
                sl_ = [sl.start, sl.stop, sl.step]
                if sl_[0] is not None and sl_[0] < 0:
                    sl_[0] = sshp[x] + sl_[0]

                if sl_[1] is not None and sl_[1] < 0:
                    sl_[1] = sshp[x] + sl_[1]

                if sl_[2] is not None and sl_[2] < 0:
                    slice_[x] = slice(sl_[1],sl_[0],-sl_[2])
                else:
                    slice_[x] = slice(*sl_)

        # Finally, make it a tuple
        return tuple(slice_)

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

    @property
    def storage_encoding(self):
        return self.parameter_type.value_encoding

    def expand_content(self, domain, origin, expansion):
        if domain == VariabilityEnum.TEMPORAL: # Temporal
            self._storage.expand(self.shape[1:], origin, expansion)
        elif domain == VariabilityEnum.SPATIAL: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __len__(self):
        # I don't think this is correct - should be the length of the total available set of values, not the length of storage...
#        return len(self._storage)
        return prod(self.shape)

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

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_)

        return self._storage[slice_]

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_)

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
        slice_ = self._fix_slice(slice_)

        total_indices = prod(self.shape)
        x = np.arange(total_indices).reshape(self.shape)[slice_] # CBM TODO: This is INDEX based evaluation!!!

        return ne.evaluate(self.content).astype(self.value_encoding)

    def __setitem__(self, slice_, value):
        if is_well_formed_where(value):
            slice_ = self._fix_slice(slice_)
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
        return self._storage[0][()] # Returned from storage as a 0-d array

    def expand_content(self, domain, origin, expansion):
        # No op storage is always 1 - appropriate domain applied during retrieval of data
        pass

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_)

        c = np.ones(self.shape)[slice_] # Make an array of ones of the appropriate shape and slice it as desired

        ret = ne.evaluate(self.content).astype(self.value_encoding)
        if ret.size==1:
            if ret.ndim==0:
                ret=ret[()]
            else:
                ret=ret[0]
        return ret

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
        self._storage.expand(self.shape, 0, self.shape[0])

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_)

        ret = self._storage[slice_]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_)

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
        self._storage.expand(self.shape, 0, self.shape[0])

    def __getitem__(self, slice_):
        slice_ = self._fix_slice(slice_)

        ret = self._storage[slice_]

        return ret

    def __setitem__(self, slice_, value):
        slice_ = self._fix_slice(slice_)

        self._storage[slice_] = value
