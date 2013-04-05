#!/usr/bin/env python

"""
@package coverage_model.parameter_values
@file coverage_model/parameter_values.py
@author Christopher Mueller
@brief Abstract and concrete value objects for parameters
"""
from ooi.logging import log
from coverage_model.basic_types import AbstractBase, InMemoryStorage, VariabilityEnum, Span
from coverage_model.numexpr_utils import is_well_formed_where, nest_wheres
from coverage_model.parameter_functions import ParameterFunctionException
from coverage_model import utils
import numpy as np
import numexpr as ne


def get_value_class(param_type, domain_set, **kwargs):
    module = __import__(param_type._value_module, fromlist=[param_type._value_class])
    classobj = getattr(module, param_type._value_class)
    return classobj(parameter_type=param_type, domain_set=domain_set, **kwargs)


def _cleanse_value(val, slice_):
    ret = np.atleast_1d(val)

    # If the array is size 1 AND a slice object was NOT part of the query
    if ret.size == 1 and not np.atleast_1d([isinstance(s, slice) for s in slice_]).all():
        val = ret[0]

    return val

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
        self._storage = storage if storage is not None else InMemoryStorage(dtype=self.parameter_type.storage_encoding, fill_value=self.parameter_type.fill_value)
        self._min = self._max = self.fill_value

    @property
    def shape(self):
        return self.domain_set.total_extents

    @property
    def bounds(self):
        return self.min, self.max

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def storage(self):
        return self._storage

    @property
    def content(self):
        return self._storage[:]

    @property
    def value_encoding(self):
        return self.parameter_type.value_encoding

    @property
    def fill_value(self):
        return self.parameter_type.fill_value

    def expand_content(self, domain, origin, expansion):
        if domain == VariabilityEnum.TEMPORAL: # Temporal
            self._storage.expand(self.shape[1:], origin, expansion)
        elif domain == VariabilityEnum.SPATIAL: # Spatial
            raise NotImplementedError('Expansion of the Spatial Domain is not yet supported')

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        return _cleanse_value(self._storage[slice_], slice_)

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)

        self._storage[slice_] = value

        self._update_min_max(value)

    def _update_min_max(self, value):

        # TODO: There is a flaw here when OVERWRITING:
        # overwritten values may still appear to be a min/max value as
        # recalculation of the full array does not occur...
        if np.dtype(self.value_encoding).kind != 'S':  # No min/max for strings
            v = np.atleast_1d(value)
            # All values are fill_values, leave what we have!
            if np.atleast_1d(v == self.fill_value).all():
                return
            # Mask fill_value so it's not included in the calculation
            v = np.atleast_1d(np.ma.masked_equal(v, self.fill_value, copy=False))
            # Update min
            self._min = min(v.min(), self._min) if self._min != self.fill_value else v.min()
            # Update max
            self._max = max(v.max(), self._max) if self._min != self.fill_value else v.max()

    def __len__(self):
        # I don't think this is correct - should be the length of the total available set of values, not the length of storage...
#        return len(self._storage)
        return utils.prod(self.shape)

    # def __str__(self):
    #     return str(self.content)


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

    def _update_min_max(self, value):
        # No-op - Must override in concrete class to make functional
        # By default, these value types present fill_value for min & max
        pass


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


class BooleanValue(AbstractSimplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractSimplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        self._storage.expand(self.shape, 0, self.shape[0])

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)

        if self.parameter_type.is_valid_value(value):
            self._storage[slice_] = np.asanyarray(value, dtype='bool')
            self._update_min_max(np.asanyarray(value, dtype='int8'))


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

        return _cleanse_value(ne.evaluate(self.content).astype(self.value_encoding), slice_)

    def __setitem__(self, slice_, value):
        if is_well_formed_where(value):
            slice_ = utils.fix_slice(slice_, self.shape)
            if len(slice_) == 1 and isinstance(slice_[0], int) and slice_ < len(self._storage[0]):
                self._storage[0][slice_[0]] = value
            else:
                self._storage[0].append(value)


class ParameterFunctionValue(AbstractSimplexParameterValue):

    def __init__(self, parameter_type, domain_set, storage=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractComplexParameterValue; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractSimplexParameterValue.__init__(self, parameter_type, domain_set, storage, **kwc)
        # Do NOT expand storage - no need to store anything here!!

        # Grab a local pointer to the coverage's _cov_range_value object
        self._pval_callback = self.parameter_type._pval_callback
        self._memoized_values = None

    @property
    def content(self):
        return self.parameter_type.function

    def expand_content(self, domain, origin, expansion):
        # No op, storage is not used
        pass

    def _update_min_max(self, value):
        # TODO: Can possibly do something here?  Only if memoized?
        # No-op - What's the min/max for this value class?
        pass

    def __getitem__(self, slice_):
        if self._memoized_values is not None:
            return self._memoized_values
        else:
            if self._pval_callback is None:
                raise ParameterFunctionException('\'_pval_callback\' is None; cannot evaluate!!')

            slice_ = utils.fix_slice(slice_, self.shape)

            try:
                r = self.content.evaluate(self._pval_callback, slice_, self.parameter_type.fill_value)
            except Exception as ex:
                import sys
                raise ParameterFunctionException(ex.message, type(ex)), None, sys.exc_traceback

            return _cleanse_value(r, slice_)

    def __setitem__(self, slice_, value):
        self._memoized_values = value
#        raise ValueError('Values cannot be set against ParameterFunctionValues!')


class SparseConstantValue(AbstractComplexParameterValue):

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
        # No op - storage expanded in __setitem__
        pass

    def __indexify_slice(self, slice_, total_shape):
        ## ONLY WORKS FOR 1D ARRAYS!!!
        fsl = utils.fix_slice(slice_, total_shape)
        ss = utils.slice_shape(slice_, total_shape)
        ret = np.empty(ss, dtype=int)
        rf = ret.flatten()

        ci = 0
        for s, shape in zip(fsl, total_shape):
            if isinstance(s,slice):
                ind = range(*s.indices(shape))
                ll = len(ind)
                rf[ci:ll] = ind
                ci += ll
            elif isinstance(s, (list,tuple)):
                ll = len(s)
                rf[ci:ll] = s
                ci += ll
            elif isinstance(s, int):
                rf[ci] = s
                ci += 1
            else:
                raise TypeError('Unsupported slice method') # TODO: Better error message

        return rf.reshape(ss)

    def _apply_value(self, stor_sub):
        v_arr = np.empty(0, dtype=self.value_encoding)
        max_i = self.shape[0]
        for s in stor_sub:
            # log.trace('s: %s, max_i: %s', s, max_i)
            st = s.lower_bound or 0
            en = s.upper_bound or max_i
            # log.trace('st: %s, en: %s, offset: %s', st, en, s.offset)

            if st == en == max_i:
                break

            if isinstance(s.value, AbstractParameterValue):
                st += s.offset
                en += s.offset
                e = s.value[st:en]
            else:
                sz = en - st
                e = np.empty(sz, dtype=self.value_encoding)
                e.fill(s.value)

            v_arr = np.append(v_arr, e)

        return v_arr

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        # Nothing asked for!
        if len(slice_) is 0:
            return np.empty(0, dtype=self.value_encoding)

        try:
            spans = self._storage[0]
        except ValueError, ve:
            if ve.message != 'No Bricks!':
                raise

            return np.empty(0, dtype=self.value_encoding)

        if not hasattr(spans, '__iter__') and spans == self.fill_value:
            ret = np.empty(utils.slice_shape(slice_, self.shape), dtype=self.value_encoding)
            ret.fill(self.fill_value)
            return ret

        # Build the index array
        ind_arr = self.__indexify_slice(slice_, self.shape)
        # Empty index array!
        if len(ind_arr) == 0:
            return np.empty(0, dtype=self.value_encoding)

        # Get first and last index
        fi, li = ind_arr.min(), ind_arr.max()

        # Find the first storage needed
        strt_i = None
        end_i = None
        enum = enumerate(spans)
        for i, s in enum:
            if fi in s:
                strt_i = i
                break

        if fi == li:
            end_i = strt_i + 1
        else:
            for i, s in reversed(list(enum)):
                if li in s:
                    end_i = i + 1
                    break

        # log.trace('srt: %s, end: %s, fi: %s, li: %s', strt_i, end_i, fi, li)

        stor_sub = spans[strt_i:end_i]
        # Build the array of stored values
        v_arr = self._apply_value(stor_sub)

        if stor_sub[0].lower_bound is None:
            offset = 0
        else:
            offset = stor_sub[0].lower_bound
        io = ind_arr - offset

        return _cleanse_value(v_arr[io], slice_)

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)

        try:
            spans = self._storage[0]
        except ValueError, ve:
            if ve.message != 'No Bricks!':
                raise

            spans = self.fill_value

        if isinstance(value, SparseConstantValue):  # RDT --> Coverage style assignment
            value = value[0]

        if not isinstance(value, AbstractParameterValue):
            # If the value is an array/iterable, we only take the first one
            value = np.atleast_1d(value)[0]

        if not hasattr(spans, '__iter__') and spans == self.fill_value:
            spans = [Span(value=value)]
            slice_ = (self.shape[0] - 1,)

        # Get the last span
        lspn = spans[-1]

        if slice_[0] == self.shape[0] - 1:  # -1 was used for slice
            # Change the value of the last span and return
            lspn.value = value
            self._storage[0] = spans
            return

        nspn_offset = 0
        if isinstance(slice_[0], Span):
            # TODO: This could be used to alter previous span objects, but for now, just use it to pass the offset
            nspn_offset = slice_[0].offset
        elif utils.slice_shape(slice_, self.shape) == self.shape:  # Full slice
            nspn_offset = -self.shape[0]

        if not isinstance(value, AbstractParameterValue) and not isinstance(lspn.value, AbstractParameterValue):
            if value == lspn.value:
                # The previous value equals the new value - do not add a new span!
                return

        # The current index becomes the upper_bound of the previous span and the start of the next span
        curr_ind = self.shape[0]

        # Reset the upper_bound of the previous span
        spans[-1] = Span(lspn.lower_bound, curr_ind, offset=lspn.offset, value=lspn.value)

        # Create the new span
        nspn = Span(curr_ind, None, nspn_offset, value=value)

        # Add the new span
        spans.append(nspn)

        # Reset the storage
        self._storage[0] = spans


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

    def _update_min_max(self, value):
        self._min = value
        self._max = value

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret_shape = utils.slice_shape(slice_, self.shape)
        ret = np.empty(ret_shape, dtype=np.dtype(self.value_encoding))
        ret.fill(self.content)

        return _cleanse_value(ret, slice_)

    def __setitem__(self, slice_, value):
        if self.parameter_type.is_valid_value(value):
            if np.iterable(value) and not isinstance(value, basestring):
                value = value[0]
            if np.dtype(self.value_encoding).kind == 'S': # If we're dealing with a str
                value = str(value) # Ensure the value is a str!!
            self._storage[0] = value

            self._update_min_max(value)


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
        self._storage.expand((1,), 0, 2)

    @property
    def content(self):
        # If it's the fill_value, return None
        if self._storage[0] == self.fill_value:
            ret = self.fill_value
        else:
            ret = tuple(self._storage[:2])

        return ret

    def expand_content(self, domain, origin, expansion):
        # No op storage is always 1 - appropriate domain applied during retrieval of data
        pass

    def _update_min_max(self, value):
        self._min = value
        self._max = value

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret_shape = utils.slice_shape(slice_, self.shape)
        ret = np.empty(ret_shape, dtype=np.dtype(object)) # Always object type because it's 2 values / element!!
        ret.fill(self.content)

        return _cleanse_value(ret, slice_)

    def __setitem__(self, slice_, value):
        if self.parameter_type.is_valid_value(value):
            if value == self.fill_value:
                self._storage[:2] = self.fill_value
                return

            # We already know it's either a list or tuple, that it's length is >= 2, and that both of
            # the first two values are of the correct type...so...just deal with funky nesting...
            va = np.atleast_1d(value).flatten()  # Flatten the whole thing - deals with nD arrays
            if isinstance(va[0], tuple):  # Array of tuples, likely from another ConstantRangeValue
                va = np.array(va[0])

            if np.dtype(self.value_encoding).kind == 'S':
                va = [str(v) for v in va]

            self._storage[:2] = np.array(va[:2], dtype=self.value_encoding)

            self._update_min_max(self.content)


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

        ret = np.atleast_1d(self._storage[slice_])
        cats = self.parameter_type.categories
        if np.iterable(ret):
            ret = np.array([cats[x] for x in ret], dtype=object)
        else:
            ret = cats[ret]

        return _cleanse_value(ret, slice_)

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)
        self.parameter_type.is_valid_value(value)
        value = np.atleast_1d(value)

        # Replace any None with fill_value
        np.place(value, value == np.array([None]), self.fill_value)

        if value.dtype.kind == np.dtype(self.parameter_type.value_encoding).kind:
            # Set as ordinals
            self._storage[slice_] = value
        else:
            # Set as categories
            rcats={v:k for k,v in self.parameter_type.categories.iteritems()}
            try:
                vals=[rcats[v] if v != self.fill_value else v for v in value]
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

    def __getitem__(self, slice_):
        slice_ = utils.fix_slice(slice_, self.shape)

        ret = _cleanse_value(self._storage[slice_], slice_)

        return ret

    def __setitem__(self, slice_, value):
        slice_ = utils.fix_slice(slice_, self.shape)

        value = np.atleast_1d(value)
        if len(value.shape) > 1:
            v = np.empty(value.shape[0], dtype=object)
            for i in xrange(value.shape[0]):
                iv = value[i,:]
                if isinstance(iv, np.ndarray):
                    v[i] = iv.tolist()
                else:
                    v[i] = iv

            value = v

        self._storage[slice_] = value[:]

        self._update_min_max(value)