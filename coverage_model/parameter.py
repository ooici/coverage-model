#!/usr/bin/env python

"""
@package coverage_model.parameter
@file coverage_model/parameter.py
@author Christopher Mueller
@brief Concrete parameter classes
"""

from ooi.logging import log
from coverage_model.basic_types import AbstractIdentifiable, VariabilityEnum, AxisTypeEnum
from coverage_model.parameter_types import AbstractParameterType, QuantityType
from coverage_model.parameter_functions import ParameterFunctionException
from collections import OrderedDict
import copy

#==================
# Parameter Objects
#==================

class Parameter(AbstractIdentifiable):
    """
    Container class to hold a ParameterContext and it's associated value
    """
    def __init__(self, parameter_context, shape, value):
        """
        Construct a new Parameter

        @param parameter_context The ParameterContext of the parameter
        @param shape    The shape of the parameter
        @param value    The AbstractParameterValue of the parameter
        """
        AbstractIdentifiable.__init__(self)
        self.context = parameter_context
        self.value = value
        self.shape = shape

    # Expose a couple of the context attributes at this level as "read only"
    @property
    def name(self):
        """
        Retrieve the name of the Parameter (via it's ParameterContext)

        @returns    The string name for the Parameter
        """
        return self.context.name

    @property
    def is_coordinate(self):
        """
        Indicates if the Parameter is a coordinate (via it's ParameterContext)

        @returns    True if it is a coordinate; False otherwise
        """
        return self.context.is_coordinate

class ParameterContext(AbstractIdentifiable):
    """
    Context for a parameter

    The ParameterContext is intended to fully describe a parameter in terms of:
    - typing: the param_type attribute should be one of the concrete implementations of AbstractParameterType (i.e. QuantityType)
        - The type may impart certain attributes to the ParameterContext.  For example, a ParameterContext of type QuantityType
     will have a uom (units of measure) attribute.
    - structure: from the perspective of the container (coverage-model, granule), this includes things such as axis
    (a.k.a. 'axis') and variability (one of the VariabilityEnum members).
    - data-critical-metadata: 'metadata' that is critical to the storage/processing of the data - such as fill_value, nil_value, etc.

    @todo: This will likely undergo moderate to significant changes as things continue to mature
    """

    # Dynamically added attributes - from ObjectType CI Attributes (2480139)
    ATTRS = [
#             'attributes',
#             'index_key',
#             'ion_name',
#             'name', # accounted for as 'name'
#             'units', # accounted for as 'uom'
#             'standard_name',
#             'long_name',
#             'ooi_short_name',
#             'display_name',
#             'missing_value', # accounted for as 'fill_value'
#             'cdm_data_type',
#             'variable_reports',
#             'axis', # accounted for as 'axis'
#             'references',
#             'comment', # accounted for as description
#             'code_reports',]
            'reference_urls',
            'internal_name',
            'display_name', 
            'standard_name',
            'ooi_short_name',
            'precision',
            #'description', # Warning - Overrides AbstractIdentifiable.description
    ]

    def __init__(self, name, param_type=None, axis=None, fill_value=None, variability=None, uom=None, **kwargs):
        """
        Construct a new ParameterContext object

        Must provide the 'name' argument.  It can be either a string indicating the name of the parameter, or an exising
        ParameterContext object that should be used as a template.
        If 'name' is a ParameterContext, a keyword argument 'new_name' can be provided which will be used as the name for
        the new ParameterContext.  If 'new_name' is not provided, the name will be the same as the template.

        When 'name' is a ParameterContext - the provided ParameterContext is utilized as a 'template' and it's attributes are copied into the new
        ParameterContext.  If additional constructor arguments are provided (i.e. axis, fill_value, etc), they will be used preferentially
        over those in the 'template' ParameterContext.  If param_type is specified, it must be compatible (i.e. equivalent) to the param_type
        in the 'template' ParameterContext.

        @param name The local name OR a 'template' ParameterContext.
        @param param_type   The concrete AbstractParameterType; defaults to QuantityType if not provided
        @param axis The axis, typically a member of AxisTypeEnum; if not None, associated with the appropriate Domain
        @param fill_value  The default fill value
        @param variability Indicates if the parameter is a function of time, space, both, or neither; Default is VariabilityEnum.BOTH
        @param **kwargs Keyword arguments matching members of ParameterContext.ATTRS are applied.  Additional keyword arguments are copied and the copy is passed up to AbstractIdentifiable; see documentation for that class for details
        """
        kwc=kwargs.copy()
        my_kwargs = {x:kwc.pop(x) for x in self.ATTRS if x in kwc}
        new_name = kwc.pop('new_name') if 'new_name' in kwc else None
        param_context = None
        if not isinstance(name, (basestring, ParameterContext)):
            raise SystemError('\'name\' must be an instance of either basestring or ParameterContext')

        if isinstance(name, ParameterContext):
            param_context = name
            name = new_name or param_context.name

        # TODO: If additional required constructor parameters are added, make sure they're dealt with in _load
        if param_context is None and name is None:
            raise SystemError('Must specify \'name\', which can be either a string or ParameterContext.')
        AbstractIdentifiable.__init__(self, **kwc)
        if not param_context is None:
            self._derived_from_name = param_context._derived_from_name
            self.name = name or self._derived_from_name
            # CBM TODO: Is this right?  If param_type is provided, AND is equivalent to the clone's param_type, use it
            if not param_type is None and (param_type == param_context.param_type):
                self.param_type = param_type
            else:
                self.param_type = copy.deepcopy(param_context.param_type)

            # Ensure the param_type's name is the same as yours
            self.param_type.name = self.name

            self.axis = axis or param_context.axis
            if uom is not None:
                self.uom = uom
            elif param_context.uom is not None:
                self.uom = param_context.uom
            self.fill_value = fill_value or param_context.fill_value
            self.variability = variability or param_context.variability

            for a in self.ATTRS:
                setattr(self, a, kwargs[a] if a in my_kwargs else getattr(param_context, a))

        else:
            # TODO: Should this be None?  potential for mismatches if the self-given name happens to match a "blessed" name...
            self._derived_from_name = name
            self.name = name
            if param_type and not isinstance(param_type, AbstractParameterType):
                raise SystemError('\'param_type\' must be a concrete subclass of AbstractParameterType')
            self.param_type = param_type or QuantityType()

            # Ensure the param_type's name is the same as yours
            self.param_type.name = self.name

            self.axis = axis or None
            if uom is not None:
                self.uom = uom
            if fill_value is not None: # Provided by ParameterType
                self.fill_value = fill_value
            self.variability = variability or VariabilityEnum.BOTH

            for a in self.ATTRS:
                setattr(self, a, kwargs[a] if a in my_kwargs else None)

    @property
    def is_coordinate(self):
        """
        Indicates if the ParameterContext is a coordinate

        @returns    True if it is a coordinate; False otherwise
        """
        return not self.axis is None

    def get_module_dependencies(self):
        return self.param_type.get_module_dependencies()

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Derived from name: {1}'.format(indent, self._derived_from_name))
        lst.append('{0}Name: {1}'.format(indent, self.name))
        if self.is_coordinate:
            lst.append('{0}Is Coordinate: {1}'.format(indent, self.axis))
        lst.append('{0}Type: {1}'.format(indent, self.param_type))
        lst.append('{0}Fill Value: {1}'.format(indent, self.fill_value))
        if hasattr(self, 'uom'): #TODO: This should be dealt with by the ParameterType...
            lst.append('{0}Unit of Measure: {1}'.format(indent, self.uom))

        return '\n'.join(lst)

    def __eq__(self, other):
        if isinstance(other, ParameterContext):
            if self._derived_from_name == other._derived_from_name:
                if self.param_type == other.param_type:
                    return True

        return False

    def __ne__(self, other):
        return not self == other

    def __dir__(self):
        lst = dir(super(ParameterContext))
        map(lst.append, self.__dict__.keys())
        map(lst.append, self.param_type._template_attrs.keys())
        return lst

    def __getattr__(self, name):
        if 'param_type' in self.__dict__ and name in self.__dict__['param_type']._template_attrs.keys():
            return getattr(self.__dict__['param_type'], name)
        else:
            return getattr(super(ParameterContext, self), name)

    def __setattr__(self, key, value):
        if 'param_type' in self.__dict__ and key in self.__dict__['param_type']._template_attrs.keys():
            setattr(self.__dict__['param_type'], key, value)
        else:
            super(ParameterContext, self).__setattr__(key, value)

class ParameterDictionary(AbstractIdentifiable):
    """
    Contains a set of ParameterContext objects keyed by name

    The ParameterDictionary is used to compose both Streams (specifically Data or Parameter Streams) as well as the
    internal Dataset (via the Coverage API).
    """


    def __init__(self, contexts=None):
        """
        Construct a new ParameterDictionary

        @param contexts an iterable collection of ParameterContext objects to add to the ParameterDictionary
        """
        AbstractIdentifiable.__init__(self)
        self._map = OrderedDict()
        self.__count=0
        self.temporal_parameter_name = None

        if not contexts is None and hasattr(contexts, '__iter__'):
            for pc in contexts:
                if isinstance(pc, ParameterContext):
                    self.add_context(pc)

    def add_context(self, param_ctxt, is_temporal=False):
        """
        Add a ParameterContext

        @param param_ctxt The ParameterContext object to add
        @param is_temporal  If this ParameterContext should be used as the temporal parameter
        """
        if not isinstance(param_ctxt, ParameterContext):
            raise TypeError('param_ctxt must be a ParameterContext object')

        # CBM TODO: Fix this logic - never reaches "param_ctxt.axis = None",
        # that line and "if claims_time:" should replace the "raise NameError" which should be turned into a warning
        claims_time = param_ctxt.axis == AxisTypeEnum.TIME
        if is_temporal or claims_time:
            if self.temporal_parameter_name is None:
                self.temporal_parameter_name = param_ctxt.name
            else:
                raise NameError('This dictionary already has a parameter designated as \'temporal\': %s', self.temporal_parameter_name)
            param_ctxt.axis = AxisTypeEnum.TIME
        else:
            if claims_time:
                param_ctxt.axis = None

        self.__count += 1
        self._map[param_ctxt.name] = (self.__count, param_ctxt)

    def get_context(self, param_name):
        """
        Retrieve a ParameterContext by name

        @param param_name   The name of the ParameterContext
        @returns    The ParameterContext at key 'param_name'
        @throws KeyError    If 'param_name' is not found within the object
        """
        if not param_name in self._map:
            raise KeyError('The ParameterDictionary does not contain the specified key \'{0}\''.format(param_name))

        return self._map[param_name][1]

    def get_temporal_context(self):
        if self.temporal_parameter_name is not None:
            return self._map[self.temporal_parameter_name][1]
        else:
            raise KeyError('This dictionary does not have a parameter designated as \'temporal\'')

    def get_context_by_ord(self, ordinal):
        """
        Retrieve a ParameterContext by ordinal

        @param ordinal   The ordinal of the ParameterContext
        @returns    The ParameterContext with the ordinal 'ordinal'
        @throws KeyError    A parameter with the provided ordinal does not exist
        """
        return self.get_context(self.key_from_ord(ordinal))

    def ord_from_key(self, param_name):
        """
        Retrieve the ordinal for a ParameterContext by name

        @param param_name   The name of the ParameterContext
        @returns    The ordinal of the ParameterContext at key 'param_name'
        @throws KeyError    If 'param_name' is not found within the object
        """
        if not param_name in self._map:
            raise KeyError('The ParameterDictionary does not contain the specified key \'{0}\''.format(param_name))

        return self._map[param_name][0]

    def key_from_ord(self, ordinal):
        """
        Retrieve the parameter name for an ordinal

        @param ordinal   The ordinal of the ParameterContext
        @returns    The parameter name for the provided ordinal
        @throws KeyError    A parameter with the provided ordinal does not exist
        """
        for k, v in self.iteritems():
            if v[0] == ordinal:
                return k

        raise KeyError('Ordinal \'{0}\' not found in ParameterDictionary'.format(ordinal))

    def _todict(self):
        """
        Overrides Dictable._todict() to properly handle ordinals
        """
        #CBM TODO: try/except this to inform more pleasantly if it bombs
        res = dict((k,(v[0],v[1]._todict())) for k, v in self._map.iteritems())
        res.update((k,v._todict() if hasattr(v, '_todict') else v) for k, v in self.__dict__.iteritems() if k != '_map')
        res['cm_type'] = (self.__module__, self.__class__.__name__)
        return res

    @classmethod
    def _fromdict(cls, cmdict, arg_masks=None):
        """
        Overrides Dictable._fromdict() to properly handle ordinals
        """
        #CBM TODO: try/except this to inform more pleasantly if it bombs
        ret = ParameterDictionary()
        if isinstance(cmdict, dict):
            d=cmdict.copy()
            _=d.pop('cm_type')
            for k, v in d.iteritems():
                if isinstance(v, (tuple,list)) and len(v) == 2 and isinstance(v[0],int) and isinstance(v[1],dict) and 'cm_type' in v[1]:
                    pc = ParameterContext._fromdict(v[1])
                    ret._map[pc.name] = (v[0], pc)
                elif isinstance(v, dict) and 'cm_type' in v: # CBM TODO: Don't think we ever get here?!
                    ms, cs = v['cm_type']
                    module = __import__(ms, fromlist=[cs])
                    classobj = getattr(module, cs)
                    setattr(ret,k,classobj._fromdict(v))
                else:
                    setattr(ret, k, v)


        return ret

    def size(self):
        return self.__count

    def __len__(self):
        return self.size()

    def __iter__(self):
        return self._map.__iter__()

    def iteritems(self):
        return self._map.iteritems()

    def itervalues(self):
        return self._map.itervalues()

    def keys(self):
        return self._map.keys()

    def compare(self, other):
        """
        Performs a cross-wise comparison of the two ParameterDictionary objects.  Each member of self is compared to each member of other.

        @param other    A ParameterDictionary instance to compare to self
        @returns    A dictionary with keys from self and lists of matching keys from other
        """
        if not isinstance(other, ParameterDictionary):
            raise TypeError('\'other\' must be an instance of ParameterDictionary')

        ret = {}
        other_keys = other.keys()
        for ks, vs in self.iteritems():
            log.debug('Checking self key \'%s\'', ks)
            loop = True
            ret[ks] = []
            if ks in other_keys:
                log.debug('Found key \'%s\' in other, comparing values', ks)
                vo = other.get_context(ks)
                if vs[1] == vo:
                    log.debug('Value of \'%s\' in self == value of \'%s\' in other', ks, ks)
                    ret[ks].append(ks)
                    del other_keys[other_keys.index(ks)]
                    loop=False
            if loop:
                for ko in list(other_keys):
                    vo=other.get_context(ko)
                    if vs[1] == vo:
                        log.debug('Value of \'%s\' in self == value of \'%s\' in other', ks, ko)
                        ret[ks].append(ks)
                        del other_keys[other_keys.index(ks)]

        # Back-check any leftover keys
        log.debug('Back-checking leftover keys from other')
        for ko in list(other_keys):
            for ks, vs in self.iteritems():
                vo=other.get_context(ko)
                if vs[1] == vo:
                    log.debug('Value of \'%s\' in self == value of \'%s\' in other', ks, ko)
                    ret[ks].append(ko)
                    del other_keys[other_keys.index(ko)]

        ret[None] = other_keys

        return ret


    def __eq__(self, other):
        try:
            res = self.compare(other)
            for k,v in res.iteritems():
                if k is None:
                    if len(v) > 0:
                        return False
                elif len(v) != 1 or k != v[0]:
                    return False

            return True
        except TypeError:
            return False

    def __ne__(self, other):
        return not self == other

class ParameterFunctionValidator(object):
    """
    Class for validating that the union of multiple sets of parameter contexts
    is capable of fulfilling given dependent parameters
    """

    def __init__(self, contexts_with_values, *pd_contexts):
        """
        Constructor for ParameterFunctionValidator

        @param contexts_with_values single or iterable of ParameterContext objects that have obtainable values
        @param pd_contexts  one or more iterable sets of ParameterContext objects
        @return a FunctionValidator instance
        """
        self._ctxts = {}
        if not hasattr(contexts_with_values, '__iter__'):
            contexts_with_values = [contexts_with_values]
        self._cwv = [p for p in contexts_with_values if isinstance(p, ParameterContext)]
        self._cwvn = [p.name for p in self._cwv]

        import itertools
        ctxts = contexts_with_values
        ctxts += [c for c in itertools.chain(*pd_contexts)]

        for p in [p for p in ctxts if isinstance(p, ParameterContext)]:
            if p.name not in self._ctxts:
                np = ParameterContext(p) # Copy
                if hasattr(np.param_type, '_pctxt_callback'):
                    np._pctxt_callback = self._ctxt_callback
                self._ctxts[p.name] = np

    def _ctxt_callback(self, context_name):
        return self._ctxts[context_name]

    def validate(self, context_name):
        """
        Validate that the ParameterContext indicated by <i>context_name</i> can be fulfilled by the set of ParameterContexts
        available to this ParameterFunctionValidator

        @param context_name the name of the parameter to validate, must be a member of self._ctxts
        """
        try:
            # Attempt to build the graph
            g = self._ctxts[context_name].param_type.get_dependency_graph()

            # First ensure all of the things we have values for are made forestgreen
            for n in [n for n in self._cwvn if n in g.node]:
                g.node[n]['color'] = g.node[n]['fontcolor'] = 'forestgreen'

            def _prune(g, n, parent_green=False, ind=''):
    #            print '{0}node: {1}'.format(ind,n)
                for s in g.successors(n):
                    preds=g.predecessors(s)
    #                print '{0}succ: {1}'.format(ind,s)
                    c = g.node[n]['color']
                    if c == 'forestgreen' or parent_green:
    #                    print '{0}is_green'.format(ind)
                        _prune(g, s, True, ind=ind+'  ')
                        if len(preds) == 1:
                            if c != 'forestgreen':
    #                            print '{0}remove: {1}'.format(ind,s)
                                g.remove_node(s)
                    else:
                        _prune(g, s, parent_green, ind=ind+'  ')

                preds=g.predecessors(n)
                if len(g.successors(n)) == 0 and len(preds) == 1 and g.node[preds[0]]['color'] == 'forestgreen':
                    g.remove_node(n)

            _prune(g, context_name)

            # Get the leaf nodes again - if any AREN'T green, we have a problem
            missing = [n for n,d in g.out_degree().items() if d==0 and g.node[n]['color'] != 'forestgreen']

            if len(missing) > 0:
                raise ValueError('Unable to calculate \'{0}\', missing values or functions: {1}'.format(context_name, list(missing)))

            return g
        except Exception as ex:
            import sys
            raise ParameterFunctionException(ex.message, type(ex)), None, sys.exc_traceback



"""

from coverage_model.parameter import *
from coverage_model.parameter_types import *
pd=ParameterDictionary()
pd.add_context(ParameterContext('p1',param_type=QuantityType(value_encoding='f', uom='m/s')))
pd.add_context(ParameterContext('p2',param_type=QuantityType(value_encoding='d', uom='km')))
pd.add_context(ParameterContext('p3',param_type=QuantityType(value_encoding='i', uom='s')))
dout=pd.dump()

pd2=ParameterDictionary.load(dout)

from pyon.net.endpoint import Publisher
pub=Publisher()

pub.publish(dout,('ex','gohere'))

"""
