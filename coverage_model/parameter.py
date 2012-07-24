#!/usr/bin/env python

"""
@package coverage_model.parameter
@file coverage_model/parameter.py
@author Christopher Mueller
@brief Concrete parameter classes
"""

from pyon.public import log
from coverage_model.basic_types import AbstractIdentifiable, VariabilityEnum
from coverage_model.parameter_types import AbstractParameterType

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
    - structure: from the perspective of the container (coverage-model, granule), this includes things such as reference_frame
    (a.k.a. 'axis') and variability (one of the VariabilityEnum members).
    - data-critical-metadata: 'metadata' that is critical to the storage/processing of the data - such as fill_value, nil_value, etc.

    @todo: This will likely undergo moderate to significant changes as things continue to mature
    """
    # TODO: Need to incorporate some indication of if the parameter is a function of temporal, spatial, both, or None
    def __init__(self, name, param_type, reference_frame=None, fill_value=None, variability=None):
        """
        Construct a new ParameterContext object

        @param name The local name
        @param param_type   The concrete AbstractParameterType
        @param reference_frame The reference frame, often a coordinate axis identifier
        @param fill_value  The default fill value
        @param variability Indicates if the parameter is a function of time, space, both, or neither
        """
        # TODO: If additional required constructor parameters are added, make sure they're dealt with in _load
        AbstractIdentifiable.__init__(self)
        self.name = name
        if not isinstance(param_type, AbstractParameterType):
            raise SystemError('\'param_type\' must be a concrete subclass of AbstractParameterType')
        self.param_type = param_type

        # Expose the template_attrs from the param_type
        for k,v in self.param_type.template_attrs.iteritems():
            setattr(self,k,v)

        self.reference_frame = reference_frame or None
        self.fill_value = fill_value or -999
        self.variability = variability or VariabilityEnum.TEMPORAL

    @property
    def is_coordinate(self):
        """
        Indicates if the ParameterContext is a coordinate

        @returns    True if it is a coordinate; False otherwise
        """
        return not self.reference_frame is None

    def _dump(self):
        """
        Get a composable type (dict) representation of the ParameterContext and all internal objects

        @returns    A dict representation of this ParameterContext
        """
        ret={'cm_type': self.__class__.__name__}
        for k,v in self.__dict__.iteritems():
            #TODO: If additional objects are added to ParameterContext, add their attr name here as well
            if isinstance(v, (AbstractParameterType,)):
                ret[k] = v._dump()
            else:
                ret[k] = v

        return ret

    @classmethod
    def _load(cls, pcdict):
        """
        Create a ParameterContext from a dict representation

        @param cls  A ParemeterContext instance
        @param pcdict   A dict containing ParameterContext information (requires 'cm_type' and 'p_type' keys)
        """
        if isinstance(pcdict, dict) and 'cm_type' in pcdict and pcdict['cm_type'] == ParameterContext.__name__:
            #TODO: If additional required constructor parameters are added, make sure they're dealt with here
            ptd = AbstractParameterType._load(pcdict['param_type'])
            n = pcdict['name']
            pc=ParameterContext(n,ptd)
            for k, v in pcdict.iteritems():
                if not k in ('name','param_type','cm_type'):
                    setattr(pc,k,v)

            return pc
        else:
            raise TypeError('pcdict is not properly formed, must be of type dict and contain ' \
                            'a \'cm_type\' key with the value \'{0}\': {1}'.format(ParameterContext.__name__, pcdict))

    def __str__(self, indent=None):
        indent = indent or ' '
        lst = []
        lst.append('{0}ID: {1}'.format(indent, self._id))
        lst.append('{0}Name: {1}'.format(indent, self.name))
        if self.is_coordinate:
            lst.append('{0}Is Coordinate: {1}'.format(indent, AxisTypeEnum._str_map[self.reference_frame]))
        lst.append('{0}Type: {1}'.format(indent, self.param_type))
        lst.append('{0}Fill Value: {1}'.format(indent, self.fill_value))
        if hasattr(self, 'uom'): #TODO: This should be dealt with by the ParameterType...
            lst.append('{0}Unit of Measure: {1}'.format(indent, self.uom))

        return '\n'.join(lst)

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
        self._map = {}
        self.__count=-1

        if not contexts is None and hasattr(contexts, '__iter__'):
            for pc in contexts:
                if isinstance(pc, ParameterContext):
                    self.add_context(pc)

    def add_context(self, param_ctxt):
        """
        Add a ParameterContext

        @param param_ctxt The ParameterContext object to add
        """
        if not isinstance(param_ctxt, ParameterContext):
            raise TypeError('param_ctxt must be a ParameterContext object')

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

        @returns    The parameter name for the provided ordinal
        @throws KeyError    A parameter with the provided ordinal does not exist
        """
        for k, v in self.iteritems():
            if v[0] == ordinal:
                return k

        raise KeyError('Ordinal \'{0}\' not found in ParameterDictionary'.format(ordinal))


    def dump(self):
        """
        Retrieve a standard dict object representing the ParameterDictionary and all sub-objects

        @returns    A dict containing all information in the ParameterDictionary
        """
        #TODO: try/except this to inform more pleasantly if it bombs
        res = dict((k,(v[0],v[1]._dump())) for k, v in self._map.iteritems())

        return res

    @classmethod
    def load(cls, pdict):
        """
        Create a ParameterDictionary from a dict

        @param cls  A ParameterDictionary instance
        @param pdict    A dict object containing valid ParameterDictionary content
        """
        ret = ParameterDictionary()
        if isinstance(pdict, dict):
            for k, v in pdict.iteritems():
                pc = ParameterContext._load(v[1])
                ret._map[pc.name] = (v[0], pc)

        return ret

    def __iter__(self):
        return self._map.__iter__()

    def iteritems(self):
        return self._map.iteritems()


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