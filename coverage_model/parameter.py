#!/usr/bin/env python

"""
@package 
@file parameter
@author Christopher Mueller
@brief 
"""

from pyon.public import log
from coverage_model.basic_types import AbstractIdentifiable
from coverage_model.parameter_types import *

#################
# Parameter Objects
#################

class Parameter(AbstractIdentifiable):
    """

    """
    def __init__(self, parameter_context, shape, value):
        AbstractIdentifiable.__init__(self)
        self.context = parameter_context
        self.value = value
        self.shape = shape

    # Expose a couple of the context attributes at this level as "read only"
    @property
    def name(self):
        return self.context.name

    @property
    def is_coordinate(self):
        return self.context.is_coordinate

class ParameterContext(AbstractIdentifiable):
    # TODO: Need to incorporate some indication of if the parameter is a function of temporal, spatial, both, or None
    """

    """
    def __init__(self, name, param_type, reference_frame=None, fill_value=None):
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

    @property
    def is_coordinate(self):
        return not self.reference_frame is None

    def _dump(self):
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

class ParameterDictionary(object):

    def __init__(self):
    #        if not param_dict is None and not isinstance(param_dict, dict):
    #            raise TypeError('param_dict must be a ParameterDictionary object')
        self._map = {}

    def add_context(self, param_ctxt):
        if not isinstance(param_ctxt, ParameterContext):
            raise TypeError('param_ctxt must be a ParameterContext object')

        self._map[param_ctxt.name] = param_ctxt

    def get_context(self, param_name):
        if not param_name in self._map:
            raise KeyError('The ParameterDictionary does not contain the specified key \'{0}\''.format(param_name))

        return self._map[param_name]

    def dump(self):
        #TODO: try/except this to inform more pleasantly if it bombs
        res = dict((k,v._dump()) for k, v in self._map.iteritems())

        return res

    @classmethod
    def load(cls, pdict):
        ret = ParameterDictionary()
        if isinstance(pdict, dict):
            for k, v in pdict.iteritems():
                ret.add_context(ParameterContext._load(v))

        return ret

    def __iter__(self):
        return self._map.__iter__()

"""

from coverage_model.parameter import *
from coverage_model.parameter_types import *
pd=ParameterDictionary()
pd.add_context(ParameterContext('p1',param_type=QuantityType(value_encoding='f', uom='m/s')))
dout=pd.dump()

pd2=ParameterDictionary.load(dout)

from pyon.net.endpoint import Publisher
pub=Publisher()

pub.publish(dout,('ex','gohere'))

"""