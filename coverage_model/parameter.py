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
