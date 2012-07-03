#!/usr/bin/env python

"""
@package 
@file common
@author Christopher Mueller
@brief 
"""

import uuid
def create_guid():
    """
    @retval Return global unique id string
    """
    # guids seem to be more readable if they are UPPERCASE
    return str(uuid.uuid4()).upper()

class AbstractBase(object):
    """

    """
    def __init__(self):
        self.mutable = False
        self.extension = {}

class AbstractIdentifiable(AbstractBase):
    """

    """
    def __init__(self):
        AbstractBase.__init__(self)
        self._id = create_guid()
        self.label = ''
        self.description = ''

    @property
    def id(self):
        return self._id