__author__ = 'casey'

import json
from coverage_model.basic_types import Dictable
from coverage_model.util.jsonable import Jsonable, unicode_convert


class ReferenceCoverageExtents(Jsonable):

    def __init__(self, name, reference_coverage_id, time_extents=None, domain_extents=None):
        self.name = str(name)
        self.cov_id = str(reference_coverage_id)

        if time_extents is not None:
            if not isinstance(time_extents, (tuple, set, list)) or len(time_extents) != 2:
                raise ValueError('Time extents should be a tuple (min,max). Found: %s' % repr(domain_extents))
            self.time_extents = tuple(time_extents)
        else:
            self.time_extents = time_extents

        self.domain_extents = {}
        if domain_extents is not None:
            if not isinstance(domain_extents, dict):
                raise ValueError('Domain extents should be a dictionary of name/tuple(min,max) key/value pairs.  Found:', domain_extents)
            for k,v in domain_extents.iteritems():
                if not isinstance(v, (tuple, list, set)) or len(v) != 2:
                    raise ValueError('Domain extents should be a dictionary of name/tuple(min,max) key/value pairs.  Found:', domain_extents)
                self.domain_extents[str(k)] = tuple(v)

    @classmethod
    def from_json(cls, json_str):
        d = json.loads(json_str, object_hook=unicode_convert)
        return cls.from_dict(d)

    @staticmethod
    def from_dict(json_object):
        if not set(['name', 'cov_id', 'time_extents', 'domain_extents']).issubset(set(json_object.keys())):
            raise KeyError("Dictionary cannot be used to create object: %s" % json_object)
        return ReferenceCoverageExtents(json_object['name'], json_object['cov_id'], json_object['time_extents'], json_object['domain_extents'])

    def __str__(self):
        return '%s: %s' % (self.__class__.__name__, repr(self.__dict__))


class ExtentsDict(Dictable):

    def __init__(self, extent_dict=None):
        self.data = {}
        if extent_dict is not None:
            for k,v in extent_dict.iteritems():
                self.add_extents(k, v)

    def add_extents(self, cov_id, extents):
        if isinstance(extents, ReferenceCoverageExtents):
            extents = [extents]
        if cov_id in self.data:
            self.data[cov_id].extend(extents)
        else:
            self.data[cov_id] = extents

    def replace_extents(self, cov_id, extents):
        if isinstance(extents, ReferenceCoverageExtents):
            extents = [extents]
        self.data[cov_id] = extents