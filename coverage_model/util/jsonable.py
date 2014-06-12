__author__ = 'casey'

import json
from coverage_model.basic_types import Dictable

class Jsonable(Dictable):

    def to_json(self, sort=True, indent=None, separators=None):
        return json.dumps(self.__dict__, sort_keys=sort, indent=indent, separators=separators)

    @classmethod
    def from_json(cls, json_str):
        return NotImplementedError('Not implemented in base class')


def unicode_convert(data):
    if isinstance(data, dict):
        return {unicode_convert(key): unicode_convert(value) for key, value in data.iteritems()}
    elif isinstance(data, list):
        return [unicode_convert(value) for value in data]
    elif isinstance(data, unicode):
        return data.encode('utf-8')
    else:
        return data
