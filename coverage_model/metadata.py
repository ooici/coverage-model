#!/usr/bin/env python

from ooi.logging import log

from coverage_model.data_span import SpanCollectionByFile, SpanStats
from coverage_model.address import BrickFileAddress
import numpy


class MetadataManager(object):

    def __init__(self, **kwargs):
        super(MetadataManager, self).__setattr__('_hmap',{})
        super(MetadataManager, self).__setattr__('_dirty',set())
        super(MetadataManager, self).__setattr__('_ignore',set())
        self._ignore.add('span_collection')
        super(MetadataManager, self).__setattr__('span_collection', SpanCollectionByFile())

    def __setattr__(self, key, value):
        super(MetadataManager, self).__setattr__(key, value)

    @staticmethod
    def isPersisted(directory, guid):
        raise NotImplementedError('Not implemented by base class')

    def storage_type(self):
        return 'db'

    def flush(self):
        raise NotImplementedError('Not implemeted by base class')

    def _load(self):
        raise NotImplementedError('Not implemented by base class')

    def _base_load(self, f):
        raise NotImplementedError('Not implemented by base class')

    def is_dirty(self, force_deep=False):
        raise NotImplementedError('Not implemented by base class')

    def update_rtree(self, count, extents, obj):
        raise NotImplementedError('Not implemented by base class')

    def _init_rtree(self, bD):
        raise NotImplementedError('Not implemented by base class')

    def add_external_link(self, link_path, rel_ext_path, link_name):
        raise NotImplementedError('Not implemented by base class')

    def create_group(self, group_path):
        raise NotImplementedError('Not implemented by base class')

    def track_data_written_to_brick(self, brick_id, brick_slice, param_name, min_val, max_val):
        log.trace("Extending parameter %s min/max to %s/%s %s type", param_name, min_val, max_val, type(min_val))
        if type(min_val).__module__ == numpy.__name__:
            min_val = numpy.asscalar(min_val)
        if type(max_val).__module__ == numpy.__name__:
            max_val = numpy.asscalar(max_val)
        log.trace("After numpy conversion %s min/max to %s/%s %s type", param_name, min_val, max_val, type(min_val))
        address = BrickFileAddress(self.guid, brick_id)
        span = self.span_collection.get_span(address)
        min_max = (min_val, max_val)
        if span is None:
            params = dict()
            params[param_name] = min_max
            span = SpanStats(address, params)
            self.span_collection.add_span(span)
        else:
            span.extend_param(param_name, min_max)
        tup = span.params[param_name]
        log.trace("From span: %s min/max to %s/%s %s type", param_name, tup[0], tup[1], type(tup[0]))

    def dump_span_collection(self):
        log.trace("Spans for guid %s", self.guid)
        for key in self.span_collection.span_dict.keys():
            for k2 in self.span_collection.span_dict[key]:
                span = self.span_collection.span_dict[key][k2]
                log.trace("Span: %s", span.as_tuple_str())

    def __eq__(self, other):
        return 0 == len(self.hdf_conversion_key_diffs(other))

    def __ne__(self, other):
        return not self.__eq__(other)

    def hdf_conversion_key_diffs(self, other):
        key_diffs = set(self.__dict__.keys()) - set(other.__dict__.keys())
        for key in self.__dict__:
            if key in key_diffs:
                continue # know it isn't in other.__dict__
            if key is '_hmap':
                continue # calculated value by object these will never be equal for different objects
            if key is '_ignore':
                continue  # This is an HDF artifact.  Ignored flush values are different between versions
            if key in ['sdom', 'tdom', 'brick_domains', 'brick_list', '_dirty', '_is_dirty']:
                continue # Tuple/List conversion by Postgres prevents comparison
            elif self.__dict__[key] != other.__dict__[key]:
                key_diffs.add(key)

        return key_diffs
