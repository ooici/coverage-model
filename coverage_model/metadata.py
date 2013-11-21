#!/usr/bin/env python

from ooi.logging import log


class MetadataManager(object):

    def __init__(self, **kwargs):
        super(MetadataManager, self).__setattr__('_hmap',{})
        super(MetadataManager, self).__setattr__('_dirty',set())
        super(MetadataManager, self).__setattr__('_ignore',set())
        super(MetadataManager, self).__setattr__('_stored',set())
        super(MetadataManager, self).__setattr__('_filesWritten',set())

    def __setattr__(self, key, value):
        super(MetadataManager, self).__setattr__(key, value)

    @staticmethod
    def isPersisted(directory, guid):
        raise NotImplementedError('Not implemented by base class')

    def flush(self):
        '''
        if len(self._stored) > 0:
            for k in self._stored:
                print k, " = ", getattr(self, k)

            print self._stored
        '''
        pass

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
