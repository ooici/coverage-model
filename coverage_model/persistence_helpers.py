#!/usr/bin/env python

"""
@package coverage_model.persistence_helpers
@file coverage_model/persistence_helpers.py
@author Christopher Mueller
@brief Helper functions and classes for the PersistenceLayer
"""

from pyon.core.interceptor.encode import encode_ion, decode_ion
from ooi.logging import log
from coverage_model.basic_types import Dictable
from coverage_model import utils
from coverage_model.metadata import MetadataManager
from coverage_model.data_span import SpanStats, SpanStatsCollection
from coverage_model.address import BrickAddress

import os
import h5py
from coverage_model.hdf_utils import HDFLockingFile
import msgpack
import numpy as np


def pack(payload):
    return msgpack.packb(payload, default=encode_ion).replace('\x01','\x01\x02').replace('\x00','\x01\x01')


def unpack(msg):
    return msgpack.unpackb(msg.replace('\x01\x01','\x00').replace('\x01\x02','\x01'), object_hook=decode_ion)


def get_coverage_type(path):
    ctype = 'simplex'
    if os.path.exists(path):
        with HDFLockingFile(path) as f:
            if 'coverage_type' in f.attrs:
                ctype = unpack(f.attrs['coverage_type'][0])

    return ctype


class RTreeItem(object):

    def __init__(self, item_id, obj):
        self.id = item_id
        self.object = obj


# Proxy the properties.dimension property, pia...
class HoldDim(object):
    def __init__(self, dim=2):
        self.dimension = dim


class RTreeProxy(object):

    def __init__(self):
        self._spans = []

        self.properties = HoldDim()

    def insert(self, count, extents, obj=None):
        # The extents from the old rtree impl are [xmin,ymin,xmax,ymax]
        minval = extents[0]
        maxval = extents[2]

        from coverage_model.basic_types import Span
        span = Span(minval, maxval, value=obj)

        if span not in self._spans:
            self._spans.append(span)

    def intersection(self, coords, objects=True):
        minval = coords[0]
        maxval = coords[2]

        si = 0
        ei = len(self._spans)
        for i, s in enumerate(self._spans):
            if minval in s:
                si = i
                break

        for i, s in enumerate(self._spans):
            if maxval in s:
                ei = i+1
                break

        ret = []
        for i, s in enumerate(self._spans[si:ei]):
            ret.append(RTreeItem(si+i, s.value))
        return ret

    @property
    def bounds(self):
        lb = float(self._spans[0].lower_bound) if len(self._spans) > 0 else 0.0
        ub = float(self._spans[0].upper_bound) if len(self._spans) > 0 else 0.0
        return [lb, 0.0, ub, 0.0]

    def serialize(self):
        out = ''
        if len(self._spans) > 0:
            out = self.__class__.__name__
            for span in self._spans:
                tup_str = ''
                for i in span.tuplize(with_value=True):
                    tup_str = '%(orig)s_%(val)s' % {'orig': tup_str, 'val': i}
                out = '%(orig)s%(sep)s%(new)s' % {'orig': out, 'sep': '::span::', 'new': tup_str}
        return out

    @classmethod
    def deserialize(cls, src_str):
        if isinstance(src_str, basestring) and src_str.startswith(cls.__name__):
            tmp = src_str.strip(cls.__name__)
            rtp = RTreeProxy()
            for span_tpl in tmp.split('::span::'):
                if span_tpl == '':
                    continue
                a, b, c, d, e = span_tpl.split('_')
                span_tpl = (int(b), int(c), int(d), e)
                from coverage_model.basic_types import Span
                span = Span.from_iterable(span_tpl)
                rtp._spans.append(span)
            return rtp
        raise TypeError('Improper formatting for RTreeProxy deserialization: %s' % src_str)

    def __eq__(self, other):
        return self.serialize() == other.serialize()

    def __ne__(self, other):
        return not self.__eq__(other)


class BaseManager(MetadataManager):

    @staticmethod
    def dirExists(directory):
        os.path.exists(directory)

    @staticmethod
    def isPersisted(directory, guid):
        if os.path.exists(directory):
            file_path = os.path.join(directory, guid)
            if os.path.exists(file_path):
                return True

        return False

    def storage_type(self):
        return 'hdf'

    @staticmethod
    def getCoverageType(directory, guid):
        return get_coverage_type(os.path.join(directory, guid, '{0}_master.hdf5'.format(guid)))

    def __init__(self, root_dir, file_name, **kwargs):
        MetadataManager.__init__(self, **kwargs)
        self.root_dir = root_dir
        self.file_path = os.path.join(root_dir, file_name)

        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)

        if os.path.exists(self.file_path):
            self._load()

        for k, v in kwargs.iteritems():
            # Don't overwrite with None
            if hasattr(self, k) and v is None:
                continue

            setattr(self, k, v)

    def flush(self):
        if self.is_dirty(True):
            try:
                with HDFLockingFile(self.file_path, 'a') as f:
                    for k in list(self._dirty):
                        v = getattr(self, k)
    #                    log.debug('FLUSH: key=%s  v=%s', k, v)
                        if isinstance(v, Dictable):
                            prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                            value = prefix + pack(v.dump())
                        else:
                            value = pack(v)

                        f.attrs[k] = np.array([value])

                        # Update the hash_value in _hmap
                        self._hmap[k] = utils.hash_any(v)
                        # Remove the key from the _dirty set
                        self._dirty.remove(k)
            except IOError, ex:
                if "unable to create file (File accessability: Unable to open file)" in ex.message:
                    log.info('Issue writing to hdf file during master_manager.flush - this is not likely a huge problem: %s', ex.message)
                else:
                    raise

            super(BaseManager, self).__setattr__('_is_dirty',False)

    def _load(self):
        raise NotImplementedError('Not implemented by base class')

    def _base_load(self, f):
        for key, val in f.attrs.iteritems():
            val = val[0]
            if isinstance(val, basestring) and val.startswith('DICTABLE'):
                i = val.index('|', 9)
                smod, sclass = val[9:i].split(':')
                value = unpack(val[i+1:])
                module = __import__(smod, fromlist=[sclass])
                classobj = getattr(module, sclass)
                value = classobj._fromdict(value)
            elif key in ('root_dir', 'file_path'):
                # No op - set in constructor
                continue
            else:
                value = unpack(val)

            if isinstance(value, tuple):
                value = list(value)

            setattr(self, key, value)

    def is_dirty(self, force_deep=False):
        """
        Tells if the object has attributes that have changed since the last flush

        @return: True if the BaseMananager object is dirty and should be flushed
        """
        if not force_deep and self._is_dirty: # Something new was set, easy-peasy
            return True
        else: # Nothing new has been set, need to check hashes
            self._dirty.difference_update(self._ignore) # Ensure any ignored attrs are gone...
            for k, v in [(k,v) for k, v in self.__dict__.iteritems() if not k in self._ignore and not k.startswith('_')]:
                chv = utils.hash_any(v)
                # log.trace('key=%s:  cached hash value=%s  current hash value=%s', k, self._hmap[k], chv)
                if self._hmap[k] != chv:
                    self._dirty.add(k)
            return len(self._dirty) != 0

    def __setattr__(self, key, value):
        super(BaseManager, self).__setattr__(key, value)
        if not key in self._ignore and not key.startswith('_'):
            self._hmap[key] = utils.hash_any(value)
            self._dirty.add(key)
            super(BaseManager, self).__setattr__('_is_dirty',True)


class MasterManager(BaseManager):

    def __init__(self, root_dir, guid, **kwargs):
        BaseManager.__init__(self, root_dir=os.path.join(root_dir,guid), file_name='{0}_master.hdf5'.format(guid), **kwargs)
        self.guid = guid
        if hasattr(self, 'parameter_bounds') and self.parameter_bounds is None:
            self.parameter_bounds = {}

        # Add attributes that should NEVER be flushed
        self._ignore.update(['param_groups', 'guid', 'file_path', 'root_dir', 'brick_tree'])
        if not hasattr(self, 'param_groups'):
            self.param_groups = set()

    def update_rtree(self, count, extents, obj):
        log.debug('MM count: {0}'.format(count))
        if not hasattr(self, 'brick_tree'):
            raise AttributeError('Cannot update rtree; object does not have a \'brick_tree\' attribute!!')

        log.debug('self.file_path: {0}'.format(self.file_path))
        with HDFLockingFile(self.file_path, 'a') as f:
            rtree_ds = f.require_dataset('rtree', shape=(count,), dtype=h5py.special_dtype(vlen=str), maxshape=(None,))
            rtree_ds.resize((count+1,))
            rtree_ds[count] = pack((extents, obj))

            self.brick_tree.insert(count, extents, obj=obj)

    def _init_rtree(self, bD):
        self.brick_tree = RTreeProxy()

    def _load(self):
        with HDFLockingFile(self.file_path, 'r') as f:
            self._base_load(f)

            self.param_groups = set()
            f.visit(self.param_groups.add)
            # TODO: Use valid parameter list to compare against inspected param_groups and discard all that are invalid
            self.param_groups.discard('rtree')

            # Don't forget brick_tree!
            if 'rtree' in f.keys():
                # Populate brick tree from the 'rtree' dataset
                ds = f['/rtree']

                def tree_loader(darr):
                    for i, x in enumerate(darr):
                        ext, obj = unpack(x)
                        yield (i, ext, obj)

                rtp = RTreeProxy()
                for x in tree_loader(ds[:]):
                    rtp.insert(*x)

                setattr(self, 'brick_tree', rtp)
            else:
                setattr(self, 'brick_tree', RTreeProxy())

    def add_external_link(self, link_path, rel_ext_path, link_name):
        with HDFLockingFile(self.file_path, 'r+') as f:
            f[link_path] = h5py.ExternalLink(rel_ext_path, link_name)

    def create_group(self, group_path):
        with HDFLockingFile(self.file_path, 'r+') as f:
            f.create_group(group_path)


class ParameterManager(BaseManager):

    def __init__(self, root_dir, parameter_name, read_only=True, **kwargs):
        BaseManager.__init__(self, root_dir=root_dir, file_name='{0}.hdf5'.format(parameter_name), **kwargs)
        self.parameter_name = parameter_name
        self.read_only = read_only

        # Add attributes that should NEVER be flushed
        self._ignore.update(['brick_tree', 'file_path', 'root_dir', 'read_only'])

    def thin_origins(self, origins):
        pass

    def flush(self):
        if not self.read_only:
            super(ParameterManager, self).flush()

    def _load(self):
        with HDFLockingFile(self.file_path, 'r') as f:
            self._base_load(f)

