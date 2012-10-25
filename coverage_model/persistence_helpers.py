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

import os
import rtree
import h5py
import msgpack


def pack(payload):
    return msgpack.packb(payload, default=encode_ion).replace('\x01','\x01\x02').replace('\x00','\x01\x01')

def unpack(msg):
    return msgpack.unpackb(msg.replace('\x01\x01','\x00').replace('\x01\x02','\x01'), object_hook=decode_ion)

class BaseManager(object):

    def __init__(self, root_dir, file_name, **kwargs):
        super(BaseManager, self).__setattr__('_hmap',{})
        super(BaseManager, self).__setattr__('_dirty',set())
        super(BaseManager, self).__setattr__('_ignore',set())
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
            with h5py.File(self.file_path, 'a') as f:
                for k in list(self._dirty):
                    v = getattr(self, k)
                    log.trace('FLUSH: key=%s  v=%s', k, v)
                    if isinstance(v, Dictable):
                        prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                        value = prefix + pack(v.dump())
                    else:
                        value = pack(v)

                    f.attrs[k] = value

                    # Update the hash_value in _hmap
                    self._hmap[k] = self._dohash(v)
                    # Remove the key from the _dirty set
                    self._dirty.remove(k)

            super(BaseManager, self).__setattr__('_is_dirty',False)

    def _load(self):
        raise NotImplementedError('Not implemented by base class')

    def _base_load(self, f):
        for key, val in f.attrs.iteritems():
            if val.startswith('DICTABLE'):
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
                chv = self._dohash(v)
                log.trace('key=%s:  cached hash value=%s  current hash value=%s', k, self._hmap[k], chv)
                if self._hmap[k] != chv:
                    self._dirty.add(k)

            return len(self._dirty) != 0

    def _dohash(self, value, hv=None):
        hv = hv or 0
        if value is None or isinstance(value, (str, unicode, int, long, float, bool)):
            log.trace('is primitive:  value=%s  hv=%s', value, hv)
            hv = hash(value) ^ hv
        elif isinstance(value, (list, tuple, set)):
            log.trace('is list/tuple/set:  value=%s  hv=%s', value, hv)
            for x in value:
                hv = self._dohash(x, hv)
        elif isinstance(value, dict):
            log.trace('is dict:  value=%s  hv=%s', value, hv)
            for k,v in value.iteritems():
                hv = self._dohash(k, hv)
                hv = self._dohash(v, hv)
        elif isinstance(value, object):
            log.trace('is object:  value=%s  hv=%s', value, hv)
            hv = self._dohash(value.__dict__, hv)

        return hv

    def __setattr__(self, key, value):
        self.__dict__[key] = value
        if not key in self._ignore and not key.startswith('_'):
            self._hmap[key] = self._dohash(value)
            self._dirty.add(key)
            super(BaseManager, self).__setattr__('_is_dirty',True)

class MasterManager(BaseManager):

    def __init__(self, root_dir, guid, **kwargs):
        BaseManager.__init__(self, root_dir=os.path.join(root_dir,guid), file_name='{0}_master.hdf5'.format(guid), **kwargs)
        self.guid = guid

        # Add attributes that should NEVER be flushed
        self._ignore.add('param_groups')
        if not hasattr(self, 'param_groups'):
            self.param_groups = set()

    def _load(self):
        with h5py.File(self.file_path, 'r') as f:
            self._base_load(f)

            self.param_groups = set()
            f.visit(self.param_groups.add)

    def add_external_link(self, link_path, rel_ext_path, link_name):
        with h5py.File(self.file_path, 'r+') as f:
            f[link_path] = h5py.ExternalLink(rel_ext_path, link_name)

    def create_group(self, group_path):
        with h5py.File(self.file_path, 'r+') as f:
            f.create_group(group_path)


class ParameterManager(BaseManager):

    def __init__(self, root_dir, parameter_name, **kwargs):
        BaseManager.__init__(self, root_dir=root_dir, file_name='{0}.hdf5'.format(parameter_name), **kwargs)
        self.parameter_name = parameter_name

        # Add attributes that should NEVER be flushed
        self._ignore.add('brick_tree')

    def thin_origins(self, origins):
        pass

    def update_rtree(self, count, extents, obj):
        if not hasattr(self, 'brick_tree'):
            raise AttributeError('Cannot update rtree; object does not have a \'brick_tree\' attribute!!')

        with h5py.File(self.file_path, 'a') as f:
            rtree_ds = f.require_dataset('rtree', shape=(count,), dtype=h5py.special_dtype(vlen=str), maxshape=(None,))
            rtree_ds.resize((count+1,))
            rtree_ds[count] = pack((extents, obj))

            self.brick_tree.insert(count, extents, obj=obj)

    def _load(self):
        with h5py.File(self.file_path, 'r') as f:
            self._base_load(f)

            # Don't forget brick_tree!
            p = rtree.index.Property()
            p.dimension = self.tree_rank

            if 'rtree' in f.keys():
                # Populate brick tree from the 'rtree' dataset
                ds = f['/rtree']

                def tree_loader(darr):
                    for i, x in enumerate(darr):
                        ext, obj = unpack(x)
                        yield (i, ext, obj)

                setattr(self, 'brick_tree', rtree.index.Index(tree_loader(ds[:]), properties=p))
            else:
                setattr(self, 'brick_tree', rtree.index.Index(properties=p))
