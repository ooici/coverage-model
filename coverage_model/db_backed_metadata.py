#!/usr/bin/env python

"""
@package coverage_model.db_persistence
@file coverage_model/db_persistence.py
@author Casey Bryant
@brief The core interface for packaging and persisting metadata to a database
"""

from ooi.logging import log
import os
import msgpack
from coverage_model.basic_types import Dictable
from coverage_model.data_span import SpanCollectionByFile
from coverage_model.db_connectors import DBFactory
from coverage_model.metadata import MetadataManager
from coverage_model.config import CoverageConfig
from coverage_model.persistence_helpers import RTreeProxy, pack, unpack, MasterManager, BaseManager
from coverage_model.utils import hash_any


class DbBackedMetadataManager(MetadataManager):

    @staticmethod
    def dirExists(directory):
        return True

    @staticmethod
    def isPersisted(directory, guid):
        if DbBackedMetadataManager.is_persisted_in_db(guid) is False:
            if directory is not None:
                return MasterManager.isPersisted(directory, guid)
            else:
                return False
        return True

    @staticmethod
    def is_persisted_in_db(guid):
        return DBFactory.get_db().is_persisted(guid)

    @staticmethod
    def get_coverage_class(directory, guid):
        config = CoverageConfig()
        return config.get_coverage_class(DbBackedMetadataManager.getCoverageType(directory, guid))

    @staticmethod
    def getCoverageType(directory, guid):
        if DbBackedMetadataManager.is_persisted_in_db(guid) is True:
            cov_type = DBFactory.get_db().get_coverage_type(guid)
            if '' == cov_type:
                return ''
            else:
                cov_type = unpack(cov_type)
                return cov_type
        else:
            return BaseManager.getCoverageType(directory, guid)

    def __init__(self, filedir, guid, **kwargs):
        MetadataManager.__init__(self, **kwargs)
        self._ignore.update(['guid', 'file_path', 'root_dir'])
        self.guid = guid
        self.param_groups = set()
        self.root_dir = os.path.join(filedir,guid)
        fname = ''.join([guid, '_master.hdf5'])
        self.file_path = os.path.join(self.root_dir, fname)
        self.brick_tree = RTreeProxy()

        if self.is_persisted_in_db(guid):
            self._load()
        elif self.isPersisted(filedir, guid):
            mm = MasterManager(filedir, guid, **kwargs)
            for key, value in mm.__dict__.iteritems():
                if not key == "_dirty":
                    self.__setattr__(key, value)
            self.flush(deep=False)

        #    This is odd - You can load an object and override stored values on construction.
        #    This might lead to unexpected behavior for users.
        for k, v in kwargs.iteritems():
            if hasattr(self, k) and v is None:
                continue

            setattr(self, k, v)

        if (hasattr(self, 'parameter_bounds') and self.parameter_bounds is None) or not hasattr(self, 'parameter_bounds'):
            self.parameter_bounds = {}

    def __setattr__(self, key, value):
        super(DbBackedMetadataManager, self).__setattr__(key, value)
        if not key in self._ignore and not key.startswith('_'):
            self._hmap[key] = hash_any(value)
            self._dirty.add(key)
            super(DbBackedMetadataManager, self).__setattr__('_is_dirty',True)

    def flush(self, deep=True):
        if self.is_dirty(deep):
            try:
                # package for storage
                insert_dict = {}
                for k in list(self._dirty):
                    v = getattr(self, k)
                    log.trace('FLUSH: key=%s  v=%s', k, v)
                    if isinstance(v, Dictable):
                        prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                        value = prefix + pack(v.dump())
                    elif k == 'brick_tree':
                        if hasattr(self, 'brick_tree') and isinstance(self.brick_tree, RTreeProxy):
                            val = self.brick_tree.serialize()
                            if val != '':
                                insert_dict['brick_tree'] = val
                            continue
                    elif k == 'parameter_metadata':
                        value = pack_parameter_manager_dict(v)
                    else:
                        value = pack(v)

                    insert_dict[k] = value

                    # Update the hash_value in _hmap
                    self._hmap[k] = hash_any(v)

                dirty_spans = self.span_collection.get_dirty_spans()
                if len(dirty_spans) > 0:
                    val = str(self.span_collection)
                    log.trace("Span tuple: %s", val)
                    value = pack(val)
                    insert_dict['span_collection'] = value


                DBFactory.get_db().insert(self.guid, insert_dict, dirty_spans)

                for span in dirty_spans:
                    span.is_dirty = False
                self._dirty.clear()

            except IOError, ex:
                if "unable to create file (File accessability: Unable to open file)" in ex.message:
                    log.info('Issue writing to hdf file during master_manager.flush - this is not likely a huge problem: %s', ex.message)
                else:
                    raise

            super(DbBackedMetadataManager, self).__setattr__('_is_dirty',False)

    def _load(self):
        try:
            results = DBFactory.get_db().get(self.guid)
            for key in results:
                val = results[key]
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
                elif key == 'brick_tree':
                    setattr(self, key, RTreeProxy.deserialize(val))
                    continue
                elif key == 'span_collection':
                    unpacked = unpack(val)
                    value = SpanCollectionByFile.from_str(unpacked)
                    log.trace("Reconstructed SpanCollection for %s: %s", self.guid, str(value))
                elif key == 'parameter_metadata':
                    value = unpack_parameter_manager_dict(val)
                else:
                    value = unpack(val)

                if isinstance(value, tuple):
                    value = list(value)

                setattr(self, key, value)
                self._dirty.clear()
                super(DbBackedMetadataManager, self).__setattr__('_is_dirty',False)

        except Exception as e:
            log.error("Caught exception reconstructing metadata for guid %s : %s", self.guid, e.message)
            raise

    def _base_load(self, f):
        raise NotImplementedError('Not implemented by base class')

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
                chv = hash_any(v)
                # log.trace('key=%s:  cached hash value=%s  current hash value=%s', k, self._hmap[k], chv)
                if self._hmap[k] != chv:
                    self._dirty.add(k)

            return len(self._dirty) != 0

    def update_rtree(self, count, extents, obj):
        if not hasattr(self, 'brick_tree'):
            raise AttributeError('Cannot update rtree; object does not have a \'brick_tree\' attribute!!')
        self.brick_tree.insert(count, extents, obj=obj)
        self.brick_tree = self.brick_tree  # so the dirty bit gets set

    def _init_rtree(self, bD):
        pass

    def add_external_link(self, link_path, rel_ext_path, link_name):
        pass

    def create_group(self, group_path):
        if group_path not in self.param_groups:
            self.param_groups.add(group_path)
            self.param_groups = self.param_groups  # so the dirty bit gets set

    def add_span(self, span):
        self.span_collection.add_span(span)
        self._dirty.add('span_collection')

    #def __eq__(self, other):
    #    if self.param_groups == other.param_groups and self.guid == other.guid and \
    #        self.file_path == other.file_path and self.parameter_bounds == other.parameter_bounds and \
    #        self.type == other.type and self.root_dir == other.root_dir and \
    #        self.span_collection == other.span_collection and self.brick_tree == other.brick_tree:
    #        return True
    #    return False
    #
    #def __ne__(self, other):
    #    return not self.__eq__(other)


def pack_parameter_manager_dict(pm_dict):
    pack_dict = {}
    for k, v in pm_dict.iteritems():
        if not isinstance(v, ParameterContextWrapper):
            raise RuntimeError('Dictionary values must be type %s' % ParameterContextWrapper.__name__)
        pack_dict[k] = v.pack()

    return msgpack.packb(pack_dict)


def unpack_parameter_manager_dict(text):
    pm_dict = {}
    pack_dict = msgpack.unpackb(text)
    for k, v in pack_dict.iteritems():
        pm_dict[k] = ParameterContextWrapper.from_pack(v)

    return pm_dict


class ParameterContextWrapper(MetadataManager):
    ''' This class is meant to override the persistence behavior of ParameterManager.
        Instead, it provides packing and unpacking methods to allow other objects to persist it.
        It exists to minimize interface change ripple effects.  Consider changing the interface in the future
        if there is time.
    '''

    def __init__(self, guid, parameter_name, read_only=True, **kwargs):
        super(ParameterContextWrapper, self).__init__(**kwargs)
        for k, v in kwargs.iteritems():
            # Don't overwrite with None
            if hasattr(self, k) and v is None:
                continue

            setattr(self, k, v)

        self.guid = guid
        self.parameter_name = parameter_name
        self.read_only = read_only

        # Add attributes that should NEVER be flushed
        self._ignore.update(['read_only'])

    def pack(self):
        pack_dict = {}
        for k, v in self.__dict__.iteritems():
            if k in self._ignore or k.startswith('_'):
                continue
            if isinstance(v, Dictable):
                prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                v = prefix + pack(v.dump())
            pack_dict[k] = v

        return msgpack.packb(pack_dict)

    @staticmethod
    def from_pack(text):
        pack_dict = msgpack.unpackb(text)
        guid = pack_dict.pop('guid')
        parameter_name = pack_dict.pop('parameter_name')
        pm = ParameterContextWrapper(guid, parameter_name, read_only=True)
        for k, val in pack_dict.iteritems():
            if isinstance(val, basestring) and val.startswith('DICTABLE'):
                i = val.index('|', 9)
                smod, sclass = val[9:i].split(':')
                val = unpack(val[i+1:])
                module = __import__(smod, fromlist=[sclass])
                classobj = getattr(module, sclass)
                val = classobj._fromdict(val)
            pm.__setattr__(k,val)
        return pm

    def thin_origins(self, origins):
        pass

    def flush(self):
        if not self.read_only and self.is_dirty(True):
            self._dirty.clear()

    def _load(self):
        raise NotImplementedError('This object does not load itself')
