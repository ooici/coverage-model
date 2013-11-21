from ooi.logging import log
import os
from coverage_model.metadata import MetadataManager
from coverage_model import utils
from coverage_model.persistence_helpers import RTreeProxy, pack, unpack, get_coverage_type, RTreeItem
from coverage_model.basic_types import Dictable
from pycassa.cassandra.ttypes import NotFoundException
from pycassa import ConnectionPool, ColumnFamily


class CassandraMetadataManager(MetadataManager):
    columnFamilyName = "Entity"
    connectionPool = ConnectionPool('OOICI')

    @staticmethod
    def dirExists(directory):
        return True

    @staticmethod
    def isPersisted(directory, guid):
        cf = ColumnFamily(CassandraMetadataManager.connectionPool, CassandraMetadataManager.columnFamilyName)
        try:
            cf.get(guid, column_count=1)
            return True
        except NotFoundException:
            return False

    @staticmethod
    def getCoverageType(directory, guid):
        cf = ColumnFamily(CassandraMetadataManager.connectionPool, CassandraMetadataManager.columnFamilyName)
        try:
            col = cf.get(guid, columns=['coverage_type'])
            val = unpack(col['coverage_type'])
            return val
        except NotFoundException:
            return 'simplex'

    def __init__(self, filedir, guid, **kwargs):
        MetadataManager.__init__(self, **kwargs)
        self.guid = guid
        self.groups = {}
        self._ignore.update(['param_groups', 'guid', 'file_path', 'root_dir', 'brick_tree', 'groups'])
        self.param_groups = set()
        self.root_dir = os.path.join(filedir,guid)
        self.file_path = os.path.join(filedir, guid)

        self._load()
        for k, v in kwargs.iteritems():
            if hasattr(self, k) and v is None:
                continue

            setattr(self, k, v)

        if hasattr(self, 'parameter_bounds') and self.parameter_bounds is None:
            self.parameter_bounds = {}
        if not hasattr(self, 'parameter_bounds'):
            self.parameter_bounds = {}

    def __setattr__(self, key, value):
        super(CassandraMetadataManager, self).__setattr__(key, value)
        if not key in self._ignore and not key.startswith('_'):
            self._hmap[key] = utils.hash_any(value)
            self._dirty.add(key)
            super(CassandraMetadataManager, self).__setattr__('_is_dirty',True)

    def flush(self):
        MetadataManager.flush(self)
        if self.is_dirty(True):
            try:
                colFam = ColumnFamily(CassandraMetadataManager.connectionPool, CassandraMetadataManager.columnFamilyName)
                # put values in cassandra
                for k in list(self._dirty):
                    v = getattr(self, k)
#                    log.debug('FLUSH: key=%s  v=%s', k, v)
                    if isinstance(v, Dictable):
                        prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                        value = prefix + pack(v.dump())
                    else:
                        value = pack(v)

                    colFam.insert(self.guid, {k:value})

                    # Update the hash_value in _hmap
                    self._hmap[k] = utils.hash_any(v)
                    # Remove the key from the _dirty set
                    self._dirty.remove(k)
            except IOError, ex:
                if "unable to create file (File accessability: Unable to open file)" in ex.message:
                    log.info('Issue writing to hdf file during master_manager.flush - this is not likely a huge problem: %s', ex.message)
                else:
                    raise

            super(CassandraMetadataManager, self).__setattr__('_is_dirty',False)

    def _load(self):
        colFam = ColumnFamily(CassandraMetadataManager.connectionPool, CassandraMetadataManager.columnFamilyName)
        try:
            results = colFam.get(self.guid)
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
                else:
                    value = unpack(val)

                if isinstance(value, tuple):
                    value = list(value)

                setattr(self, key, value)

        except NotFoundException:
            pass

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
                chv = utils.hash_any(v)
                # log.trace('key=%s:  cached hash value=%s  current hash value=%s', k, self._hmap[k], chv)
                if self._hmap[k] != chv:
                    self._dirty.add(k)

            return len(self._dirty) != 0

    def update_rtree(self, count, extents, obj):
        if not hasattr(self, 'brick_tree'):
            raise AttributeError('Cannot update rtree; object does not have a \'brick_tree\' attribute!!')
# update cassandra
        self.brick_tree.insert(count, extents, obj=obj)

    def _init_rtree(self, bD):
            self.brick_tree = RTreeProxy()

    def add_external_link(self, link_path, rel_ext_path, link_name):
        group = link_path
        while group != "" and group != os.sep:
            group, tmp = os.path.split(group)

        self.groups[tmp][link_name] = os.path.abspath(rel_ext_path)


    def create_group(self, group_path):
        if group_path not in self.groups:
            self.groups[group_path] = {}
