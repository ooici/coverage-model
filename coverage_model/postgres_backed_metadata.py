from ooi.logging import log
import os
from coverage_model.metadata import MetadataManager
from coverage_model import utils
from coverage_model.persistence_helpers import RTreeProxy, pack, unpack
from coverage_model.basic_types import Dictable
import psycopg2
import psycopg2.extras


class PostgresMetadataManager(MetadataManager):
    tableName = "Entity"
    database = 'casey'
    user = 'casey'

    con = psycopg2.connect(database='casey', user='casey')
    @staticmethod
    def dirExists(directory):
        return True

    @staticmethod
    def isPersisted(directory, guid):
        con = PostgresMetadataManager.con
        try:
            cur = con.cursor()
            cur.execute("""SELECT 1 from Entity where id=%(guid)s""", {'guid': guid})
            if 0 < cur.rowcount:
                return True
        except Exception as e:
            log.warn('Caught exception %s', e.message)
            return False
        return False

    @staticmethod
    def getCoverageType(directory, guid):
        con = PostgresMetadataManager.con
        try:
            cur = PostgresMetadataManager.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("""SELECT coverage_type from Entity where id=%s""", (guid,))
            row = cur.fetchone()
            val = row['coverage_type']
            val = str.decode(val, 'hex')
            val = unpack(val)
            return val
        except Exception as e:
            log.warn('Caught exception %s', e.message)
            return ''

    def __init__(self, filedir, guid, **kwargs):
        MetadataManager.__init__(self, **kwargs)
        self.guid = guid
        self._ignore.update(['param_groups', 'guid', 'file_path', 'root_dir', 'brick_tree', 'groups'])
        self.param_groups = set()
        self.root_dir = os.path.join(filedir,guid)
        self.file_path = os.path.join(filedir, guid)
        self.brick_tree = RTreeProxy()

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
        super(PostgresMetadataManager, self).__setattr__(key, value)
        if not key in self._ignore and not key.startswith('_'):
            self._hmap[key] = utils.hash_any(value)
            self._dirty.add(key)
            super(PostgresMetadataManager, self).__setattr__('_is_dirty',True)

    def flush(self):
        if self.is_dirty(True):
            con = PostgresMetadataManager.con
            try:
                cur = con.cursor()
                exists = False
                if self.isPersisted("", self.guid):
                    exists = True
                for k in list(self._dirty):
                    v = getattr(self, k)
                    if v is None:
                        continue
#                    log.debug('FLUSH: key=%s  v=%s', k, v)
                    if isinstance(v, Dictable):
                        prefix='DICTABLE|{0}:{1}|'.format(v.__module__, v.__class__.__name__)
                        value = prefix + pack(v.dump())
                    else:
                        value = pack(v)

                    if exists:
                        statement = """UPDATE """ + PostgresMetadataManager.tableName + """ SET """ + k + """=%(val)s WHERE id=%(guid)s"""
                        cur.execute(statement, {'val': bytes.encode(value, 'hex'), 'guid': self.guid})
                    else:
                        encoded = value.encode('hex')
                        statement = """INSERT into """ + PostgresMetadataManager.tableName + """ (id, """ + k + """) VALUES(%(guid)s, %(val)s)"""
                        cur.execute(statement, {'guid': self.guid, 'val': encoded})
                        con.commit()
                        exists = self.isPersisted("", self.guid)
                    #colFam.insert(self.guid, {k:value})

                    # Update the hash_value in _hmap
                    self._hmap[k] = utils.hash_any(v)
                    # Remove the key from the _dirty set
                    self._dirty.remove(k)

                if hasattr(self, 'brick_tree') and isinstance(self.brick_tree, RTreeProxy):
                    val = self.brick_tree.serialize()
                    if val != '' and val is not None:
                        if exists:
                            statement = """UPDATE """ + PostgresMetadataManager.tableName + """ SET brick_tree=%(val)s WHERE id=%(guid)s"""
                            cur.execute(statement, {'val': bytes.encode(val, 'hex'), 'guid': self.guid})
                        else:
                            statement = """INSERT into """ + PostgresMetadataManager.tableName + """ (id, brick_tree) VALUES(%(guid)s, %(val)s)"""
                            cur.execute(statement, {'guid': self.guid, 'val': bytes.encode(val, 'hex')})
                            con.commit()
                            exists = self.isPersisted("", self.guid)

                if hasattr(self, 'param_groups') and isinstance(self.param_groups, set):
                    if isinstance(self.param_groups, set):
                        groups = ''
                        for group in self.param_groups:
                            groups = '%(groups)s::group::%(group)s' % {'groups': groups, 'group': group}
                        if exists:
                            statement = """UPDATE """ + PostgresMetadataManager.tableName + """ SET param_groups=%(val)s WHERE id=%(guid)s"""
                            cur.execute(statement, {'guid': self.guid, 'val': bytes.encode(groups, 'hex')})
                        else:
                            statement = """INSERT into """ + PostgresMetadataManager.tableName + """ (id, param_groups) VALUES(%(guid)s, %(val)s)"""
                            cur.execute(statement, {'guid': self.guid, 'val': bytes.encode(groups, 'hex')})
                            con.commit()
                            exists = self.isPersisted("", self.guid)
                if con is not None:
                    con.commit()
            except Exception, ex:
                if "unable to create file (File accessability: Unable to open file)" in ex.message:
                    log.info('Issue writing to hdf file during master_manager.flush - this is not likely a huge problem: %s', ex.message)
                else:
                    log.warn('Caught exception during flush %s', ex.message)
                    raise

            super(PostgresMetadataManager, self).__setattr__('_is_dirty',False)

    def _load(self):
        con = PostgresMetadataManager.con
        try:
            con = psycopg2.connect(database=PostgresMetadataManager.database, user=PostgresMetadataManager.user)
            cur = con.cursor(cursor_factory=psycopg2.extras.DictCursor)
#            cur.execute("""SELECT column_name from information_schema.columns where table_name='entity'""")
#            cols = cur.fetchall()
            statement = """SELECT * from """ + PostgresMetadataManager.tableName + """ WHERE id=%(guid)s"""
            cur.execute(statement, {'guid':self.guid})
            row = cur.fetchone()
            if row is not None:
                for key in row.keys():
#                for key in cols:
#                    key = key[0]
                    if key == 'id':
                        continue
                    val = row[key]
                    if val is not None:
                        val = str.decode(val, 'hex')
                        if val is None:
                            continue
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
                        elif key == 'param_groups':
                            self.param_groups.clear()
                            for group in val.split('::group::'):
                                if group != '':
                                    self.param_groups.add(group)
                            continue
                        else:
                            value = unpack(val)

                        if isinstance(value, tuple):
                            value = list(value)

                        setattr(self, key, value)

        except Exception, e:
            log.warn('Caught exception %s', e.message)
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
        self.brick_tree.insert(count, extents, obj=obj)

    def _init_rtree(self, bD):
        self.brick_tree = RTreeProxy()

    def add_external_link(self, link_path, rel_ext_path, link_name):
        pass

    def create_group(self, group_path):
        if group_path not in self.param_groups:
            self.param_groups.add(group_path)
