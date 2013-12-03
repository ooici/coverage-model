__author__ = 'casey'

from pycassa.cassandra.ttypes import NotFoundException
from pycassa import ConnectionPool, ColumnFamily
from ooi.logging import log


class DB(object):
    def insert(self, uuid, data):
        raise NotImplementedError('Not implemented by base class')

    def get(self, uuid):
        raise NotImplementedError('Not implemented by base class')


class CassandraDB(DB):
    pool = ConnectionPool('OOICI')
    col_fam = ColumnFamily(pool, 'Entity')

    def __init__(self):
        pass
#        self.pool = ConnectionPool('OOICI')
#        self.col_fam = ColumnFamily(self.pool, 'Entity')

    def is_persisted(self, guid):
        try:
            self.col_fam.get(guid, column_count=1)
            return True
        except NotFoundException:
            pass
        return False

    def get_coverage_type(self, guid):
        try:
            col = self.col_fam.get(guid, columns=['coverage_type'])
            return col['coverage_type']
        except NotFoundException:
            return ''



    def insert(self, uuid, data):
        try:
            if type(data) is dict:
                self.col_fam.insert(uuid, data)
                return True
        except Exception as e:
            log.warning('Exception writing to db:', e)
            pass
        return False

    def get(self, uuid):
        try:
            results = self.col_fam.get(uuid)
            return results
        except NotFoundException:
            return {}


class DBFactory(object):

    db = CassandraDB()

    @staticmethod
    def getDB(type=None):
        return DBFactory.db


