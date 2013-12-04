__author__ = 'casey'

from pycassa.cassandra.ttypes import NotFoundException
from pycassa import ConnectionPool, ColumnFamily
from ooi.logging import log
import psycopg2
import psycopg2.extras


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
            log.warning('Exception writing to cassandra db:', e)
            pass
        return False

    def get(self, uuid):
        try:
            results = self.col_fam.get(uuid)
            return results
        except NotFoundException:
            return {}


class PostgresDB(DB):
    tableName = "Entity"
    con = psycopg2.connect(database='casey', user='casey')
    column_names = []
    encoding = 'hex'

    def __init__(self):
        usage = "Using Postgres db. database=casey user=casey table=" + self.tableName
        log.debug(usage)
        pass

    def is_persisted(self, uuid):
        try:
            cur = self.con.cursor()
            cur.execute("""SELECT 1 from Entity where id=%(uuid)s""", {'uuid': uuid})
            if 0 < cur.rowcount:
                return True
        except Exception as e:
            log.warning('Caught exception checking Postgres existence: ', e)
            return False
        finally:
            if cur is not None and not cur.closed:
                cur.close()
        return False

    def get_coverage_type(self, uuid):
        try:
            cur = self.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
            cur.execute("""SELECT coverage_type from Entity where id=%s""", (uuid,))
            row = cur.fetchone()
            val = row['coverage_type']
            val = str.decode(val, self.encoding)
            return val
        except Exception as e:
            log.warning('Caught exception extracting coverage type from Postgres: ', e)
            return ''

    def insert(self, uuid, data, exists=False):
        rv = False
        if type(data) is not dict:
            return False

        if exists is False:
            if self.is_persisted(uuid):
                exists = True
        try:
            cur = self.con.cursor()
            for key in data:
                val = bytes.encode(data[key], self.encoding)
                if not exists:
                    statement = """INSERT into """ + self.tableName + """ (id, """ + key + """) VALUES(%(guid)s, %(val)s)"""
                    cur.execute(statement, {'guid': uuid, 'val': val})
                    self.con.commit()
                    exists = True
                else:
                    statement = """UPDATE """ + self.tableName + " SET " + key + """=%(val)s WHERE id=%(guid)s"""
                    cur.execute(statement, {'val': val, 'guid': uuid})

            self.con.commit()
            cur.close()
            rv = True
        except Exception, ex:
            log.warning('Caught exception writing to Postgres: ', ex.message)
            raise
        finally:
            if cur is not None and not cur.closed:
                cur.close()
            return rv

    def get(self, uuid):
        results = {}
        try:
            cur = self.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
            statement = """SELECT * from """ + self.tableName + """ WHERE id=%(uuid)s"""
            cur.execute(statement, {'uuid': uuid})
            row = cur.fetchone()
            if row is not None:
                for key in row.keys():
                    if key == 'id':
                        continue
                    val = row[key]
                    if val is not None:
                        val = str.decode(val, self.encoding)
                        if val is not None:
                            results[key] = val
            return results
        except Exception as e:
            log.warning('Caught exception loading id ' + uuid + ' from Postgres table ' + self.tableName)
            raise
        finally:
            if cur is not None and not cur.closed:
                cur.close()
        return results


class DBFactory(object):

#    default = CassandraDB()
    default = PostgresDB()

    @staticmethod
    def getDB(type=None):
        """
        if type is not None and isinstance(type, str):
            if type.lower() == 'cassandra':
                return DBFactory.cassandra
            elif type.lower() == 'postgres':
                return DBFactory.postgres
        """

        return DBFactory.default


