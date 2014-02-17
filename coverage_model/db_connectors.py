__author__ = 'casey'

from ooi.logging import log

from pyon.core.bootstrap import bootstrap_pyon, CFG
from pyon.datastore.datastore_common import DatastoreFactory
from pyon.datastore.postgresql.base_store import PostgresDataStore
import datetime
import psycopg2
import psycopg2.extras
from coverage_model.data_span import ParamSpan
from coverage_model.search.search_parameter import ParamValueRange, ParamValue, Param2DValueRange


class DB(object):
    def insert(self, uuid, data):
        raise NotImplementedError('Not implemented by base class')

    def index(self, uuid, params):
        raise NotImplementedError('Not implemented by base class')

    def reindex(self, uuid, param_tuples):
        raise NotImplementedError('Not implemented by base class')

    def get(self, uuid):
        raise NotImplementedError('Not implemented by base class')

    def search(self, search_criteria):
        raise NotImplementedError('Not implemented by base class')

    def list(self, limit=100):
        raise NotImplementedError('Not implemented by base class')

    def _get_collection_time(self, params):
        pass


class CassandraDB(DB):

    def __init__(self):
#        self.pool = ConnectionPool('OOICI')
#        self.col_fam = ColumnFamily(self.pool, 'Entity')
        pass

    def is_persisted(self, guid):
#        try:
#            self.col_fam.get(guid, column_count=1)
#            return True
#        except NotFoundException:
#            pass
        return False

    def get_coverage_type(self, guid):
#        try:
#            col = self.col_fam.get(guid, columns=['coverage_type'])
#            return col['coverage_type']
#        except NotFoundException:
            return ''

#    def insert(self, uuid, data):
#        try:
#            if type(data) is dict:
#                self.col_fam.insert(uuid, data)
#                return True
#        except Exception as e:
#            log.warning('Exception writing to cassandra db:', e)
#            pass
#        return False

#    def get(self, uuid):
#        try:
#            results = self.col_fam.get(uuid)
#            return results
#        except NotFoundException:
#            return {}


class PostgresDB(DB):
    encoding = 'hex'

    def __init__(self):
        bootstrap_pyon()
        self.datastore = DatastoreFactory.get_datastore(datastore_name='coverage', config=CFG)
        if self.datastore is None:
            raise RuntimeError("Unable to load datastore for coverage")
        else:
            self.entity_table_name = self.datastore._get_datastore_name()
            log.trace("Got datastore: %s type %s" % (self.datastore._get_datastore_name(), type(self.datastore)))
        self.span_store = DatastoreFactory.get_datastore(datastore_name='coverage_spans', config=CFG)
        if self.span_store is None:
            raise RuntimeError("Unable to load datastore for coverage_spans")
        else:
            self.span_table_name = self.span_store._get_datastore_name()
            log.trace("Got datastore: %s type %s", self.span_store._get_datastore_name(), type(self.span_store))

    def is_persisted(self, uuid):
        try:
            with self.datastore.pool.cursor(**self.datastore.cursor_args) as cur:
                log.trace(cur.mogrify("""SELECT 1 from """ + self.entity_table_name + " where id=%(uuid)s""",
                                      {'uuid': uuid}))
                cur.execute("""SELECT 1 from """ + self.entity_table_name + " where id=%(uuid)s""", {'uuid': uuid})
                if 0 < cur.rowcount:
                    log.trace("Record exists: %s", uuid)
                    return True
        except Exception as e:
            log.warn('Caught exception checking Postgres existence: ', e)
            return False
        return False

    def get_coverage_type(self, uuid):
        try:
            #with self.datastore.pool.cursor(**self.datastore.cursor_args) as cur:
            with self.datastore.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            #cur = self.con.cursor(cursor_factory=psycopg2.extras.DictCursor)
                log.trace(cur.mogrify("""SELECT coverage_type from """ + self.entity_table_name +
                                      """ where id=%s""", (uuid,)))
                cur.execute("""SELECT coverage_type from """ + self.entity_table_name + """ where id=%s""", (uuid,))
                row = cur.fetchone()
                val = row['coverage_type']
                val = str.decode(val, self.encoding)
                return val
        except Exception as e:
            log.warn('Caught exception extracting coverage type from Postgres: ', e)
            return ''

    def insert(self, uuid, data, spans=[], exists=False):
        rv = False
        if type(data) is not dict:
            return False

        if exists is False:
            if self.is_persisted(uuid):
                exists = True
        try:
            self.insert_spans(uuid, spans, None)
            with self.datastore.pool.cursor(**self.datastore.cursor_args) as cur:
                for key in data:
                    val = bytes.encode(data[key], self.encoding)
                    if not exists:
                        statement = """INSERT into """ + self.entity_table_name + """ (id, """ + key + \
                            """) VALUES(%(guid)s, %(val)s)"""
                        log.trace(cur.mogrify(statement, {'guid': uuid, 'val': val}))
                        cur.execute(statement, {'guid': uuid, 'val': val})
                        exists = True
                    else:
                        statement = """UPDATE """ + self.entity_table_name + " SET " + key + \
                            """=%(val)s WHERE id=%(guid)s"""
                        log.trace(cur.mogrify(statement, {'val': val, 'guid': uuid}))
                        cur.execute(statement, {'val': val, 'guid': uuid})

            rv = True
        except Exception, ex:
            log.warn('Caught exception writing to Postgres: %s', ex.message)
            raise
        return rv

    def _span_exists(self, span_address):
        with self.span_store.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            cur.execute("SELECT 1 FROM %s WHERE span_address='%s'" % (self.span_table_name, span_address))
            if cur.rowcount > 0:
                return True
        return False

    def insert_spans(self, uuid, spans, cur):
        log.debug("Inserting spans")
        try:
            for span in spans:
                cols, values = self.span_values(uuid, span)
                dic = dict(zip(cols,values))
                if len(cols) > 0:
                    span_addr = span.address.get_db_str()
                    statement = ''
                    if self._span_exists(span_addr):
                        statement = ''.join(['UPDATE ', self.span_table_name, ' SET '])
                        for k, v in dic.iteritems():
                            statement = ''.join([statement, k, '=', v, ', '])
                        statement = statement.rstrip(', ')
                        statement = ''.join([statement, " WHERE span_address = '", span_addr, "'"])
                    else:
                        statement = """INSERT into """ + self.span_table_name + """ ("""
                        for col in cols:
                            statement = ''.join([statement, col, ', '])
                        statement = statement.rstrip(', ')
                        statement = ''.join([statement, """) VALUES ("""])
                        for val in values:
                            statement = ''.join([statement, val, ', '])
                        statement = statement.rstrip(', ')
                        statement = ''.join([statement, """)"""])

                    log.trace("Inserting span into datastore: %s", statement)
                    with self.span_store.pool.cursor(**self.datastore.cursor_args) as cur:
                        cur.execute(statement)
        except Exception as ex:
            log.warn('Unable to insert spans %s %s', str(spans), ex.message)

    @staticmethod
    def span_values(coverage_id, span):
        r_cols = ()
        r_vals = ()
        cols = ['coverage_id']
        values = [''.join(["'", str(coverage_id), "'"])]
        cols.append('span_address')
        values.append(''.join(["'", span.address.get_db_str(), "'"]))
        insert = False
        if 'lat' in span.params and 'lon' in span.params:
            tmp = []
            lats = span.params['lat']
            lons = span.params['lon']
            lat_min = str(lats[0])
            lat_max = str(lats[1])
            lon_min = str(lons[0])
            lon_max = str(lons[1])
            tmp = PostgresDB.get_geo_shape(lon_min, lon_max, lat_min, lat_max)
            if len(tmp) > 0:
                cols.append('spatial_geography')
                values.append(tmp)
                insert = True

        if 'time' in span.params:
            # Just taking the raw time as utc for now.  This may work or it may not.
            # Talked with Matthew Arrott about this.  He's thinking about it/looking into things
            time_min_str = datetime.datetime.utcfromtimestamp(span.params['time'][0]).strftime('%Y-%m-%d %H:%M:%S.%f z')
            time_max_str = datetime.datetime.utcfromtimestamp(span.params['time'][1]).strftime('%Y-%m-%d %H:%M:%S.%f z')
            value = "'[", str(time_min_str), ', ', str(time_max_str), ")'"
            cols.append('time_range')
            values.append(''.join(value))
            insert = True

        if insert:
            r_cols = tuple(cols)
            r_vals = tuple(values)
        return r_cols, r_vals

    @staticmethod
    def get_geo_shape(lon_min, lon_max, lat_min, lat_max):
        if lat_min == lat_max or lon_min == lon_max:
            if lat_min == lat_max and lon_min == lon_max:
                tmp = ["ST_GeographyFromText('POINT(", str(lon_min), ' ', str(lat_min), ")')"]
            else:
                tmp = ["ST_GeographyFromText('LINESTRING(", str(lon_min), ' ', str(lat_min), ',', str(lon_max), ' ', str(lat_max), ")')"]
        else:
            tmp = ["ST_GeographyFromText('POLYGON((", str(lon_min), ' ', str(lat_min), ',', str(lon_min), ' ', str(lat_max), ',', str(lon_max),
                   ' ', str(lat_max), ',', str(lon_max), ' ', str(lat_min), ',', str(lon_min), ' ', str(lat_min), "))')"]
        return ''.join(tmp)

    def get(self, uuid):
        results = {}
        try:
            with self.datastore.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                statement = """SELECT * from """ + self.entity_table_name + """ WHERE id=%(uuid)s"""
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
        except Exception as e:
            log.warn('Caught exception loading id %s from Postgres table %s' % (uuid,  self.entity_table_name) )
            raise
        return results

    def search(self, search_criteria, limit=None):
        from coverage_model.search.coverage_search import SearchCriteria
        if not isinstance(search_criteria, SearchCriteria):
            raise ValueError("".join(["search_criteria must be of type SearchCriteria. Found: ",
                                      str(type(search_criteria))]))
        statement = ''.join(["SELECT coverage_id, span_address FROM ", self.span_table_name, ' WHERE'])
        and_str = ' and'
        for param in search_criteria.criteria:
            if param.param_name == 'time':
                if isinstance(param.param_type, ParamValue):
                    time_min = param.value
                    time_max = param.value
                if isinstance(param.param_type, ParamValueRange):
                    time_min = param.value[0]
                    time_max = param.value[1]
                time_min_str = datetime.datetime.utcfromtimestamp(time_min).strftime('%Y-%m-%d %H:%M:%S.%f z')
                time_max_str = datetime.datetime.utcfromtimestamp(time_max).strftime('%Y-%m-%d %H:%M:%S.%f z')
                statement += ''.join([' time_range && ', "'[", time_min_str, ", ", time_max_str, ")'::tsrange", and_str])
            if param.param_name == 'geo_box':
                if isinstance(param.param_type, Param2DValueRange):
                    lat_min = param.value[0][0]
                    lat_max = param.value[0][1]
                    lon_min = param.value[1][0]
                    lon_max = param.value[1][1]
                    statement += ''.join([" ST_intersects(spatial_geography, ",
                                 self.get_geo_shape(lon_min, lon_max, lat_min, lat_max), "::geometry)", and_str])

        statement = statement.rstrip(and_str)
        if len(statement) > 0:
            rows = {}
            results = {}
            with self.datastore.pool.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
                cur.execute(statement)
                if limit is None:
                    rows = cur.fetchmany()
                elif limit == -1:
                    rows = cur.fetchall()
                else:
                    rows = cur.fetchmany(limit)
            for row in rows:
                if row['coverage_id'] not in results.keys():
                    results[row['coverage_id']] = []
                results[row['coverage_id']].append(row['span_address'])
            return results

    def list(self, limit=100):
        raise NotImplementedError('Not implemented by base class')


class DBFactory(object):
    db_dict = {}

    @staticmethod
    def get_db(db_type='postgres'):
        db = None
        if db_type is None:
            db_type = 'postgres'
        if isinstance(db_type, str):
            if db_type.lower() == 'cassandra':
                if 'cassandra' not in DBFactory.db_dict.keys():
                    DBFactory.db_dict['cassandra'] = CassandraDB()
                db = DBFactory.db_dict['cassandra']
            elif db_type.lower() == 'postgres':
                if 'postgres' not in DBFactory.db_dict:
                    DBFactory.db_dict['postgres'] = PostgresDB()
                db = DBFactory.db_dict['postgres']
            else:
                raise ValueError("No db of type: %s", db_type)
        else:
            raise TypeError("Invalid type, %s. Expected %s", type(db_type), type(str))

        return db


