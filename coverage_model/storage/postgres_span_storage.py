__author__ = 'casey'

from ooi.logging import log
import psycopg2
from pyon.core.bootstrap import bootstrap_pyon, CFG
from pyon.datastore.datastore import DatastoreManager
from pyon.datastore.postgresql.base_store import PostgresDataStore
from coverage_model.data_span import Span, SpanStats
from coverage_model.config import CoverageConfig
from coverage_model.db_connectors import PostgresDB
from coverage_model.storage.span_storage import SpanStorage
from coverage_model.search.search_parameter import *
from coverage_model.db_connectors import PostgresDB


class SpanJsonDumper(psycopg2.extras.Json):
    def dumps(self, obj):
        return obj.as_json()


class PostgresSpanStorage(SpanStorage):
    span_table_name = 'coverage_span_data'
    span_stats_table_name = 'coverage_span_stats'
    bin_table_name = 'coverage_bin_stats'
    stats_store = None
    span_store = None
    bin_store = None

    def __init__(self):
        bootstrap_pyon()
        dsm = DatastoreManager()
        self.config = CoverageConfig()

        self.span_store = dsm.get_datastore(ds_name=PostgresSpanStorage.span_table_name)
        if self.span_store is None:
            raise RuntimeError("Unable to load datastore for %s" % PostgresSpanStorage.span_table_name)
        else:
            PostgresSpanStorage.span_table_name = self.span_store._get_datastore_name()
            log.trace("Got datastore: %s type %s" % (self.span_store._get_datastore_name(), str(type(self.span_store))))
        self.stats_store = dsm.get_datastore(ds_name=PostgresSpanStorage.span_stats_table_name)
        if self.stats_store is None:
            raise RuntimeError("Unable to load datastore for %s" % PostgresSpanStorage.span_stats_table_name)
        else:
            self.span_stats_table_name = self.stats_store._get_datastore_name()
            log.trace("Got datastore: %s type %s", self.stats_store._get_datastore_name(), type(self.stats_store))
        self.bin_store = dsm.get_datastore(ds_name=PostgresSpanStorage.bin_table_name)
        if self.bin_store is None:
            raise RuntimeError("Unable to load datastore for %s" % PostgresSpanStorage.bin_table_name)
        else:
            self.bin_table_name = self.stats_store._get_datastore_name()
            log.trace("Got datastore: %s type %s", self.bin_store._get_datastore_name(), type(self.bin_store))

    def write_span(self, span):
        stats_sql, bin_sql = self.get_span_stats_and_bin_insert_sql(span)
        span_sql, data_times = self.get_span_insert_sql(span)
        sql_str = "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;  %s %s %s COMMIT;" % (span_sql, stats_sql, bin_sql)
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(sql_str, [SpanJsonDumper(span) for i in range(data_times)])

    def get_span_insert_sql(self, span):
        data_times = 1
        if True:
            tbl = PostgresSpanStorage.span_table_name
            id = span.id
            cid = span.coverage_id
            it = span.ingest_time
            d = '%s'
            sql_str = """UPDATE %s SET coverage_id='%s', ingest_time=%f, data=%s WHERE id='%s';
                         INSERT INTO %s (id, coverage_id, ingest_time, data)
                          SELECT '%s', '%s', %f, %s
                          WHERE NOT EXISTS (SELECT 1 FROM %s WHERE id='%s');""" % \
                      (tbl, cid, it, d, id, \
                       tbl, \
                       id, cid, it, d, \
                       tbl, id)
            data_times = 2
        else:
            sql_str = """INSERT INTO  %s (id, coverage_id, ingest_time, data) VALUES ('%s', '%s', %f, %s);""" % \
                      (PostgresSpanStorage.span_table_name, span.id, span.coverage_id, span.ingest_time, '%s')

        return sql_str, data_times

    def get_span_stats_and_bin_insert_sql(self, span):
        time_db_key = self.config.get_time_key(span.param_dict.keys())
        lat_db_key = self.config.get_lat_key(span.param_dict.keys())
        lon_db_key = self.config.get_lon_key(span.param_dict.keys())
        vertical_db_key = self.config.get_vertical_key(span.param_dict.keys())
        span_stats = span.get_span_stats(params=[time_db_key, lat_db_key, lon_db_key, vertical_db_key]).params

        tbl = self.span_stats_table_name
        addr = span.id
        cid = span.coverage_id
        h = span.get_hash()
        stats_sql = """UPDATE %s SET coverage_id='%s', hash='%s' WHERE span_address='%s';
                     INSERT INTO %s (span_address, coverage_id, hash)
                      SELECT '%s', '%s', '%s'
                      WHERE NOT EXISTS (SELECT 1 FROM %s WHERE span_address='%s');""" % \
                  (tbl, cid, h, addr, \
                   tbl, \
                   addr, cid, h, \
                   tbl, addr)
        # stats_sql = """INSERT INTO %s (span_address, coverage_id, hash) VALUES ('%s', '%s', '%s'); """ % \
        #             (self.span_stats_table_name, span.id, span.coverage_id, span.get_hash())

        if time_db_key in span_stats:
            time_min = PostgresDB._get_time_string(span_stats[time_db_key][0])
            time_max = PostgresDB._get_time_string(span_stats[time_db_key][1])
            time_sql = """UPDATE %s SET time_range='[%s, %s)' WHERE span_address='%s'; """ % (self.span_stats_table_name, time_min, time_max, span.id )
            stats_sql = ''.join([stats_sql, time_sql])

        if lat_db_key in span_stats and lon_db_key in span_stats:
            lat_stats = span_stats[lat_db_key]
            lon_stats = span_stats[lon_db_key]
            geo_shape = PostgresDB.get_geo_shape(lon_stats[0], lon_stats[1], lat_stats[0], lat_stats[1])
            spatial_sql = """UPDATE %s SET spatial_geometry=%s WHERE span_address='%s'; """ % (self.span_stats_table_name, geo_shape, span.id)
            stats_sql = ''.join([stats_sql, spatial_sql])

        if vertical_db_key in span_stats:
            stats = span_stats[vertical_db_key]
            vertical_sql = """UPDATE %s SET vertical_range='[%s, %s)' WHERE span_address='%s'; """ % (self.span_stats_table_name, str(stats[0]), str(stats[1]), span.id)
            stats_sql = ''.join([stats_sql, vertical_sql])
        bin_sql = ""
        return stats_sql, bin_sql

    def get_spans(self, span_ids=None, coverage_ids=None, params=None, start_time=None, stop_time=None, decompressors=None):
        statement = """SELECT data::text from %s where coverage_id = '%s'""" % (self.span_table_name, coverage_ids)
        if span_ids is not None:
            from collections import Iterable
            if isinstance(span_ids, Iterable) and not isinstance(span_ids, basestring):
                span_ids = ','.join(span_ids)
            statement = ''.join([statement, """ AND id = ANY('{%s}')""" % (span_ids)])
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(statement)
            results = cur.fetchall()

        spans = []
        for row in results:
            data, = row
            spans.append(Span.from_json(data, decompressors))

        return spans

    def has_data(self, coverage_id):
        statement = """SELECT coverage_id FROM %s WHERE coverage_id = '%s'""" % (self.span_stats_table_name, coverage_id)
        results = []
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(statement)
            results = cur.fetchall()

        for row in results:
            cov_id, = row
            if coverage_id == cov_id:
                return True

        return False

    def get_stored_span_hash(self, span_id):
        statement = """SELECT hash from %s where span_address='%s'; """ % (self.span_stats_table_name, span_id)
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(statement)
            results = cur.fetchall()

        stored_hash = None
        for row in results:
            stored_hash, = row

        return stored_hash

    def search(self, search_criteria, limit=None):
        from coverage_model.search.coverage_search import SearchCriteria
        from coverage_model.search.search_constants import IndexedParameters, AllowedSearchParameters, MinimumOneParameterFrom
        from coverage_model.config import CoverageConfig
        config = CoverageConfig()
        if not isinstance(search_criteria, SearchCriteria):
            raise ValueError("".join(["search_criteria must be of type SearchCriteria. Found: ",
                                      str(type(search_criteria))]))
        if len(MinimumOneParameterFrom.intersection(search_criteria.criteria)) < 1:
            raise ValueError(''.join(['Search criteria must include a parameter from the minimum set: ',
                                      str(MinimumOneParameterFrom)]))

        if not set(search_criteria.criteria.keys()).issubset(AllowedSearchParameters):
            raise ValueError(''.join(['Search criteria can only include values from the allowed parameter set: ',
                                      str(AllowedSearchParameters)]))


        statement = ''.join(['SELECT ', config.span_coverage_id_db_key, ', ', config.span_id_db_key, ' FROM ', self.span_stats_table_name, ' WHERE'])
        and_str = ' and'
        lat_lon_handled = False
        for param in search_criteria.criteria.values():
            if param.param_name == IndexedParameters.Time:
                if isinstance(param, ParamValue):
                    time_min = param.value
                    time_max = param.value
                elif isinstance(param, ParamValueRange):
                    time_min = param.value[0]
                    time_max = param.value[1]
                else:
                    raise TypeError(''.join(['Parameter type, ', str(type(param)),
                                             " doesn't make sense for parameter name - ", param.param_name]))

                time_min_str = PostgresDB._get_time_string(time_min)
                time_max_str = PostgresDB._get_time_string(time_max)
                statement += ''.join([' ', config.time_db_key, ' && ', "'[", time_min_str, ", ", time_max_str, ")'::tsrange", and_str])

            if param.param_name == IndexedParameters.Latitude \
                or param.param_name == IndexedParameters.Longitude \
                or param.param_name == IndexedParameters.GeoBox:
                if param.param_name != IndexedParameters.GeoBox:
                    if lat_lon_handled is True:
                        continue
                    lat_min, lat_max, lon_min, lon_max = PostgresDB._get_lat_long_extents(search_criteria.criteria)
                    lat_lon_handled = True
                else:
                    if isinstance(param, Param2DValueRange):
                        lat_min = param.value[0][0]
                        lat_max = param.value[0][1]
                        lon_min = param.value[1][0]
                        lon_max = param.value[1][1]
                    else:
                        raise TypeError(''.join(['Parameter type, ', str(type(param)),
                                                 " doesn't make sense for parameter name - ", param.param_name]))

                statement += ''.join([' ST_intersects(', config.geo_db_key, ', ',
                                      PostgresDB.get_geo_shape(lon_min, lon_max, lat_min, lat_max), '::geometry)', and_str])

            if param.param_name == IndexedParameters.Vertical:
                if isinstance(param, ParamValue):
                    vertical_min = str(param.value)
                    vertical_max = str(param.value)
                elif isinstance(param, ParamValueRange):
                    vertical_min = str(param.value[0])
                    vertical_max = str(param.value[1])
                else:
                    raise TypeError(''.join(['Parameter type, ', str(type(param)),
                                             " doesn't make sense for parameter - ", param.param_name]))
                statement += ''.join([' ', config.vertical_db_key, ' && ', "'[", vertical_min, ", ", vertical_max, ")'::numrange", and_str])

            if param.param_name == IndexedParameters.CoverageId:
                if isinstance(param, ParamValue) and isinstance(param.value, basestring):
                    statement += ''.join([' ', config.span_coverage_id_db_key, " = '", param.value, "'", and_str])
                else:
                    raise TypeError(''.join(['Parameter type, ', str(type(param)), ', or value type, ',
                                             str(type(param.value)), ', is invalid for parameter ',
                                             param.param_name]))

        statement = statement.rstrip(and_str)
        if len(statement) > 0:
            found_rows = {}
            results = {}
            with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
                cur.execute(statement)
                #results_cursor = ResultsCursor(cur)
                #if limit is None:
                #    found_rows = cur.fetchmany()
                #elif limit == -1:
                #    log.trace("Fetching all")
                #    found_rows = cur.fetchall()
                #else:
                #    found_rows = cur.fetchmany(limit)
                found_rows = cur.fetchall()

            for row in found_rows:
                coverage_id, span_address = row
                if coverage_id not in results.keys():
                    results[coverage_id] = []
                log.trace("Adding row to results")
                results[coverage_id].append(span_address)
            return results


