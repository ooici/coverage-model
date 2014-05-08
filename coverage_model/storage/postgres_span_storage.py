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
        sql_str = "BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;  %s %s %s COMMIT;" % (self.get_span_insert_sql(span), stats_sql, bin_sql)
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(sql_str, [SpanJsonDumper(span)])

    def get_span_insert_sql(self, span):
        sql_str = """INSERT INTO  %s (id, coverage_id, ingest_time, data) VALUES ('%s', '%s', %f, %s);""" % \
                  (PostgresSpanStorage.span_table_name, span.id, span.coverage_id, span.ingest_time, '%s')

        return sql_str

    def get_span_stats_and_bin_insert_sql(self, span):
        time_db_key = self.config.get_time_key(span.param_dict.keys())
        lat_db_key = self.config.get_lat_key(span.param_dict.keys())
        lon_db_key = self.config.get_lon_key(span.param_dict.keys())
        vertical_db_key = self.config.get_vertical_key(span.param_dict.keys())
        span_stats = span.get_span_stats(params=[time_db_key, lat_db_key, lon_db_key, vertical_db_key]).params

        stats_sql = """INSERT INTO %s (span_address, coverage_id, hash) VALUES ('%s', '%s', '%s'); """ % \
                    (self.span_stats_table_name, span.id, span.coverage_id, span.get_hash())

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
            vertical_sql = """UPDATE %s SET vertical_range='[%f, %f)' WHERE span_address='%s'; """ % (self.span_stats_table_name, stats[0], stats[1], span.id)
            stats_sql = ''.join([stats_sql, vertical_sql])
        bin_sql = ""
        return stats_sql, bin_sql

    def get_spans(self, span_ids=None, coverage_ids=None, start_time=None, stop_time=None, decompressors=None):
        statement = """SELECT data::text from %s where coverage_id = '%s'""" % (self.span_table_name, coverage_ids)
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(statement)
            results = cur.fetchall()

        spans = []
        for row in results:
            data, = row
            spans.append(Span.from_json(data, decompressors))

        return spans

    def get_stored_span_hash(self, span_id):
        statement = """SELECT hash from %s where span_address='%s'; """ % (self.span_stats_table_name, span_id)
        with self.span_store.pool.cursor(**self.span_store.cursor_args) as cur:
            cur.execute(statement)
            results = cur.fetchall()

        stored_hash = None
        for row in results:
            stored_hash, = row

        return stored_hash
