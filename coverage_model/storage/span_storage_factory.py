__author__ = 'casey'


from coverage_model.storage.postgres_span_storage import PostgresSpanStorage


class SpanTablesFactory(object):
    span_table = None

    @classmethod
    def get_span_table_obj(cls):
        if cls.span_table is None:
            cls.span_table = PostgresSpanStorage()
        return cls.span_table