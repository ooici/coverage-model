__author__ = 'casey'


from coverage_model.storage.postgres_span_storage import PostgresSpanStorage


class SpanStorageFactory(object):
    span_table = None
    default_span_storage_name = 'postgres_span_storage'
    storage_class_dict = {default_span_storage_name: PostgresSpanStorage}
    storage_object_dict = {}

    @classmethod
    def get_span_storage_obj(cls, storage_name=None):
        if storage_name is None:
            storage_name = cls.default_span_storage_name

        if storage_name not in cls.storage_class_dict:
            raise RuntimeError('Do not know how to construct storage for %', storage_name)

        if storage_name not in cls.storage_object_dict:
            cls.storage_object_dict[storage_name] = cls.storage_class_dict[storage_name]()

        return cls.storage_object_dict[storage_name]