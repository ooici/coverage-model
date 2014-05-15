__author__ = 'casey'

from coverage_model.storage.span_storage import SpanStorage


class InstreamSpanStorage(SpanStorage):
    def write_span(self, span):
        raise NotImplementedError("Cannot write to an incoming stream")

    def get_spans(self, span_ids=None, coverage_ids=None, params=None, start_time=None, stop_time=None, decompressors=None):
        raise NotImplementedError("This should be implemented")

    def get_span_hash(self, span_id):
        raise NotImplementedError("Hash not calculated for incoming stream")
