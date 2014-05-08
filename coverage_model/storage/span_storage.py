__author__ = 'casey'


class SpanStorage(object):
    def write_span(self, span):
        raise NotImplementedError()

    def get_spans(self, span_ids=None, coverage_ids=None, start_time=None, stop_time=None, decompressors=None):
        raise NotImplementedError()

    def get_span_hash(self, span_id):
        raise NotImplementedError()
