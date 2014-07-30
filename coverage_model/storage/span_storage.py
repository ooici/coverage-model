__author__ = 'casey'


class SpanStorage(object):
    def write_span(self, span):
        raise NotImplementedError('Not implemented in base class')

    def get_spans(self, span_ids=None, coverage_ids=None, params=None, start_time=None, stop_time=None, decompressors=None):
        raise NotImplementedError('Not implemented in base class')

    def get_span_hash(self, span_id):
        raise NotImplementedError('Not implemented in base class')

    def search(self, search_criteria, limit=None):
        raise NotImplementedError('Not implemented in base class')

    def has_data(self, coverage_id):
        raise NotImplementedError('Not implemented in base class')

    def get_stored_coverage_ids(self):
        raise NotImplementedError('Not implemented in base class')

    def replace_spans(self, new_spans, old_spans):
        raise NotImplementedError('Not implemented in base class')
