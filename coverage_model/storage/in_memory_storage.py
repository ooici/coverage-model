__author__ = 'casey'

from coverage_model.storage.span_storage import SpanStorage


class InMemoryStorage(SpanStorage):
    def __init__(self):
        self.coverage_dict = {}

    def write_span(self, span):
        if span.coverage_id not in self.coverage_dict:
            self.coverage_dict[span.coverage_id] = []
        self.coverage_dict[span.coverage_id].append(span)

    def get_spans(self, span_ids=None, coverage_ids=None, params=None, start_time=None, stop_time=None, decompressors=None):
        if coverage_ids is None:
            coverage_ids = self.coverage_dict.keys()
        elif isinstance(coverage_ids, basestring):
            coverage_ids = [coverage_ids]

        return_spans = []
        for coverage_id in coverage_ids:
            if coverage_id not in self.coverage_dict:
                raise KeyError('%s is not a valid coverage' % coverage_id)
            for span in self.coverage_dict[coverage_id]:
                if span_ids is None or span.id in span_ids:
                    return_spans.append(span)
        return return_spans

    def get_span_hash(self, span_id):
        raise NotImplementedError("Hash not calculated for incoming stream")
