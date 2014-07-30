__author__ = 'casey'

import sys
import threading
from ooi.logging import log
from threading import Thread
from coverage_model.config import CoverageConfig
from coverage_model.coverage import AbstractCoverage, SimplexCoverage
from coverage_model.data_span import Span
from coverage_model.storage.span_storage_factory import SpanStorageFactory
from coverage_model.storage.span_storage import SpanStorage
from coverage_model.metadata_factory import MetadataManagerFactory


class StorageRespanTask(object):
    """
    Repackage data writes into contiguous memory packages for more optimal data access.
    Each individual data write is stored in it's own location to support asynchronous writes.
    Repackaging accounts for the possibility of out of observation chronological order writes as well as
    data write sizes that could hurt read performance (e.g. lots of small-ish writes).
    Repackaging attempts to merge many small 'files' into one optimally-sized 'file'.
    The current implementation does not split larger 'files' into multiple smaller files, though that
    functionality should be considered in the future.
    """

    def __init__(self, storage_name=None, coverage_ids=None, time_segment=None,
                 sort_parameter=None):
        store = SpanStorageFactory.get_span_storage_obj(storage_name)
        if not isinstance(store, SpanStorage):
            raise TypeError("Retrieved storage object must implement %s type.  Found %s." % (SpanStorage.__name__, self.store.__class__.__name__))
        else:
            self.store = store

        if coverage_ids is None:
            self.coverage_ids = set()
            coverage_ids = self.store.get_stored_coverage_ids()
            for cov_id in coverage_ids:
                if MetadataManagerFactory.is_persisted(cov_id):
                    self.coverage_ids.add(cov_id)

        elif isinstance(coverage_ids, (list,set)):
            self.coverage_ids = set(coverage_ids)
        elif isinstance(coverage_ids, basestring):
            self.coverage_ids = [coverage_ids]
        else:
            raise TypeError("Unhandled coverage_ids type - %s", type(coverage_ids))

        if time_segment is not None and not isinstance(time_segment, tuple) and len(time_segment) != 2:
            raise TypeError()
        self.time_segment = time_segment
        self.sort_parameter_name = sort_parameter

    def do_respan(self, asynchronous=False):
        if asynchronous:
            thread = Thread(target=self.do_respan, args=(False,))
            thread.start()
            return thread

        for id in self.coverage_ids:
            self.respan_coverage(id)

    def respan_coverage(self, cov_id):
        cov = AbstractCoverage.resurrect(cov_id, 'r')
        if not isinstance(cov, SimplexCoverage):
            return
        log.info('Respanning coverage %s' % cov_id)
        decompressors = cov._persistence_layer.value_list
        fill_value_dict = {}
        for k in decompressors:
            fill_value_dict[k] = cov.get_parameter_context(k).fill_value
            if fill_value_dict[k] is None:
                fill_value_dict[k] = -9999.0
        spans = self.store.get_spans(coverage_ids=cov_id, decompressors=decompressors)
        for span in spans:
            span.sort_parameter = cov.temporal_parameter_name
        starting_num_spans = len(spans)

        ideal_span_size = CoverageConfig().ideal_span_size
        span_sets = [[]]
        current_size = 0
        spans = sorted(spans)
        for span in spans:
            span_size = span.get_numpy_bytes()
            if (current_size > ideal_span_size and len(span_sets[-1]) > 0) or abs(ideal_span_size - current_size) < abs(ideal_span_size - span_size):
                span_sets.append([])
                current_size = 0
            span_sets[-1].append(span)
            current_size += span.get_numpy_bytes()

        new_span_ids = []
        for span_set in span_sets:
            new_span = Span.merge_spans(span_set, sort_param=self.sort_parameter_name, fill_value_dict=fill_value_dict)
            self.store.replace_spans(new_span, span_set)
            new_span_ids.append(new_span.id)

        log.info('Respaned coverage %s from %s spans to %s spans' % (cov_id, starting_num_spans, len(new_span_ids)))
        return new_span_ids