#!/usr/bin/env python

"""
@package coverage_model.coverage_search
@file coverage_model/coverage_search.py
@author Casey Bryant
@brief Interfaces and implementations for finding coverages that match criteria
"""

__author__ = 'casey'

from ooi.logging import log
import numpy as np
from search_parameter import SearchCriteria
from coverage_model.db_connectors import DBFactory
from coverage_model.metadata_factory import MetadataManagerFactory
from coverage_model.search.search_constants import *
from coverage_model.search.search_parameter import *
from coverage_model.coverage import AbstractCoverage
from coverage_model.data_span import *
from coverage_model.storage.span_storage import SpanStorage
from coverage_model.storage.span_storage_factory import SpanStorageFactory


class CoverageSearch(object):

    def __init__(self, search_criteria, order_by=None, coverage_id=None, viewable_parameters=None):
        if not isinstance(search_criteria, SearchCriteria):
            raise ValueError('Search parameters must be of type ', SearchCriteria.__class__.__name__)
        self.coverage_id = None
        if coverage_id is not None:
            search_criteria.append(ParamValue('coverage_id', coverage_id))
            self.coverage_id = coverage_id
        self.search_criteria = search_criteria
        self.order_by = order_by
        self.viewable_parameters = viewable_parameters

    def select(self, db_name=None, limit=-1):
        db = SpanStorageFactory.get_span_storage_obj(db_name)
        # db = DBFactory.get_db(db_name)
        span_dict = db.search(self.search_criteria, limit)
        return CoverageSearchResults(span_dict, self.search_criteria, viewable_parameters=self.viewable_parameters,
                                     order_by=self.order_by)

    @staticmethod
    def find(coverage_id, persistence_dir, db_name=None, limit=-1):
        db = DBFactory.get_db(db_name)
        rows = db.get(coverage_id)
        if len(rows) > 1:
            return AbstractCoverage.load(persistence_dir, persistence_guid=coverage_id, mode='r')
        return None


# class ResultsCursor(object):
#     def __init__(self, cursor):
#         self.cursor = cursor
#
#     def get_results(self, number=None):
#         found_rows = {}
#         if number is None:
#             found_rows = self.cursor.fetchall()
#         elif number == 1:
#             found_rows = self.cursor.fetchone()
#         else:
#             found_rows = self.cursor.fetchmany(number)
#
#         for row in found_rows:
#             coverage_id, span_address = row
#             if coverage_id not in results.keys():
#                 results[coverage_id] = []
#             results[coverage_id].append(span_address)
#         return results


class CoverageSearchResults(object):

    def __init__(self, coverage_dict, search_criteria, order_by=None, viewable_parameters=None):
        self.span_dict = coverage_dict
        self.search_criteria = search_criteria
        self.order_by = order_by
        self.viewable_parameters = viewable_parameters

    def get_view_coverage(self, coverage_id, working_dir):
        if coverage_id in self.span_dict:
            return self._build_view_coverage(coverage_id, self.span_dict[coverage_id], working_dir)
        return None

    def get_view_coverages(self):
        coverages = {}
        for cov_id, spans in self.span_dict:
            coverages[cov_id] = self._build_view_coverage(cov_id, spans)

    def _build_view_coverage(self, coverage_id, spans, working_dir):
        return SearchCoverage(coverage_id, base_dir=working_dir, spans=spans, view_criteria=self.search_criteria,
                            order_by=self.order_by, viewable_parameters=self.viewable_parameters)

    def get_found_coverage_ids(self):
        return self.span_dict.keys()


class SearchCoverage(object):

    def __init__(self, coverage_id, base_dir, spans=None, view_criteria=None, viewable_parameters=None, order_by=None):
        from coverage_model.coverage import AbstractCoverage
        # wrap a read only abstract coverage so we can access values in a common method
        self._cov = AbstractCoverage.load(base_dir, persistence_guid=coverage_id, mode='r')
        self.spans = spans
        self.view_criteria = view_criteria
        self.viewable_parameters = []
        self.order_by = []
        self.np_array_dict = {}
        if viewable_parameters is not None:
            if isinstance(viewable_parameters, basestring):
                viewable_parameters = [viewable_parameters]
            if not isinstance(viewable_parameters, collections.Iterable):
                raise TypeError(''.join(['Unable to create view for view_parameter type: ', str(type(viewable_parameters))]))
            for val in viewable_parameters:
                if isinstance(val, basestring):
                    self.viewable_parameters.append(val)
                else:
                    raise TypeError(''.join(['Unable to create view for view_parameter member type: ', str(type(val)),
                                             ' from view_parameters: ', str(viewable_parameters)]))
        if order_by is not None:
            if isinstance(order_by, basestring):
                order_by = [order_by]
            if not isinstance(order_by, collections.Iterable):
                raise TypeError(''.join(['Unable to order by type: ', str(type(order_by))]))
            for val in order_by:
                if isinstance(val, basestring):
                    self.order_by.append(val)
                else:
                    raise TypeError(''.join(['Unable to order by order_by member type: ', str(type(val)),
                                    ' from order_by: ', str(order_by)]))

        self._extract_parameter_data()

    def _extract_parameter_data(self):
        observation_list = []
        for span_address in self.spans:
            span_address = AddressFactory.from_db_str(span_address)
            span = self._cov._persistence_layer.master_manager.span_collection.get_span(span_address)
            intersection = None
            span_np_dict = {}
            for param_name in span.params.keys():
                val = self._cov._range_value[param_name]._storage
                span_np_dict[param_name] = val.get_brick_slice(span_address.brick_id)

                for param in self.view_criteria.criteria.values():
                    if param.param_name in span.params and param.param_name == param_name:
                        indexes = np.argwhere( (span_np_dict[param_name]>=param.value[0]) &
                                               (span_np_dict[param_name]<=param.value[1]) )
                        if len(indexes.shape) > 1:
                            indexes = indexes.ravel()
                        if intersection is None:
                            intersection = indexes
                        else:
                            intersection = np.intersect1d(intersection, indexes)
            for param_name, np_array in span_np_dict.iteritems():
                if param_name in self.np_array_dict:
                    self.np_array_dict[param_name] = np.append(self.np_array_dict[param_name], np_array[intersection])
                else:
                    self.np_array_dict[param_name] = np_array[intersection]

        dtype = []
        npas = []
        self.data_size = None
        for key, val in self.np_array_dict.iteritems():
            if self.data_size is None:
                self.data_size = len(val)
            elif len(val) != self.data_size:
                log.warn("Parameter arrays aren't consistent size results may be meaningless")
            if len(val) < self.data_size:
                self.data_size = len(val)

    #This is a convenience method to allow a view coverage to be built for a known coverage id
    #It creates a bi-directional weird dependency with CoverageSearchResults since it uses
    # CoverageSearchResults to create the ViewCoverage
    @classmethod
    def from_search_criteria(cls, coverage_id, search_criteria):
        search_results = CoverageSearchResults.search_within_one_coverage(coverage_id, search_criteria)
        if coverage_id in search_results.get_found_coverage_ids():
            return search_results.get_view_coverage(coverage_id)
        return None

    def get_num_value_elements(self):
        len(self.result)

    def get_observations(self, start_index=None, end_index=None, order_by=None):
        dtype = []
        npas = []
        result = {}
        for key, val in self.np_array_dict.iteritems():
            dtype.append( (key, val.dtype.type) )
            npas.append(val)
        result = np.asarray(zip(*npas), dtype=dtype)

        if self.order_by is not None:
            for order in self.order_by:
                if order not in self.np_array_dict.keys():
                    log.warn("Unable to order by parameter %s. Parameter is not present in results.", order)
                    self.order_by.remove(order)
            result.sort(order=self.order_by)

        if start_index is None and end_index is None:
            return result
        else:
            if start_index is None:
                start_index = 0
            if end_index is None or end_index > self.data_size:
                end_index = self.data_size
            return result[start_index:end_index]

    def get_value_dictionary(self, param_list=None, start_index=None, end_index=None):
        self._has_param(param_list)
        if param_list is None:
            param_list = self.np_array_dict.keys()
        result_dict = {}
        if start_index is None and end_index is None:
            for param in param_list:
                result_dict[param] = self.np_array_dict[param]
        else:
            if start_index is None:
                start_index = 0
            if end_index is None or start_index + end_index > self.data_size:
                end_index = self.data_size
            for param in param_list:
                result_dict[param] = self.np_array_dict[param][start_index:end_index]

        return result_dict

    def _has_param(self, params):
        for param in params:
            if param not in self.np_array_dict:
                raise ValueError("Coverage, %s, does not have parameter - %s", self._cov.persistence_guid, param)
