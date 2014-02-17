#!/usr/bin/env python

"""
@package coverage_model.coverage_search
@file coverage_model/coverage_search.py
@author Casey Bryant
@brief Interfaces and implementations for finding coverages that match criteria
"""

__author__ = 'casey'

from ooi.logging import log
from search_parameter import SearchCriteria
from coverage_model.db_connectors import DBFactory
from coverage_model.metadata_factory import MetadataManagerFactory


class CoverageSearch(object):

    @staticmethod
    def list(db_name=None, limit=None):
        db = DBFactory.get_db(db_name)
        ids = db.list(limit)
        return CoverageSearch._build_managers(ids)

    @staticmethod
    def select(search_params, db_name=None, limit=100):
        if not isinstance(search_params, SearchCriteria):
            raise ValueError('Search parameters must be of type ', SearchCriteria.__class__.__name__)
        db = DBFactory.get_db(db_name)
        return db.search(search_params, limit)

    @staticmethod
    def find(uid, db_name='postgres', limit=100):
        db = DBFactory.get_db(db_name)
        db.get(uid)
        return CoverageSearch._build_managers(uid)

    @staticmethod
    def _build_managers(ids):
        managers = []
        import collections
        if not isinstance(ids, collections.Iterable):
            ids = [ids]
        for uid in ids:
            managers.append(MetadataManagerFactory.buildMetadataManager("", uid))
        return managers