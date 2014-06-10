#!/usr/bin/env python

from coverage_model.persistence_helpers import MasterManager
from coverage_model.db_backed_metadata import DbBackedMetadataManager


class MetadataManagerFactory(object):

    mmm = DbBackedMetadataManager
#    mmm = MasterManager

    @staticmethod
    def buildMetadataManager(directory, guid, **kwargs):
        manager = MetadataManagerFactory.mmm(directory, guid, **kwargs)
        return manager

    @staticmethod
    def get_coverage_class(directory, guid):
        return MetadataManagerFactory.mmm.get_coverage_class(directory, guid)

    @staticmethod
    def getCoverageType(directory, guid):
        return MetadataManagerFactory.mmm.getCoverageType(directory, guid)

    @staticmethod
    def isPersisted(directory, guid):
        return MetadataManagerFactory.mmm.isPersisted(directory, guid)

    @staticmethod
    def dirExists(directory):
        return MetadataManagerFactory.mmm.dirExists(directory)