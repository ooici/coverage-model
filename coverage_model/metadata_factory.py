#!/usr/bin/env python

from coverage_model.persistence_helpers import MasterManager, ParameterManager
from coverage_model.db_backed_metadata import DbBackedMetadataManager, ParameterContextWrapper


class MetadataManagerFactory(object):

    mmm = DbBackedMetadataManager
    pm = ParameterContextWrapper
#    mmm = MasterManager

    @staticmethod
    def buildMetadataManager(directory, guid, **kwargs):
        manager = MetadataManagerFactory.mmm(directory, guid, **kwargs)
        return manager

    @staticmethod
    def buildParameterManager(identifier, param_name, read_only=True, **kwargs):
        manager = MetadataManagerFactory.pm(identifier, param_name, read_only, **kwargs)
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
    def is_persisted(guid):
        return MetadataManagerFactory.mmm.is_persisted_in_db(guid)

    @staticmethod
    def dirExists(directory):
        return MetadataManagerFactory.mmm.dirExists(directory)