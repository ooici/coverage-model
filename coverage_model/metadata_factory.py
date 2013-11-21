#!/usr/bin/env python

from coverage_model.persistence_helpers import MasterManager
from coverage_model.cassandra_backed_metadata import CassandraMetadataManager

class MetadataManagerFactory(object):

    mmm = CassandraMetadataManager
#    mmm = MasterManager

    @staticmethod
    def buildMetadataManager(directory, guid, **kwargs):
        manager = MetadataManagerFactory.mmm(directory,guid, **kwargs)
#        manager = MasterManager(directory,guid, **kwargs)
        return manager

    @staticmethod
    def getCoverageType(directory, guid):
        return MetadataManagerFactory.mmm.getCoverageType(directory, guid)

    @staticmethod
    def isPersisted(directory, guid):
        return MetadataManagerFactory.mmm.isPersisted(directory, guid)

    @staticmethod
    def dirExists(directory):
        return MetadataManagerFactory.mmm.dirExists(directory)