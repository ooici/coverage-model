#!/usr/bin/env python

"""
@package
@file test_crs_values.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
from coverage_model import *

@attr('UNIT',group='cov')
class TestCRSValuesUnit(CoverageModelUnitTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_crs_temporal(self):
        temporal_code = 'ISO 8601'
        tcrs = CRS([AxisTypeEnum.TIME], temporal_code=temporal_code)
        self.assertEqual(tcrs.temporal_code, temporal_code)

        tcrs = CRS.standard_temporal(temporal_code=temporal_code)
        self.assertEqual(tcrs.temporal_code, temporal_code)

    def test_crs_spatial(self):
        epsg_code = 4326
        scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT], epsg_code=epsg_code)
        self.assertEqual(scrs.epsg_code, epsg_code)

        scrs = CRS.lat_lon(epsg_code=4326)
        self.assertEqual(scrs.epsg_code, epsg_code)

        scrs = CRS.lat_lon_height(epsg_code=4326)
        self.assertEqual(scrs.epsg_code, epsg_code)

        scrs = CRS.x_y_z(epsg_code=4326)
        self.assertEqual(scrs.epsg_code, epsg_code)