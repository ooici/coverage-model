#!/usr/bin/env python

"""
@package
@file test_parameter_types.py
@author James D. Case
@brief
"""

from nose.plugins.attrib import attr
import coverage_model.parameter_types as ptypes
import coverage_model.basic_types as btypes
import coverage_model as cm
import numpy as np
import random
from unittest import TestCase

@attr('UNIT',group='cov')
class TestParameterTypesUnit(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # ReferenceType
    def test_reference_type(self):
        reference_type = ptypes.ReferenceType()

    # BooleanType
    def test_boolean_type(self):
        boolean_type = ptypes.BooleanType()

    # CategoryType
    def test_category_type(self):
        categories = {'a':0,'b':1,'c':2}
        category_type = ptypes.CategoryType(categories=categories)

    # CountType
    def test_count_type(self):
        count_type = ptypes.CountType()

    # QuantityType
    def test_quantity_type(self):
        quantity_type = ptypes.QuantityType()
        quantity_type.fill_value = -9999.99999
        #quantity_type.value_encoding = np.dtype('float32')
        quantity_type.variability = btypes.VariabilityEnum.TEMPORAL

    # TextType
    def test_text_type(self):
        text_type = ptypes.TextType()

    # TimeType
    def test_time_type(self):
        time_type = ptypes.TimeType()

    # CategoryRangeType
    def test_category_range_type(self):
        category_range_type = ptypes.CategoryRangeType()

    # CountRangeType
    def test_count_range_type(self):
        count_range_type = ptypes.CountRangeType()

    # QuantityRangeType
    def test_quantity_range_type(self):
        quantity_range_type = ptypes.QuantityRangeType()

    # TimeRangeType
    def test_time_range_type(self):
        time_range_type = ptypes.TimeRangeType()

    # FunctionType
    def test_function_type(self):
        function_type = ptypes.FunctionType(base_type=None)

    # ConstantType
    def test_constant_type(self):
        constant_type = ptypes.ConstantType(base_type=None)

    # RecordType
    def test_record_type(self):
        record_type = ptypes.RecordType()

    # VectorType
    def test_vector_type(self):
        vector_type = ptypes.VectorType()

    # ArrayType
    def test_array_type(self):
        array_type = ptypes.ArrayType()
