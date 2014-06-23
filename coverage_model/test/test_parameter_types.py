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
from coverage_model import CoverageModelUnitTestCase

# TODO: Revisit this test class and expand/elaborate the tests

@attr('UNIT',group='cov')
class TestParameterTypesUnit(CoverageModelUnitTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    # ReferenceType
    def test_reference_type(self):
        reference_type = ptypes.ReferenceType()
        self.assertIsInstance(reference_type, ptypes.ReferenceType)

    # BooleanType
    def test_boolean_type(self):
        boolean_type = ptypes.BooleanType()
        self.assertIsInstance(boolean_type, ptypes.BooleanType)

    # CategoryType
    def test_category_type(self):
        categories = {'a':0,'b':1,'c':2}
        category_type = ptypes.CategoryType(categories=categories)
        self.assertIsInstance(category_type, ptypes.CategoryType)

    # CountType
    def test_count_type(self):
        count_type = ptypes.CountType()
        self.assertIsInstance(count_type, ptypes.CountType)

    # QuantityType
    def test_quantity_type(self):
        quantity_type = ptypes.QuantityType()
        quantity_type.fill_value = -9999.99999
        #quantity_type.value_encoding = np.dtype('float32')
        quantity_type.variability = btypes.VariabilityEnum.TEMPORAL
        self.assertIsInstance(quantity_type, ptypes.QuantityType)

    # TextType
    def test_text_type(self):
        text_type = ptypes.TextType()
        self.assertIsInstance(text_type, ptypes.TextType)

    # TimeType
    def test_time_type(self):
        time_type = ptypes.TimeType()
        self.assertIsInstance(time_type, ptypes.TimeType)

    # CategoryRangeType
    def test_category_range_type(self):
        category_range_type = ptypes.CategoryRangeType()
        self.assertIsInstance(category_range_type, ptypes.CategoryRangeType)

    # CountRangeType
    def test_count_range_type(self):
        count_range_type = ptypes.CountRangeType()
        self.assertIsInstance(count_range_type, ptypes.CountRangeType)

    # QuantityRangeType
    def test_quantity_range_type(self):
        quantity_range_type = ptypes.QuantityRangeType()
        self.assertIsInstance(quantity_range_type, ptypes.QuantityRangeType)

    # TimeRangeType
    def test_time_range_type(self):
        time_range_type = ptypes.TimeRangeType()
        self.assertIsInstance(time_range_type, ptypes.TimeRangeType)

    # FunctionType
    def test_function_type(self):
        function_type = ptypes.FunctionType(base_type=None)
        self.assertIsInstance(function_type, ptypes.FunctionType)

    # ConstantType
    def test_constant_type(self):
        constant_type = ptypes.ConstantType(base_type=None)
        self.assertIsInstance(constant_type, ptypes.ConstantType)

    # RecordType
    def test_record_type(self):
        record_type = ptypes.RecordType()
        self.assertIsInstance(record_type, ptypes.RecordType)

    # VectorType
    def test_vector_type(self):
        vector_type = ptypes.VectorType()
        self.assertIsInstance(vector_type, ptypes.VectorType)

    # ArrayType
    def test_array_type(self):
        array_type = ptypes.ArrayType()
        self.assertIsInstance(array_type, ptypes.ArrayType)

    def test_array_type_creation(self):
        import numpy as np
        at5 = ptypes.ArrayType(inner_encoding='i8', inner_length=5, inner_fill_value=-99)
        at2 = ptypes.ArrayType(inner_encoding='i8', inner_length=2, inner_fill_value=-99)
        arrs = []
        arrs.append(at2.create_data_array([[1,2], [3,4]]))
        arrs.append(at5.create_data_array([[10,20,30,40,50], [50,60,70,80,90]]))
        arrs.append(at2.create_data_array([[100,200], [300,400], [500,600]]))
        merged = at2.create_merged_value_array(arrs)
        expected = np.array([[1,2,-99,-99,-99], [3,4,-99,-99,-99], [10,20,30,40,50], [50,60,70,80,90],
                             [100,200,-99,-99,-99], [300,400,-99,-99,-99], [500,600,-99,-99,-99]], dtype='<i8')
        np.testing.assert_array_equal(merged,expected)

    def test_quantity_type_creation(self):
        import numpy as np
        at5 = ptypes.QuantityType(value_encoding='i8')
        at2 = ptypes.QuantityType(value_encoding='i8')
        arrs = []
        arrs.append(at2.create_data_array([1,2,3,4]))
        arrs.append(at5.create_data_array([10,20,30,40,50,50,60,70,80,90]))
        arrs.append(at2.create_data_array([100,200,300,400,500,600]))
        merged = at2.create_merged_value_array(arrs)
        expected = np.array([1,2,3,4,10,20,30,40,50,50,60,70,80,90,100,200,300,400,500,600,], dtype='<i8')
        np.testing.assert_array_equal(merged,expected)
