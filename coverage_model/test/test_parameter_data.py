#!/usr/bin/env python

"""
@package coverage_model.postgres_persisted_storage
@file coverage_model.postgres_persisted_storage
@author Casey Bryant
@brief Persistence Layer specialized classes for storing persisted data to Postgres
"""

import numpy as np

from nose.plugins.attrib import attr
from coverage_model import *
from coverage_model.parameter_data import NumpyParameterData, ConstantOverTime


@attr('UNIT',group='cov')
class TestSpanUnit(CoverageModelUnitTestCase):

    def test_numpy_parameter(self):
        name = 'dummy'
        data = np.array([1.1, 1.2, 1.3, 1.4])
        alignment = np.array([0.01, 0.02, 0.03, 0.04])
        misalignment = np.array([0.01, 0.02, 0.03])

        with self.assertRaises(TypeError):
            pd = NumpyParameterData(1, data, alignment)
        with self.assertRaises(TypeError):
            pd = NumpyParameterData(name, 'hi', alignment)
        with self.assertRaises(TypeError):
            pd = NumpyParameterData(name, data, 'hi')
        with self.assertRaises(ValueError):
            pd = NumpyParameterData(name, data, misalignment)

        pd = NumpyParameterData(name, data, alignment)

        self.assertEqual(id(data), id(pd.get_data_as_numpy_array(pd.get_alignment())))

        self.assertEqual(0, pd.get_data_as_numpy_array(np.array([0.05, 0.015])).size)
        self.assertEqual(np.array([1.2, 1.4]).all(), pd.get_data_as_numpy_array(np.array([0.04, 0.02])).all())

    def test_constant_parameter(self):
        name = 'dummy'
        data = 3.14159
        start = 1.0
        stop = 2.0

        fill_val = np.NaN
        arr = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5])

        pd = ConstantOverTime(name, data, time_start=start)

        alignment_array = arr[10:14]
        got = pd.get_data_as_numpy_array(alignment_array)
        expected = np.empty(4)
        expected.fill(data)
        self.assertTrue(np.array_equal(expected, got))

        alignment_array = arr[10:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(fill_val)
        expected[5:alignment_array.size] = data
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:10]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(fill_val)
        expected[5:alignment_array.size] = data
        np.testing.assert_array_equal(got, expected)


        pd = ConstantOverTime(name, data, time_start=start, time_end=stop)

        alignment_array = arr[10:14]
        got = pd.get_data_as_numpy_array(alignment_array)
        expected = np.empty(4)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[10:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(11)
        expected.fill(fill_val)
        expected[0:6] = data
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(fill_val)
        expected[5:11+5] = data
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:11]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(fill_val)
        expected[5:] = data
        expected = np.array([fill_val, fill_val, fill_val, fill_val, fill_val, data, data, data, data, data, data])
        np.testing.assert_array_equal(got, expected)

        pd = ConstantOverTime(name, data, time_end=stop)

        alignment_array = arr[10:14]
        got = pd.get_data_as_numpy_array(alignment_array)
        expected = np.empty(4)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[10:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(11)
        expected.fill(fill_val)
        expected[0:6] = data
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(fill_val)
        expected[0:11+5] = data
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:11]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        pd = ConstantOverTime(name, data)

        alignment_array = arr[10:14]
        got = pd.get_data_as_numpy_array(alignment_array)
        expected = np.empty(4)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[10:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(11)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

        alignment_array = arr[0:11]
        got = pd.get_data_as_numpy_array(alignment_array, fill_value=fill_val)
        expected = np.empty(alignment_array.size)
        expected.fill(data)
        np.testing.assert_array_equal(got, expected)

    def test_constant_parameter_failures(self):
        cot = ConstantOverTime('dummy', 10, 1.1, 1.10001)
        with self.assertRaises(RuntimeError):
            cot = ConstantOverTime('dummy', 10, 1.1, 1.1)
        with self.assertRaises(RuntimeError):
            cot = ConstantOverTime('dummy', 10, 1.1, 1.099999)
