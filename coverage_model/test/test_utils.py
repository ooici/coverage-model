#!/usr/bin/env python

"""
@package 
@file test_utils.py
@author Christopher Mueller
@brief 
"""

from nose.plugins.attrib import attr
import coverage_model.utils as utils
from coverage_model import CoverageModelUnitTestCase

@attr('UNIT',group='cov')
class TestUtilsUnit(CoverageModelUnitTestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_prod(self):
        a = 3
        b = 4
        c = 6
        r = a * b * c

        self.assertEqual(r, utils.prod([a,b,c]))

    def test_create_guid(self):
        guid = utils.create_guid()

        # Ensure the guid is a str
        self.assertIsInstance(guid, str)

        # Make sure it's properly formatted - this also tests is_guid
        self.assertTrue(utils.is_guid(guid))

        # Test that is_guid fails when appropriate
        self.assertFalse(utils.is_guid(guid[:-1]))

    def test_is_valid_constraint_passes(self):
        sl = 1
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = [0,1,2]
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = (0,1,2)
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = slice(None,None,None)
        self.assertTrue(utils.is_valid_constraint(sl))

    def test_is_valid_constraint_fails(self):
        sl = 'bob'
        self.assertFalse(utils.is_valid_constraint(sl))

        sl = 43.2
        self.assertFalse(utils.is_valid_constraint(sl))

    def test_is_valid_constraint_nested_passes(self):
        sl = [[1,2,3]]
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = ((1,2,3))
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = [slice(1,2,3),slice(None,None,None)]
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = [slice(1,2,3), 2, 3]
        self.assertTrue(utils.is_valid_constraint(sl))

        sl = [1, slice(2,3,4), [1,3,5]]
        self.assertTrue(utils.is_valid_constraint(sl))

    def test_is_valid_constraint_nested_fails(self):
        sl = ['bob', 1, 4]
        self.assertFalse(utils.is_valid_constraint(sl))

        sl = [1, slice(2,3,4), 10.4]
        self.assertFalse(utils.is_valid_constraint(sl))


    def test__raise_index_error_int(self):
        size = 10
        dim = 0

        # Negative index
        self.assertRaises(IndexError, utils._raise_index_error_int, -1, size, dim)

        # Index == size
        self.assertRaises(IndexError, utils._raise_index_error_int, size, size, dim)

        # Index > size
        self.assertRaises(IndexError, utils._raise_index_error_int, size+1, size, dim)

        # Non failure
        utils._raise_index_error_int(1,size,dim)

    def test__raise_index_error_list(self):
        size = 10
        dim = 0

        # List with duplicate indices
        self.assertRaises(IndexError, utils._raise_index_error_list, [1,1], size, dim)

        # List with non-increasing indices
        self.assertRaises(IndexError, utils._raise_index_error_list, [2,1], size, dim)

        # List with an index == size
        self.assertRaises(IndexError, utils._raise_index_error_list, [1,size], size, dim)

        # List with an index > size
        self.assertRaises(IndexError, utils._raise_index_error_list, [1,size+1], size, dim)

        #### Passes ####
        utils._raise_index_error_list([1,2,8],size,dim)

    def test__raise_index_error_slice(self):
        size = 10
        dim = 0

        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(10, 2, None), size, dim)
        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(2, 2, None), size, dim)
        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(-1, 9, None), size, dim)
#        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(None, 13, None), size, dim)
        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(-1, None, None), size, dim)
        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(None, -1, None), size, dim)
        self.assertRaises(IndexError, utils._raise_index_error_slice, slice(None, 0, None), size, dim)

        #### Passes ####
        utils._raise_index_error_slice(slice(None,None,None),size,dim)
        utils._raise_index_error_slice(slice(0,None,None),size,dim)
        utils._raise_index_error_slice(slice(None,5,None),size,dim)
        utils._raise_index_error_slice(slice(2,8,None),size,dim)

    def test_fix_slice(self):
        shp = (50,)

        # Initial 'is_valid_constraint' catches bad stuff
        self.assertRaises(SystemError, utils.fix_slice, 'bob', shp)

        # Tuples are list-ified
        sl = utils.fix_slice((20,), shp)

        # Down-ranking (slice with more dims than shp)
        shp = (50,)
        sl = utils.fix_slice([5,23], shp)
        self.assertEqual(sl, (5,))

        # Up-ranking (slice with fewer dims than shp)
        shp = (50,10,)
        sl = utils.fix_slice(5, shp)
        self.assertEqual(sl, (5, slice(None, None, None)))

    def test_fix_slice_with_int(self):
        shp = (50,)

        # Verify int is tupled
        sl = utils.fix_slice(7, shp)
        self.assertEqual(sl, (7,))

        shp = (50,20)

        # Verify list of ints are tupled
        sl = utils.fix_slice([5,8], shp)
        self.assertEqual(sl, (5,8,))

        shp = (50,)

        #### Passes ####
        # Index
        sl = utils.fix_slice(0, shp)
        self.assertEqual(sl, (0,))

        # Negative index
        sl = utils.fix_slice(-4, shp)
        self.assertEqual(sl, (46,))

        #### Failures ####
        # Index > size
        self.assertRaises(IndexError, utils.fix_slice, 52, shp)

        # Negative index > size (after adjustment, index is -2)
        self.assertRaises(IndexError, utils.fix_slice, -52, shp)

    def test_fix_slice_with_list(self):
        shp = (50,)

        # Verify list is tupled
        sl = utils.fix_slice([[7,9]], shp)
        self.assertEqual(sl, ([7,9],))

        shp = (50,20)

        # Verify list of lists are tupled
        sl = utils.fix_slice([[5,8],[1,3]], shp)
        self.assertEqual(sl, ([5,8],[1,3]))

        shp = (50,)

        #### Passes ####
        # List of indices
        sl = utils.fix_slice([[5, 14, 36]], shp)
        self.assertEqual(sl, ([5, 14, 36],))

        # List with some negative indices
        sl = utils.fix_slice([[5, -36, 23]], shp)
        self.assertEqual(sl, ([5, 14, 23],))

        # List of all negative indices
        sl = utils.fix_slice([[-44, -32, -5]], shp)
        self.assertEqual(sl, ([6, 18, 45],))

        #### Failures ####
        # List of indices in non-increasing order
        self.assertRaises(IndexError, utils.fix_slice, [[20, 14, 36]], shp)

        # List of negative indices in non-increasing order (after adjustment, list is [45, 30, 5]
        self.assertRaises(IndexError, utils.fix_slice, [[-5, -20, -45]], shp)

    def test_fix_slice_with_slice(self):
        shp = (50,)

        # Verify slice is tupled
        sl = utils.fix_slice(slice(None, None, None), shp)
        self.assertEqual(sl, (slice(None, None, None),))

        shp = (50,20)

        # Verify list of slices are tupled
        sl = utils.fix_slice([slice(None, None, None), slice(None, None, None)], shp)
        self.assertEqual(sl, (slice(None, None, None), slice(None, None, None),))

        shp = (50,)

        #### Passes ####
        # Slice with all None
        sl = utils.fix_slice(slice(None, None, None), shp)
        self.assertEqual(sl, (slice(None, None, None),))

        # Slice with start=None & stop=positive
        sl = utils.fix_slice(slice(None, 30, None), shp)
        self.assertEqual(sl, (slice(None, 30, None),))

        # Slice with start=positive & stop=None
        sl = utils.fix_slice(slice(10, None, None), shp)
        self.assertEqual(sl, (slice(10, None, None),))

        # Slice with start=negative & stop=None
        sl = utils.fix_slice(slice(-10, None, None), shp)
        self.assertEqual(sl, (slice(40, None, None),))

        # Slice with start=negative & stop=negative
        sl = utils.fix_slice(slice(-10, -5, None), shp)
        self.assertEqual(sl, (slice(40, 45, None),))

        # Slice with start=None & stop=negative
        sl = utils.fix_slice(slice(None, -18, None), shp)
        self.assertEqual(sl, (slice(None, 32, None),))

        #### Failures ####
        # Slice with start > stop
        self.assertRaises(IndexError, utils.fix_slice, slice(30, 2, None), shp)

        # Slice with start == stop
        self.assertRaises(IndexError, utils.fix_slice, slice(18, 18, None), shp)

#        # Slice with start=None & stop > size
#        self.assertRaises(IndexError, utils.fix_slice, slice(None, 52, None), shp)

        # Slice with start > size & stop=None
        self.assertRaises(IndexError, utils.fix_slice, slice(53, None, None), shp)

        # Slice with start < 0 & stop=None (after adjustment, -52 => -2)
        self.assertRaises(IndexError, utils.fix_slice, slice(-52, None, None), shp)

        # Slice with start=None & stop=0
        self.assertRaises(IndexError, utils.fix_slice, slice(None, 0, None), shp)

        # Slice with start=None & stop < 0 (after adjustment, -52 => -2)
        self.assertRaises(IndexError, utils.fix_slice, slice(None, -52, None), shp)

        # Slice with start=negative & stop=negative with start > stop
        self.assertRaises(IndexError, utils.fix_slice, slice(-2, -10, None), shp)

        #### Stepping ####
        # Slice with step != None
        sl = utils.fix_slice(slice(4, 43, 6), shp)
        self.assertEqual(sl, (slice(4, 43, 6),))
        
        # Slice with negative step - reverses start/stop
        sl = utils.fix_slice(slice(10, 2, -2), shp)
        self.assertEqual(sl, (slice(2, 10, 2),))

        # Slice with all negatives (start, stop, step)
        sl = utils.fix_slice(slice(-3, -22, -5), shp)
        self.assertEqual(sl, (slice(28, 47, 5),))

        
