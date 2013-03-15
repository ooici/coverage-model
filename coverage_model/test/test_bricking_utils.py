#!/usr/bin/env python

"""
@package coverage_model.test.test_bricking_utils
@file coverage_model/test/test_bricking_utils.py
@author Christopher Mueller
@brief Tests for the bricking_utils module
"""


from nose.plugins.attrib import attr
from coverage_model import CoverageModelUnitTestCase, CoverageModelIntTestCase

from coverage_model.bricking_utils import *
from rtree import index


@attr('UNIT', group='cov')
class TestBrickingUtilsUnit(CoverageModelUnitTestCase):

    def test_calc_brick_origins_1d(self):
        brick_sizes = (5,)

        # 1d - even num of bricks
        total_domain = (15,)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0,), (5,), (10,))
        self.assertEqual(brick_origins, want)

        # 1d - uneven num of bricks
        total_domain = (13,)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0,), (5,), (10,))
        self.assertEqual(brick_origins, want)

    def test_calc_brick_origins_2d(self):
        brick_sizes = (5, 5)

        # 2d - even num of bricks
        total_domain = (15, 10)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0, 0), (0, 5), (5, 0), (5, 5), (10, 0), (10, 5))
        self.assertEqual(brick_origins, want)

        # 2d - uneven num of bricks
        total_domain = (13, 17)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0, 0), (0, 5), (0, 10), (0, 15), (5, 0), (5, 5), (5, 10), (5, 15), (10, 0), (10, 5), (10, 10), (10, 15))
        self.assertEqual(brick_origins, want)

    def test_calc_brick_origins_3d(self):
        brick_sizes = (5, 5, 5)

        # 3d - even num of bricks
        total_domain = (10, 15, 5)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0, 0, 0), (0, 5, 0), (0, 10, 0), (5, 0, 0), (5, 5, 0), (5, 10, 0))
        self.assertEqual(brick_origins, want)

        # 3d - uneven num of bricks
        total_domain = (13, 19, 3)
        brick_origins = calc_brick_origins(total_domain, brick_sizes)
        want = ((0, 0, 0), (0, 5, 0), (0, 10, 0), (0, 15, 0), (5, 0, 0), (5, 5, 0), (5, 10, 0), (5, 15, 0), (10, 0, 0), (10, 5, 0), (10, 10, 0), (10, 15, 0))
        self.assertEqual(brick_origins, want)

    def test_calc_brick_origins_errors(self):
        brick_sizes = (5, 5)
        total_domain = (10,)

        # Non-iterable total_domain
        self.assertRaises(ValueError, calc_brick_origins, 10, brick_sizes)

        # Non-iterable brick_sizes
        self.assertRaises(ValueError, calc_brick_origins, total_domain, 5)

        # Incompatible total_domain & brick_sizes
        self.assertRaises(ValueError, calc_brick_origins, total_domain, brick_sizes)

    def test_calc_brick_and_rtree_extents_1d(self):
        sizes = (5,)

        origins = ((0,), (5,), (10,))
        be, rte = calc_brick_and_rtree_extents(origins, sizes)
        be_want = (((0, 4),), ((5, 9),), ((10, 14),))
        rte_want = ((0, 0, 4, 0), (5, 0, 9, 0), (10, 0, 14, 0))
        self.assertEqual(be, be_want)
        self.assertEqual(rte, rte_want)

    def test_calc_brick_and_rtree_extents_2d(self):
        sizes = (5, 5)
        origins = ((0, 0), (0, 5), (5, 0), (5, 5), (10, 0), (10, 5))
        be, rte = calc_brick_and_rtree_extents(origins, sizes)
        be_want = (((0, 4), (0, 4)), ((0, 4), (5, 9)), ((5, 9), (0, 4)), ((5, 9), (5, 9)), ((10, 14), (0, 4)), ((10, 14), (5, 9)))
        rte_want = ((0, 0, 4, 4), (0, 5, 4, 9), (5, 0, 9, 4), (5, 5, 9, 9), (10, 0, 14, 4), (10, 5, 14, 9))
        self.assertEqual(be, be_want)
        self.assertEqual(rte, rte_want)

    def test_calc_brick_and_rtree_extents_3d(self):
        sizes = (5, 5, 5)
        origins = ((0, 0, 0), (0, 5, 0), (0, 10, 0), (5, 0, 0), (5, 5, 0), (5, 10, 0))
        be, rte = calc_brick_and_rtree_extents(origins, sizes)
        be_want = (((0, 4), (0, 4), (0, 4)), ((0, 4), (5, 9), (0, 4)), ((0, 4), (10, 14), (0, 4)), ((5, 9), (0, 4), (0, 4)), ((5, 9), (5, 9), (0, 4)), ((5, 9), (10, 14), (0, 4)))
        rte_want = ((0, 0, 0, 4, 4, 4), (0, 5, 0, 4, 9, 4), (0, 10, 0, 4, 14, 4), (5, 0, 0, 9, 4, 4), (5, 5, 0, 9, 9, 4), (5, 10, 0, 9, 14, 4))
        self.assertEqual(be, be_want)
        self.assertEqual(rte, rte_want)

    def test_calc_brick_and_rtree_extents_errors(self):
        brick_sizes = (5,)
        brick_origins = ((0, 0), (0, 5))

        # Non-iterable brick_origins
        self.assertRaises(ValueError, calc_brick_and_rtree_extents, 10, brick_sizes)

        # Non-iterable brick_sizes
        self.assertRaises(ValueError, calc_brick_and_rtree_extents, brick_origins, 5)

        # Incompatible brick_origins & brick_sizes
        self.assertRaises(ValueError, calc_brick_and_rtree_extents, brick_origins, brick_sizes)

    def test_rtree_populator_1d(self):

        brick_extents = (((0, 4),), ((5, 9),), ((10, 14),))
        rtree_extents = ((0, 0, 4, 0), (5, 0, 9, 0), (10, 0, 14, 0))

        p = index.Property()
        p.dimension = 2  # Minimum is 2 for proper functioning
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        self.assertIsInstance(rtree, index.Index)
        self.assertEqual(rtree.get_bounds(), [0.0, 0.0, 14.0, 0.0])
        self.assertEqual(rtree.leaves(), [(0, [0, 1, 2], [0.0, 0.0, 14.0, 0.0])])
        self.assertEqual(rtree.properties.dimension, 2)

    def test_rtree_populator_2d(self):

        brick_extents = (((0, 4), (0, 4)), ((0, 4), (5, 9)), ((5, 9), (0, 4)), ((5, 9), (5, 9)), ((10, 14), (0, 4)), ((10, 14), (5, 9)))
        rtree_extents = ((0, 0, 4, 4), (0, 5, 4, 9), (5, 0, 9, 4), (5, 5, 9, 9), (10, 0, 14, 4), (10, 5, 14, 9))

        p = index.Property()
        p.dimension = 2
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        self.assertIsInstance(rtree, index.Index)
        self.assertEqual(rtree.get_bounds(), [0.0, 0.0, 14.0, 9.0])
        self.assertEqual(rtree.leaves(), [(0, [0, 1, 2, 3, 4, 5], [0.0, 0.0, 14.0, 9.0])])
        self.assertEqual(rtree.properties.dimension, 2)

    def test_rtree_populator_3d(self):

        brick_extents = (((0, 4), (0, 4), (0, 4)), ((0, 4), (5, 9), (0, 4)), ((0, 4), (10, 14), (0, 4)), ((5, 9), (0, 4), (0, 4)), ((5, 9), (5, 9), (0, 4)), ((5, 9), (10, 14), (0, 4)))
        rtree_extents = ((0, 0, 0, 4, 4, 4), (0, 5, 0, 4, 9, 4), (0, 10, 0, 4, 14, 4), (5, 0, 0, 9, 4, 4), (5, 5, 0, 9, 9, 4), (5, 10, 0, 9, 14, 4))

        p = index.Property()
        p.dimension = 3
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        self.assertIsInstance(rtree, index.Index)
        self.assertEqual(rtree.get_bounds(), [0.0, 0.0, 0.0, 9.0, 14.0, 4.0])
        self.assertEqual(rtree.leaves(), [(0, [0, 1, 2, 3, 4, 5], [0.0, 0.0, 0.0, 9.0, 14.0, 4.0])])
        self.assertEqual(rtree.properties.dimension, 3)

    def _get_bricks_assert(self, slice_, rtree, total_domain, size, brick_list):
        bricks = get_bricks_from_slice(slice_, rtree, total_domain)
        self.assertEqual(len(bricks), size)
        self.assertEqual(bricks, brick_list)

    def _p_get_bricks_assert(self, slice_, rtree, total_domain, size, brick_list):
        bricks = get_bricks_from_slice(slice_, rtree, total_domain)
        print
        print len(bricks)
        print bricks

    def test_get_bricks_from_slice_1d(self):
        total_domain = (15,)
        brick_extents = (((0, 4),), ((5, 9),), ((10, 14),))
        rtree_extents = ((0, 0, 4, 0), (5, 0, 9, 0), (10, 0, 14, 0))

        brick_0 = (0, ((0, 4),))
        brick_1 = (1, ((5, 9),))
        brick_2 = (2, ((10, 14),))

        p = index.Property()
        p.dimension = 2  # Minimum is 2 for proper functioning
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        # Try a variety of slices
        self._get_bricks_assert(slice(None), rtree, total_domain, 3, [brick_0, brick_1, brick_2])

        self._get_bricks_assert(slice(None, None, 3), rtree, total_domain, 3, [brick_0, brick_1, brick_2])

        self._get_bricks_assert(slice(0, 3), rtree, total_domain, 1, [brick_0])

        self._get_bricks_assert(slice(5, 9), rtree, total_domain, 1, [brick_1])

        self._get_bricks_assert(slice(6, None), rtree, total_domain, 2, [brick_1, brick_2])

        self._get_bricks_assert(slice(None, None, 10), rtree, total_domain, 3, [brick_0, brick_1, brick_2])  # three bricks, tho the middle one isn't needed

        self._get_bricks_assert(([1, 3],), rtree, total_domain, 1, [brick_0])

        self._get_bricks_assert(([2, 4, 7],), rtree, total_domain, 2, [brick_0, brick_1])

        self._get_bricks_assert(([3, 12],), rtree, total_domain, 3, [brick_0, brick_1, brick_2])  # three bricks, tho the middle one isn't needed

        self._get_bricks_assert(1, rtree, total_domain, 1, [brick_0])

        self._get_bricks_assert(6, rtree, total_domain, 1, [brick_1])

        self._get_bricks_assert(13, rtree, total_domain, 1, [brick_2])

    def test_get_bricks_from_slice_2d(self):
        total_domain = (15, 10)
        brick_extents = (((0, 4), (0, 4)), ((0, 4), (5, 9)), ((5, 9), (0, 4)), ((5, 9), (5, 9)), ((10, 14), (0, 4)), ((10, 14), (5, 9)))
        rtree_extents = ((0, 0, 4, 4), (0, 5, 4, 9), (5, 0, 9, 4), (5, 5, 9, 9), (10, 0, 14, 4), (10, 5, 14, 9))

        brick_0 = (0, ((0, 4), (0, 4)))
        brick_1 = (1, ((0, 4), (5, 9)))
        brick_2 = (2, ((5, 9), (0, 4)))
        brick_3 = (3, ((5, 9), (5, 9)))
        brick_4 = (4, ((10, 14), (0, 4)))
        brick_5 = (5, ((10, 14), (5, 9)))

        p = index.Property()
        p.dimension = 2
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        # Get all bricks
        self._get_bricks_assert((slice(None),) * 2, rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((slice(None), slice(None, 8)), rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((slice(None), slice(None, 4)), rtree, total_domain, 3, [brick_0, brick_2, brick_4])

        self._get_bricks_assert((slice(7, 12), slice(5, 8)), rtree, total_domain, 2, [brick_3, brick_5])

        self._get_bricks_assert((slice(2, 14, 3), slice(2, 7)), rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((slice(2, 14, 10), slice(2, 7)), rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((0, slice(2, 8, 3)), rtree, total_domain, 2, [brick_0, brick_1])

        self._get_bricks_assert((6, slice(2, 7)), rtree, total_domain, 2, [brick_2, brick_3])

        self._get_bricks_assert((slice(None, 12), 7), rtree, total_domain, 3, [brick_1, brick_3, brick_5])

        self._get_bricks_assert((12, slice(2, None, 4)), rtree, total_domain, 2, [brick_4, brick_5])

        self._get_bricks_assert(([1, 2], 9), rtree, total_domain, 1, [brick_1])

        self._get_bricks_assert(([0, 14], 3), rtree, total_domain, 3, [brick_0, brick_2, brick_4])

        self._get_bricks_assert((3, [1, 8]), rtree, total_domain, 2, [brick_0, brick_1])

        self._get_bricks_assert(([2, 5], [1, 8]), rtree, total_domain, 4, [brick_0, brick_1, brick_2, brick_3])

        self._get_bricks_assert(([6, 9], [1, 8]), rtree, total_domain, 2, [brick_2, brick_3])

        self._get_bricks_assert(([2, 8, 13], [7, 8]), rtree, total_domain, 3, [brick_1, brick_3, brick_5])

    def test_get_bricks_from_slice_3d(self):
        total_domain = (10, 15, 5)
        brick_extents = (((0, 4), (0, 4), (0, 4)), ((0, 4), (5, 9), (0, 4)), ((0, 4), (10, 14), (0, 4)), ((5, 9), (0, 4), (0, 4)), ((5, 9), (5, 9), (0, 4)), ((5, 9), (10, 14), (0, 4)))
        rtree_extents = ((0, 0, 0, 4, 4, 4), (0, 5, 0, 4, 9, 4), (0, 10, 0, 4, 14, 4), (5, 0, 0, 9, 4, 4), (5, 5, 0, 9, 9, 4), (5, 10, 0, 9, 14, 4))

        brick_0 = (0, ((0, 4), (0, 4), (0, 4)))
        brick_1 = (1, ((0, 4), (5, 9), (0, 4)))
        brick_2 = (2, ((0, 4), (10, 14), (0, 4)))
        brick_3 = (3, ((5, 9), (0, 4), (0, 4)))
        brick_4 = (4, ((5, 9), (5, 9), (0, 4)))
        brick_5 = (5, ((5, 9), (10, 14), (0, 4)))

        p = index.Property()
        p.dimension = 3
        rtree = index.Index(rtree_populator(rtree_extents, brick_extents), properties=p)

        # Get all bricks
        self._get_bricks_assert((slice(None),) * 3, rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((0, 0, 0), rtree, total_domain, 1, [brick_0])

        self._get_bricks_assert((8, 5, 2), rtree, total_domain, 1, [brick_4])

        self._get_bricks_assert((4, 12, 1), rtree, total_domain, 1, [brick_2])

        self._get_bricks_assert((9, 13, [0, 2]), rtree, total_domain, 1, [brick_5])

        self._get_bricks_assert((8, [3, 5, 12], 0), rtree, total_domain, 3, [brick_3, brick_4, brick_5])

        self._get_bricks_assert(([5, 9], 10, 0), rtree, total_domain, 1, [brick_5])

        self._get_bricks_assert(([5, 9], [4, 12, 13], 0), rtree, total_domain, 3, [brick_3, brick_4, brick_5])

        self._get_bricks_assert(([2, 4], [2, 11], [1, 3, 4]), rtree, total_domain, 3, [brick_0, brick_1, brick_2])

        self._get_bricks_assert(([2, 3, 9], 12, [1, 3, 4]), rtree, total_domain, 2, [brick_2, brick_5])

        self._get_bricks_assert((slice(None), 12, [1, 3, 4]), rtree, total_domain, 2, [brick_2, brick_5])

        self._get_bricks_assert((slice(1, 7), 3, [1, 3, 4]), rtree, total_domain, 2, [brick_0, brick_3])

        self._get_bricks_assert((slice(3, 4), 7, [1, 3, 4]), rtree, total_domain, 1, [brick_1])

        self._get_bricks_assert((slice(2, 8, 7), [1, 6, 12], 4), rtree, total_domain, 6, [brick_0, brick_1, brick_2, brick_3, brick_4, brick_5])

        self._get_bricks_assert((slice(2, 4, 7), slice(None), 2), rtree, total_domain, 3, [brick_0, brick_1, brick_2])

        self._get_bricks_assert((slice(None, 4), slice(9, None, 2), slice(None)), rtree, total_domain, 2, [brick_1, brick_2])

        self._get_bricks_assert((slice(None, 6, 4), slice(12, None, 2), slice(3, None)), rtree, total_domain, 2, [brick_2, brick_5])

        self._get_bricks_assert((slice(None, 8), slice(6, 13, 4), slice(None, None, 3)), rtree, total_domain, 4, [brick_1, brick_2, brick_4, brick_5])

    def _run_test_slices(self, ba, sl_list, val_arr, verbose):
        for sl in sl_list:
            ba.reset_bricks()

            vals = val_arr[sl]
            ba.put_values_to_bricks(sl, vals)
            vo = ba.get_values_from_bricks(sl)

            self.assertTrue(np.array_equal(vals, vo) or np.array_equal(vals.squeeze(), vo))

    def test_set_get_slice_1d(self):
        from coverage_model.test.bricking_assessment import test_1d
        test_1d(self._run_test_slices, None, persist=False, verbose=False, dtype='int16')
        test_1d(self._run_test_slices, None, persist=False, verbose=False, dtype='int32')
        test_1d(self._run_test_slices, None, persist=False, verbose=False, dtype='float32')
        test_1d(self._run_test_slices, None, persist=False, verbose=False, dtype='float64')

    def test_set_get_slice_2d(self):
        from coverage_model.test.bricking_assessment import test_2d
        test_2d(self._run_test_slices, None, persist=False, verbose=False, dtype='int16')
        test_2d(self._run_test_slices, None, persist=False, verbose=False, dtype='int32')
        test_2d(self._run_test_slices, None, persist=False, verbose=False, dtype='float32')
        test_2d(self._run_test_slices, None, persist=False, verbose=False, dtype='float64')

    def test_set_get_slice_3d(self):
        from coverage_model.test.bricking_assessment import test_3d
        test_3d(self._run_test_slices, None, persist=False, verbose=False, dtype='int16')
        test_3d(self._run_test_slices, None, persist=False, verbose=False, dtype='int32')
        test_3d(self._run_test_slices, None, persist=False, verbose=False, dtype='float32')
        test_3d(self._run_test_slices, None, persist=False, verbose=False, dtype='float64')


@attr('INT', group='cov')
class TestBrickingUtilsInt(CoverageModelIntTestCase):

    def _run_test_slices(self, ba, sl_list, val_arr, verbose):
        for sl in sl_list:
            ba.reset_bricks()

            vals = val_arr[sl]
            ba.put_values_to_bricks(sl, vals)
            vo = ba.get_values_from_bricks(sl)

            self.assertTrue(np.array_equal(vals, vo) or np.array_equal(vals.squeeze(), vo))

    def test_set_get_slice_1d(self):
        from coverage_model.test.bricking_assessment import test_1d
        test_1d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int16')
        test_1d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int32')
        test_1d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float32')
        test_1d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float64')

    def test_set_get_slice_2d(self):
        from coverage_model.test.bricking_assessment import test_2d
        test_2d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int16')
        test_2d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int32')
        test_2d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float32')
        test_2d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float64')

    def test_set_get_slice_3d(self):
        from coverage_model.test.bricking_assessment import test_3d
        test_3d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int16')
        test_3d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='int32')
        test_3d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float32')
        test_3d(self._run_test_slices, self.working_dir, persist=True, verbose=False, dtype='float64')