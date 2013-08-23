#!/usr/bin/env python

"""
@package 
@file brick_split
@author Christopher Mueller
@brief

"""
from ooi.logging import log
from coverage_model import utils
from coverage_model import bricking_utils
from coverage_model import ParameterContext, QuantityType
from coverage_model.persistence_helpers import MasterManager, ParameterManager
import os
import shutil
from coverage_model import create_guid
from coverage_model.persistence_helpers import RTreeProxy

import numpy as np
import h5py
from coverage_model.hdf_utils import HDFLockingFile


class BrickingAssessor(object):
    def __init__(self, total_domain=(10, 10), brick_size=5, use_hdf=False, root_dir='test_data/multi_dim_trials',
                 guid=None, dtype='int16'):
        self.total_domain = total_domain
        self.brick_sizes = tuple(brick_size for x in total_domain)
        self.use_hdf = use_hdf
        self.dtype = np.dtype(dtype).name
        if self.use_hdf:
            self.guid = guid or create_guid()
            name = '%s_%s' % (self.guid, self.dtype)
            self.root_dir = root_dir
            if not os.path.exists(self.root_dir):
                os.makedirs(self.root_dir)

            if os.path.exists(os.path.join(self.root_dir, name)):
                shutil.rmtree(os.path.join(self.root_dir, name))

            self.master_manager = MasterManager(self.root_dir, name, name='md_test_{0}'.format(name))

            self.master_manager.flush()

            pc = ParameterContext('test_param', param_type=QuantityType(self.dtype), fill_value=-1)
            self.param_manager = ParameterManager(os.path.join(self.root_dir, name, pc.name), pc.name)
            self.param_manager.parameter_context = pc
            self.master_manager.create_group(pc.name)

            self.param_manager.flush()

        self.bricks = {}

        self.brick_origins = bricking_utils.calc_brick_origins(self.total_domain, self.brick_sizes)
        self.brick_extents, self.rtree_extents = bricking_utils.calc_brick_and_rtree_extents(self.brick_origins,
                                                                                             self.brick_sizes)
        self.build_bricks()

        self.rtree = RTreeProxy()
        for x in BrickingAssessor.rtree_populator(self.rtree_extents, self.brick_extents):
            self.rtree.insert(*x)

    @classmethod
    def rtree_populator(cls, rtree_extents, brick_extents):
        for i, e in enumerate(rtree_extents):
            yield i, e, brick_extents[i]

    def _get_numpy_array(self, shape):
        if not isinstance(shape, tuple):
            shape = tuple(shape)

        return np.arange(utils.prod(shape), dtype=self.dtype).reshape(shape)

    def build_bricks(self):
        for x in xrange(len(self.brick_origins)):
            if not self.use_hdf:
                self.bricks[x] = np.empty(self.brick_sizes, dtype=self.dtype)
                self.bricks[x].fill(-1)
            else:
                id = str(x)
                fn = '{0}.hdf5'.format(id)
                pth = os.path.join(self.param_manager.root_dir, fn)
                relpth = os.path.join(self.param_manager.root_dir.replace(self.master_manager.root_dir, '.'), fn)
                lnpth = '/{0}/{1}'.format(self.param_manager.parameter_name, id)

                self.master_manager.add_external_link(lnpth, relpth, id)
                self.bricks[x] = pth

    def reset_bricks(self):
        for i, arr in enumerate(self.bricks.itervalues()):
            if not self.use_hdf:
                arr.fill(-1)
            else:
                with HDFLockingFile(arr, mode='a') as f:
                    ds = f.require_dataset(str(i), shape=self.brick_sizes, dtype=self.dtype, chunks=None, fillvalue=-1)
                    ds[:] = -1

    def put_values_to_bricks(self, slice_, values):
        slice_ = utils.fix_slice(slice_, self.total_domain)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.rtree,
                                                      self.total_domain) # this is a list of tuples [(b_id, (bounds...),), ...]

        values = np.asanyarray(values)
        v_shp = values.shape
        log.debug('value_shape: %s', v_shp)
        s_shp = utils.slice_shape(slice_, self.total_domain)
        log.debug('slice_shape: %s', s_shp)
        is_broadcast = False
        if v_shp == ():
            log.debug('Broadcast!!')
            is_broadcast = True
            value_slice = ()
        elif v_shp != s_shp:
            if v_shp == tuple([i for i in s_shp if i != 1]): # Missing dimensions are singleton, just reshape to fit
                values = values.reshape(s_shp)
                v_shp = values.shape
            else:
                raise IndexError(
                    'Shape of \'value\' is not compatible with \'slice_\': slice_ shp == {0}\tvalue shp == {1}'.format(
                        s_shp, v_shp))
        else:
            value_slice = None

        log.debug('value_shape: %s', v_shp)

        for b in bricks:
            # b is (brick_id, (brick_bounds per dim...),)
            bid, bbnds = b
            log.debug('Determining slice for brick: %s', b)
            bexts = tuple([x + 1 for x in zip(*bbnds)[1]]) # Shift from index to size
            log.debug('bid=%s, bbnds=%s, bexts=%s', bid, bbnds, bexts)

            brick_slice, brick_mm = bricking_utils.get_brick_slice_nd(slice_, bbnds)

            if None in brick_slice: # Brick does not contain any of the requested indices
                log.debug('Brick does not contain any of the requested indices: Move to next brick')
                continue

            try:
                brick_slice = utils.fix_slice(brick_slice, bexts)
            except IndexError:
                log.debug('Malformed brick_slice: move to next brick')
                continue

            if not is_broadcast:
                value_slice = bricking_utils.get_value_slice_nd(slice_, v_shp, bbnds, brick_slice, brick_mm)

                try:
                    value_slice = utils.fix_slice(value_slice, v_shp)
                except IndexError:
                    log.debug('Malformed value_slice: move to next brick')
                    continue

            log.debug('\nbrick %s:\n\tbrick_slice %s=%s\n\tmin/max=%s\n\tvalue_slice %s=%s', b,
                      utils.slice_shape(brick_slice, bexts), brick_slice, brick_mm,
                      utils.slice_shape(value_slice, v_shp), value_slice)
            v = values[value_slice]
            log.debug('\nvalues %s=\n%s', v.shape, v)
            if not self.use_hdf:
                self.bricks[bid][brick_slice] = v
            else:
                fi = self.bricks[bid]
                with HDFLockingFile(fi, 'a') as f:
                    ds = f.require_dataset(str(bid), shape=self.brick_sizes, dtype=self.dtype, chunks=None,
                                           fillvalue=-1)
                    ds[brick_slice] = v

    def get_values_from_bricks(self, slice_):
        slice_ = utils.fix_slice(slice_, self.total_domain)
        bricks = bricking_utils.get_bricks_from_slice(slice_, self.rtree,
                                                      self.total_domain) # this is a list of tuples [(b_id, (bounds...),), ...]

        ret_shp = utils.slice_shape(slice_, self.total_domain)
        ret_arr = np.empty(ret_shp, dtype=self.dtype)

        for b in bricks:
            bid, bbnds = b
            brick_slice, brick_mm = bricking_utils.get_brick_slice_nd(slice_, bbnds)

            if None in brick_slice:
                continue

            ret_slice = bricking_utils.get_value_slice_nd(slice_, ret_shp, bbnds, brick_slice, brick_mm)

            if not self.use_hdf:
                ret_vals = self.bricks[bid][brick_slice]
            else:
                fi = self.bricks[bid]
                with HDFLockingFile(fi) as f:
                    ds = f.require_dataset(str(bid), shape=self.brick_sizes, dtype=self.dtype, chunks=None,
                                           fillvalue=-1)
                    ret_vals = ds[brick_slice]

            ret_arr[ret_slice] = ret_vals

        ret_arr = ret_arr.squeeze()

        if ret_arr.size == 1:
            if ret_arr.ndim == 0:
                ret_arr = ret_arr[()]
            else:
                ret_arr = ret_arr[0]

        return ret_arr


def _run_test_slices(ba, sl_list, val_arr, verbose):
    if not verbose:
        from sys import stdout

    for sl in sl_list:
        tstr = '*** Slice: {0} ***'.format(sl)
        if verbose:
            print '\n' + tstr
            print 'Slice Shape: {0}'.format(utils.slice_shape(sl, ba.total_domain))

        ba.reset_bricks()
        vals = val_arr[sl]
        ba.put_values_to_bricks(sl, vals)
        vo = ba.get_values_from_bricks(sl)
        eq = np.array_equal(vals, vo)
        seq = np.array_equal(vals.squeeze(), vo)
        if not eq and not seq:
            print "\n!!!!!!!! NOT EQUAL !!!!!!!!"
            print 'vals in:\n%s' % (vals,)
            print 'vals out:\n%s' % (vo,)
        else:
            if verbose:
                print "Value Shape: {0}".format(vo.shape)
                print "Equal{0}!".format(' (w/squeeze)' if not eq else '')
            else:
                if not eq:
                    stdout.write('s')
                else:
                    stdout.write('.')

                stdout.flush()
        if verbose:
            print '\n' + '*' * len(tstr)

    print

def test_1d(slice_runner, root_dir, persist=False, verbose=False, dtype='int16'):
    if root_dir is None:
        persist = False

    shp = (100,)
    ba = BrickingAssessor(total_domain=shp, brick_size=10, use_hdf=persist, guid='1d_trial', dtype=dtype, root_dir=root_dir)
    val_arr = ba._get_numpy_array(shp)

    sl_list = []
    # Single value slices
    sl_list.append(0)
    sl_list.append(9)
    sl_list.append(39)
    sl_list.append(88)

    # List slices
    sl_list.append(([1, 2],))
    sl_list.append(([1, 4],))
    sl_list.append(([6, 9],))
    sl_list.append(([1, 2, 5, 8],))
    sl_list.append(([2, 5, 6, 9],))

    # Slice slices
    sl_list.append((slice(6, 9, 2),))
    sl_list.append((slice(2, 7),))
    sl_list.append((slice(1, None),))
    sl_list.append((slice(None),))
    sl_list.append((slice(2, 8),))
    sl_list.append(slice(2, 8))
    sl_list.append(slice(None, None, 3))
    sl_list.append(slice(1, 8, 3))
    sl_list.append(slice(3, None))
    sl_list.append(slice(None, 80, 15))
    sl_list.append(slice(0, 21, 15))

    slice_runner(ba, sl_list, val_arr, verbose)

    return ba, val_arr


def test_2d(slice_runner, root_dir, persist=False, verbose=False, dtype='int16'):
    if root_dir is None:
        persist = False

    shp = (10, 10)
    ba = BrickingAssessor(total_domain=shp, brick_size=2, use_hdf=persist, guid='2d_trial', dtype=dtype, root_dir=root_dir)
    val_arr = ba._get_numpy_array(shp)

    sl_list = []
    # Single value slices
    sl_list.append((0, 1))
    sl_list.append((9, 3))
    sl_list.append((3, 8))
    sl_list.append((8, 7))

    # List slices
    sl_list.append(([1, 2], 2))
    sl_list.append(([1, 4], 6))
    sl_list.append(([6, 9], 3))
    sl_list.append(([1, 2, 5, 8], 5))
    sl_list.append((2, [2, 5, 6, 9]))

    # Slice slices
    sl_list.append((slice(6, 9, 2),))
    sl_list.append((slice(2, 7), slice(3, 8)))
    sl_list.append((slice(1, None), slice(4, 8)))
    sl_list.append((slice(None), slice(None)))
    sl_list.append((slice(2, 8), slice(None)))
    sl_list.append((slice(2, 8), slice(3, 6)))
    sl_list.append((slice(None, None, 3), slice(None, None, 2)))
    sl_list.append((slice(1, 8, 3), slice(3, None, 2)))
    sl_list.append((slice(6, None, 10), slice(0, None, 2)))
    sl_list.append((slice(2, 6, 3), slice(3, None, 6)))
    sl_list.append((slice(3, None), slice(3, 9, 2)))

    slice_runner(ba, sl_list, val_arr, verbose)

    return ba, val_arr


def test_3d(slice_runner, root_dir, persist=False, verbose=False, dtype='int16'):
    if root_dir is None:
        persist = False

    shp = (10, 10, 10)
    ba = BrickingAssessor(total_domain=shp, brick_size=2, use_hdf=persist, guid='3d_trial', dtype=dtype, root_dir=root_dir)
    val_arr = ba._get_numpy_array(shp)

    sl_list = []
    # Single value slices
    sl_list.append((0, 1, 8))
    sl_list.append((9, 3, 2))
    sl_list.append((3, 8, 6))
    sl_list.append((8, 7, 0))

    # List slices
    sl_list.append(([1, 2], 2, 0))
    sl_list.append(([1, 4], 6, 9))
    sl_list.append(([6, 9], 3, 4))
    sl_list.append(([1, 2, 5, 8], 5, 8))
    sl_list.append((2, [2, 5, 6, 9], 3))
    sl_list.append((2, 4, [2, 5, 6, 9]))

    # Slice slices
    sl_list.append((slice(6, 9, 2),))
    sl_list.append((slice(2, 7), slice(3, 8), slice(0, 5)))
    sl_list.append((slice(1, None), slice(4, 8), slice(None, 6)))
    sl_list.append((slice(None), slice(None), slice(None)))
    sl_list.append((slice(2, 8), slice(None), slice(5, 7)))
    sl_list.append((slice(2, 8), slice(3, 6), slice(7, 8)))
    sl_list.append((slice(None, None, 3), slice(None, None, 2), slice(None, None, 4)))
    sl_list.append((slice(None, None, 3), slice(None, None, 2), slice(None, None, 7)))
    sl_list.append((slice(5, None, 6), slice(None, 8, 3), slice(None, None, 7)))
    sl_list.append((slice(1, 8, 3), slice(3, None, 2), slice(4, 9, 4)))
    sl_list.append((slice(None, None, 6), slice(None, None, None), slice(4, 9, 5)))
    sl_list.append((slice(3, None), slice(3, 9, 2), slice(None, None, 2)))

    slice_runner(ba, sl_list, val_arr, verbose)

    return ba, val_arr


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Bricking & Persistence Trials")
    parser.add_argument('-p', '--persist', help='If HDF persistence should be used, otherwise uses numpy',
                        action="store_true")
    parser.add_argument('-v', '--verbose', help='Print verbose information', action='store_true')
    parser.add_argument('-d', '--dtype', help='Data type for values', nargs=1, default='int16')
    parser.add_argument('-a', '--all', help='Run all tests', action="store_true")
    parser.add_argument('-d1', '--onedim', help='Run 1D test', action="store_true")
    parser.add_argument('-d2', '--twodim', help='Run 2D test', action="store_true")
    parser.add_argument('-d3', '--threedim', help='Run 3D test', action="store_true")

    args = parser.parse_args()

    root_dir = 'test_data/multi_dim_trials'

    if args.all:
        args.onedim = True
        args.twodim = True
        args.threedim = True

    one_picked = np.any([args.onedim, args.twodim, args.threedim])
    if args.onedim or not one_picked:
        print ">>> Start 1D Test >>>"
        test_1d(_run_test_slices, root_dir, args.persist, args.verbose, args.dtype[0])
        print "<<<< End 1D Test <<<<\n"
    if args.twodim:
        print ">>> Start 2D Test >>>"
        test_2d(_run_test_slices, root_dir, args.persist, args.verbose, args.dtype[0])
        print "<<<< End 2D Test <<<<\n"
    if args.threedim:
        print ">>> Start 3D Test >>>"
        #        print "Not built yet..."
        test_3d(_run_test_slices, root_dir, args.persist, args.verbose, args.dtype[0])
        print "<<<< End 3D Test <<<<\n"
