#!/usr/bin/env python
'''
@author Luke Campbell <LCampbell at ASAScience dot com>
@file coverage_model/test/test_threads.py
@date Tue Feb  5 10:29:12 EST 2013
'''

from nose.plugins.attrib import attr

from gevent.monkey import patch_all
patch_all()

from coverage_model.threads import AsyncDispatcher

import h5py
from coverage_model.hdf_utils import HDFLockingFile
import os
import numpy as np
import time
import gevent
import coverage_model
import unittest



@attr('UNIT',group='cov')
class TestThreads(coverage_model.CoverageModelUnitTestCase):
    
    filepath = '/tmp/test_threads.hdf'
    def setUp(self):
        self.remove_file()
    
    
    def block_stuff(self):
        self.remove_file()
        with HDFLockingFile(self.filepath, 'w') as f:
            ds = f.require_dataset('test_ds', shape=(5000, 10000), dtype='float32', chunks=None)
            ds[:] = np.arange(5000*10000).reshape(5000,10000)
        self.remove_file()
        return np.arange(20)

    def remove_file(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

    @unittest.skip("Deprecated and it uses HDF5, bad example")
    def test_gevent_friendly(self):

        # Used to verify that file descriptors aren't consumed
        r,w = os.pipe()
        os.close(r)
        os.close(w)

        # Get a good benchmark without any concurrent actions
        t1 = time.time()
        with AsyncDispatcher(self.block_stuff) as dispatcher:
            v = dispatcher.wait(10)
        dt = time.time() - t1

        # Check that it takes less than 5 seconds and that it's the right value
        self.assertTrue(dt < 10)
        self.assertTrue(np.array_equal(v,np.arange(20)))

        # Try it again but this time with a gevent sleep that should run
        # Concurrently with the dispatcher thread
        t1 = time.time()
        with AsyncDispatcher(self.block_stuff) as dispatcher:
            gevent.sleep(5)
            v = dispatcher.wait(10)
        ndt = time.time() - t1

        # There is ususally some difference but should definitely be less than
        # one second
        self.assertTrue( abs(dt - ndt) < 5)

        try:
            # Make sure we're not losing file descriptors to maintain thread synchronization
            self.assertEquals((r,w), os.pipe())
        finally:
            os.close(r)
            os.close(w)

    def load_clib(self):
        from ctypes import cdll
        try:
            clib = cdll.LoadLibrary('libc.so')
        except OSError as e:
            if 'image not found' in e.message:
                clib = cdll.LoadLibrary('libc.dylib')
            else:
                raise
        return clib

    def clib_timeout(self):
        clib = self.load_clib()
        clib.sleep(5)


    def test_true_block(self):
        t0 = time.time()
        self.clib_timeout()
        t1 = time.time()
        self.assertTrue((t1 - t0) >= 5)

        t0 = time.time()
        g = gevent.spawn(self.clib_timeout)
        gevent.sleep(5)
        g.join()
        t1 = time.time()
        # If it was concurrent delta-t will be less than 10
        self.assertTrue((t1 - t0) >= 10)

        # Syncing with gevent
        t0 = time.time()
        with AsyncDispatcher(self.clib_timeout) as dispatcher:
            gevent.sleep(5)
            dispatcher.wait(10)
        t1 = time.time()

        # Proof that they run concurrently and the clib call doesn't block gevent
        self.assertTrue((t1 - t0) < 6)

