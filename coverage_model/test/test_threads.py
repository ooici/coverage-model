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
import os
import numpy as np
import time
import gevent
import coverage_model




@attr('UNIT',group='cov')
class TestThreads(coverage_model.CoverageModelUnitTestCase):
    
    filepath = '/tmp/test_threads.hdf'
    def setUp(self):
        self.remove_file()
    
    
    def block_stuff(self):
        self.remove_file()
        with h5py.File(self.filepath) as f:
            ds = f.require_dataset('test_ds', shape=(5000, 10000), dtype='float32', chunks=None)
            ds[:] = np.arange(5000*10000).reshape(5000,10000)
        self.remove_file()
        return np.arange(20)

    def remove_file(self):
        if os.path.exists(self.filepath):
            os.remove(self.filepath)

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


