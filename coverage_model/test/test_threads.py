#!/usr/bin/env python
'''
@author Luke Campbell <LCampbell at ASAScience dot com>
@file coverage_model/test/test_threads.py
@date Tue Feb  5 10:29:12 EST 2013
'''

from pyon.util.unit_test import PyonTestCase
from nose.plugins.attrib import attr

from gevent.monkey import patch_all
patch_all()

from coverage_model.threads import AsyncDispatcher

import h5py
import os
import numpy as np
import time
import gevent




@attr('UNIT',group='cov')
class TestThreads(PyonTestCase):
    
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

        t1 = time.time()
        dispatcher = AsyncDispatcher(self.block_stuff)
        v = dispatcher.wait(5)
        dt = time.time() - t1

        self.assertTrue(dt < 5)
        self.assertTrue(np.array_equal(v,np.arange(20)))

        t1 = time.time()
        dispatcher = AsyncDispatcher(self.block_stuff)
        gevent.sleep(1)
        v = dispatcher.wait(5)
        ndt = time.time() - t1

        self.assertTrue( abs(dt - ndt) < 1)


