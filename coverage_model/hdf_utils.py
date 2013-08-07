#!/usr/bin/env python

"""
@package coverage_model.hdf_utils
@file coverage_model/hdf_utils.py
@author Christopher Mueller
@brief Utility functions wrapping various HDF5 binaries
"""

import os
import re
import shutil
import subprocess
import StringIO
import fcntl
import h5py
import gevent.coros


def repack(infile_path, outfile_path=None):
    if not os.path.exists(infile_path):
        raise IOError('Input file does not exist: \'{0}\''.format(infile_path))

    replace = False
    if outfile_path is None:
        replace = True
        outfile_path = infile_path + '_out'

    try:
        subprocess.check_output(['h5repack', infile_path, outfile_path])
        if replace:
            os.remove(infile_path)
            shutil.move(outfile_path, infile_path)
    except subprocess.CalledProcessError:
        if os.path.exists(outfile_path):
            os.remove(outfile_path)
        raise


def space_ratio(infile_path):
    if not os.path.exists(infile_path):
        raise IOError('Input file does not exist: \'{0}\''.format(infile_path))

    #_metadata = r'File metadata: (\d*) bytes'
    _unaccounted = r'Unaccounted space: (\d*) bytes'
    #_raw_data = r'Raw data: (\d*) bytes'
    _total = r'Total space: (\d*) bytes'
    try:
        output = subprocess.check_output(['h5stat', '-S', infile_path])
        #meta = float(re.search(_metadata, output).group(1))
        unaccounted = float(re.search(_unaccounted, output).group(1))
        #raw = float(re.search(_raw_data, output).group(1))
        total = float(re.search(_total, output).group(1))

        return unaccounted/total
    except subprocess.CalledProcessError:
        raise


def dump(infile_path):
    if not os.path.exists(infile_path):
        raise IOError('Input file does not exist: \'{0}\''.format(infile_path))

    try:
        subprocess.check_output(['h5dump', '-BHA', infile_path], stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        raise


def has_corruption(infile_path):
    if not os.path.exists(infile_path):
        raise IOError('Input file does not exist: \'{0}\''.format(infile_path))

    try:
        # Try dumping the file - most read corruptions can be found this way
        _=dump(infile_path)
    except subprocess.CalledProcessError:
        return True

    # Other mechanisms for detecting corruption?



    return False

class HDFLockingFile(h5py.File):
    __locks={}
    __rlock=gevent.coros.RLock()

    def __init__(self, name, mode=None, driver=None, 
                 libver=None, userblock_size=None, **kwds):
        h5py.File.__init__(self, name, mode=mode, driver=driver, libver=libver, 
                userblock_size=userblock_size, **kwds)

        self.lock()

    def lock(self):
        with self.__rlock:
            if self.driver == 'sec2' and self.mode != 'r':
                if self.filename in self.__locks:
                    raise IOError('[Errno 11] Resource temporarily unavailable')

                self.__locks[self.filename] = 1 

                # Using sec2 and not reading
                fd = self.fid.get_vfd_handle()
                # Lock the file
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def unlock(self):
        with self.__rlock:
            if self.driver == 'sec2' and self.mode != 'r':
                fd = self.fid.get_vfd_handle()
                fcntl.flock(fd, fcntl.LOCK_UN | fcntl.LOCK_NB)
                del self.__locks[self.filename]

    def close(self):
        self.unlock()

        h5py.File.close(self)





