#!/usr/bin/env python

"""
@package coverage_model.brick_dispatch
@file coverage_model/brick_dispatch.py
@author Christopher Mueller
@brief Module containing classes for delegating the writing of values to persistence bricks using a pool of writer processes
"""

import os

from pyon.util.async import spawn
from coverage_model.basic_types import create_guid

from ooi.logging import log
from gevent_zeromq import zmq
from gevent import queue, coros
import time
import random
from pyon.core.interceptor.encode import encode_ion, decode_ion
from msgpack import packb, unpackb
import numpy as np
import h5py

REQUEST_WORK = 'REQUEST_WORK'
SUCCESS = 'SUCCESS'
FAILURE = 'FAILURE'
PROVISIONER_ENDPOINT = 'tcp://localhost:5071'
RESPONDER_ENDPOINT = 'tcp://localhost:5566'

BASE_DIR = 'test_data/masonry'
WORK_KEYS = ['a','b','c','d','e']

def pack(msg):
    return packb(msg, default=encode_ion)

def unpack(msg):
    return unpackb(msg, object_hook=decode_ion)

class BrickWriterDispatcher(object):

    _original = False

    def __init__(self):
        self.prep_queue = queue.Queue()
        self.work_queue = queue.Queue()
        self._pending_work = {}
        self._stashed_work = {}
        self._active_work = {}
        self._do_stop = False
        self._count = 0
        self._active_work_lock = coros.RLock()
        self._pending_work_lock = coros.RLock()

        self.context = zmq.Context(1)
        self.prov_sock = self.context.socket(zmq.REP)
        self.prov_sock.bind('tcp://*:5071')

        self.sub_sock = self.context.socket(zmq.SUB)
        self.sub_sock.bind('tcp://*:5566')
        self.sub_sock.setsockopt(zmq.SUBSCRIBE, '')

    def run(self):
        self._do_stop = False
        spawn(self.organize_work)

        spawn(self.provisioner) # zmq style workers
        spawn(self.receiver)

    def stop(self):
        self._do_stop = True

    def organize_work(self):
        while True:
            if self._do_stop and self.prep_queue.empty():
                break
            try:
                # Timeout after 1 second to allow stopage and _stashed_work cleanup
                wd = self.prep_queue.get(timeout=1)
                k, wm, w = wd
                is_list = isinstance(w, list)

                if k not in self._stashed_work and len(w) == 0:
                    log.warn('Discarding empty work')
                    continue

                log.warn('Work: %s',w)

                is_active = False
                with self._active_work_lock:
                    is_active = k in self._active_work

                if is_active:
                    log.warn('Do Stash')
                    # The work_key is being worked on
                    if k not in self._stashed_work:
                        # Create the stash for this work_key
                        self._stashed_work[k] = (wm, [])

                    # Add the work to the stash
                    if is_list:
                        self._stashed_work[k][1].extend(w[:])
                    else:
                        self._stashed_work[k][1].append(w)
                else:
                    # If there is a stash for this work_key, prepend it to work
                    if k in self._stashed_work:
                        log.warn('Was a stash, prepend: %s, %s', self._stashed_work[k], w)
                        _, sv=self._stashed_work.pop(k)
                        if is_list:
                            sv.extend(w[:])
                        else:
                            sv.append(w)
                        w = sv

                    log.warn('Work: %s',w)

                    # The work_key is not yet pending
                    with self._pending_work_lock:
                        not_in_pend = k not in self._pending_work

                        if not_in_pend:
                            # Create the pending for this work_key
                            log.debug('-> new pointer \'%s\'', k)
                            self._pending_work[k] = (wm,[])

                        # Add the work to the pending
                        log.debug('-> adding work to \'%s\': %s', k, w)
                        if is_list:
                            self._pending_work[k][1].extend(w[:])
                        else:
                            self._pending_work[k][1].append(w)

                        if not_in_pend:
                            # Add the not-yet-pending work to the work_queue
                            self.work_queue.put(k)
            except queue.Empty:
                # No new work added - see if there's anything on the stash to cleanup...
                for k in self._stashed_work:
                    log.warn('Cleanup _stashed_work...')
                    # Just want to trigger cleanup of the _stashed_work, pass an empty list of 'work', gets discarded
                    self.put_work(k, None, [])


    def put_work(self, work_key, work_metrics, work):
        log.debug('<<< put work for %s: %s', work_key, work)
        self.prep_queue.put((work_key, work_metrics, work))

    def receiver(self):
        while not self._do_stop:
            resp_type, worker_guid, work_key, work = unpack(self.sub_sock.recv())
            work = list(work) if work is not None else work
            if resp_type == SUCCESS:
                log.debug('Worker %s was successful', worker_guid)
                with self._active_work_lock:
                    self._active_work.pop(work_key)
            elif resp_type == FAILURE:
                log.debug('===> FAILURE reported for work on %s', work_key)
                if work_key is None:
                    # Worker failed before it did anything, put all work back on the prep queue to be reorganized by the organizer
                    with self._active_work_lock:
                        # Because it failed so miserably, need to find the work_key based on guid
                        for k, v in self._active_work.iteritems():
                            if v[0] == worker_guid:
                                work_key = k
                                break

                        if work_key is not None:
                            wguid, wp = self._active_work.pop(work_key)
                            self.put_work(work_key, wp[0], wp[1])
                else:
                    # Normal failure
                    with self._active_work_lock:
                        # Pop the work from active work, and queue the work returned by the worker
                        wguid, wp = self._active_work.pop(work_key)
                        self.put_work(work_key, wp[0], work)

    def provisioner(self):
        while not self._do_stop:
            _, worker_guid = unpack(self.prov_sock.recv())
            work_key = self.work_queue.get()
            log.debug('===> assign work for %s', work_key)
            with self._pending_work_lock:
                work_metrics, work = self._pending_work.pop(work_key)

            with self._active_work_lock:
                self._active_work[work_key] = (worker_guid, (work_metrics, work))

            wp = (work_key, work_metrics, work)
            log.debug('===> assigning to %s: %s', work_key, wp)
            self.prov_sock.send(pack(wp))

class BrickWriterWorker(object):

    def __init__(self, name=None):
        self.name=name or create_guid()
        self.context = zmq.Context(1)

        # Socket to get work from provisioner
        self.req_sock = self.context.socket(zmq.REQ)
        self.req_sock.connect(PROVISIONER_ENDPOINT)

        # Socket to respond to responder
        self.resp_sock = self.context.socket(zmq.PUB)
        self.resp_sock.connect(RESPONDER_ENDPOINT)

        self._do_stop = False

    def stop(self):
        self._do_stop = True

    def start(self):
        self._do_stop = False
        guid = create_guid()
        spawn(self._run, guid)
        log.debug('worker \'%s\' started', guid)

    def _run(self, guid):
        while not self._do_stop:
            try:
                log.debug('%s making work request', guid)
                self.req_sock.send(pack((REQUEST_WORK, guid)))
                brick_key, brick_metrics, work = unpack(self.req_sock.recv())
                work=list(work) # lists decode as a tuples
                try:
                    log.warn('*%s*%s* got work for %s: %s', time.time(), guid, brick_key, work)
                    brick_path, bD, cD, data_type, fill_value = brick_metrics
                    with h5py.File(brick_path, 'a') as f:
                        f.require_dataset(brick_key, shape=bD, dtype=data_type, chunks=cD, fillvalue=fill_value)
                        for w in list(work): # Iterate a copy - WARN, this is NOT deep, if the list contains objects, they're NOT copied
                            brick_slice, value = w
                            if isinstance(brick_slice, tuple):
                                brick_slice = list(brick_slice)

                            log.error('slice_=%s, value=%s', brick_slice, value)
                            f[brick_key].__setitem__(*brick_slice, val=value)
                            # Remove the work AFTER it's completed (i.e. written)
                            work.remove(w)
                    log.debug('*%s*%s* done working on %s', time.time(), guid, brick_key)
                    self.resp_sock.send(pack((SUCCESS, guid, brick_key, None)))
                except Exception as ex:
                    log.debug('Exception: %s', ex.message)
                    log.debug('%s send failure response with work %s', guid, work)
                    # TODO: Send the remaining work back
                    self.resp_sock.send(pack((FAILURE, guid, brick_key, work)))
                    time.sleep(0.001)
            except Exception as ex:
                log.debug('Exception: %s', ex.message)
                log.debug('%s send failure response with work %s', guid, None)
                # TODO: Send a response - I don't know what I was working on...
                self.resp_sock.send(pack((FAILURE, guid, None, None)))
                time.sleep(0.001)


def run_test_dispatcher(work_count):
    for x in os.listdir(BASE_DIR):
        os.remove(os.path.join(BASE_DIR,x))

    fps = {}
    for k in WORK_KEYS:
        fps[k] = os.path.join(BASE_DIR, '{0}.h5'.format(k))
#        with h5py.File(fps[k], 'a'):
#            pass

    bD = (50,)
    cD = (5,)
    fv = -9999
    dtype = 'f'

    disp = BrickWriterDispatcher()
    disp.run()

    def make_work():
        for x in xrange(work_count):
            bk = random.choice(WORK_KEYS)
            brick_metrics = (fps[bk], bD, cD, dtype, fv)
            if np.random.random_sample(1)[0] > 0.5:
                sl = int(np.random.randint(0,10,1)[0])
                w = np.random.random_sample(1)[0]
            else:
                strt = int(np.random.randint(0,bD[0] - 2,1)[0])
                stp = int(np.random.randint(strt+1,bD[0],1)[0])
                sl = slice(strt, stp)
                w = np.random.random_sample(stp-strt)
            disp.put_work(work_key=bk, work_metrics=brick_metrics, work=([sl], w))
            time.sleep(0.1)

    spawn(make_work)

    return disp

def run_test_worker():
    worker = BrickWriterWorker()
    worker.start()
    return worker


"""
from coverage_model.brick_dispatch import run_test_dispatcher;
disp=run_test_dispatcher(20)

------------

from coverage_model.brick_dispatch import run_test_worker;
worker=run_test_worker()
"""