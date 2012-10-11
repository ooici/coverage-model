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

import tempfile
from pidantic.supd.pidsupd import SupDPidanticFactory
from pidantic.state_machine import PIDanticState

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

    def __init__(self, num_workers=1, pidantic_dir=None, working_dir=None, worker_cmd=None):
        self.guid = create_guid()
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

        self.num_workers = num_workers if num_workers > 0 else 1
        self.is_single_worker = self.num_workers == 1
        self.working_dir = working_dir or '.'
        self.worker_cmd = worker_cmd or 'bin/python coverage_model/brick_worker.py'
        self.pidantic_dir = pidantic_dir or './pid_dir'
        self.workers = []

        self._configure_workers()

    def _configure_workers(self):
        # TODO: if num_workers == 1, simply run one in-line (runs in a greenlet anyhow)
        if self.is_single_worker:
            from brick_worker import run_worker
            worker = run_worker()
            self.workers.append(worker)
        else:
            if os.path.exists(self.pidantic_dir):
                bdp = os.path.join(self.pidantic_dir, 'brick_dispatch')
                if os.path.exists(bdp):
                    import zipfile, zlib
                    with zipfile.ZipFile(os.path.join(bdp, 'archived_worker_logs.zip'), 'a', zipfile.ZIP_DEFLATED) as f:
                        names = f.namelist()
                        for x in [x for x in os.listdir(bdp) if x.startswith('worker_') and x not in names]:
                            fn = os.path.join(bdp, x)
                            f.write(filename=fn, arcname=x)
                            os.remove(fn)

            else:
                os.makedirs(self.pidantic_dir)

            self.factory = SupDPidanticFactory(name='brick_dispatch', directory=self.pidantic_dir)
            # Check for old workers - FOR NOW, TERMINATE THEM TODO: These should be reusable...
            old_workers = self.factory.reload_instances()
            for x in old_workers:
                old_workers[x].cleanup()

            for x in xrange(self.num_workers):
                w = self.factory.get_pidantic(command=self.worker_cmd, process_name='worker_{0}'.format(x), directory=os.path.realpath(self.working_dir))
                w.start()
                self.workers.append(w)

            ready=False
            while not ready:
                self.factory.poll()
                for x in self.workers:
                    s = x.get_state()
                    if s is PIDanticState.STATE_STARTING:
                        break
                    elif s is PIDanticState.STATE_RUNNING:
                        continue
                    elif s is PIDanticState.STATE_EXITED:
                        self.stop()
                        raise SystemError('Error starting worker - cannot continue')
                    else:
                        raise SystemError('Problem starting worker - cannot continue')

                ready = True

    def run(self):
        self._do_stop = False
        spawn(self.organize_work)

        spawn(self.provisioner) # zmq style workers
        spawn(self.receiver)

    def stop(self):
        self._do_stop = True
        if self.is_single_worker:
            self.workers[0].stop()
        else:
            self.workers = self.factory.reload_instances()
            for x in self.workers:
                self.workers[x].cleanup()
            self.factory.terminate()

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

                log.debug('Work: %s',w)

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
                        is_list = True # Work is a list going forward!!

                    log.debug('Work: %s',w)

                    # The work_key is not yet pending
                    with self._pending_work_lock:
                        not_in_pend = k not in self._pending_work

                        if not_in_pend:
                            # Create the pending for this work_key
                            log.debug('-> new pointer \'%s\'', k)
                            self._pending_work[k] = (wm, [])

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
                    self.put_work(k, self._stashed_work[k][0], [])


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
                log.debug('===> FAILURE reported for work on %s by worker %s', work_key, worker_guid)
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
            time.sleep(0.3)
            log.debug('===> assigning to %s: %s', work_key, wp)
            self.prov_sock.send(pack(wp))

    def __del__(self):
        self.stop()

def run_test_dispatcher(work_count, num_workers=1):
    for x in [x for x in os.listdir(BASE_DIR) if x.endswith('.h5')]:
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

    disp = BrickWriterDispatcher(num_workers=num_workers, pidantic_dir='test_data/pid')
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

"""
from coverage_model.brick_dispatch import run_test_dispatcher;
disp=run_test_dispatcher(20)

"""

"""
https://github.com/nimbusproject/pidantic
https://github.com/nimbusproject/pidantic/blob/master/pidantic/nosetests/piddler_supd_basic_test.py
https://github.com/nimbusproject/epuharness/blob/master/epuharness/harness.py
https://github.com/nimbusproject/eeagent/blob/master/eeagent/execute.py#L265
https://github.com/nimbusproject/eeagent/blob/master/eeagent/execute.py#L290
"""