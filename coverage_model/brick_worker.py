#!/usr/bin/env python

"""
@package 
@file brick_worker
@author Christopher Mueller
@brief 
"""

from pyon.util.async import spawn
from ooi.logging import log, config
import logging
from coverage_model.brick_dispatch import pack, unpack, FAILURE, REQUEST_WORK, SUCCESS
from coverage_model.utils import create_guid
from gevent_zeromq import zmq
import h5py
import time
import sys

class BrickWriterWorker(object):

    def __init__(self, req_port, resp_port, name=None):
        self.name=name or create_guid()
        self.context = zmq.Context(1)

        # Socket to get work from provisioner
        self.req_sock = self.context.socket(zmq.REQ)
        self.req_port = req_port
        self.req_sock.connect('tcp://localhost:{0}'.format(self.req_port))

        # Socket to respond to responder
        self.resp_sock = self.context.socket(zmq.PUB)
        self.resp_port = resp_port
        self.resp_sock.connect('tcp://localhost:{0}'.format(self.resp_port))

        self._do_stop = False

    def stop(self):
        self._do_stop = True
        self._g.join()
        self.req_sock.close()
        self.resp_sock.close()
        log.debug('Terminating the context')
        self.context.term()
        log.debug('Context terminated')

    def start(self):
        self._do_stop = False
        self._g=spawn(self._run, self.name)
        log.info('Brick writer worker \'%s\' started: req_port=%s, resp_port=%s', self.name, 'tcp://localhost:{0}'.format(self.req_port), 'tcp://localhost:{0}'.format(self.resp_port))
        return self._g

    def _run(self, guid):
        while not self._do_stop:
            try:
                log.debug('%s making work request', guid)
                self.req_sock.send(pack((REQUEST_WORK, guid)))
                msg = None
                while msg is None:
                    try:
                        msg = self.req_sock.recv(zmq.NOBLOCK)
                    except zmq.ZMQError, e:
                        if e.errno == zmq.EAGAIN:
                            if self._do_stop:
                                break
                            else:
                                time.sleep(0.1)
                        else:
                            raise

                if msg is not None:
                    brick_key, brick_metrics, work = unpack(msg)
                    work=list(work) # lists decode as a tuples
                    try:
                        log.debug('*%s*%s* got work for %s, metrics %s: %s', time.time(), guid, brick_key, brick_metrics, work)
                        brick_path, bD, cD, data_type, fill_value = brick_metrics
                        if data_type == '|O8':
                            data_type = h5py.special_dtype(vlen=str)
                        # TODO: Uncomment this to properly turn 0 & 1 chunking into True
#                        if 0 in cD or 1 in cD:
#                            cD = True
                        with h5py.File(brick_path, 'a') as f:
                            # TODO: Due to usage concerns, currently locking chunking to "auto"
                            f.require_dataset(brick_key, shape=bD, dtype=data_type, chunks=True, fillvalue=fill_value)
                            for w in list(work): # Iterate a copy - WARN, this is NOT deep, if the list contains objects, they're NOT copied
                                brick_slice, value = w
                                if isinstance(brick_slice, tuple):
                                    brick_slice = list(brick_slice)

                                log.debug('slice_=%s, value=%s', brick_slice, value)
                                f[brick_key].__setitem__(*brick_slice, val=value)
                                # Remove the work AFTER it's completed (i.e. written)
                                work.remove(w)
                        log.debug('*%s*%s* done working on %s', time.time(), guid, brick_key)
                        self.resp_sock.send(pack((SUCCESS, guid, brick_key, None)))
                    except Exception as ex:
                        log.error('Exception: %s', ex.message)
                        log.warn('%s send failure response with work %s', guid, work)
                        # TODO: Send the remaining work back
                        self.resp_sock.send(pack((FAILURE, guid, brick_key, work)))
            except Exception as ex:
                log.error('Exception: %s', ex.message)
                log.error('%s send failure response with work %s', guid, None)
                # TODO: Send a response - I don't know what I was working on...
                self.resp_sock.send(pack((FAILURE, guid, None, None)))
            finally:
#                time.sleep(0.1)
                pass


def run_worker(req_port, resp_port):
    worker = BrickWriterWorker(req_port, resp_port)
    worker.start()
    return worker

"""
from coverage_model.brick_dispatch import run_test_worker;
worker=run_test_worker()

"""

def main(args=None):
    # Configure logging
    def_log_paths = ['res/config/logging.yml', 'res/config/logging.local.yml']
    for path in def_log_paths:
        try:
            config.add_configuration(path)
        except Exception, e:
            print 'WARNING: could not load logging configuration file %s: %s' % (path, e)

    # direct warnings mechanism to loggers
    logging.captureWarnings(True)

    args = args or sys.argv[:]
    if len(args) < 3:
        raise SystemError('Must provide request_port and response_port arguments')

    worker = BrickWriterWorker(args[1], args[2])
    g = worker.start()
    log.info('Worker %s started successfully', worker.name)

    # Doesn't work because it hits the g.join() below and hangsout...signal seems to be caught by the greenlet...
#    def signal_handler(signal, frame):
#        worker.stop()
#        g.join()
#        return 1
#    # Configure ctrl+C capture
#    signal.signal(signal.SIGINT, signal_handler)

    # Waits until the glet is finshed
    g.join()
    return 0


if __name__ == "__main__":
    sys.exit(main())