#!/usr/bin/env python
'''
@author Luke Campbell <LCampbell at ASAScience dot com>
@file coverage_model/threads/sync.py
'''

import os
import fcntl
import gevent

_pythread = None

def get_pythread():
    '''
    Loads the thread module without monkey patching
    source: https://github.com/nimbusproject/kazoo/blob/master/kazoo/sync/util.py
    '''
    global _pythread
    if _pythread:
        return _pythread
    import imp
    fp, path, desc = imp.find_module('thread')
    try:
        _pythread = imp.load_module('pythread',fp,path,desc)
    finally:
        if fp:
            fp.close()

    return _pythread


class _Event(gevent.event.Event):
    '''
    A gevent-friendly event to be signaled by os thread and respected by 
    the gevent context manager
    '''

    def __init__(self):
        gevent.event.Event.__init__(self)
        self._r, self._w = self._pipe()

        # Create a new gevent core object that will observe the
        # nonblocking pipe file descriptor for a value
        self._core_event = gevent.core.event(
                    gevent.core.EV_READ | gevent.core.EV_PERSIST,
                    self._r, 
                    self._pipe_read)
        self._core_event.add()

    def _pipe(self):
        r,w = os.pipe()
        fcntl.fcntl(r, fcntl.F_SETFD, os.O_NONBLOCK)
        fcntl.fcntl(w, fcntl.F_SETFD, os.O_NONBLOCK)
        return r,w

    def _pipe_read(self, event, eventtype):
        '''
        Non blocking gevent-friendly core event callback 
        http://www.gevent.org/gevent.core.html#events
        '''
        try:
            os.read(event.fd,1)
        except EnvironmentError:
            # file descriptors set with O_NONBLOCK return -1 and set errno to 
            # EAGAIN, we just want to ignore it and try again later
            pass

    def set(self):
        '''
        Sets the event value and writes a value to the pipe to
        trigger the gevent core event
        '''
        gevent.event.Event.set(self)
        os.write(self._w, '\0')



class AsyncDispatcher(object):
    '''
    Used to synchronize a result obtained in a pythread to a gevent thread
    '''
    _value     = None
    _exception = None
    _set       = None

    def __init__(self, callback, *args, **kwargs):

        self.event = _Event()
        pythread = get_pythread()
        self._thread = pythread.start_new_thread(self.dispatch,(callback,) + args,kwargs)
        

    def dispatch(self, callback, *args, **kwargs):
        '''
        Runs a callback, either sets an asynchronous value or an exception
        When the os thread completes, the event is set to signal gevent that it's complete
        '''
        try:
            retval = callback(*args, **kwargs)
            self._value = retval
        except Exception as e:
            self._exception = e
        self.event.set()

    def wait(self,timeout=None):
        '''
        Blocks the current gevent greenlet until a value is set or the timeout expires.
        If the timeout expires a Timeout is raised.
        If the callback raised an exception in the os thread, an exception is raised here.
        '''
        if self.event.wait(timeout):
            if self._exception:
                raise self._exception
            else:
                return self._value
        else:
            raise gevent.timeout.Timeout(timeout)



