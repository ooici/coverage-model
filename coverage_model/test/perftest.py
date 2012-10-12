#!/usr/bin/env python

"""
@package coverage_model.test.perftest
@file coverage_model/test/perftest.py
@author Christopher Mueller, James Case
@brief Performance test functions
"""

#from netCDF4 import Dataset
from coverage_model.basic_types import *
from coverage_model.coverage import *
from coverage_model.parameter_types import *
import numpy as np
import time
import rtree

def _make_cov(brick_size=None, chunk_size=None):
    bricking_scheme = None

    if brick_size and chunk_size is not None:
        bricking_scheme = {'brick_size': brick_size, 'chunk_size': chunk_size}


    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
    t_ctxt.reference_frame = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'performance test coverage_model', pdict, temporal_domain=tdom, bricking_scheme=bricking_scheme)

    return scov

def _set_data(scov, size, origin, repeat, delay):
    timer = []
    # Expand the domain
    si=time.time()
    scov.insert_timesteps(size)
    sis=time.time()-si

    # Add data for each parameter
    loc = slice(origin, origin+size)

    for i in xrange(repeat):
        st = time.time()

        scov.set_parameter_values('quantity_time', value=np.arange(size), tdoa=loc)
        timer.append(time.time()-st)

        # Sleep for delay (seconds)
        time.sleep(delay)

    return timer, sis

def _fill_coverage(scov, size, limit):
    insert_timer = []
    expand_timer = []

    # Add data for each parameter
    origins = np.arange(limit)[1::size]-1
    for origin in origins:
        # Expand the domain
        si=time.time()
        scov.insert_timesteps(size)
        expand_timer.append(time.time()-si)

        loc = slice(origin, origin+size)
        st = time.time()

        scov.set_parameter_values('quantity_time', value=np.arange(size)+origin, tdoa=loc)
        insert_timer.append(time.time()-st)

    return insert_timer, expand_timer

def run_perf_test(size=10, origin=0, repeat=1, delay=0, brick_size=None, chunk_size=None):
    scov = _make_cov(brick_size, chunk_size)
    timer, insert_time = _set_data(scov, size, origin, repeat, delay)
    avg = sum(timer)/len(timer)
    return avg, timer, insert_time

# avg, timer, insert_time = run_perf_fill_test(size=100, limit=10000, repeat=1, delay=0, brick_size=20000, chunk_size=500)
def run_perf_fill_test(sizes=[10], limits=[300], brick_size=50, chunk_size=25):
    scov = _make_cov(brick_size, chunk_size)

    t = time.localtime()
    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
    output_file = 'perf_result-{0}.csv'.format(str_time)

    header = 'Size,Limit,Brick Size,Chunk Size,Insert Average,Expand Average\n'
    with open(output_file, 'w') as f:
        f.write(header)

        for s in sizes:
            for l in limits:
                print 'Working on: {0},{1},{2},{3}'.format(s, l, brick_size, chunk_size)
                insert_timer, expand_timer = _fill_coverage(scov, s, l)
                insert_avg = sum(insert_timer)/len(insert_timer)
                expand_avg = sum(expand_timer)/len(expand_timer)
                line = '{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(s, l, brick_size, chunk_size, insert_avg, expand_avg, insert_timer, expand_timer)
                f.write(line)

def wtf():
    brick_size = 10
    chunk_size = 5
    size = 10

    loc = slice(0, size)

    scov = _make_cov(brick_size, chunk_size)
    scov.insert_timesteps(size)
    scov.set_parameter_values('quantity_time', value=np.arange(size), tdoa=loc)
    scov.set_parameter_values('quantity', value=np.random.random_sample(size)*(26-23)+23, tdoa=loc)

# test_all(max_size=100000, max_repeat=1, max_delay=0, max_brick_size=50000, max_chunk_size=500, size_step=500, fixed_size=None, brick_step=50, fixed_bricking=1)
def test_all(max_size, max_repeat, max_delay, max_brick_size, max_chunk_size, size_step=5, fixed_size=None, brick_step=5, fixed_bricking=None):
    t = time.localtime()
    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
    output_file = 'perf_result-{0}.csv'.format(str_time)

    header = 'Origin,Size,Repeat,Delay,Brick Size,Chunk Size,Average,Raw\n'
    with open(output_file, 'w') as f:
        f.write(header)

        repeats = np.arange(max_repeat+1)[1:]
#        repeats = repeats.tolist()
        if max_delay > 0:
            delays = np.arange(max_delay+1)[1:]
#            delays = delays.tolist()
        else:
            delays = [0]

        if fixed_size is None:
            sizes = np.arange(max_size+size_step)[1::size_step][1:]-1
#            sizes = sizes.tolist()
        else:
            sizes = [max_size]

        if fixed_bricking is None:
            brick_sizes = np.arange(max_brick_size+brick_step)[1::brick_step][1:]-1
            chunk_sizes = np.arange(max_chunk_size+brick_step)[1::brick_step][1:]-1
        else:
            brick_sizes = [max_brick_size]
            chunk_sizes = [max_chunk_size]

        for s in sizes:
            for r in repeats:
                for d in delays:
                    for bs in brick_sizes:
                        for cs in chunk_sizes:
                            if cs < bs:
                                print 'Working on: {0},{1},{2},{3},{4},{5}'.format(s, size_step, r, d, bs, cs)
                                avg, timer, insert_time = run_perf_test(size=size_step, origin=s, repeat=r, delay=d, brick_size=bs, chunk_size=cs)
                                line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(s, size_step, r, d, bs, cs, avg, timer, insert_time)
                                f.write(line)

def test1(sizes, max_repeat, max_delay, brick_sizes, chunk_sizes):
    t = time.localtime()
    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
    output_file = 'perf_result-{0}.csv'.format(str_time)

    header = 'Origin,Size,Repeat,Delay,Brick Size,Chunk Size,Average,Raw\n'
    with open(output_file, 'w') as f:
        f.write(header)

        repeats = np.arange(max_repeat+1)[1:]
        if max_delay > 0:
            delays = np.arange(max_delay+1)[1:]
        else:
            delays = [0]
        origin = 0
        for s in sizes:
            for r in repeats:
                for d in delays:
                    for bs in brick_sizes:
                        for cs in chunk_sizes:
                            if cs < bs:
                                print 'Working on: {0},{1},{2},{3},{4}'.format(s, r, d, bs, cs)
                                avg, timer, insert_time = run_perf_test(size=s, origin=origin, repeat=r, delay=d, brick_size=bs, chunk_size=cs)
                                line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(origin, s, r, d, bs, cs, avg, timer, insert_time)
                                f.write(line)


def rt():
    p = rtree.index.Property()
    p.dimension = 2
    brick_tree = rtree.index.Index(properties=p)
    brick_tree.insert(0,(0,0,500,0),obj='GUID1')
    start = [0,0]
    stop = [500,0]
    st1 = time.time()
    hits = list(brick_tree.intersection(tuple(start+stop), objects=True))
    stp1 = time.time() - st1

    brick_tree.insert(1,(500,0,1000,0),obj='GUID2')
    start = [500,0]
    stop = [1000,0]
    st2 = time.time()
    hits = list(brick_tree.intersection(tuple(start+stop), objects=True))
    stp2 = time.time() - st2
    return stp1, stp2