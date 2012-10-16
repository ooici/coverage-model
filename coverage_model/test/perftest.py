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

all_dtypes = ['bool','int','int8','int16','int32','int64','uint8','uint16','uint32','uint64','float32','float64']
#all_dtypes = ['float16', 'complex', 'complex64','complex128', 'complex256']  # NOT SUPPORTED - will raise an error within the coverage_model

def _make_cov(brick_size=None, chunk_size=None, dtype='int64'):
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
    t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype(dtype)), variability=VariabilityEnum.TEMPORAL)
    t_ctxt.reference_frame = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'performance test coverage_model', pdict, temporal_domain=tdom, bricking_scheme=bricking_scheme)

    return scov

def _fill(scov, size, limit):
    insert_timer = []
    expand_timer = []

    # Add data for each parameter
    origins = np.arange(limit)[1::size]-1
    for origin in origins:

        # Expand the domain
        si=time.time()
        scov.insert_timesteps(size)
        expand_timer.append(time.time()-si)

        upper_bnd = origin + size
        if upper_bnd > limit:
            upper_bnd = limit
        loc = slice(origin, upper_bnd)
        st = time.time()

        scov.set_parameter_values('quantity_time', value=np.arange(size)+origin, tdoa=loc)
        insert_timer.append(time.time()-st)

    time.sleep(0.5)

    return insert_timer, expand_timer

# run_perf_fill_test(sizes=[1,5,10,15,20,25], limit=300, brick_size=100, chunk_size=25, dtypes=['int64'])
def run_perf_fill_test(sizes=[100], limit=86400, brick_size=10000, chunk_size=1000, dtypes=all_dtypes):
    t = time.localtime()
    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
    output_file = 'perf_result-{0}.csv'.format(str_time)

    scov_dict = {}

    header = 'Path,Dtype,Data Write Size,Max Data Points,Brick Size,Chunk Size,Insert Average,Expand Average\n'
    with open(output_file, 'w') as f:
        f.write(header)

        for dtype in dtypes:
            for size in sizes:
                print 'Working on: {0},{1},{2},{3},{4}'.format(dtype, size, limit, brick_size, chunk_size)
                scov = _make_cov(brick_size, chunk_size, dtype=dtype)
                pl_scov = scov._persistence_layer
                scov_path = pl_scov.master_manager.root_dir
                insert_timer, expand_timer = _fill(scov, size, limit)
                insert_avg = sum(insert_timer)/len(insert_timer)
                expand_avg = sum(expand_timer)/len(expand_timer)
                line = '{0},{1},{2},{3},{4},{5},{6},{7}\n'.format(scov_path, dtype, size, limit, brick_size, chunk_size, insert_avg, expand_avg)
                f.write(line)
                print 'Folder Size for {0}: {1}'.format(scov_path, size_dir(scov_path))
                scov_dict[dtype] = scov_path

    time.sleep(10)
    
    for s in scov_dict:
        print 'Folder Size for dtype ({2}) at location {0}: {1}'.format(scov_dict[s], size_dir(scov_dict[s]), s)

    return scov_dict

def size_dir(d):
    import os
    from os.path import join, getsize
    s = 0
    for root, dirs, files in os.walk(d):
        s = s + sum(getsize(join(root, name)) for name in files)
    return s

#def getFolderSize(folder):
#    total_size = os.path.getsize(folder)
#    for item in os.listdir(folder):
#        itempath = os.path.join(folder, item)
#        if os.path.isfile(itempath):
#            total_size += os.path.getsize(itempath)
#        elif os.path.isdir(itempath):
#            total_size += getFolderSize(itempath)
#    return total_size

#def run_perf_test(size=10, origin=0, repeat=1, delay=0, brick_size=None, chunk_size=None):
#    scov = _make_cov(brick_size, chunk_size)
#    timer, insert_time = _set_data(scov, size, origin, repeat, delay)
#    avg = sum(timer)/len(timer)
#    return avg, timer, insert_time
#
#def _set_data(scov, size, origin, repeat, delay):
#    timer = []
#    # Expand the domain
#    si=time.time()
#    scov.insert_timesteps(size)
#    sis=time.time()-si
#
#    # Add data for each parameter
#    loc = slice(origin, origin+size)
#
#    for i in xrange(repeat):
#        st = time.time()
#
#        scov.set_parameter_values('quantity_time', value=np.arange(size), tdoa=loc)
#        timer.append(time.time()-st)
#
#        # Sleep for delay (seconds)
#        time.sleep(delay)
#
#    return timer, sis
#
## test_all(max_size=100000, max_repeat=1, max_delay=0, max_brick_size=50000, max_chunk_size=500, size_step=500, fixed_size=None, brick_step=50, fixed_bricking=1)
#def test_all(max_size, max_repeat, max_delay, max_brick_size, max_chunk_size, size_step=5, fixed_size=None, brick_step=5, fixed_bricking=None):
#    t = time.localtime()
#    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
#    output_file = 'perf_result-{0}.csv'.format(str_time)
#
#    header = 'Origin,Size,Repeat,Delay,Brick Size,Chunk Size,Average,Raw\n'
#    with open(output_file, 'w') as f:
#        f.write(header)
#
#        repeats = np.arange(max_repeat+1)[1:]
##        repeats = repeats.tolist()
#        if max_delay > 0:
#            delays = np.arange(max_delay+1)[1:]
##            delays = delays.tolist()
#        else:
#            delays = [0]
#
#        if fixed_size is None:
#            sizes = np.arange(max_size+size_step)[1::size_step][1:]-1
##            sizes = sizes.tolist()
#        else:
#            sizes = [max_size]
#
#        if fixed_bricking is None:
#            brick_sizes = np.arange(max_brick_size+brick_step)[1::brick_step][1:]-1
#            chunk_sizes = np.arange(max_chunk_size+brick_step)[1::brick_step][1:]-1
#        else:
#            brick_sizes = [max_brick_size]
#            chunk_sizes = [max_chunk_size]
#
#        for s in sizes:
#            for r in repeats:
#                for d in delays:
#                    for bs in brick_sizes:
#                        for cs in chunk_sizes:
#                            if cs < bs:
#                                print 'Working on: {0},{1},{2},{3},{4},{5}'.format(s, size_step, r, d, bs, cs)
#                                avg, timer, insert_time = run_perf_test(size=size_step, origin=s, repeat=r, delay=d, brick_size=bs, chunk_size=cs)
#                                line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(s, size_step, r, d, bs, cs, avg, timer, insert_time)
#                                f.write(line)
#
#def test1(sizes, max_repeat, max_delay, brick_sizes, chunk_sizes):
#    t = time.localtime()
#    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
#    output_file = 'perf_result-{0}.csv'.format(str_time)
#
#    header = 'Origin,Size,Repeat,Delay,Brick Size,Chunk Size,Average,Raw\n'
#    with open(output_file, 'w') as f:
#        f.write(header)
#
#        repeats = np.arange(max_repeat+1)[1:]
#        if max_delay > 0:
#            delays = np.arange(max_delay+1)[1:]
#        else:
#            delays = [0]
#        origin = 0
#        for s in sizes:
#            for r in repeats:
#                for d in delays:
#                    for bs in brick_sizes:
#                        for cs in chunk_sizes:
#                            if cs < bs:
#                                print 'Working on: {0},{1},{2},{3},{4}'.format(s, r, d, bs, cs)
#                                avg, timer, insert_time = run_perf_test(size=s, origin=origin, repeat=r, delay=d, brick_size=bs, chunk_size=cs)
#                                line = '{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'.format(origin, s, r, d, bs, cs, avg, timer, insert_time)
#                                f.write(line)

