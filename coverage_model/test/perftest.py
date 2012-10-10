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

    quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    quant_ctxt.long_name = 'example of a parameter of type QuantityType'
    quant_ctxt.uom = 'degree_Celsius'
    pdict.add_context(quant_ctxt)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'performance test coverage_model', pdict, temporal_domain=tdom, bricking_scheme=bricking_scheme)

    return scov

def _set_data(scov, size, repeat, delay):
    timer = []
    for i in xrange(repeat):
        st = time.time()
        # Expand the domain
        scov.insert_timesteps(size)

        # Add data for each parameter
        loc = slice(i*size,(i*size)+size)
        scov.set_parameter_values('quantity_time', value=np.arange(size), tdoa=loc)
        scov.set_parameter_values('quantity', value=np.random.random_sample(size)*(26-23)+23, tdoa=loc)
        timer.append(time.time()-st)

        # Sleep for delay (seconds)
        time.sleep(delay)

    return timer

def run_perf_test(size=10, repeat=1, delay=0, brick_size=None, chunk_size=None):
    scov = _make_cov(brick_size, chunk_size)
    timer = _set_data(scov, size, repeat, delay)
    avg = sum(timer)/len(timer)
    return avg, timer

def test_all(max_size, max_repeat, max_delay, max_brick_size, max_chunk_size, fixed_size=None, fixed_bricking=None):
    t = time.localtime()
    str_time = '{0}{1}{2}{3}{4}'.format(t.tm_year,t.tm_mon,t.tm_mday,t.tm_hour,t.tm_min,t.tm_sec)
    output_file = 'perf_result-{0}.csv'.format(str_time)

    header = 'Size,Repeat,Delay,Brick Size,Chunk Size,Average,Raw\n'
    with open(output_file, 'w') as f:
        f.write(header)

        repeats = np.arange(max_repeat+1)[1:]
        repeats = repeats.tolist()
        if max_delay > 0:
            delays = np.arange(max_delay+1)[1:]
            delays = delays.tolist()
        else:
            delays = [0]

        if fixed_size is None:
            sizes = np.arange(max_size+5)[1::5][1:]-1
            sizes = sizes.tolist()
        else:
            sizes = [max_size]

        if fixed_bricking is None:
            brick_sizes = np.arange(max_brick_size+5)[1::5][1:]-1
            brick_sizes = brick_sizes.tolist()
            chunk_sizes = np.arange(max_chunk_size+5)[1::5][1:]-1
            chunk_sizes = chunk_sizes.tolist()
        else:
            brick_sizes = [max_brick_size]
            chunk_sizes = [max_chunk_size]

        for s in sizes:
            for r in repeats:
                for d in delays:
                    for bs in brick_sizes:
                        for cs in chunk_sizes:
                            if cs < bs:
                                print 'Working on: {0},{1},{2},{3},{4}'.format(s, r, d, bs, cs)
                                avg, timer = run_perf_test(size=s, repeat=r, delay=d, brick_size=bs, chunk_size=cs)
                                line = '{0},{1},{2},{3},{4},{5},{6}\n'.format(s, r, d, bs, cs, avg, timer)
                                f.write(line)

