#!/usr/bin/env python

"""
@package coverage_model.test.examples
@file coverage_model/test/examples.py
@author Christopher Mueller
@brief Exemplar functions for creation, manipulation, and basic visualization of coverages
"""
import random

from ooi.logging import log
from netCDF4 import Dataset
from coverage_model import *
import numpy as np

def values_outside_coverage():
    num_rec = 10
    dom = SimpleDomainSet((num_rec,))

    # QuantityType example
    qtype = QuantityType(value_encoding=np.dtype('float32'))
    qval = get_value_class(qtype, domain_set=dom)

    # ArrayType example
    atype = ArrayType()
    aval = get_value_class(atype, domain_set=dom)

    # RecordType example
    rtype = RecordType()
    rval = get_value_class(rtype, domain_set=dom)

    # ConstantType w/ numeric QuantityType example
    ctype_n = ConstantType(QuantityType(value_encoding=np.dtype('int32')))
    cval_n = get_value_class(ctype_n, domain_set=dom)

    # ConstantType w/ fixed_string QuantityType example
    ctype_s = ConstantType(QuantityType(value_encoding=np.dtype('S21')))
    cval_s = get_value_class(ctype_s, domain_set=dom)

    # FunctionType example
    ftype = FunctionType(QuantityType(value_encoding=np.dtype('float32')))
    fval = get_value_class(ftype, domain_set=dom)

    crtype = ConstantRangeType(QuantityType(value_encoding=np.dtype('int16')))
    crval = get_value_class(crtype, domain_set=dom)

    cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
    cattype = CategoryType(categories=cat)
    catval = get_value_class(cattype, dom)

    # Add data to the values
    qval[:] = np.random.random_sample(num_rec)*(50-20)+20 # array of 10 random values between 20 & 50

    catkeys = cat.keys()
    letts='abcdefghij'
    for x in xrange(num_rec):
        aval[x] = np.random.bytes(np.random.randint(1,20)) # One value (which is a byte string) for each member of the domain
        rval[x] = {letts[x]: letts[x:]} # One value (which is a dict) for each member of the domain
        catval[x] = random.choice(catkeys)

    # Doesn't matter what index (or indices) you assign these 3 - the same value is used everywhere!!
    cval_n[0] = 200
    cval_s[0] = 'constant string value'
    crval[0] = (10, 50)

    fval[:] = make_range_expr(100, min=0, max=4, min_incl=True, max_incl=False, else_val=-9999)
    fval[:] = make_range_expr(200, min=4, max=6, min_incl=True, else_val=-9999)
    fval[:] = make_range_expr(300, min=6, else_val=-9999)

    if not (aval.shape == rval.shape == cval_n.shape):# == fval.shape):
        raise SystemError('Shapes are not equal!!')

    types = (qtype, atype, rtype, ctype_n, ctype_s, cattype, ftype)
    vals = (qval, aval, rval, cval_n, cval_s, crval, catval, fval)
#    for i in xrange(len(vals)):
#        log.info('Type: %s', types[i])
#        log.info('\tContent: %s', vals[i].content)
#        log.info('\tVals: %s', vals[i][:])

    log.info('Returning: qval, aval, rval, cval_n, cval_s, crval, catval, fval')
    return vals

def param_dict_dump_load():
    pd = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    pd.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')))
    pd.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
    pd.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
    pd.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))

    pddump = pd.dump()

    pd2 = ParameterDictionary.load(pddump)

    # Tests that a dumped/loaded PD is equal to the original
    print 'pd==pd2: {0}'.format(pd==pd2)

    # Tests that a dumped/loaded PD is ordered the same way as the original
    for o, pc in pd.itervalues():
        print 'pc.name :: pd2.get_context_by_ord(o).name: {0} :: {1}'.format(pc.name, pd2.get_context_by_ord(o).name)


def param_dict_compare():
# Instantiate a ParameterDictionary
    pdict_1 = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    pdict_1.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
    pdict_1.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
    pdict_1.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
    pdict_1.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))


    # Instantiate a ParameterDictionary
    pdict_2 = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    pdict_2.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
    pdict_2.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
    pdict_2.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
    pdict_2.add_context(ParameterContext('temp', param_type=QuantityType(uom='degree_Celsius')))


    # Instantiate a ParameterDictionary
    pdict_3 = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    pdict_3.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
    pdict_3.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
    pdict_3.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))
    pdict_3.add_context(ParameterContext('temp2', param_type=QuantityType(uom='degree_Celsius')))


    # Instantiate a ParameterDictionary
    pdict_4 = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    pdict_4.add_context(ParameterContext('time', param_type=QuantityType(value_encoding='l', uom='seconds since 01-01-1970')), is_temporal=True)
    pdict_4.add_context(ParameterContext('lat', param_type=QuantityType(uom='degree_north')))
    pdict_4.add_context(ParameterContext('lon', param_type=QuantityType(uom='degree_east')))

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(uom = 'degree_Celsius'))
    pdict_4.add_context(temp_ctxt)

    temp2_ctxt = ParameterContext(name=temp_ctxt, new_name='temp2')
    pdict_4.add_context(temp2_ctxt)


    print 'Should be equal and compare \'one-to-one\' with nothing in the None list'
    print pdict_1 == pdict_2
    print pdict_1.compare(pdict_2)

    print '\nShould be unequal and compare with an empty list for \'temp\' and \'temp2\' in the None list'
    print pdict_1 == pdict_3
    print pdict_1.compare(pdict_3)

    print '\nShould be unequal and compare with both \'temp\' and \'temp2\' in \'temp\' and nothing in the None list'
    print pdict_1 == pdict_4
    print pdict_1.compare(pdict_4)

    print "Returning: pdict_1, pdict_2, pdict_3, pdict_4"
    return pdict_1, pdict_2, pdict_3, pdict_4

def samplecov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt, is_temporal=True)

    lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lat_ctxt.axis = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    pdict.add_context(lat_ctxt)

    lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lon_ctxt.axis = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    pdict.add_context(lon_ctxt)

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
    temp_ctxt.uom = 'degree_Celsius'
    pdict.add_context(temp_ctxt)

    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    cond_ctxt.uom = 'unknown'
    pdict.add_context(cond_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert some timesteps (automatically expands other arrays)
    nt = 30
    scov.insert_timesteps(nt)

    # Add data for each parameter
    scov.set_parameter_values('time', value=np.arange(nt))
    scov.set_parameter_values('lat', value=45)
    scov.set_parameter_values('lon', value=-71)
    # make a random sample of 10 values between 23 and 26
    # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
    # --> To sample  multiply the output of random_sample by (b-a) and add a
    tvals=np.random.random_sample(nt)*(26-23)+23
    scov.set_parameter_values('temp', value=tvals)
    scov.set_parameter_values('conductivity', value=np.random.random_sample(nt)*(110-90)+90)

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/sample.cov')

    return scov

def samplecov2(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt, is_temporal=True)

    lat_ctxt = ParameterContext('lat', param_type=ConstantType(), variability=VariabilityEnum.NONE)
    lat_ctxt.axis = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    pdict.add_context(lat_ctxt)

    lon_ctxt = ParameterContext('lon', param_type=ConstantType(), variability=VariabilityEnum.NONE)
    lon_ctxt.axis = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    pdict.add_context(lon_ctxt)

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
    temp_ctxt.uom = 'degree_Celsius'
    pdict.add_context(temp_ctxt)

    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    cond_ctxt.uom = 'unknown'
    pdict.add_context(cond_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    nt = 20
    tdom = GridDomain(GridShape('temporal', [nt]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert some timesteps (automatically expands other arrays)
#    scov.insert_timesteps(10)

    # Add data for each parameter
    scov.set_parameter_values('time', value=np.arange(nt))
    scov.set_parameter_values('lat', value=make_range_expr(45.32))
    scov.set_parameter_values('lon', value=make_range_expr(-71.11))
    # make a random sample of 10 values between 23 and 26
    # Ref: http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
    # --> To sample  multiply the output of random_sample by (b-a) and add a
    tvals=np.random.random_sample(nt)*(26-23)+23
    scov.set_parameter_values('temp', value=tvals)
    scov.set_parameter_values('conductivity', value=np.random.random_sample(nt)*(110-90)+90)

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/sample2.cov')

    return scov

def manyparamcov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    for x in range(500):
        pdict.add_context(ParameterContext(str(x), param_type=QuantityType(value_encoding=np.dtype('float32'))))

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    bricking_scheme = {'brick_size':1000,'chunk_size':500}
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

    # Insert some timesteps (automatically expands other arrays)
#    nt = 1000
#    scov.insert_timesteps(nt)
#
#    # Add data for the parameter
#    scov.set_parameter_values('time', value=np.arange(nt))

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/sample.cov')

    return scov

def oneparamcov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    bricking_scheme = {'brick_size':1000,'chunk_size':500}
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

    # Insert some timesteps (automatically expands other arrays)
    nt = 1000
    scov.insert_timesteps(nt)

    # Add data for the parameter
    scov.set_parameter_values('time', value=np.arange(nt))

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/sample.cov')

    return scov

def oneparamcov_noautoflush(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, auto_flush_values=False)

    # Insert some timesteps (automatically expands other arrays)
    nt = 100
    scov.insert_timesteps(nt)

    # Add data for the parameter
    scov.set_parameter_values('time', value=np.arange(nt))

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/sample.cov')

    return scov

def emptysamplecov(save_coverage=False, in_memory=False, inline_data_writes=True, brick_size=None):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    lat_ctxt = ParameterContext('lat', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lat_ctxt.axis = AxisTypeEnum.LAT
    lat_ctxt.uom = 'degree_north'
    pdict.add_context(lat_ctxt)

    lon_ctxt = ParameterContext('lon', param_type=QuantityType(value_encoding=np.dtype('float32')))
    lon_ctxt.axis = AxisTypeEnum.LON
    lon_ctxt.uom = 'degree_east'
    pdict.add_context(lon_ctxt)

    temp_ctxt = ParameterContext('temp', param_type=QuantityType(value_encoding=np.dtype('float32')))
    temp_ctxt.uom = 'degree_Celsius'
    pdict.add_context(temp_ctxt)

    cond_ctxt = ParameterContext('conductivity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    cond_ctxt.uom = 'unknown'
    pdict.add_context(cond_ctxt)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    if brick_size is not None:
        bricking_scheme = {'brick_size':brick_size, 'chunk_size':10}
    else:
        bricking_scheme = None

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'empty sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory, bricking_scheme=bricking_scheme)

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/emptysample.cov')

    return scov

def ptypescov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    quant_t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
    quant_t_ctxt.axis = AxisTypeEnum.TIME
    quant_t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(quant_t_ctxt)

    cnst_flt_ctxt = ParameterContext('const_float', param_type=ConstantType(), variability=VariabilityEnum.NONE)
    cnst_flt_ctxt.long_name = 'example of a parameter of type ConstantType, base_type float (default)'
    cnst_flt_ctxt.axis = AxisTypeEnum.LON
    cnst_flt_ctxt.uom = 'degree_east'
    pdict.add_context(cnst_flt_ctxt)

    cnst_int_ctxt = ParameterContext('const_int', param_type=ConstantType(QuantityType(value_encoding=np.dtype('int32'))), variability=VariabilityEnum.NONE)
    cnst_int_ctxt.long_name = 'example of a parameter of type ConstantType, base_type int32'
    cnst_int_ctxt.axis = AxisTypeEnum.LAT
    cnst_int_ctxt.uom = 'degree_north'
    pdict.add_context(cnst_int_ctxt)

    cnst_str_ctxt = ParameterContext('const_str', param_type=ConstantType(QuantityType(value_encoding=np.dtype('S21'))), fill_value='', variability=VariabilityEnum.NONE)
    cnst_str_ctxt.long_name = 'example of a parameter of type ConstantType, base_type fixed-len string'
    pdict.add_context(cnst_str_ctxt)

    cnst_rng_flt_ctxt = ParameterContext('const_rng_flt', param_type=ConstantRangeType(), variability=VariabilityEnum.NONE)
    cnst_rng_flt_ctxt.long_name = 'example of a parameter of type ConstantRangeType, base_type float (default)'
    pdict.add_context(cnst_rng_flt_ctxt)

    cnst_rng_int_ctxt = ParameterContext('const_rng_int', param_type=ConstantRangeType(QuantityType(value_encoding='int16')), variability=VariabilityEnum.NONE)
    cnst_rng_int_ctxt.long_name = 'example of a parameter of type ConstantRangeType, base_type int16'
    pdict.add_context(cnst_rng_int_ctxt)

    cat = {0:'turkey',1:'duck',2:'chicken',99:'None'}
    cat_ctxt = ParameterContext('category', param_type=CategoryType(categories=cat), variability=VariabilityEnum.TEMPORAL)
    pdict.add_context(cat_ctxt)

#    func_ctxt = ParameterContext('function', param_type=FunctionType(QuantityType(value_encoding=np.dtype('float32'))))
#    func_ctxt.long_name = 'example of a parameter of type FunctionType'
#    pdict.add_context(func_ctxt)

    quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    quant_ctxt.long_name = 'example of a parameter of type QuantityType'
    quant_ctxt.uom = 'degree_Celsius'
    pdict.add_context(quant_ctxt)

    arr_ctxt = ParameterContext('array', param_type=ArrayType())
    arr_ctxt.long_name = 'example of a parameter of type ArrayType, will be filled with variable-length \'byte-string\' data'
    pdict.add_context(arr_ctxt)

    rec_ctxt = ParameterContext('record', param_type=RecordType())
    rec_ctxt.long_name = 'example of a parameter of type RecordType, will be filled with dictionaries'
    pdict.add_context(rec_ctxt)

    fstr_ctxt = ParameterContext('fixed_str', param_type=QuantityType(value_encoding=np.dtype('S8')), fill_value='')
    fstr_ctxt.long_name = 'example of a fixed-length string parameter'
    pdict.add_context(fstr_ctxt)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert some timesteps (automatically expands other arrays)
    nt = 20
    scov.insert_timesteps(nt)

    # Add data for each parameter
    scov.set_parameter_values('quantity_time', value=np.arange(nt))
    scov.set_parameter_values('const_float', value=-71.11) # Set a constant with correct data type
    scov.set_parameter_values('const_int', value=45.32) # Set a constant with incorrect data type (fixed under the hood)
    scov.set_parameter_values('const_str', value='constant string value') # Set with a string
    scov.set_parameter_values('const_rng_flt', value=(12.8, 55.2)) # Set with a tuple
    scov.set_parameter_values('const_rng_int', value=[-10, 10]) # Set with a list

    scov.set_parameter_values('quantity', value=np.random.random_sample(nt)*(26-23)+23)

#    # Setting three range expressions such that indices 0-2 == 10, 3-7 == 15 and >=8 == 20
#    scov.set_parameter_values('function', value=make_range_expr(10, 0, 3, min_incl=True, max_incl=False, else_val=-999.9))
#    scov.set_parameter_values('function', value=make_range_expr(15, 3, 8, min_incl=True, max_incl=False, else_val=-999.9))
#    scov.set_parameter_values('function', value=make_range_expr(20, 8, min_incl=True, max_incl=False, else_val=-999.9))

    arrval = []
    recval = []
    catval = []
    fstrval = []
    catkeys = cat.keys()
    letts='abcdefghijklmnopqrstuvwxyz'
    while len(letts) < nt:
        letts += 'abcdefghijklmnopqrstuvwxyz'
    for x in xrange(nt):
        arrval.append(np.random.bytes(np.random.randint(1,20))) # One value (which is a byte string) for each member of the domain
        d = {letts[x]: letts[x:]}
        recval.append(d) # One value (which is a dict) for each member of the domain
        catval.append(random.choice(catkeys))
        fstrval.append(''.join([random.choice(letts) for x in xrange(8)])) # A random string of length 8
    scov.set_parameter_values('array', value=arrval)
    scov.set_parameter_values('record', value=recval)
    scov.set_parameter_values('category', value=catval)
    scov.set_parameter_values('fixed_str', value=fstrval)

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/ptypes.cov')

    return scov

def nospatialcov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('quantity_time', param_type=QuantityType(value_encoding=np.dtype('int64')), variability=VariabilityEnum.TEMPORAL)
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    quant_ctxt = ParameterContext('quantity', param_type=QuantityType(value_encoding=np.dtype('float32')))
    quant_ctxt.long_name = 'example of a parameter of type QuantityType'
    quant_ctxt.uom = 'degree_Celsius'
    pdict.add_context(quant_ctxt)

    const_ctxt = ParameterContext('constant', param_type=ConstantType())
    const_ctxt.long_name = 'example of a parameter of type ConstantType'
    pdict.add_context(const_ctxt)

    arr_ctxt = ParameterContext('array', param_type=ArrayType())
    arr_ctxt.long_name = 'example of a parameter of type ArrayType with base_type ndarray (resolves to \'object\')'
    pdict.add_context(arr_ctxt)

    arr2_ctxt = ParameterContext('array2', param_type=ArrayType())
    arr2_ctxt.long_name = 'example of a parameter of type ArrayType with base_type object'
    pdict.add_context(arr2_ctxt)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert some timesteps (automatically expands other arrays)
    nt = 20
    scov.insert_timesteps(nt)

    # Add data for each parameter
    scov.set_parameter_values('quantity_time', value=np.arange(nt))
    scov.set_parameter_values('quantity', value=np.random.random_sample(nt)*(26-23)+23)
    scov.set_parameter_values('constant', value=20)

    arrval = []
    arr2val = []
    for x in xrange(nt): # One value (which IS an array) for each member of the domain
        arrval.append(np.random.bytes(np.random.randint(1,20)))
        arr2val.append(np.random.random_sample(np.random.randint(1,10)))
    scov.set_parameter_values('array', value=arrval)
    scov.set_parameter_values('array2', value=arr2val)

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/ptypes.cov')

    return scov

def ncgrid2cov(save_coverage=False, in_memory=False, inline_data_writes=True):
    if True:
        raise NotImplementedError('Multidimensional support is not available at this time')
    # Open the netcdf dataset
    ds = Dataset('test_data/ncom.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','depth','water_u','water_v','salinity','water_temp',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionary
    for v in var_names:
        var = ds.variables[v]

        pcontext = ParameterContext(v, param_type=QuantityType(value_encoding=ds.variables[v].dtype.char))
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the axis for the coordinate parameters
        if v == 'time':
            pcontext.variability = VariabilityEnum.TEMPORAL
            pcontext.axis = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.axis = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.axis = AxisTypeEnum.LON
        elif v == 'depth':
            pcontext.axis = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT, AxisTypeEnum.HEIGHT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal'), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [34,57,89]), scrs, MutabilityEnum.IMMUTABLE) # 3d spatial topology (grid)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample grid coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        log.debug('Assign values to %s', v)
        var = ds.variables[v]
        var.set_auto_maskandscale(False)
        arr = var[:]
        # TODO: Sort out how to leave these sparse internally and only broadcast during read
        if v == 'depth':
            z,_,_ = my_meshgrid(arr,np.zeros([57]),np.zeros([89]),indexing='ij',sparse=True)
            scov._range_value[v][:] = z
        elif v == 'lat':
            _,y,_ = my_meshgrid(np.zeros([34]),arr,np.zeros([89]),indexing='ij',sparse=True)
            scov._range_value[v][:] = y
        elif v == 'lon':
            _,_,x = my_meshgrid(np.zeros([34]),np.zeros([57]),arr,indexing='ij',sparse=True)
            scov._range_value[v][:] = x
        else:
            scov._range_value[v][:] = var[:]

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/ncom.cov')

    return scov

def ncstation2cov(save_coverage=False, in_memory=False, inline_data_writes=True):
    # Open the netcdf dataset
    ds = Dataset('test_data/usgs.nc')
    # Itemize the variable names that we want to include in the coverage
    var_names = ['time','lat','lon','z','streamflow','water_temperature',]

    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a ParameterContext object for each of the variables in the dataset and add them to the ParameterDictionaryl
    for v in var_names:
        var = ds.variables[v]

        ptype = QuantityType(var.dtype.char)
        if v in ('lat','lon','z'):
            ptype=ConstantType(ptype)

        pcontext = ParameterContext(v, param_type=ptype)
        if 'units' in var.ncattrs():
            pcontext.uom = var.getncattr('units')
        if 'long_name' in var.ncattrs():
            pcontext.description = var.getncattr('long_name')
        if '_FillValue' in var.ncattrs():
            pcontext.fill_value = var.getncattr('_FillValue')

        # Set the axis for the coordinate parameters
        if v == 'time':
            pcontext.variability = VariabilityEnum.TEMPORAL
            pcontext.axis = AxisTypeEnum.TIME
        elif v == 'lat':
            pcontext.axis = AxisTypeEnum.LAT
        elif v == 'lon':
            pcontext.axis = AxisTypeEnum.LON
        elif v == 'z':
            pcontext.axis = AxisTypeEnum.HEIGHT

        pdict.add_context(pcontext)

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS.standard_temporal()
    scrs = CRS.lat_lon_height()

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 1d spatial topology (station/trajectory)

    # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
    scov = SimplexCoverage('test_data', create_guid(), 'sample station coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom, inline_data_writes=inline_data_writes, in_memory_storage=in_memory)

    # Insert the timesteps (automatically expands other arrays)
    tvar=ds.variables['time']
    scov.insert_timesteps(tvar.size)

    # Add data to the parameters - NOT using setters at this point, direct assignment to arrays
    for v in var_names:
        var = ds.variables[v]
        var.set_auto_maskandscale(False)

        if v in ('lat','lon','z'):
            scov._range_value[v][0] = make_range_expr(var[0])
        else:
            scov._range_value[v][:] = var[:]

    if in_memory and save_coverage:
        SimplexCoverage.pickle_save(scov, 'test_data/usgs.cov')

    return scov

def benchmark_value_setting(num_params=10, num_insertions=100, repeat=1, bulk_ts_insert=False):
    # Instantiate a ParameterDictionary
    pdict = ParameterDictionary()

    # Create a set of ParameterContext objects to define the parameters in the coverage, add each to the ParameterDictionary
    t_ctxt = ParameterContext('time', param_type=QuantityType(value_encoding=np.dtype('int64')))
    t_ctxt.axis = AxisTypeEnum.TIME
    t_ctxt.uom = 'seconds since 01-01-1970'
    pdict.add_context(t_ctxt)

    for i in xrange(num_params-1):
        pdict.add_context(ParameterContext('param_{0}'.format(i)))

    # Construct temporal and spatial Coordinate Reference System objects
    tcrs = CRS([AxisTypeEnum.TIME])
    scrs = CRS([AxisTypeEnum.LON, AxisTypeEnum.LAT])

    # Construct temporal and spatial Domain objects
    tdom = GridDomain(GridShape('temporal', [0]), tcrs, MutabilityEnum.EXTENSIBLE) # 1d (timeline)
    sdom = GridDomain(GridShape('spatial', [0]), scrs, MutabilityEnum.IMMUTABLE) # 0d spatial topology (station/trajectory)

    import time
    counter = 1
    insert_times = []
    per_rep_times = []
    full_time = time.time()
    for r in xrange(repeat):
        # Instantiate the SimplexCoverage providing the ParameterDictionary, spatial Domain and temporal Domain
        cov = SimplexCoverage('test_data', create_guid(), 'empty sample coverage_model', parameter_dictionary=pdict, temporal_domain=tdom, spatial_domain=sdom)

        rep_time = time.time()
        if bulk_ts_insert:
            cov.insert_timesteps(num_insertions)
        for x in xrange(num_insertions):
            in_time = time.time()
            if not bulk_ts_insert:
                cov.insert_timesteps(1)
            slice_ = slice(cov.num_timesteps - 1, None)
            cov.set_parameter_values('time', 1, tdoa=slice_)
            for i in xrange(num_params-1):
                cov.set_parameter_values('param_{0}'.format(i), 1.1, tdoa=slice_)

            in_time = time.time() - in_time
            insert_times.append(in_time)
            counter += 1
        rep_time = time.time() - rep_time
        per_rep_times.append(rep_time)

        cov.close()

    print 'Average Value Insertion Time (%s repetitions): %s' % (repeat, sum(insert_times) / len(insert_times))
    print 'Average Total Expansion Time (%s repetitions): %s' % (repeat, sum(per_rep_times) / len(per_rep_times))
    print 'Full Time (includes cov creation/closing): %s' % (time.time() - full_time)

    return cov


def cov_get_by_integer():
    cov = oneparamcov()
    dat = cov._range_value.time
    for s in range(len(dat)):
        log.info(s)
        log.info('\t%s', dat[s])

    return cov

def cov_get_by_slice():
    cov = oneparamcov()
    dat = cov._range_value.time
    for s in range(len(dat)):
        for e in range(len(dat)):
            e+=1
            if s < e:
                sl = slice(s, None, None)
                log.info(sl)
                log.info('\t%s', dat[s])
                sl = slice(None, e, None)
                log.info(sl)
                log.info('\t%s', dat[s])
                sl = slice(s, e, None)
                log.info(sl)
                log.info('\t%s', dat[s])
                for st in range(e-s):
                    sl = slice(s, e, st+1)
                    log.info(sl)
                    log.info('\t%s', dat[sl])

    return cov

def cov_get_by_list():
    cov = oneparamcov()
    dat = cov._range_value.time
    dl = len(dat)
    for x in range(5):
        lst = list(set([np.random.randint(0,dl) for s in xrange(np.random.randint(1,dl-1))]))
        lst.sort()
        log.info(lst)
        log.info('\t%s', dat[[lst]])

    return cov



def direct_read():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.info('<========= Query =========>')
    log.info('\n>> All data for first timestep\n')
    slice_ = 0
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data\n')
    slice_ = (slice(None))
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data for every other timestep from 0 to 10\n')
    slice_ = (slice(0,10,2))
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

    log.debug('\n>> All data for first, sixth, eighth, thirteenth, and fifty-sixth timesteps\n')
    slice_ = [[(0,5,7,12,55)]]
    log.debug('sflow <shape %s> sliced with: %s', shp,slice_)
    log.debug(scov.range_value['streamflow'][slice_])

def direct_write():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.debug('<========= Assignment =========>')

    slice_ = (slice(None))
    value = 22
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.range_value['streamflow'][slice_] = value
    log.debug(scov.range_value['streamflow'][slice_])

    slice_ = [[(1,5,7,)]]
    value = [10, 20, 30]
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.range_value['streamflow'][slice_] = value
    log.debug(scov.range_value['streamflow'][slice_])

def methodized_read():
    from coverage_model import SimplexCoverage
    from coverage_model.basic_types import DomainOfApplication
    import numpy as np
    import os

    log.debug('============ Station ============')
    pth = 'test_data/usgs.cov'
    if not os.path.exists(pth):
        raise SystemError('Cannot proceed, \'{0}\' file must exist.  Run the \'ncstation2cov()\' function to generate the file.'.format(pth))

    cov=SimplexCoverage.load(pth)
    ra=np.zeros([0])
    log.debug('\n>> All data for first timestep\n')
    log.debug(cov.get_parameter_values('water_temperature',0,None,ra))
    log.debug('\n>> All data\n')
    log.debug(cov.get_parameter_values('water_temperature',None,None,None))
    log.debug('\n>> All data for second, fifth and sixth timesteps\n')
    log.debug(cov.get_parameter_values('water_temperature',[[1,4,5]],None,None))
    log.debug('\n>> First datapoint (in x) for every 5th timestep\n')
    log.debug(cov.get_parameter_values('water_temperature',slice(0,None,5),0,None))
    log.debug('\n>> First datapoint for first 10 timesteps, passing DOA objects\n')
    tdoa = DomainOfApplication(slice(0,10))
    sdoa = DomainOfApplication(0)
    log.debug(cov.get_parameter_values('water_temperature',tdoa,sdoa,None))

    log.debug('\n============ Grid ============')
    pth = 'test_data/ncom.cov'
    if not os.path.exists(pth):
        raise SystemError('Cannot proceed, \'{0}\' file must exist.  Run the \'ncstation2cov()\' function to generate the file.'.format(pth))

    cov=SimplexCoverage.load(pth)
    ra=np.zeros([0])
    log.debug('\n>> All data for first timestep\n')
    log.debug(cov.get_parameter_values('water_temp',0,None,ra))
    log.debug('\n>> All data\n')
    log.debug(cov.get_parameter_values('water_temp',None,None,None))
    log.debug('\n>> All data for first, fourth, and fifth timesteps\n')
    log.debug(cov.get_parameter_values('water_temp',[[0,3,4]],None,None))
    log.debug('\n>> Data from z=0, y=10, x=10 for every 2nd timestep\n')
    log.debug(cov.get_parameter_values('water_temp',slice(0,None,2),[0,10,10],None))
    log.debug('\n>> Data from z=0-10, y=10, x=10 for the first 2 timesteps, passing DOA objects\n')
    tdoa = DomainOfApplication(slice(0,2))
    sdoa = DomainOfApplication([slice(0,10),10,10])
    log.debug(cov.get_parameter_values('water_temp',tdoa,sdoa,None))

def methodized_write():
    scov, ds = ncstation2cov()
    shp = scov.range_value.streamflow.shape

    log.debug('<========= Assignment =========>')

    slice_ = (slice(None))
    value = 22
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.set_parameter_values('streamflow',value=value,tdoa=slice_)

    slice_ = [[(1,5,7,)]]
    value = [10, 20, 30]
    log.debug('sflow <shape %s> assigned with slice: %s and value: %s', shp,slice_,value)
    scov.set_parameter_values('streamflow',value=value,tdoa=slice_)

#    raise NotImplementedError('Example not yet implemented')

def test_plot_1():
    from coverage_model.test.examples import SimplexCoverage
    import matplotlib.pyplot as plt

    cov=SimplexCoverage.load('test_data/usgs.cov')

    log.debug('Plot the \'water_temperature\' and \'streamflow\' for all times')
    wtemp = cov.get_parameter_values('water_temperature')
    wtemp_pc = cov.get_parameter_context('water_temperature')
    sflow = cov.get_parameter_values('streamflow')
    sflow_pc = cov.get_parameter_context('streamflow')
    times = cov.get_parameter_values('time')
    time_pc = cov.get_parameter_context('time')

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(times,wtemp)
    ax1.set_xlabel('{0} ({1})'.format(time_pc.name, time_pc.uom))
    ax1.set_ylabel('{0} ({1})'.format(wtemp_pc.name, wtemp_pc.uom))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(times,sflow)
    ax2.set_xlabel('{0} ({1})'.format(time_pc.name, time_pc.uom))
    ax2.set_ylabel('{0} ({1})'.format(sflow_pc.name, sflow_pc.uom))

    plt.show(0)

def test_plot_2():
    from coverage_model.test.examples import SimplexCoverage
    import matplotlib.pyplot as plt

    cov=SimplexCoverage.load('test_data/usgs.cov')

    log.debug('Plot the \'water_temperature\' and \'streamflow\' for all times')
    wtemp_param = cov.get_parameter('water_temperature')
    sflow_param = cov.get_parameter('streamflow')
    time_param = cov.get_parameter('time')

    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(time_param.value[:],wtemp_param.value[:])
    ax1.set_xlabel('{0} ({1})'.format(time_param.name, time_param.context.uom))
    ax1.set_ylabel('{0} ({1})'.format(wtemp_param.name, wtemp_param.context.uom))

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(time_param.value[:],sflow_param.value[:])
    ax2.set_xlabel('{0} ({1})'.format(time_param.name, time_param.context.uom))
    ax2.set_ylabel('{0} ({1})'.format(sflow_param.name, sflow_param.context.uom))

    plt.show(0)

# Based on scitools meshgrid
def my_meshgrid(*xi, **kwargs):
    """
    Return coordinate matrices from two or more coordinate vectors.

    Make N-D coordinate arrays for vectorized evaluations of
    N-D scalar/vector fields over N-D grids, given
    one-dimensional coordinate arrays x1, x2,..., xn.

    Parameters
    ----------
    x1, x2,..., xn : array_like
        1-D arrays representing the coordinates of a grid.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of output.
        See Notes for more details.
    sparse : bool, optional
         If True a sparse grid is returned in order to conserve memory.
         Default is False.
    copy : bool, optional
        If False, a view into the original arrays are returned in
        order to conserve memory.  Default is True.  Please note that
        ``sparse=False, copy=False`` will likely return non-contiguous arrays.
        Furthermore, more than one element of a broadcast array may refer to
        a single memory location.  If you need to write to the arrays, make
        copies first.

    Returns
    -------
    X1, X2,..., XN : ndarray
        For vectors `x1`, `x2`,..., 'xn' with lengths ``Ni=len(xi)`` ,
        return ``(N1, N2, N3,...Nn)`` shaped arrays if indexing='ij'
        or ``(N2, N1, N3,...Nn)`` shaped arrays if indexing='xy'
        with the elements of `xi` repeated to fill the matrix along
        the first dimension for `x1`, the second for `x2` and so on.

    Notes
    -----
    This function supports both indexing conventions through the indexing keyword
    argument.  Giving the string 'ij' returns a meshgrid with matrix indexing,
    while 'xy' returns a meshgrid with Cartesian indexing.  In the 2-D case
    with inputs of length M and N, the outputs are of shape (N, M) for 'xy'
    indexing and (M, N) for 'ij' indexing.  In the 3-D case with inputs of
    length M, N and P, outputs are of shape (N, M, P) for 'xy' indexing and (M,
    N, P) for 'ij' indexing.  The difference is illustrated by the following
    code snippet::

        xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
        for i in range(nx):
            for j in range(ny):
                # treat xv[i,j], yv[i,j]

        xv, yv = meshgrid(x, y, sparse=False, indexing='xy')
        for i in range(nx):
            for j in range(ny):
                # treat xv[j,i], yv[j,i]

    See Also
    --------
    index_tricks.mgrid : Construct a multi-dimensional "meshgrid"
                     using indexing notation.
    index_tricks.ogrid : Construct an open multi-dimensional "meshgrid"
                     using indexing notation.

    Examples
    --------
    >>> nx, ny = (3, 2)
    >>> x = np.linspace(0, 1, nx)
    >>> y = np.linspace(0, 1, ny)
    >>> xv, yv = meshgrid(x, y)
    >>> xv
    array([[ 0. ,  0.5,  1. ],
           [ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.,  0.,  0.],
           [ 1.,  1.,  1.]])
    >>> xv, yv = meshgrid(x, y, sparse=True)  # make sparse output arrays
    >>> xv
    array([[ 0. ,  0.5,  1. ]])
    >>> yv
    array([[ 0.],
           [ 1.]])

    `meshgrid` is very useful to evaluate functions on a grid.

    >>> x = np.arange(-5, 5, 0.1)
    >>> y = np.arange(-5, 5, 0.1)
    >>> xx, yy = meshgrid(x, y, sparse=True)
    >>> z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
    >>> h = plt.contourf(x,y,z)

    """
    if len(xi) < 2:
        msg = 'meshgrid() takes 2 or more arguments (%d given)' % int(len(xi) > 0)
        raise ValueError(msg)

    args = np.atleast_1d(*xi)
    ndim = len(args)

    copy_ = kwargs.get('copy', True)
    sparse = kwargs.get('sparse', False)
    indexing = kwargs.get('indexing', 'xy')
    if not indexing in ['xy', 'ij']:
        raise ValueError("Valid values for `indexing` are 'xy' and 'ij'.")

    s0 = (1,) * ndim
    output = [x.reshape(s0[:i] + (-1,) + s0[i + 1::]) for i, x in enumerate(args)]

    shape = [x.size for x in output]

    if indexing == 'xy':
        # switch first and second axis
        output[0].shape = (1, -1) + (1,)*(ndim - 2)
        output[1].shape = (-1, 1) + (1,)*(ndim - 2)
        shape[0], shape[1] = shape[1], shape[0]

    if sparse:
        if copy_:
            return [x.copy() for x in output]
        else:
            return output
    else:
        # Return the full N-D matrix (not only the 1-D vector)
        if copy_:
            mult_fact = np.ones(shape, dtype=int)
            return [x * mult_fact for x in output]
        else:
            return np.broadcast_arrays(*output)


if __name__ == "__main__":
#    scov, _ = ncstation2cov()
#    log.debug(scov)
#
#    log.debug('\n=======\n')
#
#    gcov, _ = ncgrid2cov()
#    log.debug(gcov)

#    direct_read_write()
    methodized_read()

#    from coverage_model.coverage_model import AxisTypeEnum
#    axis = 'TIME'
#    log.debug(axis == AxisTypeEnum.TIME)

    pass

"""

from coverage_model.test.simple_cov import *
scov, ds = ncstation2cov()


"""