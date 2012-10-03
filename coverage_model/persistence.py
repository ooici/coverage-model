#!/usr/bin/env python

"""
@package coverage_model.persistence
@file coverage_model/persistence.py
@author James Case
@brief The core classes comprising the Persistence Layer
"""

from pyon.public import log
from coverage_model.basic_types import create_guid, AbstractStorage, InMemoryStorage, Dictable
from coverage_model.parameter import ParameterContext, ParameterDictionary
import numpy as np
import h5py
import os
import rtree
import itertools
import msgpack
from copy import deepcopy
import ast

# TODO: Make persistence-specific error classes
class PersistenceError(Exception):
    pass

def pack(payload):
    return msgpack.packb(payload).replace('\x01','\x01\x02').replace('\x00','\x01\x01')

def unpack(msg):
    return msgpack.unpackb(msg.replace('\x01\x01','\x00').replace('\x01\x02','\x01'))

class ParameterManager(object):
    def __init__(self, root_path, parameter_name, **kwargs):
        object.__setattr__(self, '_map', {})
        self.root_path = root_path
        self.parameter_name = parameter_name
        self.file_path = os.path.join(self.root_path, self.parameter_name, '{0}.hdf5'.format(self.parameter_name))

        if os.path.exists(self.file_path):
            # TODO: Do load - iterate over attributes in hdf file and call setattr(self, key, value)
            pass
        else:
            # Touch the file
#            with h5py.File(self.file_path, 'a') as f:
                # Set any kwargs
            for k, v in kwargs.iteritems():
                setattr(self, k, v)

    def __setattr__(self, key, value):
        # 0 If it's Dictable, dump it and pack it...
        if isinstance(value, Dictable):
            value = 'PACKED:' + pack(value.dump())

        # 1 TODO: logic to sort out what the value type is
        # TODO: may need to pack other things

        log.error(value)
        # 2 TODO: Write it to the hdf file

        # 3 assign it to the __dict__
        self._map[key] = value

    def __getattr__(self, item):
        try:
            val = self._map[item]
        except KeyError:
            raise AttributeError('object has no attribute \'{0}\''.format(item))

        # TODO: Is this OK? does this also catch str objects?
        log.warn(type(val))
        if isinstance(val, str):
            if val.startswith('PACKED:'):
                val = unpack(val.replace('PACKED:',''))

        # TODO: Deal with other types appropriately as necessary

        if isinstance(val, dict) and 'cm_type' in val:
            ms, cs = val['cm_type']
            module = __import__(ms, fromlist=[cs])
            classobj = getattr(module, cs)
            val = classobj._fromdict(val)

        return val

    def __delattr__(self, item):
        try:
            del self._map[item] # Will raise an error if the key isn't there
        except KeyError:
            raise AttributeError('object has no attribute \'{0}\''.format(item))

        # TODO: Remove the attribute from the hdf file

class ParameterMetadata():
    def __init__(self, root, guid, parameter_name, meta_title, meta_path, meta_object=None, **kwargs):
        self.root = root
        self.guid = guid
        self.parameter_name = parameter_name
        self.meta_title = meta_title
        self.meta_object = meta_object
        self.meta_path = meta_path

        self.file_path = '{0}/{1}/{2}/{2}.hdf5'.format(self.root,self.guid,self.parameter_name)

    def __setitem__(self):
        if self.meta_object is not None:
            if isinstance(self.meta_object, (list, dict, tuple)):
                self.meta_object = pack(self.meta_object)
            elif isinstance(self.meta_object, ParameterContext):
                self.meta_object = str(self.meta_object.dump())
            elif self.meta_path == 'rtree':
            #            Insert into Rtree
            #            log.debug('Inserting into Rtree %s:%s:%s', brick_count, rtree_extents, brick_guid)
            #            self.parameter_metadata[parameter_name]['brick_tree'].insert(brick_count, rtree_extents, obj=brick_guid)
            #            _brick_tree_dataset = _parameter_file.require_dataset('rtree', shape=(brick_count,), dtype=h5py.new_vlen(str), maxshape=(None,))
            #            _brick_tree_dataset.resize((brick_count+1,))
            #            rtree_payload = pack((rtree_extents, brick_guid))
            #            log.debug('Inserting into brick tree dataset: [%s]: %s', brick_count, unpack(rtree_payload))
            #            _brick_tree_dataset[brick_count] = rtree_payload
            #            log.debug('Rtree inserted successfully.')
                pass
        _parameter_file = h5py.File(self.file_path, 'a')
        _parameter_file[self.meta_path].attrs[self.meta_title] = self.meta_object
        _parameter_file.close()

    def __getitem__(self):
        _parameter_file = h5py.File(self.file_path, 'r+')
        meta = _parameter_file[self.meta_path].attrs[self.meta_title]
        _parameter_file.close()

        if isinstance(meta, bytes):
            return unpack(meta)
        elif isinstance(meta, str):
            return ast.literal_eval(meta)
        else:
            return meta

class PersistenceLayer(object):
    def __init__(self, root, guid, name=None, tdom=None, sdom=None, **kwargs):
        """
        Constructor for Persistence Layer
        @param root: Where to save/look for HDF5 files
        @param guid: CoverageModel GUID
        @param name: CoverageModel Name
        @param tdom: Temporal Domain
        @param sdom: Spatial Domain
        @param kwargs:
        @return:
        """
        self.root = '.' if root is ('' or None) else root
        self.guid = guid
        log.debug('Persistence GUID: %s', self.guid)
        self.name = name
        self.tdom = tdom
        self.sdom = sdom

        self.global_bricking_scheme = {}
        self.temporal_param_name = ''

        self.parameter_metadata = {} # {parameter_name: [brick_list, parameter_domains, rtree]}
        self.parameter_dictionary = ParameterDictionary()

        # Setup Master HDF5 File using the root path and the supplied coverage guid
        self.master_file_path = '{0}/{1}_master.hdf5'.format(self.root,self.guid)

        # Make sure the root path exists, if not, make it
        # TODO: Should probably be a test way before it gets here and fail outright instead of making it.
        # TODO: Then we need to check for existing files to avoid problems
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        if name is not None and tdom is not None and sdom is not None:
            # Creating new coverage model
            # Verify the master file does not exist
            if not os.path.exists(self.master_file_path):
                # Create the master file and add the master file-level metadata
                master_file = h5py.File(self.master_file_path, 'a')
                master_file.attrs['root'] = self.root
                master_file.attrs['guid'] = self.guid
                master_file.attrs['name'] = self.name
                master_file.attrs['tdom'] = str(self.tdom.dump())
                master_file.attrs['sdom'] = str(self.sdom.dump())
                master_file.attrs['global_bricking_scheme'] = str(self.global_bricking_scheme)
                master_file.attrs['temporal_param_name'] = self.temporal_param_name
            else:
                raise ValueError('A master file already exists for this coverage.')
        else:
            # Loading existing coverage model
            self._load()

        self.value_list = {}

    def _load(self):
        from coverage_model.coverage import AbstractDomain
        # Open the master file
        _master_file = h5py.File(self.master_file_path, 'r+')

        self.root = _master_file.attrs['root']
        self.guid = _master_file.attrs['guid']
        self.name = _master_file.attrs['name']
        self.tdom = AbstractDomain.load(ast.literal_eval(_master_file.attrs['tdom']))
        self.sdom = AbstractDomain.load(ast.literal_eval(_master_file.attrs['sdom']))
        self.global_bricking_scheme = ast.literal_eval(_master_file.attrs['global_bricking_scheme'])
        self.temporal_param_name = _master_file.attrs['temporal_param_name']

        # Get all the groups
        master_groups = []
        _master_file.visit(master_groups.append)

        # Get all parameter's external links to datasets
        # TODO: Verify the external liks gathered below actually exist on the filesystem
        master_links = {}
        for g in master_groups:
            grp = _master_file[g]
            for v in grp.values():
                if isinstance(v,h5py.Dataset):
                    master_links[v.name] = v.file.filename
                    v.file.close()

        # Populate the in-memory objects from the stored master and parameter metadata
        for g in master_groups:
            log.debug('master group: %s', g)
            self.parameter_metadata[g] = {}
            pfile = h5py.File('{0}/{1}/{2}/{3}.hdf5'.format(self.root, self.guid, g, g), 'r+')
            for an, at in pfile.attrs.iteritems():
                if an == 'brick_list':
                    log.debug('Unpacking brick list metadata into parameter_metadata.')
                    val = unpack(at)
                    self.parameter_metadata[g]['brick_list'] = val
                    log.debug('%s: %s', an, val)
                if an == 'brick_domains':
                    log.debug('Unpacking brick domain metadata into parameter_metadata.')
                    val = unpack(at)
                    self.parameter_metadata[g]['brick_domains'] = list(val)
                    log.debug('%s: %s', an, val)
                if an == 'parameter_context':
                    log.debug('Unpacking parameter context into parameter metadata.')
                    self.parameter_metadata[g]['parameter_context'] = ParameterContext.load(ast.literal_eval(at))
                    self.parameter_dictionary.add_context(self.parameter_metadata[g]['parameter_context'])

            # Populate the brick tree
            tree_rank = len(self.parameter_metadata[g]['brick_domains'][1])
            log.debug('tree_rank: %s', tree_rank)
            if tree_rank == 1:
                tree_rank += 1
            log.debug('tree_rank: %s', tree_rank)
            p = rtree.index.Property()
            p.dimension = tree_rank

            # TODO: Populate brick tree from dataset 'rtree' in parameter.hdf5 file
            ds = pfile['/rtree']

            def tree_loader(darr):
                for i, x in enumerate(darr):
                    ext, obj = unpack(x)
                    yield (i, ext, obj)

            brick_tree = rtree.index.Index(tree_loader(ds[:]), properties=p)

            self.parameter_metadata[g]['brick_tree'] = brick_tree

        # Close the master file
        _master_file.close()

    def calculate_brick_size(self, tD, bricking_scheme):
        """
        Calculate brick domain size given a target file system brick size (Mbytes) and dtype
        @param tD:
        @param bricking_scheme:
        @return:
        """
        log.debug('Calculating the size of a brick...')
        log.debug('Bricking scheme: %s', bricking_scheme)
        log.debug('tD: %s', tD)
        bD = [bricking_scheme['brick_size'] for x in tD]
        cD = [bricking_scheme['chunk_size'] for x in tD]
        log.debug('bD: %s', bD)
        log.debug('cD: %s', cD)
        return bD,tuple(cD)

    def init_parameter(self, parameter_context, bricking_scheme, is_temporal_param=False):
        parameter_name = parameter_context.name
        if is_temporal_param:
            self.temporal_param_name = parameter_name

        self.global_bricking_scheme = bricking_scheme

        self.parameter_metadata[parameter_name] = {}
        log.debug('Initialize %s', parameter_name)

        self.parameter_dictionary.add_context(parameter_context)

        log.debug('Performing Rtree dict setup')
        tD = parameter_context.dom.total_extents
        bD,cD = self.calculate_brick_size(tD, bricking_scheme) #remains same for each parameter
        # Verify domain is Rtree friendly
        tree_rank = len(bD)
        log.debug('tree_rank: %s', tree_rank)
        if tree_rank == 1:
            tree_rank += 1
        log.debug('tree_rank: %s', tree_rank)
        p = rtree.index.Property()
        p.dimension = tree_rank

        # TODO: Sort out the path to the bricks for this parameter
        brick_path = '{0}/{1}/{2}'.format(self.root, self.guid, parameter_name)
        # Make sure the root path exists, if not, make it
        if not os.path.exists(brick_path):
            os.makedirs(brick_path)

        brick_tree = rtree.index.Index(properties=p)

        self.parameter_metadata[parameter_name]['brick_list'] = {} # brick_list {brick_guid: [brick_extents, origin, tuple(bD), brick_active_size]
        self.parameter_metadata[parameter_name]['brick_domains'] = [tD, bD, cD, bricking_scheme] # brick_domain_dict [tD, bD, cD, bricking_scheme]
        self.parameter_metadata[parameter_name]['brick_tree'] = brick_tree # brick_tree
        self.parameter_metadata[parameter_name]['parameter_context'] = parameter_context
        v = PersistedStorage(brick_path=brick_path,
            brick_tree=self.parameter_metadata[parameter_name]['brick_tree'],
            brick_list=self.parameter_metadata[parameter_name]['brick_list'],
            brick_domains=self.parameter_metadata[parameter_name]['brick_domains'],
            dtype=parameter_context.param_type.value_encoding,
            fill_value=parameter_context.param_type.fill_value)
        self.value_list[parameter_name] = v

        self.expand_domain(parameter_context)

        return v

    def calculate_extents(self, origin, bD, parameter_name):
        """
        Calculates the Rtree extents, brick extents and active brick size for the parameter
        @param origin:
        @param bD:
        @param parameter_name:
        @return:
        """
        log.debug('origin: %s', origin)
        log.debug('bD: %s', bD)
        log.debug('parameter_name: %s', parameter_name)

        # Calculate the brick extents
        origin = list(origin)

        pc = self.parameter_dictionary.get_context(parameter_name)
        total_extents = pc.dom.total_extents # index space
        log.debug('Total extents for parameter %s: %s', parameter_name, total_extents)

        # Calculate the extents for the Rtree (index space)
        rtree_extents = origin + map(lambda o,s: o+s-1, origin, bD)
        # Fake out the rtree if rank == 1
        if len(origin) == 1:
            rtree_extents = [e for ext in zip(rtree_extents,[0 for x in rtree_extents]) for e in ext]
        log.debug('Rtree extents: %s', rtree_extents)

        # Calculate the extents of the brick (index space)
        brick_extents = zip(origin,map(lambda o,s: o+s-1, origin, bD))
        log.debug('Brick extents: %s', brick_extents)

        # Calculate active size using the inner extent of the domain within a brick (value space)
        brick_active_size = map(lambda o,s: min(o,s[1]+1)-s[0], total_extents, brick_extents)
        log.debug('Brick active size: %s', brick_active_size)

        return rtree_extents, brick_extents, brick_active_size

    def _brick_exists(self, parameter_name, brick_extents):
        # Make sure the brick doesn't already exist if we already have some bricks
        do_write = True
        brick_guid = ''
        log.debug('Check bricks for parameter \'%s\'',parameter_name)
        if parameter_name in self.parameter_metadata:
            for x,v in self.parameter_metadata[parameter_name]['brick_list'].iteritems():
                if brick_extents == v[0]:
                    log.debug('Brick found with matching extents: guid=%s', x)
                    do_write = False
                    brick_guid = x
                    break

        return do_write, brick_guid

    # Write empty HDF5 brick to the filesystem
    def write_brick(self,origin,bD,cD,parameter_name,data_type):
        root_path = '{0}/{1}/{2}'.format(self.root,self.guid,parameter_name)
        # Create a GUID for the brick
        brick_guid = create_guid()
        rtree_extents, brick_extents, brick_active_size = self.calculate_extents(origin, bD, parameter_name)

        do_write, bguid = self._brick_exists(parameter_name, brick_extents)
        if not do_write:
            log.debug('Brick already exists!  Updating brick metadata...')
            self.parameter_metadata[parameter_name]['brick_list'][bguid] = [brick_extents, origin, tuple(bD), brick_active_size]
            return

        log.debug('Writing virtual brick for parameter %s', parameter_name)

        # Set HDF5 file and group
        brick_file_name = '{0}.hdf5'.format(brick_guid)
        brick_file_path = '{0}/{1}'.format(root_path,brick_file_name)

        # Add brick to Master HDF file
        log.debug('Adding %s external link to %s.', brick_file_path, self.master_file_path)
        _master_file = h5py.File(self.master_file_path, 'r+')
        _master_file['/{0}/{1}'.format(parameter_name, brick_guid)] = h5py.ExternalLink('./{0}/{1}/{2}'.format(self.guid, parameter_name, brick_file_name), brick_guid)

        # Update the brick listing
        log.debug('Updating brick list[%s] with (%s, %s)', parameter_name, brick_guid, brick_extents)
        brick_count = self.parameter_brick_count(parameter_name)
        self.parameter_metadata[parameter_name]['brick_list'][brick_guid] = [brick_extents, origin, tuple(bD), brick_active_size]
        log.debug('Brick count for %s is %s', parameter_name, brick_count)

        # Close the master file
        _master_file.close()

        # Parameter Metadata
        _parameter_file_name = '{0}.hdf5'.format(parameter_name)
        _parameter_file_path = '{0}/{1}'.format(root_path,_parameter_file_name)
        _parameter_file = h5py.File(_parameter_file_path, 'a')
        _parameter_file.attrs['brick_list'] = pack(self.parameter_metadata[parameter_name]['brick_list'])
        _parameter_file.attrs['brick_domains'] = pack(self.parameter_metadata[parameter_name]['brick_domains'])
        _parameter_file.attrs['parameter_context'] = str(self.parameter_metadata[parameter_name]['parameter_context'].dump())
        # Insert into Rtree
        log.debug('Inserting into Rtree %s:%s:%s', brick_count, rtree_extents, brick_guid)
        self.parameter_metadata[parameter_name]['brick_tree'].insert(brick_count, rtree_extents, obj=brick_guid)
        _brick_tree_dataset = _parameter_file.require_dataset('rtree', shape=(brick_count,), dtype=h5py.new_vlen(str), maxshape=(None,))
        _brick_tree_dataset.resize((brick_count+1,))
        rtree_payload = pack((rtree_extents, brick_guid))
        log.debug('Inserting into brick tree dataset: [%s]: %s', brick_count, unpack(rtree_payload))
        _brick_tree_dataset[brick_count] = rtree_payload
        log.debug('Rtree inserted successfully.')
        _parameter_file.close()

    # Expand the domain
    def expand_domain(self, parameter_context):
        parameter_name = parameter_context.name
        log.debug('Expand %s', parameter_name)

        if self.parameter_metadata[parameter_name]['brick_domains'][0] is not None:
            log.debug('Expanding domain (n-dimension)')

            # Check if the number of dimensions of the total domain has changed
            # TODO: Will this ever happen???  If so, how to handle?
            if len(parameter_context.dom.total_extents) != len(self.parameter_metadata[parameter_name]['brick_domains'][0]):
                raise SystemError('Number of dimensions for parameter cannot change, only expand in size! No action performed.')
            else:
                tD = self.parameter_metadata[parameter_name]['brick_domains'][0]
                bD = self.parameter_metadata[parameter_name]['brick_domains'][1]
                cD = self.parameter_metadata[parameter_name]['brick_domains'][2]
                new_domain = parameter_context.dom.total_extents

                delta_domain = [(x - y) for x, y in zip(new_domain, tD)]
                log.debug('delta domain: %s', delta_domain)

                tD = [(x + y) for x, y in zip(tD, delta_domain)]
                self.parameter_metadata[parameter_name]['brick_domains'][0] = tD
        else:
            tD = parameter_context.dom.total_extents
            bricking_scheme = self.parameter_metadata[parameter_name]['brick_domains'][3]
            bD,cD = self.calculate_brick_size(tD, bricking_scheme)
            self.parameter_metadata[parameter_name]['brick_domains'] = [tD, bD, cD, bricking_scheme]

        try:
            # Gather block list
            log.debug('tD, bD, cD: %s, %s, %s', tD, bD, cD)
            lst = [range(d)[::bD[i]] for i,d in enumerate(tD)]

            # Gather brick vertices
            vertices = list(itertools.product(*lst))

            if len(vertices)>0:
                log.debug('Number of Bricks to Create: %s', len(vertices))

                # Write brick to HDF5 file
                map(lambda origin: self.write_brick(origin,bD,cD,parameter_name,parameter_context.param_type.value_encoding), vertices)

                log.info('Persistence Layer Successfully Initialized')
            else:
                log.debug('No bricks to create yet since the total domain is empty...')
        except Exception:
            raise

    # Returns a count of bricks for a parameter
    def parameter_brick_count(self, parameter_name):
        ret = 0
        if parameter_name in self.parameter_metadata:
            ret = len(self.parameter_metadata[parameter_name]['brick_list'])
        else:
            log.debug('No bricks found for parameter: %s', parameter_name)

        return ret

class PersistedStorage(AbstractStorage):

    def __init__(self, brick_path, brick_tree, brick_list, brick_domains, dtype=None, fill_value=None, **kwargs):
        """

        @param **kwargs Additional keyword arguments are copied and the copy is passed up to AbstractStorage; see documentation for that class for details
        """
        kwc=kwargs.copy()
        AbstractStorage.__init__(self, dtype=dtype, fill_value=fill_value, **kwc)

        # Rtree of bricks for parameter
        self.brick_tree = brick_tree

        # Filesystem path to HDF brick file(s)
        self.brick_path = brick_path

        # Listing of bricks and their metadata for parameter
        self.brick_list = brick_list

        self.brick_domains = brick_domains

    # Calculates the bricks from Rtree (brick_tree) using the slice_
    def _bricks_from_slice(self, slice_):
        # Make sure we don't modify the global slice_ object
        sl = deepcopy(slice_)

        # Ensure the slice_ is iterable
        if not isinstance(sl, (list,tuple)):
            sl = [sl]

        # Check the rank of the slice and pad if == 1 to satisfy Rtree requirement of rank >= 2
        rank = len(sl)
        if rank == 1:
            rank += 1
            sl += (0,)

        if self.brick_tree.properties.dimension != rank:
            raise ValueError('slice_ is of incorrect rank: is {0}, must be {1}'.format(rank, self.brick_tree.properties.dimension))

        bnds = self.brick_tree.bounds

        # Perform the calculations for the slice_ start and stop bounds in Rtree format
        start = []
        end = []
        for x in xrange(rank):
            sx=sl[x]
            if isinstance(sx, slice):
                si=sx.start if sx.start is not None else bnds[x::rank][0]
                start.append(si)
                ei=sx.stop-1 if sx.stop is not None else bnds[x::rank][1]
                end.append(ei)
            elif isinstance(sx, (list, tuple)):
                start.append(min(sx))
                end.append(max(sx))
            elif isinstance(sx, int):
                start.append(sx)
                end.append(sx)

        hits = list(self.brick_tree.intersection(tuple(start+end), objects=True))
        return [(h.id,h.object) for h in hits]

    def __getitem__(self, slice_):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('getitem slice_: %s', slice_)

        arr_shp = self._get_array_shape_from_slice(slice_)

        ret_arr = np.empty(arr_shp, dtype=self.dtype)
        ret_arr.fill(self.fill_value)
        ret_origin = [0 for x in range(ret_arr.ndim)]
        log.debug('Shape of returned array: %s', ret_arr.shape)

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
            # Figuring out which part of brick to set values
            log.debug('Return array origin: %s', ret_origin)
            brick_slice, value_slice = self._calc_slices(slice_, brick_guid, ret_arr, ret_origin)
            log.debug('Brick slice to extract: %s', brick_slice)
            log.debug('Value slice to fill: %s', value_slice)

            if not os.path.exists(brick_file_path):
                log.debug('Expected brick_file \'%s\' not found, passing back empty array...', brick_file_path)
#                v = np.empty(ret_arr.shape, dtype=self.dtype)
            else:
                log.debug('Found brick file: %s', brick_file_path)
                brick_file = h5py.File(brick_file_path, 'r+')
                v = brick_file[brick_guid].__getitem__(*brick_slice)

                # Check if object type
                if self.dtype == '|O8':
                    if not hasattr(v, '__iter__'):
                        v = [v]
                    v = [unpack(x) for x in v]

                ret_arr[value_slice] = v

                brick_file.close()

        return ret_arr

    def __setitem__(self, slice_, value):
        if not isinstance(slice_, (list,tuple)):
            slice_ = [slice_]
        log.debug('setitem slice_: %s', slice_)
        val = np.asanyarray(value)
        val_origin = [0 for x in range(val.ndim)]

        bricks = self._bricks_from_slice(slice_)
        log.debug('Slice %s indicates bricks: %s', slice_, bricks)

        for idx, brick_guid in bricks:
            brick_file_path = '{0}/{1}.hdf5'.format(self.brick_path, brick_guid)
#            if os.path.exists(brick_file):
            log.debug('Found brick file: %s', brick_file_path)

            # Figuring out which part of brick to set values
            brick_slice, value_slice = self._calc_slices(slice_, brick_guid, val, val_origin)
            log.debug('Brick slice to fill: %s', brick_slice)
            log.debug('Value slice to extract: %s', value_slice)

            # TODO: Move this to writer function

            brick_file = h5py.File(brick_file_path, 'a')

            # Check for object type
            data_type = self.dtype
            if data_type == '|O8':
                data_type = h5py.new_vlen(str)

            # Create the HDF5 dataset that represents one brick
            bD = tuple(self.brick_domains[1])
            cD = self.brick_domains[2]
            fv = pack(self.fill_value) if self.dtype == '|O8' else self.fill_value
            brick_cubes = brick_file.require_dataset(brick_guid, shape=bD, dtype=data_type, chunks=cD, fillvalue=fv)

            v = val if value_slice is None else val[value_slice]

            # Check for object type
            if self.dtype == '|O8':
                if not hasattr(v, '__iter__'):
                    v = [v]
                v = [pack(x) for x in v]

            brick_file[brick_guid].__setitem__(*brick_slice, val=v)

            brick_file.close()

#            else:
#                raise SystemError('Can\'t find brick: %s', brick_file)

    def _calc_slices(self, slice_, brick_guid, value, val_origin):
        brick_origin, _, brick_size = self.brick_list[brick_guid][1:]
        log.debug('Brick %s:  origin=%s, size=%s', brick_guid, brick_origin, brick_size)
        log.debug('Slice set: %s', slice_)

        brick_slice = []
        value_slice = []

        # Get the value into a numpy array - should do all the heavy lifting of sorting out what's what!!!
        val_arr = np.asanyarray(value)
        val_shp = val_arr.shape
        val_rank = val_arr.ndim
        log.debug('Value asanyarray: rank=%s, shape=%s', val_rank, val_shp)

        if val_origin is None or len(val_origin) == 0:
            val_ori = [0 for x in range(len(slice_))]
        else:
            val_ori = val_origin

        for i, sl in enumerate(slice_):
            bo=brick_origin[i]
            bs=brick_size[i]
            bn=bo+bs
            vo=val_ori[i]
            vs=val_shp[i] if len(val_shp) > 0 else None
            log.debug('i=%s, sl=%s, bo=%s, bs=%s, bn=%s, vo=%s, vs=%s',i,sl,bo,bs,bn,vo,vs)
            if isinstance(sl, int):
                if bo <= sl < bn: # The slice is within the bounds of the brick
                    brick_slice.append(sl-bo) # brick_slice is the given index minus the brick origin
                    value_slice.append(0 + vo)
                    val_ori[i] = vo + 1
                else: # TODO: If you specify a brick boundary this occurs [10] or [6]
                    raise ValueError('Specified index is not within the brick: {0}'.format(sl))
            elif isinstance(sl, (list,tuple)):
                lb = [x - bo for x in sl if bo <= x < bn]
                if len(lb) == 0: # TODO: In a list the last brick boundary seems to break it [1,3,10]
                    raise ValueError('None of the specified indices are within the brick: {0}'.format(sl))
                brick_slice.append(lb)
                value_slice.append(slice(vo, vo+len(lb), None)) # Everything from the appropriate index to the size needed
                val_ori[i] = len(lb) + vo
            elif isinstance(sl, slice):
                if sl.start is None:
                    start = 0
                else:
                    if bo <= sl.start < bn:
                        start = sl.start - bo
                    elif bo > sl.start:
                        start = 0
                    else: #  sl.start > bn
                        raise ValueError('The slice is not contained in this brick (sl.start > bn)')

                if sl.stop is None:
                    stop = bs
                else:
                    if bo < sl.stop <= bn:
                        stop = sl.stop - bo
                    elif sl.stop > bn:
                        stop = bs
                    else: #  bo > sl.stop
                        raise ValueError('The slice is not contained in this brick (bo > sl.stop)')

                log.debug('start=%s, stop=%s', start, stop)
                nbs = slice(start, stop, sl.step)
                nbsl = len(range(*nbs.indices(stop)))
                log.debug('nbsl=%s',nbsl)
                if vs is not None and vs != vo:
                    while nbsl > (vs-vo):
                        stop -= 1
                        nbs = slice(start, stop, sl.step)
                        nbsl = len(range(*nbs.indices(stop)))

                log.debug('nbsl=%s',nbsl)
                brick_slice.append(nbs)
                vstp = vo+nbsl
                log.debug('vstp=%s',vstp)
                if vs is not None and vstp > vs: # Don't think this will ever happen, should be dealt with by active_brick_size logic...
                    log.warn('Value set not the proper size for setter slice (vstp > vs)!')
#                    vstp = vs

                value_slice.append(slice(vo, vstp, None))
                val_ori[i] = vo + nbsl

        if val_origin is not None and len(val_origin) != 0:
            val_origin = val_ori
        else:
            value_slice = None

        return brick_slice, value_slice
    # TODO: Does not support n-dimensional
    def _get_array_shape_from_slice(self, slice_):
        log.debug('Getting array shape for slice_: %s', slice_)

        vals = self.brick_list.values()
        log.debug('vals: %s', vals)
        # Calculate the min and max brick value indices for each dimension
        if len(vals[0][1]) > 1:
            min_len = min([min(*x[0][i])+1 for i,x in enumerate(vals)])
            max_len = max([min(*x[0][i])+min(x[3]) for i,x in enumerate(vals)])
        else:
            min_len = min([min(*x[0])+1 for i,x in enumerate(vals)])
            max_len = max([min(*x[0])+min(x[3]) for i,x in enumerate(vals)])

        maxes = [max_len, min_len]

        # Calculate the shape base on the type of slice_
        shp = []
        for i, s in enumerate(slice_):
            if isinstance(s, int):
                shp.append(1)
            elif isinstance(s, (list,tuple)):
                shp.append(len(s))
            elif isinstance(s, slice):
                shp.append(len(range(*s.indices(maxes[i])))) # TODO: Does not support n-dimensional

        return tuple(shp)

    def expand(self, arrshp, origin, expansion, fill_value=None):
        pass # No op

    def fill(self, value):
        pass # No op

    def __len__(self):
        # TODO: THIS IS NOT CORRECT
        return 1

    def __iter__(self):
        # TODO: THIS IS NOT CORRECT
        return [1,1].__iter__()

class InMemoryPersistenceLayer(object):

    def expand_domain(self, *args, **kwargs):
        # No Op - storage expanded by *Value classes
        pass

    def init_parameter(self, parameter_context, *args, **kwargs):
        return InMemoryStorage(dtype=parameter_context.param_type.value_encoding, fill_value=parameter_context.param_type.fill_value)

