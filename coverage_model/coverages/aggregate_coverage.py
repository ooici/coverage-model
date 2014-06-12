__author__ = 'casey'

from ooi.logging import log
from coverage_model.coverage import *
from coverage_model.parameter import ParameterDictionary
from coverage_model.parameter_data import NumpyDictParameterData
from coverage_model.parameter_values import get_value_class
from coverage_model.persistence import is_persisted
from coverage_model.storage.parameter_persisted_storage import PostgresPersistenceLayer
from coverage_model.utils import Interval


class AggregateCoverage(AbstractCoverage):
    """
    References 1-n coverages
    """
    def __init__(self, root_dir, persistence_guid, name=None, reference_coverage_locs=None, reference_coverage_extents=None, parameter_dictionary=None,
                 mode=None, complex_type=ComplexCoverageType.PARAMETRIC_STRICT, temporal_domain=None, spatial_domain=None):

        # Initializes base class with proper mode.
        super(AggregateCoverage, self).__init__(mode)

        try:
            # Make sure root_dir and persistence_guid are both not None and are strings
            if not isinstance(root_dir, basestring) or not isinstance(persistence_guid, basestring):
                raise TypeError('\'root_dir\' and \'persistence_guid\' must be instances of basestring')

            root_dir = root_dir if not root_dir.endswith(persistence_guid) else os.path.split(root_dir)[0]

            if is_persisted(root_dir, persistence_guid):
                self._existing_coverage(root_dir, persistence_guid)
            else:
                self._new_coverage(root_dir, persistence_guid, name, reference_coverage_locs, parameter_dictionary, complex_type, reference_coverage_extents)
        except:
            self._closed = True
            raise
        self._do_build()

    def _existing_coverage(self, root_dir, persistence_guid):
        if not is_persisted(root_dir, persistence_guid):
            raise SystemError('Cannot find specified coverage: {0}'.format(persistence_guid))
        self._persistence_layer = PostgresPersistenceLayer(root_dir, persistence_guid, mode=self.mode)
        if self._persistence_layer.version != self.version:
            raise IOError('Coverage Model Version Mismatch: %s != %s' %(self.version, self._persistence_layer.version))
        self.name = self._persistence_layer.name
        self.mode = self.mode
        self._reference_covs = collections.OrderedDict()

    def _new_coverage(self, root_dir, persistence_guid, name, reference_coverage_locs, parameter_dictionary, complex_type, reference_coverage_extents={}):
        # Coverage doesn't exist, make a new one
        if reference_coverage_locs is None or name is None:
            raise SystemError('\'reference_coverages\' and \'name\' cannot be None')
        if not isinstance(name, basestring):
            raise TypeError('\'name\' must be of type basestring')
        self.name = name
        if parameter_dictionary is None:
            parameter_dictionary = ParameterDictionary()

        # Must be in 'a' for a new coverage
        self.mode = 'a'

        self._reference_covs = collections.OrderedDict()

        if not hasattr(reference_coverage_locs, '__iter__'):
            reference_coverage_locs = [reference_coverage_locs]

        self._persistence_layer = PostgresPersistenceLayer(root_dir,
                                                           persistence_guid,
                                                           name=self.name,
                                                           mode=self.mode,
                                                           param_dict=parameter_dictionary,
                                                           rcov_locs=reference_coverage_locs,
                                                           rcov_extents=reference_coverage_extents,
                                                           complex_type=complex_type,
                                                           coverage_type='complex',
                                                           version=self.version)

        for pc in parameter_dictionary.itervalues():
            self.append_parameter(pc[1])

    def close(self, force=False, timeout=None):
        if not hasattr(self, '_closed'):
            # _closed is the first attribute added to the coverage object (in AbstractCoverage)
            # If it's not there, a TypeError has likely occurred while instantiating the coverage
            # nothing else exists and we can just return
            return
        if not self._closed:
            for cov_pth, cov in self._reference_covs.iteritems():
                log.info('Closing reference coverage \'%s\'', cov.name if hasattr(cov,'name') else 'unnamed')
                cov.close(force, timeout)

        AbstractCoverage.close(self, force, timeout)

    def append_parameter(self, parameter_context):
        raise NotImplementedError('Parameter value retrieval not implemented.')

    def append_reference_coverage(self, path, **kwargs):
        ncov = AbstractCoverage.load(path)
        ncov.close()

        # Loading the coverage worked - proceed...
        # Get the current set of reference coverages
        if path in self._persistence_layer.rcov_locs:
            # Already there, note it and just return
            log.info('Coverage already referenced: \'%s\'', path)
        else:
            self._persistence_layer.rcov_locs.append(path)

        self._do_build()

    def _do_build(self):
        # Reset things to ensure we don't munge everything
        self._reference_covs = collections.OrderedDict()
        self._range_dictionary = ParameterDictionary()
        self._range_value = RangeValues()
        self._reference_covs = self._build_ordered_coverage_dict()

        from coverage_model.storage.parameter_persisted_storage import PostgresPersistedStorage
        for parameter_name in self._persistence_layer.parameter_metadata:
            md = self._persistence_layer.parameter_metadata[parameter_name]
            mm = self._persistence_layer.master_manager
            pc = md.parameter_context

            # Assign the coverage's domain object(s)
            self._assign_domain(pc)

            # Get the callbacks for ParameterFunctionType parameters
            if hasattr(pc, '_pval_callback'):
                pc._pval_callback = self.get_parameter_values
                pc._pctxt_callback = self.get_parameter_context
            self._range_dictionary.add_context(pc)
            s = PostgresPersistedStorage(md, metadata_manager=mm, parameter_context=pc, dtype=pc.param_type.storage_encoding, fill_value=pc.param_type.fill_value, mode=self._persistence_layer.mode)
            self._persistence_layer.value_list[parameter_name] = s
            self._range_value[parameter_name] = get_value_class(param_type=pc.param_type, domain_set=pc.dom, storage=s)


    def _build_ordered_coverage_dict(self):
        covs = self._verify_rcovs(self._persistence_layer.rcov_locs)
        cov_dict = collections.OrderedDict()
        cov_list = []
        for i in covs:
            cov = i[1]
            if isinstance(cov, AbstractCoverage):
                temporal_bounds = cov.get_data_bounds(cov.temporal_parameter_name)
                cov_list.append((temporal_bounds[0], temporal_bounds[1], cov))

        cov_list.sort(key=lambda tup: tup[0])

        for start, end, cov in cov_list:
            if isinstance(cov, AbstractCoverage):
                cov_dict[cov.persistence_guid] = cov
                self._head_coverage_path = cov.head_coverage_path

        return cov_dict

    def interval_map(self):
        '''
        Builds a reference structure and returns the bounds and the associated reference coverages
        note: only works for 1-d right now
        '''
        interval_map = []
        for scov in self._reference_covs.itervalues():
            interval = scov.get_data_bounds(scov.temporal_parameter_name)
            interval = Interval(interval[0], interval[1], None, None)
            interval_map.append((interval, scov))
        self._interval_qsort(interval_map)
        return interval_map

    @classmethod
    def _interval_swap(cls, arr, x0, x1):
        if x0 != x1:
            t = arr[x0]
            arr[x0] = arr[x1]
            arr[x1] = t
    @classmethod
    def _interval_pivot(cls, arr, left, right, pivot):
        val = arr[pivot][0]
        cls._interval_swap(arr, pivot, right)
        store_index = left
        for i in xrange(left, right):
            if arr[i][0] < val:
                cls._interval_swap(arr, i, store_index)
                store_index += 1
        cls._interval_swap(arr, store_index, right)
        return store_index

    @classmethod
    def _interval_qsort(cls, arr, left=None, right=None):
        '''
        Quicksort for the interval map
        '''
        if left is None:
            left = 0
        if right is None:
            right = len(arr) - 1
        if left < right:
            pivot = (right - left) / 2 + left
            pivot = cls._interval_pivot(arr, left, right, pivot)
            cls._interval_qsort(arr, left, pivot-1)
            cls._interval_qsort(arr, pivot+1, right)

    def _merge_value_dicts(self, value_dicts, override_temporal_key=None):
        total_size = 0
        dtype_map = {}
        for param_dict, coverage in value_dicts:
            skip_coverage = False
            temporal_key = coverage.temporal_parameter_name
            if override_temporal_key is not None:
                temporal_key = override_temporal_key
            cov_dict_size = param_dict[temporal_key].size
            for key, np_arr in param_dict.iteritems():
                if np_arr.size != cov_dict_size:
                    log.error("Internal coverage parameter dictionaries don't align! Skipping coverage")
                    skip_coverage = True
                    break
                if key not in dtype_map:
                    dtype_map[key] = np_arr.dtype
                else:
                    if dtype_map[key] != np_arr.dtype:
                        dtype_map[key] = np.dtype('object')
            if not skip_coverage:
                total_size += cov_dict_size

        return_dict = {}
        for key, dt in dtype_map.iteritems():
            arr = np.empty(total_size, dtype=dt)
            arr[:] = None
            return_dict[key] = arr

        current_index = 0
        for param_dict, coverage in value_dicts:
            if isinstance(coverage, SimplexCoverage):
                temporal_key = coverage.temporal_parameter_name
                if override_temporal_key is not None:
                    temporal_key = override_temporal_key
                size = param_dict[temporal_key].size
                for key in dtype_map.keys():
                    if key in param_dict:
                        return_dict[key][current_index:current_index+size] = param_dict[key]
                    elif key in coverage.list_parameters():
                        return_dict[key][current_index:current_index+size] = coverage.get_parameter_context(key).param_type.fill_value()
                current_index += size

        return return_dict

    def _add_coverage_array(cls, param_dict, size, cov_id):
        arr = np.chararray(size, len(cov_id))
        arr[:] = cov_id
        tmp_dict = {'coverage_id': arr}
        param_dict.update(tmp_dict)

    def get_time_values(self, time_segement=None, stride_length=None, return_value=None):
        cov_value_list = []
        dummy_key = "stripped_later"
        for coverage in self._reference_covs.values():
            if isinstance(coverage, AbstractCoverage):
                params = coverage.get_time_values(time_segement, stride_length, return_value)
                cov_dict = {dummy_key: params}
                cov_value_list.append((cov_dict, coverage))

        combined_data = self._merge_value_dicts(cov_value_list, override_temporal_key=dummy_key)
        from coverage_model.util.numpy_utils import sort_flat_arrays
        if dummy_key in combined_data:
            combined_data = sort_flat_arrays(combined_data, dummy_key)
            return combined_data[dummy_key] #TODO: Handle case where 'time' may not be temporal parameter name of all sub-coverages
        else:
            return np.array([])

    def get_parameter_values(self, param_names=None, time_segment=None, time=None,
                             sort_parameter=None, stride_length=None, return_value=None, fill_empty_params=False,
                             function_params=None, as_record_array=False):
        '''
        Obtain the value set for a given parameter over a specified domain
        '''

        get_times_too = self.temporal_parameter_name in param_names
        cov_value_list = []
        for coverage in self._reference_covs.values():
            if isinstance(coverage, SimplexCoverage):
                if param_names is not None:
                    this_param_names = set(param_names)
                    this_param_names = this_param_names.intersection(set(coverage.list_parameters()))
                    this_param_names = list(this_param_names)
                params = coverage.get_parameter_values(this_param_names, time_segment, time, sort_parameter, stride_length,
                                                       return_value, fill_empty_params, function_params, as_record_array=False)
                if len(params.get_data()) == 1 and coverage.temporal_parameter_name in params.get_data() and not get_times_too:
                    continue
                cov_dict = params.get_data()
                size = cov_dict[coverage.temporal_parameter_name].size
                self._add_coverage_array(cov_dict, size, coverage.persistence_guid)
                if time is not None and time_segment is None:
                    new = cov_dict[coverage.temporal_parameter_name][0]
                    old = cov_value_list[0][0][coverage.temporal_parameter_name][0]
                    if abs(new-time) < abs(old-time):
                        cov_value_list = [(cov_dict, coverage)]
                else:
                    cov_value_list.append((cov_dict, coverage))
        combined_data = self._merge_value_dicts(cov_value_list)
        return NumpyDictParameterData(combined_data, alignment_key=sort_parameter, as_rec_array=as_record_array)


    @classmethod
    def _value_dict_swap(cls, value_dict, x0, x1):
        '''
        Value dictionary array swap
        '''
        if x0 != x1:
            for name,arr in value_dict.iteritems():
                t = arr[x0]
                arr[x0] = arr[x1]
                arr[x1] = t

    @classmethod
    def _value_dict_pivot(cls, value_dict, axis, left, right, pivot):
        '''
        Pivot algorithm, part of quicksort
        '''
        axis_arr = value_dict[axis]
        idx_arr = value_dict['__idx__']
        val = axis_arr[pivot]
        cls._value_dict_swap(value_dict, pivot, right)
        store_index = left
        for i in xrange(left, right):
            if axis_arr[i] < val:
                cls._value_dict_swap(value_dict, i, store_index)
                store_index += 1
            # This part is critical to maintaining the precedence :)
            if axis_arr[i] == val and idx_arr[i] < idx_arr[right]:
                cls._value_dict_swap(value_dict, i, store_index)
                store_index += 1
                
        cls._value_dict_swap(value_dict, store_index, right)
        return store_index

    @classmethod
    def _value_dict_qsort(cls, value_dict, axis, left=None, right=None):
        '''
        Quicksort, value dictionary edition
        modifications are in-place for a stable search
        '''
        top_call = left is None and right is None
        if top_call:
            value_dict['__idx__'] = np.arange(len(value_dict[axis]))
        if left is None:
            left = 0
        if right is None:
            right = len(value_dict[axis]) - 1
        if left < right:
            pivot = (right - left) / 2 + left
            pivot = cls._value_dict_pivot(value_dict, axis, left, right, pivot)
            cls._value_dict_qsort(value_dict, axis, left, pivot-1)
            cls._value_dict_qsort(value_dict, axis, pivot+1, right)
        if top_call:
            del value_dict['__idx__']

    @classmethod
    def _value_dict_unique(cls, value_dict, axis):
        '''
        A naive unique copy algorithm

        Notes:
        - Last unique axis value has precedence
        '''
        tarray = value_dict[axis]
        truth_array = np.ones(tarray.shape, dtype=np.bool)
        for i in xrange(1, len(tarray)):
            if tarray[i-1] == tarray[i]:
                truth_array[i-1] = False

        vd_copy = {}
        for k,v in value_dict.iteritems():
            vd_copy[k] = v[truth_array]
        return vd_copy

    def _verify_rcovs(self, rcovs):
        for cpth in rcovs:
            if not os.path.exists(cpth):
                log.warn('Cannot find coverage \'%s\'; ignoring', cpth)
                continue

            pth, uuid = get_dir_and_id_from_path(cpth)
            if uuid in self._reference_covs:
                yield uuid, self._reference_covs[uuid]
                continue

            try:
                cov = AbstractCoverage.load(cpth)
            except Exception as ex:
                log.warn('Exception loading coverage \'%s\'; ignoring.  Exception: %s' % (cpth, ex.message))
                continue

            if cov.temporal_parameter_name is None:
                log.warn('Coverage \'%s\' does not have a temporal_parameter; ignoring' % cpth)
                continue

            yield cov.persistence_guid, cov

    def get_complex_type(self):
        return self._persistence_layer.complex_type

    def set_parameter_values(self, values):
        self._append_to_coverage(values)

    def set_time_values(self, values):
        self.set_parameter_values({self.temporal_parameter_name: values})

    def insert_value_set(self, value_dictionary):
        self._append_to_coverage(value_dictionary)

    def append_value_set(self, value_dictionary):
        self._append_to_coverage(value_dictionary)

    def _append_to_coverage(self, values):
        raise NotImplementedError('Aggregate coverages are read-only')

    def num_timesteps(self):
        ts = 0
        for coverage in self._reference_covs.values():
            ts += coverage.num_timesteps()
        return ts

    def get_data_bounds(self, parameter_name=None):
        if parameter_name is None:
            parameter_name = self.list_parameters()
        if isinstance(parameter_name, Iterable) and not isinstance(parameter_name, basestring) and len(parameter_name)>1:
            rt = {}
            for coverage in self._reference_covs.values():
                for param in parameter_name:
                    if param in coverage.list_parameters():
                        bounds = coverage.get_data_bounds(param)
                        if param in rt:
                            rt[param] = (min(bounds[0], rt[param][0]), max(bounds[1], rt[param][1]))
                        else:
                            rt[param] = bounds
            return rt
        else:
            rt = None
            for coverage in self._reference_covs.values():
                bounds = coverage.get_data_bounds(parameter_name)
                if rt is None:
                    rt = bounds
                else:
                    rt = (min(bounds[0], rt[0]), max(bounds[1], rt[1]))
            return rt