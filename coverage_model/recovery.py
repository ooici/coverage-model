#!/usr/bin/env python

"""
@package coverage_model.recovery
@file coverage_model/recovery.py
@author Christopher Mueller
@author James Case
@brief Contains utility functions for attempting to recover corrupted coverages
"""
from coverage_model import hdf_utils, SimplexCoverage, AbstractCoverage, ParameterDictionary, GridDomain
from coverage_model.persistence_helpers import pack
from coverage_model.basic_types import BaseEnum, Span
import os
import shutil
import tempfile
import h5py
from pyon.public import log

MASTER_ATTRS = [
    'auto_flush_values',
    'file_path',
    'global_bricking_scheme',
    'guid',
    'inline_data_writes',
    'name',
    'root_dir',
    'sdom',
    'tdom',]

PARAMETER_ATTRS = [
    'brick_domains',
    'brick_list',
    'file_path',
    'parameter_context',
    'parameter_name',
    'root_dir',
    'tree_rank',]

class StatusEnum(BaseEnum):
    UNSET = 'UNSET'

    UNKNOWN = 'UNKNOWN'

    NORMAL = 'NORMAL'

    CORRUPT = 'CORRUPT'


class AnalysisResult(object):

    def __init__(self):
        self._master = StatusEnum.UNSET
        self._params = set()
        self._bricks = {}

        self._results = {}
        self._metrics = {}

    def _safe_del_metrics(self, key):
        if self._metrics.has_key(key):
            del self._metrics[key]

    @property
    def master_status(self):
        return self._results['master']

    def set_master_status(self, pth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))
        self._results['master'] = (pth, status, size_ratio)

        # Remove the total entries from _metrics so they can be regenerated
        self._safe_del_metrics('tfc')
        self._safe_del_metrics('tcorr')
        self._safe_del_metrics('tsr')

    def add_param_status(self, pname, ppth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))

        self._results[pname] = {'param': (ppth, status, size_ratio),
                                  'bricks': []}

        # Remove the parameter entries from _metrics so they can be regenerated
        self._safe_del_metrics('pfc')
        self._safe_del_metrics('pcorr')
        self._safe_del_metrics('psr')

    def add_brick_status(self, pname, bpth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))

        if not pname in self._results:
            raise TypeError('Parameter \'{0}\' has not been added, please add parameters before bricks'.format(pname))

        self._results[pname]['bricks'].append((bpth, status, size_ratio))

        # Remove the brick entries from _metrics so they can be regenerated
        self._safe_del_metrics('bfc')
        self._safe_del_metrics('bcorr')
        self._safe_del_metrics('bsr')

    @property
    def total_file_count(self):
        if not 'tfc' in self._metrics:
            self._metrics['tfc'] = sum([1, self.brick_file_count, self.param_file_count])
        return self._metrics['tfc']

    @property
    def brick_file_count(self):
        if not 'bfc' in self._metrics:
            self._metrics['bfc'] = sum([len(self._results[p]['bricks']) for p in self._results.keys() if p != 'master'])

        return self._metrics['bfc']

    @property
    def param_file_count(self):
        if not 'pfc' in self._metrics:
            self._metrics['pfc'] = len(self._results) - 1 # Length of results minus 1 for 'master'
        return self._metrics['pfc']

    def get_master_corruption(self):
        if not 'mcorr' in self._metrics:
            corruptions = set()
            if self._results['master'][1] == StatusEnum.CORRUPT:
                corruptions.add(self._results['master'][0])

            self._metrics['mcorr'] = list(corruptions)
        return self._metrics['mcorr']

    def get_param_corruptions(self):
        if not 'pcorr' in self._metrics:
            corruptions = set()

            corruptions.update([self._results[p]['param'][0] for p in self._results if p != 'master' and self._results[p]['param'][1] == StatusEnum.CORRUPT])

            ret = list(corruptions)
            ret.sort()
            self._metrics['pcorr'] = ret
        return self._metrics['pcorr']

    def get_brick_corruptions(self):
        if not 'bcorr' in self._metrics:
            corruptions = set()

            for p in self._results:
                if p != 'master':
                    for b in self._results[p]['bricks']:
                        if b[1] == StatusEnum.CORRUPT:
                            corruptions.add(b[0])

    #        corruptions.update([b[0] for p in self._results for b in self._results[p]['bricks'] if p != 'master' and b[1] == StatusEnum.CORRUPT])

            ret = list(corruptions)
            ret.sort()
            self._metrics['bcorr'] = ret
        return self._metrics['bcorr']

    def get_corruptions(self):
        if not 'tcorr' in self._metrics:
            corruptions = set()
            corruptions.update(self.get_master_corruption())
            corruptions.update(self.get_param_corruptions())
            corruptions.update(self.get_brick_corruptions())

            ret = list(corruptions)
            ret.sort()
            self._metrics['tcorr'] = ret
        return self._metrics['tcorr']

    def get_master_size_ratio(self):
        if not 'msr' in self._metrics:
            ratios = set()

            ratios.add((self._results['master'][0], self._results['master'][2]))

            self._metrics['msr'] = list(ratios)
        return self._metrics['msr']

    def get_param_size_ratios(self):
        if not 'psr' in self._metrics:
            ratios = set()

            ratios.update([(self._results[p]['param'][0], self._results[p]['param'][2]) for p in self._results if p != 'master'])

            ret = list(ratios)
            ret.sort()
            self._metrics['psr'] = ret
        return self._metrics['psr']

    def get_brick_size_ratios(self):
        if not 'bsr' in self._metrics:
            ratios = set()

            for p in self._results:
                if p != 'master':
                    for b in self._results[p]['bricks']:
                        ratios.add((b[0],b[2]))

    #        ratios.update([(b[0], b[1]) for p in self._results for b in self._results[p]['bricks'] if p != 'master'])

            ret = list(ratios)
            ret.sort()
            self._metrics['bsr'] = ret
        return self._metrics['bsr']

    def get_size_ratios(self):
        if not 'tsr' in self._metrics:
            ratios = set()
            ratios.update(self.get_master_size_ratio())
            ratios.update(self.get_param_size_ratios())
            ratios.update(self.get_brick_size_ratios())

            ret = list(ratios)
            ret.sort()
            self._metrics['tsr'] = list(ratios)
        return self._metrics['tsr']

    @property
    def is_corrupt(self):
        if len(self.get_corruptions()) == 0:
            return False

        return True

class CoverageDoctor(object):

    def __init__(self, coverage_path, data_product_obj, dataset_obj):
        if not os.path.exists(coverage_path):
            raise TypeError('\'coverage_path\' does not exist or is unreachable')

#        if not isinstance(data_product_obj, object):
#            raise TypeError('\'data_product_obj\' must be an instance of DataProduct')
#
#        if not isinstance(dataset_obj, object):
#            raise TypeError('\'dataset_obj\' must be an instance of DataSet')

        self.cov_pth = coverage_path
        self._root, self._guid = os.path.split(self.cov_pth)
        self._inner_dir = os.path.join(self.cov_pth, self._guid)
        self._dpo = data_product_obj
        self._dso = dataset_obj

        self._ar = None

    @property
    def total_file_count(self):
        if self._ar:
            return self._ar.total_file_count

    @property
    def brick_file_count(self):
        if self._ar:
            return self._ar.brick_file_count

    @property
    def param_file_count(self):
        if self._ar:
            return self._ar.param_file_count

    def _do_analysis(self, analyze_bricks=False):
        ar = AnalysisResult()

        master_pth = os.path.join(self.cov_pth, self._guid + '_master.hdf5')

        st = StatusEnum.CORRUPT if hdf_utils.has_corruption(master_pth) else StatusEnum.NORMAL
        if st == StatusEnum.CORRUPT:
                sz = -1
        else:
            sz=hdf_utils.space_ratio(master_pth)
        ar.set_master_status(master_pth, st, sz)

        for p in os.walk(self._inner_dir).next()[1]:
            pset = self._get_parameter_fileset(p)

            # Check parameter file
            st = StatusEnum.CORRUPT if hdf_utils.has_corruption(pset['param']) else StatusEnum.NORMAL
            if st == StatusEnum.CORRUPT:
                sz = -1
            else:
                sz=hdf_utils.space_ratio(pset['param'])
            ar.add_param_status(p, pset['param'], st, sz)

            # Check each brick file
            for b_pth in pset['bricks']:
                if analyze_bricks:
                    st = StatusEnum.CORRUPT if hdf_utils.has_corruption(b_pth) else StatusEnum.NORMAL
                    if st == StatusEnum.CORRUPT:
                        sz = -1
                    else:
                        sz=hdf_utils.space_ratio(b_pth)
                else:
                    st = StatusEnum.UNKNOWN
                    sz = -1

                ar.add_brick_status(p, b_pth, st, sz)

        return ar

    def analyze(self, analyze_bricks=False, reanalyze=False):
        if self._ar is None or reanalyze:
            ar = self._do_analysis(analyze_bricks=analyze_bricks)
            self._ar = ar
        return self._ar

    def repair(self, reanalyze=False):
        """
        Heavy repair tool that recreates a blank persisted Coverage from the broken coverage's
        original construction parameters, then reconstructs the Master and Parameter metadata
        files by inspection of the ION objects and "valid" brick files.
        @return:
        """
        if self._ar is None or reanalyze:
            self._ar = self._do_analysis(analyze_bricks=True)

        if self._ar.is_corrupt:
            if len(self._ar.get_brick_corruptions()) > 0:
                raise NotImplementedError('Brick corruption.  Cannot repair at this time!!!')
            else:
                # Repair the Master and Parameter metadata files

                # Need the ParameterDictionary, TemporalDomain and SpatialDomain
                pdict = ParameterDictionary.load(self._dso.parameter_dictionary)
                tdom = GridDomain.load(self._dso.temporal_domain)
                sdom = GridDomain.load(self._dso.spatial_domain)

                # Set up the working directory for the recovered coverage
                tempcov_dir = tempfile.mkdtemp('covs')

                # Create the temporary Coverage
                tempcov = SimplexCoverage(root_dir=tempcov_dir, persistence_guid=self._guid, name=self._guid, parameter_dictionary=pdict, spatial_domain=sdom, temporal_domain=tdom)

                # Set up the original and temporary coverage path strings
                orig_dir = os.path.join(self.cov_pth, self._guid)
                temp_dir = os.path.join(tempcov.persistence_dir, tempcov.persistence_guid)

                # Insert same number of timesteps into temporary coverage as in broken coverage
                brick_domains_new, new_brick_list, brick_list_spans, tD, bD, min_data_bound, max_data_bound = self.inspect_bricks(self.cov_pth, self._guid, 'time')
                bls = [s.value for s in brick_list_spans]
                maxes = [sum(b[3]) for b in new_brick_list.values()]
                tempcov.insert_timesteps(sum(maxes))

                # Replace metadata is the Master file
                pl = tempcov._persistence_layer
                pl.master_manager.brick_domains = brick_domains_new
                pl.master_manager.brick_list = new_brick_list

                # Repair ExternalLinks to brick files
                f = h5py.File(pl.master_manager.file_path, 'a')
                for param_name in pdict.keys():
                    del f[param_name]
                    f.create_group(param_name)
                    for brick in bls:
                        link_path = '/{0}/{1}'.format(param_name, brick[0])
                        brick_file_name = '{0}.hdf5'.format(brick[0])
                        brick_rel_path = os.path.join(pl.parameter_metadata[param_name].root_dir.replace(tempcov.persistence_dir, '.'), brick_file_name)
                        log.debug('link_path: %s', link_path)
                        log.debug('brick_rel_path: %s', brick_rel_path)
                        pl.master_manager.add_external_link(link_path, brick_rel_path, brick[0])

                pl.flush_values()
                pl.flush()
                tempcov.close()

                # Remove 'rtree' dataset from Master file if it already exists (post domain expansion)
                # to make way for reconstruction
                f = h5py.File(pl.master_manager.file_path, 'a')
                if 'rtree' in f.keys():
                    del f['rtree']
                f.close()

                # Reconstruct 'rtree' dataset
                # Open temporary Coverage and PersistenceLayer objects
                fixed_cov = AbstractCoverage.load(tempcov.persistence_dir, mode='a')
                pl_fixed = fixed_cov._persistence_layer

                # Call update_rtree for each brick using PersistenceLayer builtin
                brick_count = 0

                for brick in bls:
                    rtree_extents, brick_extents, brick_active_size = pl_fixed.calculate_extents(brick[1][1],bD,tD)
                    pl_fixed.master_manager.update_rtree(brick_count, rtree_extents, obj=brick[0])
                    brick_count += 1

                # Update parameter_bounds property based on each parameter's brick data using deep inspection
                valid_bounds_types = [
                    'BooleanType',
                    'ConstantType',
                    'QuantityType',
                    'ConstantRangeType'
                ]

                for param in pdict.keys():
                    if pdict.get_context(param).param_type.__class__.__name__ in valid_bounds_types:
                        brick_domains_new, new_brick_list, brick_list_spans, tD, bD, min_data_bound, max_data_bound = self.inspect_bricks(self.cov_pth, self._guid, param)
                        # Update the metadata
                        pl_fixed.update_parameter_bounds(param, [min_data_bound, max_data_bound])
                pl_fixed.flush()
                fixed_cov.close()

                # Create backup copy of original Master and Parameter files
                import datetime
                orig_master_file = os.path.join(self.cov_pth, '{0}_master.hdf5'.format(self._guid))

                # Generate the timestamp
                tstamp_format = '%Y%m%d%H%M%S'
                tstamp = datetime.datetime.now().strftime(tstamp_format)

                backup_master_file = os.path.join(self.cov_pth, '{0}_master.{1}.hdf5'.format(self._guid, tstamp))

                shutil.copy2(orig_master_file, backup_master_file)

                for param in pdict.keys():
                    param_orig = os.path.join(orig_dir, param, '{0}.hdf5'.format(param))
                    param_backup = os.path.join(orig_dir, param, '{0}.{1}.hdf5'.format(param, tstamp))
                    shutil.copy2(param_orig, param_backup)

                # Copy Master and Parameter metadata files back to original/broken coverage (cov_pth) location
                shutil.copy2(os.path.join(tempcov.persistence_dir, '{0}_master.hdf5'.format(self._guid)), os.path.join(self.cov_pth, '{0}_master.hdf5'.format(self._guid)))
                for param in pdict.keys():
                    shutil.copy2(os.path.join(temp_dir, param, '{0}.hdf5'.format(param)), os.path.join(orig_dir, param, '{0}.hdf5'.format(param)))

                # Reanalyze the repaired coverage
                self._ar = self._do_analysis(analyze_bricks=True)

                # Verify repair worked, clean up if not
                if self._ar.is_corrupt:
                    # Remove backed up files and clean up the repair attempt
                    log.info('Repair attempt failed.  Reverting to pre-repair state.')
                    # Use backup copy to replace post-repair file.
                    shutil.copy2(backup_master_file, orig_master_file)
                    # Delete the backup
                    os.remove(backup_master_file)

                    # Iterate over parameters and revert to pre-repair state
                    for param in pdict.keys():
                        param_orig = os.path.join(orig_dir, param, '{0}.hdf5'.format(param))
                        param_backup = os.path.join(orig_dir, param, '{0}.{1}.hdf5'.format(param, tstamp))
                        # Use backup copy to replace post-repair file.
                        shutil.copy2(param_backup, param_orig)
                        # Delete the backup
                        os.remove(param_backup)

                    raise ValueError('Coverage repair failed! Revert to stored backup version, if possible.')

                # Remove temporary coverage
                shutil.rmtree(tempcov_dir)
        else:
            log.info('Coverage is not corrupt, nothing to repair!')

    def inspect_bricks(self, cov_pth, dataset_id, param_name):
        brick_domains_new = None
        new_brick_list = None
        brick_list_spans = None
        tD = None
        bD = None
        min_data_bound = None
        max_data_bound = None

        spans = []
        pdir = os.path.join(cov_pth, dataset_id, param_name)
        # TODO: Check for brick files, if none then skip this entirely
        if os.path.exists(pdir) and len(os.listdir(pdir)) > 0:
            for brick in [os.path.join(pdir, x) for x in os.listdir(pdir) if (not param_name in x) and ('.hdf5' in x)]:
                brick_guid = os.path.basename(brick).replace('.hdf5', '')
                with h5py.File(brick, 'r') as f:
                    ds = f[brick_guid]
                    fv = ds.fillvalue
                    low = ds[0]
                    up = ds.value.max()
                    low = low if low != fv else None
                    if low < up:
                        spans.append(Span(lower_bound=low, upper_bound=up, value=brick_guid))


            if len(spans) > 0:
                spans.sort()
                min_data_bound = min([s.lower_bound for s in spans])
                max_data_bound = max([s.upper_bound for s in spans])

                bricks_sorted = [[s.value, s.lower_bound, s.upper_bound, int(s.upper_bound-s.lower_bound+1)] for s in spans]

                # brick_list
                # {'0BC3FF45-60FB-440A-980D-80C9CEB6F799': (((0, 99999),),
                #                                           (0,),
                #                                           (100000,),
                #                                           (100000,)),
                #  '5AF8B7A8-49DE-47FB-8671-B9BE8164CC8F': (((100000, 199999),),
                #                                           (100000,),
                #                                           (100000,),
                #                                           (29600,))}
                # TODO: Surely we can get a hold of the bricking_scheme from some global location???
                brick_size = 100000
                chunk_size = 100000
                start = 0
                stop = brick_size - 1
                brick_list_spans = []
                new_brick_list = {}
                for brick in bricks_sorted:
                    new_brick = [brick[0], (((start, stop),),
                                            (start,),
                                            (brick_size,),
                                            (brick[3],))]
                    brick_list_spans.append(Span(lower_bound=start, upper_bound=stop, value=new_brick))
                    new_brick_list[brick[0]] = (((start, stop),),
                                                (start,),
                                                (brick_size,),
                                                (brick[3],))
                    start = start + brick_size
                    stop = start + brick_size - 1
                brick_list_spans.sort()

                # brick_domains
                # [(129600,), (100000,), (100000,), {'brick_size': 100000, 'chunk_size': 100000}]
                maxes = [sum(b[3]) for b in new_brick_list.values()]
                tD = (sum(maxes),)
                bD = (brick_size,)
                cD = (chunk_size,)
                bricking_scheme = {}
                bricking_scheme['brick_size'] = brick_size
                bricking_scheme['chunk_size'] = chunk_size
                brick_domains_new = [tD, bD, cD, bricking_scheme]

        return brick_domains_new, new_brick_list, brick_list_spans, tD, bD, min_data_bound, max_data_bound

    def _copy_original_bricks(self, pdict, orig_dir, temp_dir):
        """
        Copies all parameter brick files from broken coverage to temporary coverage for analysis
        @param pdict:
        @param orig_dir:
        @param temp_dir:
        @return:
        """
        # TODO: This is not really necessary except for debugging.  We just look at the source for inspection anyway!!!
        for param_name in pdict.keys():
            brick_files = [os.path.join(orig_dir, param_name, x) for x in os.listdir(os.path.join(orig_dir, param_name)) if not param_name in x]

            for brick in brick_files:
                shutil.copy2(brick, os.path.join(temp_dir, param_name, os.path.basename(brick)))

    def _get_repr_file_path(self, orig):
        return orig.replace('.hdf5','_repr.hdf5')

    def _repair_file_attrs(self, orig_file, rpr_file, attr_callback, attr_list):
        # Sort out which attributes are bad and which aren't
        good_atts, bad_atts = self._diagnose_attrs(orig_file, attr_list)
        with h5py.File(rpr_file) as f:
            # Copy the good attributes
            for a in good_atts:
                f.attrs[a] = good_atts[a]

            # Add new values for the bad attributes
            for a in bad_atts:
                f.attrs[a] = attr_callback(a)

    def _get_parameter_attribute(self, att):
        raise NotImplementedError('Not sure what to do with attribute: {0}'.format(att))

    def _get_master_attribute(self, att):
        if att == 'inline_data_writes' or 'auto_flush_values':
            return pack(True)
        else:
            raise NotImplementedError('Not sure what to do with attribute: {0}'.format(att))

    def _diagnose_attrs(self, fpath, atts):
        good_atts = {}
        bad_atts = []
        for a in atts:
            with h5py.File(fpath, 'r') as f:
                try:
                    good_atts[a] = f.attrs[a]
                except IOError:
                    bad_atts.append(a)

        return good_atts, bad_atts

    def _get_parameter_fileset(self, pname):
        pset = {}
        pdir = os.path.join(self._inner_dir, pname)
        pset['param'] = os.path.join(pdir, pname + '.hdf5')
        pset['bricks'] = [os.path.join(pdir, x) for x in os.listdir(pdir) if not pname in x]
        return pset

    def _compare_to_tree(self, have_groups, have_datasets, error_groups):
        missing = []

        for p in os.walk(self._inner_dir).next()[1]:
            pset = self._get_parameter_fileset(p)
            if p in have_groups:
                pset.pop('param')
            else:
                missing.append(pset.pop('param'))

            if p in have_datasets:
                for b in list(pset['bricks']):
                    bid = os.path.split(os.path.splitext(b)[0])[1]
                    if bid not in have_datasets[p]:
                        missing.append(b)
            else:
                missing.extend(pset['bricks'])

        return missing

    def repack_above(self, min_size_ratio=0.33):
        if self._ar is None:
            self.analyze()

        if self._ar.is_corrupt:
            raise ValueError('The coverage is corrupt!!  Cannot repack a corrupt coverage, please run CoverageDoctor.repair first.')

        above = [x for x in self._ar.get_size_ratios() if x[1] > min_size_ratio]

        return above
