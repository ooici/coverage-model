#!/usr/bin/env python

"""
@package coverage_model.coverage_recovery
@file coverage_model/coverage_recovery.py
@author Christopher Mueller
@brief Contains utility functions for attempting to recover corrupted coverages
"""

from coverage_model import hdf_utils
from coverage_model.persistence_helpers import pack
from coverage_model.basic_types import BaseEnum
import os
import h5py


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

        root, guid = os.path.split(self.cov_pth)
        inner_dir = os.path.join(self.cov_pth, guid)
        master_pth = os.path.join(self.cov_pth, guid + '_master.hdf5')

        st = StatusEnum.CORRUPT if hdf_utils.has_corruption(master_pth) else StatusEnum.NORMAL
        sz=hdf_utils.space_ratio(master_pth)
        ar.set_master_status(master_pth, st, sz)

        for p in os.walk(inner_dir).next()[1]:
            pdir = os.path.join(inner_dir, p)
            p_pth = os.path.join(pdir, p + '.hdf5')
            st = StatusEnum.CORRUPT if hdf_utils.has_corruption(p_pth) else StatusEnum.NORMAL
            sz=hdf_utils.space_ratio(p_pth)
            ar.add_param_status(p, p_pth, st, sz)
            for b_pth in [os.path.join(pdir, x) for x in os.listdir(pdir) if not p in x]:
                if analyze_bricks:
                    st = StatusEnum.CORRUPT if hdf_utils.has_corruption(b_pth) else StatusEnum.NORMAL
                    sz=hdf_utils.space_ratio(b_pth)
                else:
                    st = StatusEnum.UNKNOWN
                    sz = StatusEnum.UNKNOWN

                ar.add_brick_status(p, b_pth, st, sz)

        return ar

    def analyze(self, analyze_bricks=False, reanalyze=False):
        if self._ar is None or reanalyze:
            ar = self._do_analysis(analyze_bricks=analyze_bricks)
            self._ar = ar
        return self._ar

    def repair(self, reanalyze=False):
        if self._ar is None or reanalyze:
            self._ar = self._do_analysis()

        if self._ar.is_corrupt:
            if len(self._ar.get_brick_corruptions()) == 0:
                if len(self._ar.get_corruptions()) > 1 and len(self._ar.get_master_corruption()) == 1:
                    raise NotImplementedError('Corruption in master and one or more parameters.  Cannot repair at this time!! :(')
                else:
                    # Rename the top level directory

                    # Repair the corrupt file(s)
                    if len(self._ar.get_master_corruption()) == 1:
                        # Get the path to the master file
                        mfile = self._ar.get_master_corruption()[0]
                        # Sort out which attributes are bad and which aren't
                        good, bad = self._retrieve_uncorrupt_attrs(mfile)
                        if len(bad) == 0:
                            print 'Attributes all appear good!!'
                        else:
                            mfile_out = mfile.replace('.hdf5','_new.hdf5')
                            with h5py.File(mfile_out) as f:
                                for a in good:
                                    f.attrs[a] = good[a]

                                for a in bad:
                                    f.attrs[a] = self._get_master_attribute(a)

                            print 'Fixed Master!  New file at: %s' % mfile_out

                    elif len(self._ar.get_param_corruptions()) > 0:
                        raise NotImplementedError('Param repair not ready yet')
                    else:
                        raise NotImplementedError('Don\'t really know how you got here!!')

                    # Rename the top level directory back to it's original value
            else:
                raise NotImplementedError('Brick corruption.  Cannot repair at this time!!! :(')
        else:
            print 'Coverage is not corrupt, nothing to repair!! :)'

        return True

    def _get_master_attribute(self, att):
        if att == 'inline_data_writes' or 'auto_flush_values':
            return pack(True)
        else:
            raise NotImplementedError('Not sure what to do with attribute: {0}'.format(att))

    def _retrieve_uncorrupt_attrs(self, fpath, atts=None):
        if atts is None:
            atts = MASTER_ATTRS

        good_atts = {}
        bad_atts = []
        for a in atts:
            try:
                with h5py.File(fpath, 'r') as f:
                    good_atts[a] = f.attrs[a]
            except:
                bad_atts.append(a)

        return good_atts, bad_atts

    def repack_above(self, min_size_ratio=0.33):
        if self._ar is None:
            self.analyze()

        if self._ar.is_corrupt:
            raise ValueError('The coverage is corrupt!!  Cannot repack a corrupt coverage, please run CoverageDoctor.repair first.')

        above = [x for x in self._ar.get_size_ratios() if x[1] > min_size_ratio]

        return above
