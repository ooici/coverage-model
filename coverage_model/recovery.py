#!/usr/bin/env python

"""
@package coverage_model.coverage_recovery
@file coverage_model/coverage_recovery.py
@author Christopher Mueller
@brief Contains utility functions for attempting to recover corrupted coverages
"""

from coverage_model import hdf_utils
from coverage_model.coverage import AbstractCoverage
from coverage_model.basic_types import BaseEnum
from interface.objects import DataProduct, DataSet
import os

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

    @property
    def master_status(self):
        return self._results['master']

    def set_master_status(self, pth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))
        self._results['master'] = (pth, status, size_ratio)

    def add_param_status(self, pname, ppth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))

        self._results[pname] = {'param': (ppth, status, size_ratio),
                                  'bricks': []}

    def add_brick_status(self, pname, bpth, status, size_ratio=None):
        if not StatusEnum.has_member(status):
            raise TypeError('Unknown status: {0}'.format(status))

        if not pname in self._results:
            raise TypeError('Parameter \'{0}\' has not been added, please add parameters before bricks'.format(pname))

        self._results[pname]['bricks'].append((bpth, status, size_ratio))

    @property
    def total_file_count(self):
        return sum([1, self.brick_file_count, self.param_file_count])

    @property
    def brick_file_count(self):
        return sum([len(self._results[p]['bricks']) for p in self._results.keys() if p != 'master'])

    @property
    def param_file_count(self):
        return len(self._results) - 1 # Length of results minus 1 for 'master'

    def get_master_corruption(self):
        corruptions = set()
        if self._results['master'][1] == StatusEnum.CORRUPT:
            corruptions.add(self._results['master'][0])

        return list(corruptions)

    def get_param_corruptions(self):
        corruptions = set()

        corruptions.update([self._results[p]['param'][0] for p in self._results if p != 'master' and self._results[p]['param'][1] == StatusEnum.CORRUPT])

        ret = list(corruptions)
        ret.sort()
        return ret

    def get_brick_corruptions(self):
        corruptions = set()

        for p in self._results:
            if p != 'master':
                for b in self._results[p]['bricks']:
                    if b[1] == StatusEnum.CORRUPT:
                        corruptions.add(b[0])

#        corruptions.update([b[0] for p in self._results for b in self._results[p]['bricks'] if p != 'master' and b[1] == StatusEnum.CORRUPT])

        ret = list(corruptions)
        ret.sort()
        return ret

    def get_corruptions(self):
        corruptions = set()
        corruptions.update(self.get_master_corruption())
        corruptions.update(self.get_param_corruptions())
        corruptions.update(self.get_brick_corruptions())

        ret = list(corruptions)
        ret.sort()
        return ret

    def get_master_size_ratio(self):
        ratios = set()

        ratios.add((self._results['master'][0], self._results['master'][2]))

        return list(ratios)

    def get_param_size_ratios(self):
        ratios = set()
        ratios.update([(self._results[p]['param'][0], self._results[p]['param'][2]) for p in self._results if p != 'master'])

        ret = list(ratios)
        ret.sort()
        return ret

    def get_brick_size_ratios(self):
        ratios = set()

        for p in self._results:
            if p != 'master':
                for b in self._results[p]['bricks']:
                    ratios.add((b[0],b[2]))

#        ratios.update([(b[0], b[1]) for p in self._results for b in self._results[p]['bricks'] if p != 'master'])

        ret = list(ratios)
        ret.sort()
        return ret

    def get_size_ratios(self):
        ratios = set()
        ratios.update(self.get_master_size_ratio())
        ratios.update(self.get_param_size_ratios())
        ratios.update(self.get_brick_size_ratios())

        ret = list(ratios)
        ret.sort()
        return ret

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
        self.dp = data_product_obj
        self.ds = dataset_obj

        self._ar = None
        self._master_path = None

    def _do_analysis(self, incl_bricks=False):
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
            if incl_bricks:
                for b_pth in [os.path.join(pdir, x) for x in os.listdir(pdir) if not p in x]:
                    st = StatusEnum.CORRUPT if hdf_utils.has_corruption(b_pth) else StatusEnum.NORMAL
                    sz=hdf_utils.space_ratio(b_pth)
                    ar.add_brick_status(p, b_pth, st, sz)

        return ar

    def analyze(self, reanalyze=False, incl_bricks=False):
        if self._ar is None or reanalyze:
            ar = self._do_analysis(incl_bricks=incl_bricks)
            self._ar = ar
        return self._ar

    def repair(self, reanalyze=False):
        if not self._ar is None or reanalyze:
            self._ar = self._do_analysis()

        if self._ar.is_corrupt:
            if len(self._ar.get_brick_corruptions()) > 0:
                corr = self._ar.get_corruptions()
                if len(corr) > 1 and self._ar.get_master_corruption() is not None:
                    print 'Corruption in master and one or more parameters.  Cannot repair at this time!! :('
                else:
                    print 'Repairing the coverage!!! :D'
            else:
                print 'Brick corruption.  Cannot repair at this time!!! :('
        else:
            print 'Coverage is not corrupt, nothing to repair!! :)'

        return True

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

    def repack_above(self, min_size_ratio=0.33):
        if self._ar is None:
            self.analyze()

        if self._ar.is_corrupt:
            raise ValueError('The coverage is corrupt!!  Cannot repack a corrupt coverage, please run CoverageDoctor.repair first.')

        above = [x for x in self._ar.get_size_ratios() if x[1] > min_size_ratio]

        return above
