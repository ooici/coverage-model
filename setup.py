#!/usr/bin/env python

try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup

import os
import sys

# Add /usr/local/include to the path for macs, fixes easy_install for several packages (like gevent and pyyaml)
if sys.platform == 'darwin':
    os.environ['C_INCLUDE_PATH'] = '/usr/local/include'

version = '0.1'

setup(  name = 'coverage-model',
    version = version,
    description = 'OOI ION Coverage Model',
    url = 'https://github.com/blazetopher/coverage-model',
    download_url = 'http://sddevrepo.oceanobservatories.org/releases/',
    license = 'Apache 2.0',
    author = 'Christopher Mueller',
    author_email = 'cmueller@asascience.com',
    keywords = ['ooici','coverage model'],
    packages = find_packages(),
    dependency_links = [
        'http://sddevrepo.oceanobservatories.org/releases/',
        'http://sddevrepo.oceanobservatories.org/releases/h5py-2.1.1a1.tar.gz#egg=h5py',
    ],
    test_suite = '',
    install_requires = [
        'pyon',
        'pyzmq==2.2.0',
        'gevent_zeromq==0.2.5',
        'netCDF4>=0.9.8',
        'numexpr==2.0.1',
        'h5py==2.1.1a1', 
        'rtree==0.7.0',
        'pidantic',
        'gsw==3.0.1',
        'pydot==1.0.28',
        'networkx==1.7',
#        'scipy==0.10.1',
    ],
)
