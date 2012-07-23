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
    download_url = 'http://ooici.net/releases',
    license = 'Apache 2.0',
    author = 'Christopher Mueller',
    author_email = 'cmueller@asascience.com',
    keywords = ['ooici','coverage model'],
    packages = find_packages(),
    dependency_links = [
        'http://ooici.net/releases',
        'https://github.com/ooici/pyon/tarball/master#egg=pyon',
    ],
    test_suite = '',
    install_requires = [
        'pyon',
        'netCDF4>=0.9.8',
        'scipy==0.10.1',
    ],
)
