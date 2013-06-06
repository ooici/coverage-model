OOICI Coverage Model
==============

Initial coverage model implementation

Reference Information: https://confluence.oceanobservatories.org/display/CIDev/R2+Construction+Data+Model+Implementation


#Prerequisites

This assumes basic development environment setup (git, directory structure). Please follow the
"New Developers Tutorial" for basic steps.


**Install the following if not yet present:**

*OS Packages and package management:*
For Mac, use homebrew

    /usr/bin/ruby -e "$(curl -fsSL https://raw.github.com/gist/323731)"

  * git
  * python 2.7
  * hdf5
  * netcdf


**Install** git, python, hdf5 and netcdf with Homebrew
    
    brew install git python hdf5 netcdf
    
You'll also need the various dependencies of the pyon project:

    brew install libevent libyaml zeromq couchdb rabbitmq pkg-config

You can even reinstall git using brew to clean up your /usr/local directory
Be sure to read the pyon README for platform specific guidance to installing
dependent libraries and packages.
Linux: Note that many installs have much older versions installed by default.
You will need to upgrade couchdb to at least 1.1.0.

**Python packages and environment management:**

**Install** pip

    easy_install pip

**Install** virtualenv and virtualenvwrapper modules for your python 2.7 installation
*Note: This may require Mac's XCode (use XCode 3.3 free version*

    easy_install --upgrade virtualenv
    easy_install --upgrade virtualenvwrapper


Setup a virtualenv to run coverage-model (use any name you like):

    mkvirtualenv --python=python2.7 coverage_model
    workon coverage_model
    pip install numpy==1.6.2

**Optional:** *To run any of the example functions that result in graphical output, you must also install matplotlib*

    pip install matplotlib

#Source

Obtain the coverage-model project by running:  

    git clone git@github.com:ooici/coverage-model.git
    cd coverage-model

#Installation
**Ensure you are in a virtualenv prior to running the steps below**

***NOTE**: The repository uses submodules for dependent OOI-CI projects.  For more information on submodules, refer [here](http://git-scm.com/book/en/Git-Tools-Submodules)*

From the *coverage-model* directory, run the following commands:

    git submodule init
    git submodule update
    python bootstrap.py
    bin/buildout
    bin/generate_interfaces

Once those steps complete, you should be able to import the coverage model from the python shell:

    bin/pycc # or bin/ipython
    from coverage_model import *

#Running unit tests (via [nose](https://nose.readthedocs.org/en/latest/))

From the *coverage-model* directory, run the following command:

    bin/nosetests -v

This will run all UNIT and INT tests for the coverage-model repository.  The **-v** flag denotes verbose output (the name of each test prints as it runs).  For more *nose* options, refer to the [nose documentation](https://nose.readthedocs.org/en/latest/man.html)

#Using the *coverage_model.test.examples* functions

The *coverage_model/test/examples.py* module contains numerous functions that showcase the functionality of the Coverage Model - including generation of exemplar coverages, usage of ParameterTypes and more.  
***NOTE**:  The functions in this module are NOT guaranteed to work at all times!!*

Start a pycc or ipython shell session from the root *coverage-model* directory:

    cd your/code/path/coverage-model
    bin/pycc # or bin/ipython

Next, simply import functions from the module and then try them out!

For example:

    from coverage_model.test.examples import *
    sample_cov = samplecov() # Generates a sample coverage with a few simple parameters
    print sample_cov
   
    ptypes_cov = ptypescov() # Generates a coverage with a parameter of each of the parameter types
    print grid_cov
    
#Saving and Loading coverages

By default, the Coverage Model automatically persists data to disk using HDF5.  The "top level" of a coverage is a directory.  When loading a coverage, **always refer to this top level directory**.  
Coverages can be loaded in two ways.  Given the following directory structure:

    .../my_data
    |---ooi_coverages
        |---cov_one
            |--- ...
        |---cov_two
            |--- ...
        |---cov_three
            |--- ...

The coverage *cov_one* can be loaded by:    

    # Load method 1
    cov = SimplexCoverage.load('.../my_data/ooi_coverages/cov_one')

    # Load method 2
    cov = SimplexCoverage('.../my_data/ooi_coverages', 'cov_one')

It is also possible to save coverages to single files using python pickle:  
**WARNING - DEPRECATED: pickle saving is not guaranteed to work and will be removed in future versions**  
***NOTE**: In order to save a coverage using this method, it MUST be an **in-memory** coverage (see below)*

    cov = SimplexCoverage.load('.../my_data/ooi_coverages/cov_one')

    # Save to a pickle file
    cov.pickle_save('.../my_data/ooi_coverages/pickled_cov_one.cov')

    # Load from a pickle file
    cov = SimplexCoverage.pickle_load('.../my_data/ooi_coverages/pickled_cov_one.cov')


#In-memory Coverage

Coverages can be created that do not automatically persist to HDF5 by setting the **in\_memory\_storage=True** when constructing the coverage.  
***NOTE**: Using the Coverage Model in this manner can result in loss of information and should be used sparingly and/or only in certain situations*

    from coverage_model import *
    
    # Assuming appropriate ParameterDictionary (pdict) and Domain (tdom) objects exist
    cov = SimplexCoverage('test_data', 'mycov', 'my_coverage_name', parameter_dictionary=pdict, temporal_domain=tdom, in_memory_storage=True)
