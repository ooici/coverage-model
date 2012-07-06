OOICI Coverage Model
==============

Initial coverage model implementation

Reference Information: https://confluence.oceanobservatories.org/display/CIDev/R2+Construction+Data+Model+Implementation


#Prerequisites

This assumes basic development environment setup (git, directory structure). Please follow the
"New Developers Tutorial" for basic steps.

Pyon: The main dependency of this repository is the pyon Capability Container. Follow the listed
steps to install the minimal needed dependencies to run pyon on a Mac. For more details and Linux
install instructions, check out the [pyon README](https://github.com/ooici/pyon/blob/master/README)



**Install the following if not yet present:**

**Install** git 1.7.7:
*Download the Mac or Linux installer and run it*

*OS Packages and package management:*
For Mac, use homebrew

    /usr/bin/ruby -e "$(curl -fsSL https://raw.github.com/gist/323731)"

  * python 2.7
  *  hdf5
  * netcdf


**Install** python, hdf5 and netcdf with Homebrew
    
    brew install python hdf5 netcdf

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

Use pip to install ipython, numpy and netCDF4:

    pip install ipython numpy netCDF4


#Source

Obtain the coverage-model project by running:  

    git clone git@github.com:blazetopher/coverage-model.git
    cd coverage-model

#Running the example functions

The *coverage/test/simple_cov.py* module contains 2 functions that can be used to generate 'exemplar' coverages from sample netcdf files (in the *test_data* directory).

Start an ipython shell session from the root *coverage-model* directory:

    cd your/code/path/coverage-model
    ipython

From the ipython prompt, run:

    from coverage.test.examples import *
    stn_cov, _ = ncstation2cov() # Generates a coverage from the test_data/usgs.nc file
    print stn_cov
   
    grid_cov, _ = ncgrid2cov() # Generates a coverage from the test_data/ncom.nc file
    print grid_cov