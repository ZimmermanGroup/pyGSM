# pyGSM

=======> Please see LICENCE for licensing and copyright information <= \
============> Zimmerman Group, University of Michigan  <==============

## Overview
pyGSM (Python + GSM) combines the powerful tools of python with the
Growing String Method to allow for rapid prototyping and improved
readability.

## Download instructions
git clone git@github.com:ZimmermanGroup/pyGSM.git\
(need to have github ssh key activated) 

## Install instructions

Install the code using `python setup.py install`. \ 
You can also install locally without sudo permissions like:
`python setup.py install --prefix ~/.local`
You might need to create the folder `~/.local/lib/` if setup.py complains about the folder not existing.

It's also recommended to do the installation within a conda environment e.g.:
   1. `conda create -n gsm_env`
   2. `source activate gsm_env`
   3. `python setup.py install --prefix ~/.local`


## Requirements 
anaconda 3 
python 3.5 or greater\
numpy\
matplotlib\
six\
networkx

## Running as Executable
setup.py installs all the required packages, and creates an executable.  \
To execute the gsm, run `gsm`. Use `-h` to see the list of command line options.
Generally, you will need a .xyz file for the coordinates and one of the supported quantum chemistry software packages installed on your system.

## Running as API
within a python script run `import pygsm` (assuming you've run setup.py).\


## LICENCE Notifications
This project contains source material from the geomeTRIC package.\
Copyright 2016-2019 Regents of the University of California and the Authors\
Authors: Lee-Ping Wang, Chenchen Song\ 
https://github.com/leeping/geomeTRIC/blob/master/LICENSE


## Contributors: 
Cody Aldaz (lead author) \
Prof. Paul Zimmerman \
Prof. Todd Martinez \
