# pyGSM

=> Please see LICENCE for licensing and copyright information <= \
=>    Zimmerman Group, University of Michigan <= 

## Overview
pyGSM (Python + GSM) combines the powerful tools of python with the
Growing String Method to allow for rapid prototyping and improved
readability.

## Download instructions
git clone git@github.com:ZimmermanGroup/pyGSM.git\
(need to have github ssh key activated) \

## Install instructions
install the code from source, run "python setup.py install". \ 
To install the latest release from conda-forge, run ? (to be done)

## Requirements 
anaconda 2.5.1 or greater\
python 2.7\
numpy\
networkx\

## Running as Executable
setup.py installs all the required packages, and creates an executable.  \
To execute the gsm, run "gsm". Use "-h" to see the list of command line options.
Generally, you will need a .xyz file for the coordinates and one of the supported quantum chemistry software packages installed on your system.

## Running as API
if not running pygsm as a script then a conda environment should be created that installs the required packages \
1. create a conda environment if necessary
   1. conda create -n gsm_env python=2.7
   1. source activate gsm_env
   1. conda install -c networkx
   1. conda install numpy
1. source conda environmet
   1. source activate gsm_env

## LICENCE Notifications
This project contains source material from the geomeTRIC package. \
Copyright 2016-2019 Regents of the University of California and the Authors \
Authors: Lee-Ping Wang, Chenchen Song \ 
https://github.com/leeping/geomeTRIC/blob/master/LICENSE


## Contributors: 
Cody Aldaz (lead author) \
Khoi Dang \
Prof. Paul Zimmerman \
Josh Kammeraad \
Prof. Todd Martinez Group 
