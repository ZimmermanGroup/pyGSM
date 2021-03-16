# pyGSM

=======> Please see LICENCE for licensing and copyright information <===== \
============> Zimmerman Group, University of Michigan  <==============

## Overview
pyGSM (Python + GSM) is a reaction path and photochemistry tool. 
Key features
It combines the powerful tools of python with the Growing String Method to allow for rapid prototyping and improved
readability.

## Documentation
See https://zimmermangroup.github.io/pyGSM/


## Install instructions

Install the code using `python setup.py install`.\
You can also install locally without sudo permissions like:
`python setup.py install --prefix ~/.local`
You might need to create the folder `~/.local/lib/` if setup.py complains about the folder not existing.

It's also recommended to do the installation within a conda environment e.g.:
   1. `conda create -n gsm_env`
   2. `source activate gsm_env`
   3. `python setup.py install --prefix ~/.local`


## Requirements 
Python 3 is preferred\
numpy\
matplotlib\
six\
networkx


## LICENCE Notifications
This project contains source material from the geomeTRIC package.\
Copyright 2016-2019 Regents of the University of California and the Authors\
Authors: Lee-Ping Wang, Chenchen Song\ 
https://github.com/leeping/geomeTRIC/blob/master/LICENSE


## Credits: 
Cody Aldaz (lead author) \
Prof. Paul Zimmerman \
Prof. Lee-Ping Wang \
Prof. Todd Martinez
