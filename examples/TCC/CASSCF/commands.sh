#!/bin/bash
#This file was created on 06/06/2019

#if load
#module load pygsm
#simply call gsm

python ../../../pygsm/wrappers/main.py -xyzfile ../../../data/ethylene.xyz -mode SE_Cross -package TeraChemCloud -lot_inp_file tcc_lot_inp_file.txt -isomers isomers.txt -mult 1 1 -adiab 0 1  -DQMAG_MAX 0.4 -ADD_NODE_TOL 0.02 > log &
