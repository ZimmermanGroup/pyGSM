#!/bin/bash
#This file was created on 06/06/2019

#module load pygsm
#module load tcc
#if load pygsm simply call gsm

python ../../../pygsm/wrappers/main.py -xyzfile ../../../data/diels_alder.xyz -mode DE_GSM -package TeraChemCloud -lot_inp_file tcc_lot_inp_file.txt > log &
