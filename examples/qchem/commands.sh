#!/bin/bash
#This file was created on 06/06/2019

module load pygsm

gsm -xyzfile ../../data/diels_alder.xyz -mode DE_GSM -package QChem -lot_inp_file qstart  > log &

