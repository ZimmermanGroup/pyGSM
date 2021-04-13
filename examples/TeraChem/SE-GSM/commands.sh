#!/bin/bash

gsm  -xyzfile ../../../data/diels_alder.xyz \
    -mode SE_GSM \
    -isomers driving_coordinate.txt \
    -package TeraChem \
    -lot_inp_file tc_options.txt \
    -coordinate_type DLC > log 2>&1

