#!/bin/bash

gsm  -xyzfile ../../../data/diels_alder.xyz \
    -mode DE_GSM \
    -num_nodes 11 \
    -package TeraChem \
    -lot_inp_file tc_options.txt \
    -interp_method Geodesic \
    -coordinate_type DLC > log 2>&1

