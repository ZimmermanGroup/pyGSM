#!/bin/bash
#SBATCH -p zimintel --job-name=DE_GSM
#SBATCH --array=1
#SBATCH --output=std.output
#SBATCH --error=std.error
#SBATCH --nodes=1
#SBATCH -c 12
#SBATCH --time=48:00:00

# load modules
. /etc/profile.d/slurm.sh
module load qchem
module load pygsm

# pygsm will automatically read the number of processors
# use the -c option to specify threads. python will use 
# the threads to make calculations faster

#run job
gsm  -coordinate_type DLC \
    -xyzfile ../../../data/diels_alder.xyz \
    -mode DE_GSM \
    -package QChem \
    -lot_inp_file qstart \
    -ID $SLURM_ARRAY_TASK_ID > log 2>&1

ID=`printf "%0*d\n" 3 $SLURM_ARRAY_TASK_ID`
rm -rf $QCSCRATCH/string_$ID

exit

