import sys
sys.path.insert(0,'/home/caldaz/module/')

from pyGSM.dlc import *
from pyGSM.pytc import *
from pyGSM.pes import *
from pyGSM.base_gsm import *

reffile="scratch/reffile.xyz"
filepath1="scratch/initial0000.xyz"
nocc1=37
nactive=6
nocc2=37

# => create lot objects <=
lot=PyTC.from_options(states=[(1,0),(1,1)],do_coupling=False,nocc=nocc2,nactive=nactive,basis='6-31gs',from_template=True)

# => initialize lot <=
#lot.casci_from_file_from_template(reffile,filepath1,nocc1,nocc2)

# => pes <=
pes = PES.from_options(lot=lot,ad_idx=1,multiplicity=1)

# => openbabel pybel <=
mol1=pb.readfile("xyz",filepath1).next()

## => create ic object <= #
ic1=DLC.from_options(mol=mol1,PES=pes,print_level=1,EXTRA_BONDS=[(10,6)])

# => gsm <=
gsm=Base_Method.from_options(ICoord1=ic1)
gsm.icoords[0].PES.initial_energy(reffile=reffile,filepath=filepath1,ref_nocc=nocc1,nocc=nocc2)
gsm.DMAX=0.01
gsm.optimize(n=0,nsteps=100)
gsm.icoords[0].mol.write("xyz","s1minima.xyz",overwrite=True)
print "Finished" 
