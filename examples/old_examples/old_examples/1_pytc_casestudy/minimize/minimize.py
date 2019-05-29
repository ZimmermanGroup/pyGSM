import sys
sys.path.insert(0,'/home/caldaz/module/')

from pyGSM.dlc import *
from pyGSM.pytc import *
from pyGSM.pes import *
from pyGSM.base_gsm import *

filepath = 'scratch/ethylene.xyz'

##### => initialize lot parameters <= #####3
states = [(1,0),(1,1)]
charge=0
nocc=7
nactive=2
#the from_template thing is SVD overlap of current molecule with reference orbitals
lot = PyTC.from_options(states=states,nocc=nocc,nactive=nactive,basis='6-31gs',from_template=True)

# for pytc it is necessary to initialize lot/ saved memory variables are used later, ok to just use molecule (substituted mols can from_template fxn)
lot.casci_from_file(filepath)

# => pes <=
# HERE IS HOW I SET THE ADIABATIC LABEL
pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

# => openbabel pybel <=
mol=pb.readfile("xyz",filepath).next()

# => Create DLC object <= #
ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1)

# => gsm <=
gsm=Base_Method.from_options(ICoord1=ic1)
gsm.optimize(n=0,nsteps=100)
gsm.icoords[0].mol.write("xyz","s0minima.xyz",overwrite=True)
print "Finished" 
