import sys
#sys.path.append('/export/zimmerman/khoidang/pyGSM')
sys.path.insert(0,'/home/caldaz/module/pyGSM')
from dlc import *
from pytc import *
from de_gsm import *
import numpy as np


states = [(1,0),(1,1)]
charge=0

filepath1 = 'scratch/tw_pyr_meci.xyz'
filepath2 = 'scratch/et_meci.xyz'
nocc=7
nactive=2

mol1 = pb.readfile('xyz',filepath1).next()
mol2 = pb.readfile('xyz',filepath2).next()
lot = PyTC.from_options(states=states,nocc=nocc,nactive=nactive,basis='6-31gs',from_template=True,do_coupling=True)

# Fixed orbitals for MOM
dat = np.load('Cmom.npz')
Cocc_mom = ls.Tensor.array(dat['Cocc'])
Cact_mom = ls.Tensor.array(dat['Cact'])
lot.casci_from_file(filepath1,Cocc=Cocc_mom,Cact=Cact_mom)

#
pes1 = PES.from_options(lot=lot,ad_idx=states[0][1],multiplicity=states[0][0])
pes2 = PES.from_options(lot=lot,ad_idx=states[1][1],multiplicity=states[1][0])
pes = Avg_PES(pes1,pes2,lot)

print ' IC1 '
ic1 = DLC.from_options(mol=mol1,PES=pes,print_level=1,resetopt=False)
ic2 = DLC.from_options(mol=mol2,PES=pes,print_level=1,resetopt=False)

print ' Starting GSM '
gsm = GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=7,growth_direction=1,ADD_NODE_TOL=0.1)

gsm.go_gsm(opt_steps=3,rtype=0)

