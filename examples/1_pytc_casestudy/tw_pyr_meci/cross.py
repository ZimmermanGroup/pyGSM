import sys
#sys.path.append('/export/zimmerman/khoidang/pyGSM')
sys.path.insert(0,'/home/caldaz/module/pyGSM')
from dlc import *
from pytc import *
from se_xing import *
import numpy as np

# => read molecule from file <= #
filepath = 'scratch/ethylene.xyz'
mol = pb.readfile('xyz',filepath).next()

##### => initialize lot parameters <= #####3
states = [(1,0),(1,1)]
charge=0
nocc=7
nactive=2
#the from_template thing is SVD overlap of current molecule with reference orbitals
lot = PyTC.from_options(states=states,nocc=nocc,nactive=nactive,basis='6-31gs',from_template=True)

# for pytc it is necessary to initialize lot/ saved memory variables are used later, ok to just use molecule (substituted mols can from_template fxn)
lot.casci_from_file(filepath)

# => Create PES objects <= #
pes1 = PES.from_options(lot=lot,ad_idx=states[0][1],multiplicity=states[0][0])
pes2 = PES.from_options(lot=lot,ad_idx=states[1][1],multiplicity=states[1][0])
pes = Penalty_PES(pes1,pes2,lot)

# => Create DLC object <= #
ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1)

# => Create GSM object <= #
driving_coords = [('TORSION',2,1,4,6,120.),('TORSION',2,1,4,5,180.)] #extra torsion here to ensure proper orientation w.r.t et_meci
gsm = SE_Cross.from_options(ICoord1=ic1,nnodes=20,driving_coords=driving_coords,DQMAG_MAX=0.4,BDIST_RATIO=0.75)
#BDIST_RATIO controls when string will terminate, good when know exactly what you want
#DQMAG_MAX controls max step size for adding node

# => Run GSM <= #
print ' Starting GSM '
gsm.go_gsm(opt_steps=5)

# => post processing <= #
gsm.icoords[gsm.nR].mol.write("xyz","tw_pyr_meci.xyz",overwrite=True)

#Save a npz file
Cocc= ls.Tensor.array(gsm.icoords[gsm.nR].PES.lot.psiw.casci.reference.tensors['Cocc'])
Cact= ls.Tensor.array(gsm.icoords[gsm.nR].PES.lot.psiw.casci.reference.tensors['Cact'])
D   = ls.Tensor.array(gsm.icoords[gsm.nR].PES.lot.psiw.casci.reference.tensors['D'])
np.savez(
    'Cmom.npz',
    Cocc=Cocc,
    Cact=Cact,
    D=D
    )
