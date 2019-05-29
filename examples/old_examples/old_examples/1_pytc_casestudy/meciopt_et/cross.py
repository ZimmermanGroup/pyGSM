import sys
#sys.path.append('/export/zimmerman/khoidang/pyGSM')
sys.path.insert(0,'/home/caldaz/module/pyGSM')
from dlc import *
from pytc import *
from base_gsm import *

# => read molecule from file <= #
filepath = 'scratch/meci.xyz'
mol = pb.readfile('xyz',filepath).next()

##### => initialize lot parameters <= #####3
states = [(1,0),(1,1)]
charge=0
nocc=7
nactive=2
#the from_template thing is SVD overlap of current molecule with reference orbitals
lot = PyTC.from_options(states=states,nocc=nocc,nactive=nactive,basis='6-31gs',from_template=True,do_coupling=True)

# for pytc it is necessary to initialize lot/ saved memory variables are used later, ok to just use molecule (substituted mols can from_template fxn)
lot.casci_from_file(filepath)

# => Create PES objects <= #
pes1 = PES.from_options(lot=lot,ad_idx=states[0][1],multiplicity=states[0][0])
pes2 = PES.from_options(lot=lot,ad_idx=states[1][1],multiplicity=states[1][0])
pes = Avg_PES(pes1,pes2,lot)

# => Create DLC object <= #
ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1)

# => Create GSM object <= #
gsm = Base_Method.from_options(ICoord1=ic1,nnodes=1)
gsm.optimize(nsteps=50,opt_type=5)

# => post processing <= #
gsm.icoords[0].mol.write("xyz","et_meci.xyz",overwrite=True)

#Save a npz file
Cocc= ls.Tensor.array(gsm.icoords[0].PES.lot.psiw.casci.reference.tensors['Cocc'])
Cact= ls.Tensor.array(gsm.icoords[0].PES.lot.psiw.casci.reference.tensors['Cact'])
D   = ls.Tensor.array(gsm.icoords[0].PES.lot.psiw.casci.reference.tensors['D'])
np.savez(
    'Cmom.npz',
    Cocc=Cocc,
    Cact=Cact,
    D=D
    )
