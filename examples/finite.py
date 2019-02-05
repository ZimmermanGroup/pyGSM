import sys
sys.path.insert(0,'/home/caldaz/module/pyGSM')
from dlc import *
from pytc import *
import manage_xyz

# => read molecule from file <= #
filepath = 'tests/ethylene.xyz'
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

geom = manage_xyz.read_xyz(filepath,scale=1)
hess = pes1.get_finite_difference_hessian(geom)
print hess
