import sys
#sys.path.append('/export/zimmerman/khoidang/pyGSM')
sys.path.insert(0,'/home/caldaz/module/pyGSM')

from dlc import *
from pytc import *
from se_xing import *

SE_XING = True

if SE_XING:
    states = [(1,0),(1,1)]
    charge=0

    filepath = 'tests/ethylene.xyz'
    nocc=7
    nactive=2
    driving_coords = [('TORSION',2,1,4,6,90.)]

    mol = pb.readfile('xyz',filepath).next()
    lot = PyTC.from_options(states=states,nocc=nocc,nactive=nactive,basis='6-31gs')
    lot.cas_from_file(filepath)

    pes1 = PES.from_options(lot=lot,ad_idx=states[0][1],multiplicity=states[0][0])
    pes2 = PES.from_options(lot=lot,ad_idx=states[1][1],multiplicity=states[1][0])
    pes = Penalty_PES(pes1,pes2)

    print ' IC1 '
    ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1,resetopt=False)

    if True:
        print ' Starting GSM '
        gsm = SE_Cross.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,CONV_TOL=0.001,driving_coords=driving_coords,ADD_NODE_TOL=0.5,tstype=3,DQMAG_MAX=0.4)
        gsm.go_gsm(opt_steps=5)
