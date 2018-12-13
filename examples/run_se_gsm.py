import sys
import os
sys.path.append(os.popen('cd ..;pwd').read().rstrip('\n'))
from se_gsm import *

if True:
    ORCA=False
    QCHEM=True
    PYTC=False
    nproc=4

    if QCHEM:
        from qchem import *
    if ORCA:
        from orca import *
    if PYTC:
        from pytc import *
    import manage_xyz
    if True:
        filepath="tests/butadiene_ethene.xyz"
        nocc=21
        nactive=4

    mol=pb.readfile("xyz",filepath).next()
    if QCHEM:
        lot=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
    if ORCA:
        lot=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
    if PYTC:
        lot=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31g')
        lot.cas_from_file(filepath)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

    print "\n IC1 \n"
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)

    if True:
        print "\n Starting GSM \n"
        gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,driving_coords=[("ADD",6,4),("ADD",5,1)],ADD_NODE_TOL=0.05,tstype=0)
        gsm.optimize(n=6,nsteps=2,nconstraints=0,ictan=gsm.ictan[6],follow_overlap=True)
