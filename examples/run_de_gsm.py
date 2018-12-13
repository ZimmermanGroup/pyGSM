import sys
sys.path.append('/export/zimmerman/craldaz/pyGSM/')

from de_gsm import *

#    from icoord import *
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

if False:
    filepath="tests/fluoroethene.xyz"
    filepath2="tests/stretched_fluoroethene.xyz"
    nocc=11
    nactive=2

if False:
    filepath2="tests/SiH2H2.xyz"
    filepath="tests/SiH4.xyz"
    nocc=8
    nactive=2
if True:
    filepath="tests/butadiene_ethene.xyz"
    filepath2="tests/cyclohexene.xyz"
    nocc=21
    nactive=4

mol=pb.readfile("xyz",filepath).next()
mol2=pb.readfile("xyz",filepath2).next()
basis = '6-31G*'
if QCHEM:
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='B3LYP',nproc=nproc)
    lot2=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='B3LYP',nproc=nproc)

if ORCA:
    lot=Orca.from_options(states=[(1,0)],charge=0,basis=basis,functional='wB97X-D3',nproc=nproc)
    lot2=Orca.from_options(states=[(1,0)],charge=0,basis=basis,functional='wB97X-D3',nproc=nproc)
if PYTC:
    lot=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
    lot.cas_from_file(filepath)
    #lot.casci_from_file_from_template(filepath,filepath,nocc,nocc) 
    lot2=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
    #lot2.casci_from_file_from_template(filepath,filepath2,nocc,nocc)
    lot2.cas_from_file(filepath2)

pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
pes2 = PES.from_options(lot=lot2,ad_idx=0,multiplicity=1)

print "\n IC1 \n"
ic1=DLC.from_options(mol=mol,PES=pes,print_level=0)
print "\n IC2 \n"
ic2=DLC.from_options(mol=mol2,PES=pes2,print_level=0)
    
nnodes=9
if True:
    print "\n Starting GSM \n"
    gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=nnodes,nconstraints=1,tstype=0)
    #gsm.restart_string()
    #gsm.ic_reparam(ic_reparam_steps=25)
    #gsm.write_xyz_files(iters=1,base='initial_ic_reparam',nconstraints=1)
    gsm.go_gsm(max_iters=50)

