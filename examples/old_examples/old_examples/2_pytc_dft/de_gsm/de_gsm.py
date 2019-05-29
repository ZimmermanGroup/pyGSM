import sys
sys.path.insert(0,'/home/caldaz/pyGSM/')
from de_gsm import *
from dlc import *
from pytc import *
from pes import *
from rhf_lot import *
import lightspeed as ls
import psiw
from contextlib import contextmanager
@contextmanager
def custom_redirection(fileobj):
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old

filepath1="scratch/butadiene_ethene.xyz"
filepath2="scratch/cyclohexene.xyz"
basis='6-31gs'
states = [(1,0)]
resources = ls.ResourceList.build()

with open('psiw_jobs.txt','w') as out:
    with custom_redirection(out):
        molecule1 = ls.Molecule.from_xyz_file(filepath1)    
        geom1 =psiw.Geometry.build(
            resources=resources,
            molecule=molecule1,
            basisname=basis,
            )
        rhf1 = psiw.RHF.from_options(geometry=geom1,dft_functional='B3LYP',print_level=1)
        psiw1 = RHF_LOT.from_options(rhf=rhf1)

# = > molecule 1 <= #
lot1 = PyTC.from_options(states=states,psiw=psiw1)
pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
print "\n IC1 \n"
mol=pb.readfile("xyz",filepath1).next()
ic1=DLC.from_options(mol=mol,PES=pes1,print_level=0)

# = > molecule 2 <= #
print "\n IC2 \n"
molecule2 = ls.Molecule.from_xyz_file(filepath2)    
geom2 =psiw.Geometry.build(
    resources=resources,
    molecule=molecule2,
    basisname=basis,
    )
rhf2 = psiw.RHF.from_options(geometry=geom2,dft_functional='B3LYP',print_level=0)
psiw2 = RHF_LOT.from_options(rhf=rhf2)
lot2 = PyTC.from_options(states=states,psiw=psiw2)
pes2 = PES.from_options(lot=lot2,ad_idx=0,multiplicity=1)
mol2=pb.readfile("xyz",filepath2).next()
ic2=DLC.from_options(mol=mol2,PES=pes2,print_level=0)
    
print "\n Starting GSM \n"
gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9)
gsm.go_gsm(max_iters=50)
gsm.icoords[gsm.TSnode].mol.write("xyz","tsnode.xyz",overwrite=True)

