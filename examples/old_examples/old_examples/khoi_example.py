import sys
sys.path.append('/export/zimmerman/khoidang/pyGSM')

SE_XING = True
SE_GSM = False
DE_GSM = False

ORCA=False
QCHEM=True
PYTC=False
nproc=8
if QCHEM: from qchem import *
elif ORCA: from orca import *
elif PYTC: from pytc import *

if SE_XING: from se_xing import *
if SE_GSM: from se_gsm import *
if DE_GSM: from de_gsm import *

ethylene=False
sih4=False
fluoroethene=False
butadiene_ethene=False
FeCO5=True
FeO_H2=False
NiL2Br2=False
NiL2Br2_tetr=False

if SE_GSM:
    states = [(1,0)]
    basis = '6-31G(d)'
    charge=0

    if fluoroethene:
        filepath = '../tests/fluoroethene.xyz'
        nocc=11
        nactive=2
        driving_coords = [('BREAK',1,2,0.1)]
    elif ethylene:
        filepath = '../tests/ethylene.xyz'
        nocc=7
        nactive=2
        driving_coords = [('BREAK',1,2,0.1)]
    elif sih4:
        filepath = '../tests/SiH4.xyz'
        nocc=8
        nactive=2
        driving_coords = [('ADD',3,4,0.1),('BREAK',1,3,0.1),("BREAK",1,4,0.1)]
    elif butadiene_ethene:
        filepath = '../tests/butadiene_ethene.xyz'
        nocc=21
        nactive=4
    elif FeCO5:
        filepath = '../tests/FeCO5.xyz'
        driving_coords = [('ADD',1,6,0.2)]
    elif FeO_H2:
        filepath = '../tests/FeO_H2.xyz'
        driving_coords = [('Add',1,3,0.1),('Add',2,4,0.1)]
        charge=1
    elif NiL2Br2:
        filepath = '../tests/NiL2Br2_sqpl.xyz'
        driving_coords = [('Torsion',18,12,13,23,10.)]
    elif NiL2Br2_tetr:
        filepath = '../tests/NiL2Br2_tetr.xyz'
        driving_coords = [('Torsion',16,14,1,13,10.)]

    mol = pb.readfile('xyz',filepath).next()
    if QCHEM:
        lot = QChem.from_options(states=states,charge=charge,basis=basis,functional='B3LYP',nproc=nproc)
    elif ORCA:
        lot = Orca.from_options(states=states,charge=charge,basis=basis,functional='B3LYP',nproc=nproc)
    elif PYTC:
        lot = PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31g')
        lot.cas_from_file(filepath)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=states[0][0])
    print ' IC1 '
    ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1,resetopt=False)

    if True:
        print ' Starting GSM '
        gsm = SE_GSM.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,CONV_TOL=0.001,driving_coords=driving_coords,ADD_NODE_TOL=0.05)
        gsm.go_gsm(30,3)
        if ORCA:
            os.system('rm temporcarun/*')

if SE_XING:
    states = [(1,0),(3,0)]
    basis = '6-31G(d)'
    charge=0

    if fluoroethene:
        filepath = '../tests/fluoroethene.xyz'
        nocc=11
        nactive=2
        driving_coords = [('BREAK',1,2,0.1)]
    elif ethylene:
        filepath = '../tests/ethylene.xyz'
        nocc=7
        nactive=2
        driving_coords = [('BREAK',1,2,0.1)]
    elif sih4:
        filepath = '../tests/SiH4.xyz'
        nocc=8
        nactive=2
        driving_coords = [('ADD',3,4,0.1),('BREAK',1,3,0.1),("BREAK",1,4,0.1)]
    elif butadiene_ethene:
        filepath = '../tests/butadiene_ethene.xyz'
        nocc=21
        nactive=4
    elif FeCO5:
        filepath = '../tests/FeCO5.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('ADD',1,6,0.2)]
    elif FeO_H2:
        filepath = '../tests/FeO_H2.xyz'
        states = [(4,0),(6,0)]
        driving_coords = [('Add',1,3,0.1),('Add',2,4,0.1)]
        charge=1
    elif NiL2Br2:
        filepath = '../tests/NiL2Br2_sqpl.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('Torsion',18,12,13,23,10.)]
#        basis = 'def2-svp'
    elif NiL2Br2_tetr:
        filepath = '../tests/NiL2Br2_tetr.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('Torsion',16,14,1,13,10.)]

    mol = pb.readfile('xyz',filepath).next()
    if QCHEM:
        lot = QChem.from_options(states=states,charge=charge,basis=basis,functional='B3LYP',nproc=nproc)
    elif ORCA:
        lot = Orca.from_options(states=states,charge=charge,basis=basis,functional='B3LYP',nproc=nproc)
    elif PYTC:
        lot = PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31g')
        lot.cas_from_file(filepath)

    pes1 = PES.from_options(lot=lot,ad_idx=0,multiplicity=states[0][0])
    pes2 = PES.from_options(lot=lot,ad_idx=0,multiplicity=states[1][0])
    pes = Penalty_PES(pes1,pes2)
    print ' IC1 '
    ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1,resetopt=False)

    if True:
        print ' Starting GSM '
        gsm = SE_Cross.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,CONV_TOL=0.001,driving_coords=driving_coords,ADD_NODE_TOL=0.05)
        gsm.go_gsm(20)
        if ORCA:
            os.system('rm temporcarun/*')

