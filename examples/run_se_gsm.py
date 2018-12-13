import sys
sys.path.append('/export/zimmerman/craldaz/pyGSM/')

from se_gsm import *

if True:
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
        nocc=11
        nactive=2
    if False:
        filepath="tests/ethylene.xyz"
        nocc=7
        nactive=2

    if False:
        filepath="tests/SiH4.xyz"
        nocc=8
        nactive=2

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
        #gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=9,nconstraints=1,CONV_TOL=0.001,driving_coords=[("TORSION",2,1,4,6,90.)])
        gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,driving_coords=[("ADD",6,4),("ADD",5,1)],ADD_NODE_TOL=0.05,tstype=0)
        gsm.restart_string()
        #gsm.go_gsm(max_iters=50,max_steps=20)
        print "getting tangents"
        gsm.get_tangents_1()
        #gsm.icoords[6].isTSnode=True
        #print "making initial Hint"
        #gsm.icoords[6].make_Hint()
        #eig,tmph = np.linalg.eigh(gsm.icoords[6].Hint)
        #print "initial eigenvalues"
        #print eig

        print "getting eigenv finite"
        gsm.get_eigenv_finite(6)
        #print "taking follow eigenvector step"
        #gsm.icoords[6].opt_step(1,[False],gsm.ictan[6],True)
        #gsm.icoords[6].newHess=2
        #grad=np.loadtxt("grad.txt")
        #grad = np.reshape(grad,(len(grad),1))
        #gsm.icoords[6].gradq = gsm.icoords[6].grad_to_q(grad)
        #gsm.icoords[6].update_ic_eigen_ts(gsm.ictan[6])
        gsm.icoords[6].newHess=2
        gsm.optimize(n=6,nsteps=2,nconstraints=0,ictan=gsm.ictan[6],follow_overlap=True)
        #gsm.icoords[6].update_bofill()
