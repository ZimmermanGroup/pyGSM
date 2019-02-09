import numpy as np
import openbabel as ob
import pybel as pb
import options
import os
from units import *
import itertools
from copy import deepcopy
import manage_xyz
from _obutils import Utils
from _opt_utils import OStep_utils
from _icoord import *
from _bmat import Bmat
import elements 
from sklearn import preprocessing
import StringIO
from pes import *
from penalty_pes import *
from avg_pes import *

class DLC(object,Bmat,Utils,ICoords,OStep_utils):

    @staticmethod
    def default_options():
        """ DLC default options. """

        if hasattr(DLC, '_default_options'): return DLC._default_options.copy()
        opt = options.Options() 
        opt.add_option(
            key='isOpt',
            value=1,
            required=False,
            allowed_types=[int],
            doc='Something to do with how coordinates are setup? Ask Paul')

        opt.add_option(
            key='print_level',
            value=1,
            required=False,
            allowed_types=[int],
            doc='0-- no printing, 1-- printing')

        opt.add_option(
            key='MAX_FRAG_DIST',
            value=12.0,
            required=False,
            allowed_types=[float],
            doc='Maximum fragment distance considered for making fragments')

        opt.add_option(
            key='resetopt',
            value=True,
            required=False,
            allowed_types=[bool],
            doc='Whether to reset geom during optimization')

        opt.add_option(
                key="mol",
                required=False,
                allowed_types=[pb.Molecule],
                doc='Pybel molecule object (not OB.Mol)')

        opt.add_option(
                key="PES",
                required=True,
                doc='Potential energy surface object')

        opt.add_option(
                key="bonds",
                value=None,
                required=False,
                )

        opt.add_option(
                key="angles",
                value=None,
                required=False,
                )

        opt.add_option(
                key="torsions",
                value=None,
                required=False,
                )

        opt.add_option(
                key="FZN_ATOMS",
                value=None,
                required=False,
                )

        opt.add_option(
            key='OPTTHRESH',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
                key='EXTRA_BONDS',
                value='',
                required=False,
                doc='extra bond internal coordinate for creating DLC.'
                )

        opt.add_option(
                key='IC_region',
                required=False,
                doc='for hybrid dlc, what residues are to be used to form ICs'
                )

        DLC._default_options = opt
        return DLC._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return DLC(DLC.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes
        self.mol = self.options['mol']
        self.isOpt = self.options['isOpt']
        self.MAX_FRAG_DIST = self.options['MAX_FRAG_DIST']
        self.PES = self.options['PES']
        self.bonds = self.options['bonds']
        self.angles = self.options['angles']
        self.torsions = self.options['torsions']
        self.print_level=self.options['print_level']
        self.resetopt=self.options['resetopt']
        self.OPTTHRESH=self.options['OPTTHRESH']
        self.FZN_ATOMS=self.options['FZN_ATOMS']
        self.EXTRA_BONDS=self.options['EXTRA_BONDS']
        self.IC_region=self.options['IC_region']
        self.madeBonds = False
        self.isTSnode = False
        self.update_hess=False
        self.buf = StringIO.StringIO() 
        self.natoms= len(self.mol.atoms)
        #self.xyzatom_bool = np.zeros(3*self.natoms,dtype=bool)
        self.xyzatom_bool = np.zeros(self.natoms,dtype=bool)
        self.nxyzatoms=0
        self.get_nxyzics()
        if self.bonds is not None:
            self.BObj = Bond_obj(self.bonds,None,None)
            self.BObj.update(self.mol)
            self.madeBonds = True
            self.AObj = Ang_obj(self.angles,None,None)
            self.AObj.update(self.mol)
            self.TObj = Tor_obj(self.torsions,None,None)
            self.TObj.update(self.mol)
        self.setup()


    def setup(self):
        """ setup extra variables etc.,"""
        print "in setup"
        self.HESS_TANG_TOL_TS=0.5 #was 0.35
        self.HESS_TANG_TOL=0.75
        self.path_overlap=0.0
        self.V0 = 0.0
        self.coords = np.zeros((len(self.mol.atoms),3))
        self.isTSnode=False
        for i,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
            self.coords[i,0] = a.GetX()
            self.coords[i,1] = a.GetY()
            self.coords[i,2] = a.GetZ()
        self.ic_create()
        self.bmatp=self.bmatp_create()
        self.bmatp_to_U()
        self.bmat_create()
        self.make_Hint()  
        self.pgradqprim = np.zeros((self.num_ics,1),dtype=float)
        self.gradqprim = np.zeros((self.num_ics,1),dtype=float)
        self.gradq = np.zeros((self.nicd,1),dtype=float)
        self.gradrms = 1000.
        self.SCALEQN = 1.0
        self.MAXAD = 0.075
        self.ixflag = 0
        self.energy = 0.
        self.DMAX = 0.1
        self.nretry = 0 
        self.DMIN0 =0.001#self.DMAX/10.
        # TODO might be a Pybel way to do 
        atomic_nums = self.getAtomicNums()
        Elements = elements.ElementData()
        myelements = [ Elements.from_atomic_number(i) for i in atomic_nums]
        atomic_symbols = [ele.symbol for ele in myelements]
        self.geom=manage_xyz.combine_atom_xyz(atomic_symbols,self.coords)
        if self.FZN_ATOMS is not None:
            print "Freezing atoms",self.FZN_ATOMS
            for a in self.FZN_ATOMS:
                assert a>0, "Frozen atom index is 1 indexed"
                assert a<len(atomic_nums)+1, "Frozen atom index must be in set of atoms."

    @staticmethod
    def union_ic(
            icoordA,
            icoordB,
            ):

        bondA = icoordA.BObj.bonds
        bondB = icoordB.BObj.bonds
        angleA = icoordA.AObj.angles
        angleB = icoordB.AObj.angles
        torsionA = icoordA.TObj.torsions
        torsionB = icoordB.TObj.torsions
    
        for bond in bondB:
            if bond in bondA:
                pass
            elif (bond[1],bond[1]) in bondA:
                pass
            else:
                bondA.append(bond)
        permAngle = list(itertools.permutations([0,2]))
        permTor = list(itertools.permutations([0,3]))
        for angle in angleB:
            foundA=False
            for perm in permAngle:
                if (angle[perm[0]],angle[1],angle[perm[1]]) in angleA:
                    foundA=True
                    break
            if foundA==False:
                angleA.append(angle)
        for torsion in torsionB:
            foundA=False
            for perm in permTor:
                if (torsion[perm[0]],torsion[1],torsion[2],torsion[perm[1]]) in torsionA:
                    foundA=True
                    break
                elif (torsion[perm[0]],torsion[2],torsion[1],torsion[perm[1]]) in torsionA:
                    foundA=True
                    break
            if foundA==False:
                torsionA.append(torsion)

        print "printing Union ICs"
        print bondA
        print angleA
        print "Number of bonds,angles and torsions is %i %i %i" % (len(bondA),len(angleA),len(torsionA))
        print torsionA
        icoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1=pb.readfile('xyz','tmp1.xyz').next()

        lot1 = icoordA.PES.lot.copy(icoordA.PES.lot,icoordA.PES.lot.node_id)
        if icoordA.PES.__class__.__name__=="Avg_PES":
            PES1 = Avg_PES(icoordA.PES.PES1,icoordA.PES.PES2,lot1)
        else:
            PES1 = PES(icoordA.PES.options.copy().set_values({
                "lot": lot1,
                }))

        return DLC(icoordA.options.copy().set_values({
            "bonds":bondA,
            "angles":angleA,
            "torsions":torsionA,
            'mol':mol1,
            'PES':PES1,
            }))


    def ic_create(self):
        self.coordn = self.coord_num()

        if self.madeBonds==False:
            print " making bonds"
            self.BObj = self.make_bonds()
            #TODO  not sure what this isOpt thing is for 1/30/2019 CRA
            if self.isOpt>0:
                print(" isOpt: %i" %self.isOpt)
                self.nfrags,self.frags = self.make_frags()
                self.BObj.update(self.mol)
                self.bond_frags()
                self.AObj = self.make_angles()
                self.TObj = self.make_torsions()
                self.linear_ties()
                self.AObj.update(self.mol)
                self.TObj.update(self.mol)
        else:
            self.BObj.update(self.mol)
            self.AObj.update(self.mol)
            self.TObj.update(self.mol)

        self.num_ics_p = self.BObj.nbonds + self.AObj.nangles + self.TObj.ntor
        print "nxyzatoms=",self.nxyzatoms
        self.num_ics = self.BObj.nbonds + self.AObj.nangles + self.TObj.ntor + self.nxyzatoms*3
        #self.make_imptor()
        #self.make_nonbond() 

    def update_xyz(self):
        """ Updates the mol.OBMol object coords: Important for ICs"""
        for i,xyz in enumerate(self.coords):
            self.mol.OBMol.GetAtom(i+1).SetVector(xyz[0],xyz[1],xyz[2])

    def linear_ties(self):
        maxsize=0
        for anglev in self.AObj.anglev:
            if anglev>160.:
                maxsize+=1
        blist=[]
        n=0
        for anglev,angle in zip(self.AObj.anglev,self.AObj.angles):
            if anglev>160.:
                blist.append(angle)
                print(" linear angle %i of %i: %s (%4.2f)" %(n+1,maxsize,angle,anglev))
                n+=1

        # atoms attached to linear atoms
        clist=[[]]*n
        m =[]
        for i in range(n):
            # b is the vertex 
            a=self.mol.OBMol.GetAtom(blist[i][0])
            b=self.mol.OBMol.GetAtom(blist[i][1])
            c=self.mol.OBMol.GetAtom(blist[i][2])
            tmp=0
            tmplist=[]
            for nbr in ob.OBAtomAtomIter(a):
                if nbr.GetIndex() != b.GetIndex():
                    tmplist.append(nbr.GetIndex()+1)
                    print nbr.GetIndex(),
                    tmp+=1
            for nbr in ob.OBAtomAtomIter(c):
                if nbr.GetIndex() != b.GetIndex():
                    print nbr.GetIndex(),
                    tmplist.append(nbr.GetIndex()+1)
                    tmp+=1
            clist[i]=tmplist
            m.append(tmp)
            print
        for i in range(n):
            a1=blist[i][0]
            a2=blist[i][2] # not vertices
            bond=(a1,a2)
            if self.bond_exists(bond) == False:
                print(" adding bond via linear ties %s" % (bond,))
                self.BObj.bonds.append(bond)
                self.BObj.nbonds +=1
            for j in range(m[i]):
                for k in range(j):
                    b1=clist[i][j]
                    b2=clist[i][k]
                    found=False
                    for angle in self.AObj.angles:
                        if b1==angle[0] and b2==angle[2]: 
                            found=True
                        elif b2==angle[0] and b1==angle[2]:
                            found=True
                    if found==False:
                        if self.bond_exists((b1,a1))==True:
                            c1=b1
                        if self.bond_exists((b2,a1))==True:
                            c1=b2
                        if self.bond_exists((b1,a2))==True:
                            c2=b1
                        if self.bond_exists((b2,a2))==True:
                            c2=b2
                        #SHould add a try except here?
                        torsion= (c1,a1,a2,c2)
                        if not self.torsion_exists(torsion) and len(set(torsion))==4:
                            print(" adding torsion via linear ties %s" %(torsion,))
                            self.TObj.torsions.append(torsion)
                            self.TObj.ntor +=1
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

    def bmatp_to_U(self):
        G=np.matmul(self.bmatp,np.transpose(self.bmatp))
        # Singular value decomposition
        v_temp,e,vh  = np.linalg.svd(G)
        v = np.transpose(v_temp)
        lowev=0

        #make function called set_nicd --> one for dlc and one for hdlc
        self.set_nicd()
        redset = self.num_ics - self.nicd

        for eig in e[:self.nicd]:
            if eig<0.001:
                lowev+=1

        #if lowev>0:
        #    ev=np.diag(e)
        #    print " lowev = ",lowev
        #    np.savetxt('test3.out', ev, delimiter=',',fmt='%1.2e')
        #    print e
        #    print "shape G=",np.shape(G)
        #    print "numics=",self.num_ics
        #    print "numics_p=",self.num_ics_p
        #    print "3N=",self.natoms*3
        #    print "nicd=",self.nicd
        #    print "redset=",redset

        if lowev>3:
            print(" Error: optimization space less than 3N DOF")
            print "lowev=",lowev
            print "shape G=",np.shape(G)
            print "numics=",self.num_ics
            print "numics_p=",self.num_ics_p
            print "3N=",self.natoms*3
            print "nicd=",self.nicd
            print "redset=",redset
            print e[:self.nicd]
            exit(-1)
        self.nicd -= lowev

        #idx = e.argsort()[::-1]
        #v = v[idx[::-1]]
        #self.Ut=v[redset+lowev:,:]
        #print e[:self.nicd]
        self.Ut =v[:self.nicd]
        np.savetxt('test3.out', self.Ut, delimiter=',',fmt='%1.2e')
        self.torv0 = list(self.TObj.torv)
        
    def bmat_create(self):
        #print(" In bmat create")
        self.q = self.q_create()
        if self.print_level==2:
            print "printing q"
            print self.q.T
        bmat = np.matmul(self.Ut,self.bmatp)
        bbt = np.matmul(bmat,np.transpose(bmat))
        bbti = np.linalg.inv(bbt)
        self.bmatti= np.matmul(bbti,bmat)
        if self.print_level==2:
            print "bmatti"
            print self.bmatti

    def update_ic_eigen_ts(self,ictan):
        """ this method follows the overlap with reaction tangent"""
        opt_type=3
        lambda1 = 0.
        SCALE = self.SCALEQN
        if self.newHess>0: SCALE = self.SCALEQN*self.newHess
        if SCALE > 10:
            SCALE = 10.
        #TODO buf print SCALE

        #testing
        unit_test=False
        if unit_test:
            self.prepare_unit_test()
        else:
            norm = np.linalg.norm(ictan)
            C = ictan/norm
            dots = np.dot(self.Ut,C) #(nicd,numic)(numic,1)
            Cn = np.dot(self.Ut.T,dots) #(numic,nicd)(nicd,1) = numic,1
            norm = np.linalg.norm(Cn)
            Cn = Cn/norm
       
        # => get eigensolution of Hessian <= 
        eigen,tmph = np.linalg.eigh(self.Hint) #nicd,nicd
        tmph = tmph.T

        #TODO nneg should be self and checked
        nneg = 0
        for i in range(self.nicd):
            if eigen[i] < -0.01:
                nneg += 1

        #=> Overlap metric <= #
        overlap = np.dot(np.dot(tmph,self.Ut),Cn) #(nicd,nicd)(nicd,num_ic)(num_ic,1) = (nicd,1)
        #print " printing overlaps ", overlap[:4].T

        # Max overlap metrics
        self.maxol_w_Hess(overlap[0:4])

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1(eigen,4)

        # => if overlap is small use Cn as Constraint <= #
        if self.check_overlap_good(opt_type=4):
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph,self.gradq)
            path_overlap_e_g = gqe[self.path_overlap_n]
            if self.print_level>0:
                print ' gtse: {:1.4f} '.format(path_overlap_e_g[0])
            self.buf.write(' gtse: {:1.4f}'.format(path_overlap_e_g[0]))
            # => calculate eigenvector step <=#
            dqe0 = self.eigenvector_follow_step(SCALE,lambda1,gqe,eigen,4)
            # => Convert step back to DLC basis <= #
            dq = self.convert_dqe0_to_dq(dqe0,tmph)
        else:
            self.form_constrained_DLC(ictan) 
            self.Hint = self.Hintp_to_Hint()
            dq,tmp = self.update_ic_eigen(1)
            opt_type=2

        return dq,opt_type

    def maxol_w_Hess(self,overlap):
        # Max overlap metrics
        absoverlap = np.abs(overlap)
        self.path_overlap = np.max(absoverlap)
        self.path_overlap_n = np.argmax(absoverlap)
        #maxols = overlap[maxoln]
        if self.print_level>-1:
            print " t/ol %i: %3.2f" % (self.path_overlap_n,self.path_overlap)
        self.buf.write(" t/ol %i: %3.2f" % (self.path_overlap_n,self.path_overlap))


    def bond_frags(self):
        if self.nfrags<2:
            return 
        found=found2=found3=found4=0

        frags= [i[0] for i in self.frags]
        isOkay=False
        for n1 in range(self.nfrags):
            for n2 in range(n1):
                print(" Connecting frag %i to %i" %(n1,n2))
                found=found2=found3=found4=0
                close=0.
                a1=a2=b1=b2=c1=c2=d1=d2=-1
                mclose=1000.
                mclose2=1000.
                mclose3=1000.
                mclose4 = 1000.

                frag0 = filter(lambda x: x[0]==n1, self.frags)
                frag1 = filter(lambda x: x[0]==n2, self.frags)
                combs = list(itertools.product(frag0,frag1))
                for comb in combs: 
                    close=self.distance(comb[0][1],comb[1][1])
                    if close < mclose and close < self.MAX_FRAG_DIST:
                        mclose=close
                        a1=comb[0][1]
                        a2=comb[1][1]
                        found=1

                #connect second pair heavies or H-Bond only, away from first pair
                for comb in combs: 
                    close=self.distance(comb[0][1],comb[1][1])
                    dia1 = self.distance(comb[0][1],a1)
                    dja1 = self.distance(comb[1][1],a1)
                    dia2 = self.distance(comb[0][1],a2)
                    dja2 = self.distance(comb[1][1],a2)
                    dist21 = (dia1+dja1)/2.
                    dist22 = (dia2+dja2)/2.

                    #TODO changed from 4.5 to 4
                    #TODO what is getIndex doing here?
                    if (self.getIndex(comb[0][1]) > 1 or self.getIndex(comb[1][1])>1) and dist21 > 4.5 and dist22 >4. and close<mclose2 and close < self.MAX_FRAG_DIST: 
                        mclose2 = close
                        b1=comb[0][1]
                        b2=comb[1][1]
                        found2=1
    
                #TODO
                """
                for i in range(self.natoms):
                    for j in range(self.natoms):
                        if self.frags[i][0]==n1 and self.frags[j][0]==n2 and b1>0 and b2>0:
                            close=self.distance(i,j)
                            #connect third pair, heavies or H-Bond only, away from first pair //TODO what does this mean?
                            dia1 = self.distance(i,a1)
                            dja1 = self.distance(j,a1)
                            dia2 = self.distance(i,a2)
                            dja2 = self.distance(j,a2)
                            dib1 = self.distance(i,b1)
                            djb1 = self.distance(j,b1)
                            dib2 = self.distance(i,b2)
                            djb2 = self.distance(j,b2)
                            dist31 = (dia1+dja1)/2.;
                            dist32 = (dia2+dja2)/2.;
                            dist33 = (dib1+djb1)/2.;
                            dist34 = (dib2+djb2)/2.;
                            if (self.getIndex(i) > 1 or self.getIndex(j)>1) and dist31 > 4.5 and dist32 >4.5 and dist33>4.5 and dist34>4. and close<mclose3 and close < self.MAX_FRAG_DIST:
                                mclose3=close
                                c1=i
                                c2=j
                                found3=1

                for i in range(self.natoms):
                    for j in range(self.natoms):
                        if self.frags[i]==n1 and self.frags[j]==n2 and self.isOpt==2:
                            #connect fourth pair, TM only, away from first pair
                            if c1!=i and c2!=i and c1!=j and c2!=j: #don't repeat 
                                if self.isTM(i) or self.isTM(j):
                                    close=self.distance(i,j)
                                    if close<mclose4 and close<self.MAX_FRAG_DIST:
                                        mclose4=close
                                        d1=i
                                        d2=j
                                        found4=1
                """

                bond1=(a1,a2)
                print "found",found
                print "found2",found2
                print bond1
                if found>0 and self.bond_exists(bond1)==False:
                    print(" bond pair1 added : %s" % (bond1,))
                    self.BObj.bonds.append(bond1)
                    self.BObj.nbonds+=1
                    self.BObj.bondd.append(mclose)
                    print " bond dist: %1.4f" % mclose
                    #TODO check this
                    isOkay = self.mol.OBMol.AddBond(bond1[0],bond1[1],1)
                    print " Bond added okay? %r" % isOkay
                bond2=(b1,b2)
                if found2>0 and self.bond_exists(bond2)==False:
                    self.BObj.bonds.append(bond2)
                    print(" bond pair2 added : %s" % (bond2,))
                bond3=(c1,c2)
                if found3>0 and self.bond_exists(bond3)==False:
                    self.BObj.bonds.append(bond3)
                    print(" bond pair2 added : %s" % (bond3,))
                bond4=(d1,d2)
                if found4>0 and self.bond_exists(bond4)==False:
                    self.BObj.bonds.append(bond4)
                    print(" bond pair2 added : %s" % (bond24,))


                if self.isOpt==2:
                    print(" Checking for linear angles in newly added bond")
                    #TODO
        return isOkay

    @staticmethod
    def add_node_SE(ICoordA,driving_coordinate,dqmag_max=0.8,dqmag_min=0.2):

        dq0 = np.zeros((ICoordA.nicd,1))
        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        lot1 = ICoordA.PES.lot.copy(ICoordA.PES.lot,ICoordA.PES.lot.node_id+1)
        if ICoordA.PES.__class__.__name__=="Avg_PES":
            PES1 = Avg_PES(ICoordA.PES.PES1,ICoordA.PES.PES2,lot1)
        else:
            PES1 = PES(ICoordA.PES.options.copy().set_values({
                "lot": lot1,
                }))
        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol" : mol1,
            "bonds" : ICoordA.BObj.bonds,
            "angles" : ICoordA.AObj.angles,
            "torsions" : ICoordA.TObj.torsions,
            "PES" : PES1,
            }))

        ICoordC.setup()
        ictan,bdist = DLC.tangent_SE(ICoordA,driving_coordinate)
        ICoordC.opt_constraint(ictan)
        bdist = np.linalg.norm(ictan)
        #bdist = np.dot(ICoordC.Ut[-1,:],ictan)
        ICoordC.bmatp=ICoordC.bmatp_create()
        ICoordC.bmat_create()
        dqmag_scale=1.5
        minmax = dqmag_max - dqmag_min
        a = bdist/dqmag_scale
        if a>1:
            a=1
        dqmag = dqmag_min+minmax*a
        print " dqmag: %4.3f from bdist: %4.3f" %(dqmag,bdist)

        dq0[ICoordC.nicd-1] = -dqmag

        print " dq0[constraint]: %1.3f" % dq0[ICoordC.nicd-1]
        ICoordC.ic_to_xyz(dq0)
        ICoordC.update_ics()
        ICoordC.bmatp=ICoordC.bmatp_create()
        ICoordC.bmatp_to_U()
        ICoordC.bmat_create()
        ICoordC.mol.write('xyz','after.xyz',overwrite=True)
        
        # => stash bdist <= #
        ictan,bdist = DLC.tangent_SE(ICoordC,driving_coordinate,quiet=True)
        ICoordC.bdist = bdist
        if np.all(ictan==0.0):
            raise RuntimeError
        #ICoordC.dqmag = dqmag
        ICoordC.Hintp = ICoordA.Hintp

        return ICoordC

    @staticmethod
    def add_node_SE_X(ICoordA,driving_coordinate,dqmag_max=0.8,dqmag_min=0.2,BDISTMIN=0.05):

        dq0 = np.zeros((ICoordA.nicd,1))
        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        lot1 = ICoordA.PES.lot.copy(ICoordA.PES.lot,ICoordA.PES.lot.node_id+1)
        pes1 = PES(ICoordA.PES.PES1.options.copy().set_values({
            "lot": lot1,
            }))
        pes2 = PES(ICoordA.PES.PES2.options.copy().set_values({
            "lot": lot1,
            }))
        pes = Penalty_PES(pes1,pes2,lot1)

        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol" : mol1,
            "bonds" : ICoordA.BObj.bonds,
            "angles" : ICoordA.AObj.angles,
            "torsions" : ICoordA.TObj.torsions,
            "PES" : pes,
            }))

        ICoordC.setup()
        ictan,bdist = DLC.tangent_SE(ICoordA,driving_coordinate)
        if bdist<BDISTMIN:
            print "bdist too small"
            return 0
        ICoordC.opt_constraint(ictan)
        #bdist = np.linalg.norm(ictan)
        ICoordC.bmatp=ICoordC.bmatp_create()
        ICoordC.bmat_create()
        dqmag_scale=1.5
        minmax = dqmag_max - dqmag_min
        a = bdist/dqmag_scale
        if a>1:
            a=1
        dqmag = dqmag_min+minmax*a
        print " dqmag: %4.3f from bdist: %4.3f" %(dqmag,bdist)

        dq0[ICoordC.nicd-1] = -dqmag

        print " dq0[constraint]: %1.3f" % dq0[ICoordC.nicd-1]
        ICoordC.ic_to_xyz(dq0)
        ICoordC.update_ics()
        ICoordC.bmatp_create()
        ICoordC.bmatp_to_U()
        ICoordC.bmat_create()
        ICoordC.mol.write('xyz','after.xyz',overwrite=True)
    
        # => stash bdist <= #
        ictan,bdist = DLC.tangent_SE(ICoordC,driving_coordinate,quiet=True)
        ICoordC.bdist = bdist
        if np.all(ictan==0.0):
            raise RuntimeError
        
        #ICoordC.dqmag = dqmag
        ICoordC.Hintp = ICoordA.Hintp

        return ICoordC

    @staticmethod
    def add_node(ICoordA,ICoordB,nmax,ncurr):
        dq0 = np.zeros((ICoordA.nicd,1))

        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        if ICoordB.PES.lot.node_id > ICoordA.PES.lot.node_id:
            node_id = ICoordA.PES.lot.node_id + 1
        else:
            node_id = ICoordA.PES.lot.node_id - 1
        lot1 = ICoordA.PES.lot.copy(ICoordA.PES.lot,node_id)

        #TODO can make this better, ask Josh
        if ICoordA.PES.__class__.__name__=="Avg_PES":
            PES1 = Avg_PES(ICoordA.PES.PES1,ICoordA.PES.PES2,lot1)
        else:
            PES1 = PES(ICoordA.PES.options.copy().set_values({
                "lot": lot1,
                }))
        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol" : mol1,
            "bonds" : ICoordA.BObj.bonds,
            "angles" : ICoordA.AObj.angles,
            "torsions" : ICoordA.TObj.torsions,
            "PES" : PES1,
            }))

        ICoordC.setup()
        ictan = DLC.tangent_1(ICoordA,ICoordB)
        ICoordC.form_constrained_DLC(ictan)
        dqmag = np.dot(ICoordC.Ut[-1,:],ictan)
        print " dqmag: %1.3f"%dqmag
        if nmax-ncurr > 1:
            dq0[ICoordC.nicd-1] = -dqmag/float(nmax-ncurr)
        else:
            dq0[ICoordC.nicd-1] = -dqmag/2.0;

        print " dq0[constraint]: %1.3f" % dq0[ICoordC.nicd-1]
        ICoordC.ic_to_xyz(dq0)
        ICoordC.update_ics()
        ICoordC.form_unconstrained_DLC()
        assert ICoordC.PES.lot.hasRanForCurrentCoords==False,"WTH1"

        ICoordC.Hintp = ICoordA.Hintp

        return ICoordC

    @staticmethod
    def copy_node(ICoordA,new_node_id,rtype=0):
        if isinstance(ICoordA.PES,Penalty_PES):
            ICoordC = DLC.copy_node_X(ICoordA,new_node_id,rtype)
            return ICoordC
        else:
            ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
            mol1 = pb.readfile('xyz','tmp1.xyz').next()
            lot1 = ICoordA.PES.lot.copy(
                    ICoordA.PES.lot,
                    new_node_id)
            if ICoordA.PES.__class__.__name__=="Avg_PES":
                PES1 = Avg_PES(ICoordA.PES.PES1,ICoordA.PES.PES2,lot1)
            else:
                PES1 = PES(ICoordA.PES.options.copy().set_values({
                    "lot": lot1,
                    }))

            ICoordC = DLC(ICoordA.options.copy().set_values({
                "mol" : mol1,
                "bonds" : ICoordA.BObj.bonds,
                "angles" : ICoordA.AObj.angles,
                "torsions" : ICoordA.TObj.torsions,
                "PES" : PES1,
                }))
            ICoordC.Hintp = ICoordA.Hintp

            return ICoordC

    @staticmethod
    def copy_node_X(ICoordA,new_node_id,rtype=0):
        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        lot1 = ICoordA.PES.lot.copy(ICoordA.PES.lot,new_node_id)
        if rtype>=5:
            lot1.do_coupling=True
        pes1 = PES(ICoordA.PES.PES1.options.copy().set_values({
            "lot": lot1,
            }))
        pes2 = PES(ICoordA.PES.PES2.options.copy().set_values({
            "lot": lot1,
            }))
        if rtype>=5:
            pes = Avg_PES(pes1,pes2,lot1)
        else:
            pes = Penalty_PES(pes1,pes2,lot1)
        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol":mol1,
            "bonds":ICoordA.BObj.bonds,
            "angles":ICoordA.AObj.angles,
            "torsions":ICoordA.TObj.torsions,
            "PES":pes,
            }))
        ICoordC.setup()
        ICoordC.Hintp = ICoordA.Hintp
        return ICoordC

    def update_ics(self):
        self.update_xyz()
        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords)
        self.PES.lot.hasRanForCurrentCoords= False
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

    def ic_to_xyz(self,dq):
        """ Transforms ic to xyz, used by addNode"""
        assert np.shape(dq) == np.shape(self.q),"operands could not be broadcas"
        self.bmatp=self.bmatp_create()
        self.bmat_create()
        SCALEBT = 1.5
        N3=self.natoms*3
        qn = self.q + dq  #target IC values
        xyzall=[]
        magall=[]
        magp=100

        opt_molecules=[]
        xyzfile=os.getcwd()+"/ic_to_xyz.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

        for n in range(10):
            btit = np.transpose(self.bmatti)
            xyzd = np.dot(btit,dq)
            assert len(xyzd)==3*self.natoms,"xyzd is not N3 dimensional"
            xyzd = np.reshape(xyzd,(self.natoms,3))

            # => Frozen <= #
            if self.FZN_ATOMS is not None:
                for a in [3*i for i in self.FZN_ATOMS]:
                    xyzd[a:a+3]=0.

            # => Calc Mag <= #
            mag=np.dot(np.ndarray.flatten(xyzd),np.ndarray.flatten(xyzd))
            magall.append(mag)

            if mag>magp:
                SCALEBT *=1.5
            magp=mag

            # update coords
            xyz1 = self.coords + xyzd/SCALEBT 
            xyzall.append(xyz1)
            self.coords = np.copy(xyz1)
            self.update_ics()
            self.bmatp=self.bmatp_create()
            self.bmat_create()
            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            dq = qn - self.q

            if mag<0.00005: break

        #write convergence
        largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        for mol in opt_molecules:
            largeXyzFile.write(pb.readstring("xyz",mol))

        #print xyzall
        #self.mol.OBMol.GetAtom(i+1).SetVector(result[0],result[1],result[2])

        #TODO implement mag check here

        return 

    def ic_to_xyz_opt(self,dq0):
        MAX_STEPS = 8
        rflag = 0 
        retry = False
        SCALEBT = 1.5
        N3 = self.natoms*3
        xyzall=[]
        magall=[]
        dqmagall=[]
        self.update_ics()

        #Current coords
        xyzall.append(self.coords)

        magp=100
        dqmagp=100.

        dq = dq0
        #target IC values
        qn = self.q + dq 

        #primitive internal values
        qprim = np.concatenate((self.BObj.bondd,self.AObj.anglev,self.TObj.torv))

        opt_molecules=[]
        xyzfile=os.getcwd()+"/ic_to_xyz.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

        # => Calc Change in Coords <= #
        for n in range(MAX_STEPS):
            #print "ic iteration %i" % n
            btit = np.transpose(self.bmatti)
            xyzd=np.dot(btit,dq)
            if self.print_level==2:
                print "xyzd"
                print xyzd.T
            assert len(xyzd)==3*self.natoms,"xyzd is not N3 dimensional"
            xyzd = np.reshape(xyzd,(self.natoms,3))

            # => Frozen <= #
            if self.FZN_ATOMS is not None:
                for a in [3*i for i in self.FZN_ATOMS]:
                    xyzd[a:a+3]=0.

            # => Add Change in Coords <= #
            xyz1 = self.coords + xyzd/SCALEBT 

            # => Calc Mag <= #
            mag=np.dot(np.ndarray.flatten(xyzd),np.ndarray.flatten(xyzd))
            magall.append(mag)
            xyzall.append(xyz1)

            # update coords
            xyzp = np.copy(self.coords) # note that when we modify coords, xyzp will not change
            self.coords = xyz1

            self.update_ics()
            self.bmatp=self.bmatp_create()
            self.bmat_create()
            if self.print_level==2:
                print self.bmatti

            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            #calc new dq
            dq = qn - self.q
            if self.print_level==2:
                print "dq is ", dq.T

            dqmag = np.linalg.norm(dq)
            dqmagall.append(dqmag)
            if dqmag<0.0001: break

            if dqmag>dqmagp*10.:
                print(" Q%i" % n)
                SCALEBT *= 2.0
                self.coords = np.copy(xyzp)
                self.update_ics()
                self.bmatp=self.bmatp_create()
                self.bmat_create()
                dq = qn - self.q
            magp = mag
            dqmagp = dqmag

            if mag<0.00005: break

        MAXMAG = 0.025*self.natoms
        if np.sqrt(mag)>MAXMAG:
            self.ixflag +=1
            maglow = 100.
            nlow = -1
            for n,mag in enumerate(magall):
                if mag<maglow:
                    maglow=mag
                    nlow =n
            if maglow<MAXMAG:
                coords = xyzall[nlow]
                print("Wb(%6.5f/%i)" %(maglow,nlow))
            else:
                coords=xyzall[0]
                rflag = 1
                print("Wr(%6.5f/%i)" %(maglow,nlow))
                dq0 = dq0/2
                retry = True
                self.nretry+=1
                if self.nretry>100:
                    retry=False
                    print "Max retries"
        elif self.ixflag>0:
            self.ixflag = 0

        if retry==False:
            self.update_ics()
            torsion_diff=[]
            for i,j in zip(self.TObj.torv,qprim[self.BObj.nbonds+self.AObj.nangles:]):
                tordiff = i-j
                if tordiff>180.:
                    torfix=-360.
                elif tordiff<-180.:
                    torfix=360.
                else:
                    torfix=0.
                torsion_diff.append(tordiff+torfix)

            bond_diff = self.BObj.bondd - qprim[:self.BObj.nbonds]
            angle_diff = self.AObj.anglev - qprim[self.BObj.nbonds:self.AObj.nangles+self.BObj.nbonds]
            angle_diff=[a*np.pi/180. for a in angle_diff]
            torsion_diff=[t*np.pi/180. for t in torsion_diff]
            self.dqprim = np.concatenate((bond_diff,angle_diff,torsion_diff))
            self.dqprim = np.reshape(self.dqprim,(self.num_ics,1))

        #write convergence geoms to file 
        #largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        #for mol in opt_molecules:
        #    largeXyzFile.write(pb.readstring("xyz",mol))
        if self.print_level==2:
            print "dqmagall,magall"
            print dqmagall
            print magall
       
        if retry==True:
            self.ic_to_xyz_opt(dq0)
        else:
            return rflag

    def make_Hint(self):
        self.newHess = 5
        Hdiagp = []
        for bond in self.BObj.bonds:
            Hdiagp.append(0.35*self.close_bond(bond))
        for angle in self.AObj.angles:
            Hdiagp.append(0.2)
        for tor in self.TObj.torsions:
            Hdiagp.append(0.035)
        for xyzic in range(self.nxyzatoms*3):
            Hdiagp.append(1.0)

        self.Hintp=np.diag(Hdiagp)
        Hdiagp=np.asarray(Hdiagp)
        Hdiagp=np.reshape(Hdiagp,(self.num_ics,1))

        tmp = np.zeros((self.nicd,self.num_ics),dtype=float)
        for i in range(self.nicd): 
            for k in range(self.num_ics):
                tmp[i,k] = self.Ut[i,k]*Hdiagp[k]

        self.Hint = np.matmul(tmp,np.transpose(self.Ut))
        try:
            self.Hinv = np.linalg.inv(self.Hint)
        except:
            print "nicd=",self.nicd
            print "numic=",self.num_ics
            print np.shape(self.Ut)
            print np.shape(tmp)
            print np.shape(self.Hintp)
            print np.shape(self.Hint)
            exit(1)

        #TODO ?
        #if self.optCG==False or self.isTSnode==False:
        #    print "Not implemented"

    def update_for_step(self,opt_type):
        self.energy=self.energyp=self.PES.get_energy(self.geom)
        grad = self.PES.get_gradient(self.geom)
        nconstraints=self.get_nconstraints(opt_type)
        if opt_type!=3 and opt_type!=4:
            self.Hint = self.Hintp_to_Hint()
        # =>grad in ics<= #
        self.gradq = self.grad_to_q(grad)
        self.pgradq = np.copy(self.gradq)
        if self.print_level==2:
            print "gradq"
            print self.gradq.T
        self.gradrms = np.sqrt(np.dot(self.gradq.T[0,:self.nicd-nconstraints],self.gradq[:self.nicd-nconstraints,0])/(self.nicd-nconstraints))
        self.pgradrms = self.gradrms

        # => Update Hessian <= #
        self.pgradqprim=self.gradqprim
        self.gradqprim = np.dot(np.transpose(self.Ut),self.gradq)
    
        mode=1
        if opt_type in [3,4]:
            mode=2
        if self.update_hess == True:
            self.update_Hessian(mode)
        self.update_hess = True


    def step_controller(self,opt_type):

        #do this if close to seam if coupling, don't do this if isTSnode or exact TS search (opt_type 4)
        if ( self.dEstep>0.1 and not self.isTSnode and (opt_type in [0,1,2,3] or (self.PES.lot.do_coupling and self.PES.dE<0.1))):
            if self.print_level>0:
                print("decreasing DMAX"),
            self.buf.write(" decreasing DMAX")
            if self.smag <self.DMAX:
                self.DMAX = self.smag/1.5
            else: 
                self.DMAX = self.DMAX/1.5
            if self.dEstep > 2.0 and self.resetopt==True:
                #if self.print_level>0:
                print "resetting coords to coorp"
                self.coords = self.coorp
                self.update_ics()
                self.energy = self.PES.get_energy(self.geom)
                grad = self.PES.get_gradient(self.geom)
                self.gradq = self.grad_to_q(grad)
                nconstraints=self.get_nconstraints(opt_type)
                self.gradrms = np.sqrt(np.dot(self.gradq.T[0,:self.nicd-nconstraints],self.gradq[:self.nicd-nconstraints,0])/(self.nicd-nconstraints))
                self.update_hess=False

        elif opt_type==4 and self.ratio<0. and abs(self.dEpre)>0.05:
            if self.print_level>0:
                print("sign problem, decreasing DMAX"),
            self.buf.write(" sign problem, decreasing DMAX")
            self.DMAX = self.DMAX/1.35

        elif (self.ratio<0.25 or self.ratio>1.5): #can also check that dEpre >0.05?
            if self.print_level>0:
                print("decreasing DMAX"),
            self.buf.write(" decreasing DMAX")
            if self.smag<self.DMAX:
                self.DMAX = self.smag/1.1
            else:
                self.DMAX = self.DMAX/1.2

        elif self.ratio>0.75 and self.ratio<1.25 and self.smag > self.DMAX and self.gradrms<(self.pgradrms*1.35):
            if self.print_level>0:
                print("increasing DMAX"),
            self.buf.write(" increasing DMAX")
            self.DMAX=self.DMAX*1.1 + 0.01
            if self.DMAX>0.25:
                self.DMAX=0.25

        elif self.DMAX==self.DMIN0 and self.ratio>0.5 and self.dEstep<0.:
            if self.print_level>0:
                print("increasing DMAX"),
            self.buf.write(" increasing DMAX")
            self.DMAX=self.DMAX*1.1 + 0.01

        if self.DMAX<self.DMIN0:
            self.DMAX=self.DMIN0

    def update_DLC(self,opt_type,ictan):
        if opt_type==0:
            self.form_unconstrained_DLC()
        elif opt_type in [1,2]:
            self.form_constrained_DLC(ictan)
        elif opt_type==5:
            self.form_CI_DLC()
        elif opt_type in [6,7]:
            self.form_constrained_CI_DLC(constraints=ictan)
            #raise NotImplementedError #TODO for seams

    def get_constraint_steps(self,opt_type):
        nconstraints=self.get_nconstraints(opt_type)
        constraint_steps=[0]*nconstraints
        # => normal,ictan opt,follow
        if opt_type in [0,1,3,4]:
            return constraint_steps
        # => ictan climb
        elif opt_type==2: 
            constraint_steps[0]=self.walk_up(self.nicd-1)
        # => MECI
        elif opt_type==5: 
            constraint_steps[1] = self.dgrad_step() #last vector is x
        # => seam opt
        elif opt_type==6:
            constraint_steps[1] = self.dgrad_step()  #0 is dvec, 1 is dgrad, 3 is ictan
        # => seam climb
        elif opt_type==7:
            constraint_steps[1] = self.dgrad_step()  #0 is dvec, 1 is dgrad, 3 is ictan
            constraint_steps[0]=self.walk_up(self.nicd-1)

        return constraint_steps

    def opt_step(self,opt_type=0,ictan=None,refE=0):
        # => update PES info <= #
        if opt_type!=3 and opt_type!=4:
            self.update_DLC(opt_type,ictan)
        else:
            self.bmatp = self.bmatp_create()
            self.bmat_create()

        # => update DLC, grad, Hess, etc
        self.update_for_step(opt_type)
        if self.gradrms<self.OPTTHRESH:
            return 0.

        # => form eigenvector step in non-constrained space <= #
        self.dq,opt_type = self.eigenvector_step(opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)

        # => calculate constraint step <= #
        constraint_steps = self.get_constraint_steps(opt_type)
        # => add constraint_step to step <= #
        for n in range(nconstraints):
            self.dq[-nconstraints+n]=constraint_steps[n]
        if self.print_level>1:
            print "dq for step is "
            print self.dq.T

        # => update geometry <=#
        self.coorp = np.copy(self.coords)
        rflag = self.ic_to_xyz_opt(self.dq)

        #TODO if rflag and ixflag
        if rflag==True:
            print "rflag" 
            self.DMAX=self.DMAX/1.6
            self.dq=self.update_ic_eigen(nconstraints)
            self.ic_to_xyz_opt(self.dq)
            self.update_hess=False

        ## => update ICs,xyz <= #
        self.update_ics()
     
        # => calc energy at new position <= #
        self.energy = self.PES.get_energy(self.geom)

        #form DLC at new position
        if opt_type!=3 and opt_type!=4:
            self.update_DLC(opt_type,ictan)
        else:
            self.bmatp = self.bmatp_create()
            self.bmat_create()

        # check goodness of step
        self.dEstep = self.energy - self.energyp
        self.dEpre = self.compute_predE(self.dq,nconstraints)

        # constraint contribution
        for n in range(nconstraints):
            self.dEpre +=self.gradq[-n-1]*self.dq[-n-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq

        # ratio and gradrms
        self.ratio = self.dEstep/self.dEpre
        grad = self.PES.get_gradient(self.geom)
        self.gradq = self.grad_to_q(grad)
        self.gradrms = np.sqrt(np.dot(self.gradq.T[0,:self.nicd-nconstraints],self.gradq[:self.nicd-nconstraints,0])/(self.nicd-nconstraints))

        # => step controller  <= #
        self.step_controller(opt_type)

        self.buf.write(" E(M): %3.4f" %(self.energy - refE))
        if self.print_level>0:
            print " E(M): %3.5f" % (self.energy-refE),
        self.buf.write(" predE: %1.4f ratio: %1.4f" %(self.dEpre, self.ratio))
        if self.print_level>0:
            print " ratio is %1.4f" % self.ratio,
            print " predE: %1.4f" %self.dEpre,
            print " dEstep = %3.2f" %self.dEstep,

        for n in range(nconstraints):
            self.buf.write(" cg[%i] %1.3f" %(n,self.gradq[-n-1]))
        if self.print_level>0:
            print("gradrms = %1.5f" % self.gradrms),
        self.buf.write(" gRMS=%1.5f" %(self.gradrms))

        return  self.smag

    def update_Hessian(self,mode=1):
        #print("In update bfgsp")
        ''' mode 1 is BFGS, mode 2 is Bofill'''
        assert mode==1 or mode==2, "no update implemented with that mode"
        self.newHess-=1

        # do this even if mode==2
        change = self.update_bfgsp()

        self.Hintp += change
        if self.print_level==2:
            print "Hintp"
            print self.Hintp

        if mode==1:
            self.Hint=self.Hintp_to_Hint()
        if mode==2:
            change=self.update_bofill()
            self.Hint+=change
            self.Hinv=np.linalg.inv(self.Hint)
    

    def opt_constraint(self,C):
        """
        This function takes a matrix of vectors wrtiten in the basis of ICs
        same as U vectors, and returns a new normalized Ut with those vectors as 
        basis vectors.
        """
        # normalize all constraints
        Cn = preprocessing.normalize(C.T,norm='l2')
        #dots = np.matmul(Cn,Cn.T)

        # orthogonalize
        Cn = self.orthogonalize(Cn) 
        #print "shape of Cn is %s" %(np.shape(Cn),)

        # write Cn in terms of C_U?
        dots = np.matmul(self.Ut,Cn.T)
        C_U = np.matmul(self.Ut.T,dots)

        #print "Cn written in terms of U"
        #print C_U
        # normalize C_U
        try:
            C_U = preprocessing.normalize(C_U.T,norm='l2')
            C_U = self.orthogonalize(C_U) 
            dots = np.matmul(C_U,np.transpose(C_U))
        except:
            print C
            exit(-1)
        #print C_U
        #print "shape of overlaps is %s, shape of Ut is %s, shape of C_U is %s" %(np.shape(dots),np.shape(self.Ut),np.shape(C_U))

        basis=np.zeros((self.nicd,self.num_ics),dtype=float)
        for n,row in enumerate(C_U):
            basis[self.nicd-len(C_U)+n,:] =row 
        count=0
        for v in self.Ut:
            w = v - np.sum( np.dot(v,b)*b  for b in basis )
            tmp = w/np.linalg.norm(w)
            if (abs(w) > 1e-4).any():  
                basis[count,:] =tmp
                count +=1
        self.Ut = np.array(basis)
        if self.print_level>1:
            print "printing Ut"
            print self.Ut
        #print "Check if Ut is orthonormal"
        #print dots
        dots = np.matmul(self.Ut,np.transpose(self.Ut))
        assert (np.allclose(dots,np.eye(dots.shape[0],dtype=float))),"error in orthonormality"

    def orthogonalize(self,vecs):
        basis=np.zeros_like(vecs)
        basis[-1,:] = vecs[-1,:] # orthogonalizes with respect to the last
        for i,v in enumerate(vecs):
            w = v - np.sum( np.dot(v,b)*b  for b in basis)
            if (abs(w) > 1e-10).any():  
                tmp = w/np.linalg.norm(w)
                basis[i,:]=tmp
        return basis

    def form_CI_DLC(self,constraints=None):
        self.form_unconstrained_DLC()
        dvec = self.PES.get_coupling(self.geom)
        dgrad = self.PES.get_dgrad(self.geom)
        dvecq = self.grad_to_q(dvec)
        dgradq = self.grad_to_q(dgrad)
        dvecq_U = self.fromDLC_to_ICbasis(dvecq)
        dgradq_U = self.fromDLC_to_ICbasis(dgradq)
        constraints = np.zeros((len(dvecq_U),2),dtype=float)
        constraints[:,0] = dvecq_U[:,0]
        constraints[:,1] = dgradq_U[:,0]
        self.opt_constraint(constraints)
        self.bmat_create()
        #self.Hint = self.Hintp_to_Hint()

    def form_constrained_CI_DLC(self,constraints):
        self.form_unconstrained_DLC()
        dvec = self.PES.get_coupling(self.geom)
        dgrad = self.PES.get_dgrad(self.geom)
        dvecq = self.grad_to_q(dvec)
        dgradq = self.grad_to_q(dgrad)
        dvecq_U = self.fromDLC_to_ICbasis(dvecq)
        dgradq_U = self.fromDLC_to_ICbasis(dgradq)
        extra_constraints = np.shape(constraints)[1]
        new_constraints = np.zeros((len(dvecq_U),3),dtype=float) #extra constraints=1
        new_constraints[:,0] = dvecq_U[:,0]
        new_constraints[:,1] = dgradq_U[:,0]
        new_constraints[:,2] = constraints[:,0]
        self.opt_constraint(new_constraints)
        self.bmat_create()

    def form_constrained_DLC(self,constraints):
        self.form_unconstrained_DLC()
        self.opt_constraint(constraints)
        self.bmat_create()
        #self.Hint = self.Hintp_to_Hint()

    def form_unconstrained_DLC(self):
        self.bmatp = self.bmatp_create()
        self.bmatp_to_U()
        self.bmat_create()
        #self.Hint = self.Hintp_to_Hint()

    def get_nxyzics(self):
        pass
    def set_nicd(self):
        self.nicd=(self.natoms*3)-6


