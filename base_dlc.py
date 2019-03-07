import numpy as np
import openbabel as ob
import pybel as pb
import options
import os
from units import *
import itertools
import manage_xyz
from _obutils import Utils
from _icoord import *
from _bmat import Bmat
import elements 
import StringIO
from pes import PES
from penalty_pes import Penalty_PES
from avg_pes import Avg_PES

class Base_DLC(object,Bmat,Utils,ICoords):

    @staticmethod
    def default_options():
        """ Base_DLC default options. """

        if hasattr(Base_DLC, '_default_options'): return Base_DLC._default_options.copy()
        opt = options.Options() 

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
                doc='a list of bonds to form the DLC -- this is handled by the program'
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
                doc='Atoms to be left unoptimized/unmoved',
                )

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

        Base_DLC._default_options = opt
        return Base_DLC._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Base_DLC(Base_DLC.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes
        self.mol = self.options['mol']
        self.isOpt = 1
        self.MAX_FRAG_DIST = self.options['MAX_FRAG_DIST']
        self.PES = self.options['PES']
        self.print_level=self.options['print_level']
        self.FZN_ATOMS=self.options['FZN_ATOMS']
        self.EXTRA_BONDS=self.options['EXTRA_BONDS']
        self.IC_region=self.options['IC_region']
        self.madeBonds = False
        self.isTSnode = False
        self.buf = StringIO.StringIO() 
        self.natoms= len(self.mol.atoms)
        self.xyzatom_bool = np.zeros(self.natoms,dtype=bool)
        self.nxyzatoms=self.get_nxyzics()
        if self.options['bonds'] is not None:
            self.BObj = Bond_obj(self.options['bonds'],None,None)
            self.BObj.update(self.mol)
            self.madeBonds = True
            self.AObj = Ang_obj(self.options['angles'],None,None)
            self.AObj.update(self.mol)
            self.TObj = Tor_obj(self.options['torsions'],None,None)
            self.TObj.update(self.mol)
        self.setup()
        self.newHess = 5

        # tmp
        self.xs=[]
        self.g=[]
        self.geoms=[]
        self.fx=[]
        self.xnorm=[]
        self.gnorm=[]
        self.step=[]

    def setup(self):
        """ setup extra variables etc.,"""
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
        self.gradq = np.zeros((self.nicd,1),dtype=float)
        self.Hintp = self.make_Hint()
        self.Hintp_constraint = self.make_Hint()
        #self.Hint = None
        #self.Hinv = None
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

    @classmethod
    def union_ic(
            cls,
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
        # I don't like this ... Can I try the constructor again? mol=pb.Molecule(icoordA.mol.OBMol)
        mol1=pb.readfile('xyz','tmp1.xyz').next()

        lot1 = icoordA.PES.lot.copy(icoordA.PES.lot,icoordA.PES.lot.node_id)
        if icoordA.PES.__class__.__name__=="Avg_PES":
            PES1 = Avg_PES(icoordA.PES.PES1,icoordA.PES.PES2,lot1)
        else:
            PES1 = PES(icoordA.PES.options.copy().set_values({
                "lot": lot1,
                }))

        ICoordC = cls.create_DLC(icoordA,bondA,angleA,torsionA,mol1,PES1)
        ICoordC.Hintp = ICoordC.make_Hint()
        ICoordC.Hintp_constraint = ICoordC.make_Hint()

        return ICoordC

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
                        #Should add a try except here?
                        torsion= (c1,a1,a2,c2)
                        if not self.torsion_exists(torsion) and len(set(torsion))==4:
                            print(" adding torsion via linear ties %s" %(torsion,))
                            self.TObj.torsions.append(torsion)
                            self.TObj.ntor +=1
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

    def bmatp_to_U(self):
        self.set_nicd()
        G=np.matmul(self.bmatp,np.transpose(self.bmatp))
        # diagonalize G
        e,v = self.diagonalize_G(G)
        self.lowev=0

        #make function called set_nicd --> one for dlc and one for hdlc
        redset = self.num_ics - self.nicd

        for eig in e[:self.nicd]:
            if eig<0.001:
                self.lowev+=1

        if self.lowev>3:
            print(" Error: optimization space less than 3N DOF")
            print "lowev=",self.lowev
            print "shape G=",np.shape(G)
            print "numics=",self.num_ics
            print "numics_p=",self.num_ics_p
            print "3N=",self.natoms*3
            print "nicd=",self.nicd
            print "redset=",redset
            print e[:self.nicd]
            exit(-1)
        self.nicd -= self.lowev

        self.Ut =v[:self.nicd]
        self.torv0 = list(self.TObj.torv)
        
    def bmat_create(self):
        #print(" In bmat create")
        self.q = self.q_create()
        if self.print_level>1:
            print "printing q"
            print self.q.T
        bmat = np.matmul(self.Ut,self.bmatp)
        bbt = np.matmul(bmat,np.transpose(bmat))
        bbti = np.linalg.inv(bbt) #this is G in DLC
        self.bmatti= np.matmul(bbti,bmat)
        if self.print_level>2:
            print "bmatti"
            print self.bmatti

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

    @classmethod
    def add_node_SE(cls,ICoordA,driving_coordinate,dqmag_max=0.8,dqmag_min=0.2):

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

        ICoordC = cls.create_DLC(ICoordA,ICoordA.BObj.bonds,ICoordA.AObj.angles,ICoordA.TObj.torsions,mol1,PES1)

        ictan,bdist = Base_DLC.tangent_SE(ICoordA,driving_coordinate)

        ICoordC.opt_constraint(ictan)
        bdist = np.linalg.norm(ictan)
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
        ictan,bdist = Base_DLC.tangent_SE(ICoordC,driving_coordinate,quiet=True)
        ICoordC.bdist = bdist
        if np.all(ictan==0.0):
            raise RuntimeError
        #ICoordC.dqmag = dqmag
        ICoordC.Hintp = ICoordA.Hintp
        ICoordC.Hintp_constraint = ICoordA.Hintp_constraint

        return ICoordC

    @classmethod
    def add_node_SE_X(cls,ICoordA,driving_coordinate,dqmag_max=0.8,dqmag_min=0.2,BDISTMIN=0.05):

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

        ICoordC = cls.create_DLC(ICoordA,ICoordA.BObj.bonds,ICoordA.AObj.angles,ICoordA.TObj.torsions,mol1,PES1)

        ictan,bdist = Base_DLC.tangent_SE(ICoordA,driving_coordinate)
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
        ictan,bdist = Base_DLC.tangent_SE(ICoordC,driving_coordinate,quiet=True)
        ICoordC.bdist = bdist
        if np.all(ictan==0.0):
            raise RuntimeError
        
        ICoordC.Hintp = ICoordA.Hintp
        ICoordC.Hintp_constraint = ICoordA.Hintp_constraint

        return ICoordC

    @classmethod
    def add_node(cls,ICoordA,ICoordB,nmax,ncurr):
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
        ICoordC = cls.create_DLC(ICoordA,ICoordA.BObj.bonds,ICoordA.AObj.angles,ICoordA.TObj.torsions,mol1,PES1)
        ictan = Base_DLC.tangent_1(ICoordA,ICoordB)
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
        ICoordC.Hintp = ICoordA.Hintp
        ICoordC.Hintp_constraint = ICoordA.Hintp_constraint

        return ICoordC

    @classmethod
    def copy_node(cls,ICoordA,new_node_id,rtype=0):
        if isinstance(ICoordA.PES,Penalty_PES):
            ICoordC = Base_DLC.copy_node_X(ICoordA,new_node_id,rtype)
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

            ICoordC = cls.create_DLC(ICoordA,ICoordA.BObj.bonds,ICoordA.AObj.angles,ICoordA.TObj.torsions,mol1,PES1)
            ICoordC.Hintp = ICoordA.Hintp
            ICoordC.Hintp_constraint = ICoordA.Hintp_constraint

            return ICoordC

    # TODO can combine with copy_node
    @classmethod
    def copy_node_X(cls,ICoordA,new_node_id,rtype=0):
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
        
        ICoordC = cls.create_DLC(ICoordA,ICoordA.BObj.bonds,ICoordA.AObj.angles,ICoordA.TObj.torsions,mol1,PES1)
        ICoordC.Hintp = ICoordA.Hintp
        ICoordC.Hintp_constraint = ICoordA.Hintp_constraint
        return ICoordC

    def update_ics(self):
        self.update_xyz()
        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords)
        self.PES.lot.hasRanForCurrentCoords= False
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

    # can combine with ic_to_xyz
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
                for a in [(i-1) for i in self.FZN_ATOMS]:
                    xyzd[a,:]=0.

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

    def ic_to_xyz_opt(self,dq0,MAX_STEPS=8,quiet=True):
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
        
        if not quiet:
            print " dq is "
            print dq.T 

        #target IC values
        qn = self.q + dq 
        #print "printing new q"
        #print qn.T

        #primitive internal values
        qprim = self.primitive_internal_values()

        opt_molecules=[]
        xyzfile=os.getcwd()+"/ic_to_xyz.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

        # => Calc Change in Coords <= #
        for n in range(MAX_STEPS):
            if self.print_level>1:
                print "ic iteration %i" % n
            btit = np.transpose(self.bmatti)
            xyzd=np.dot(btit,dq)
            if self.print_level>2:
                print "xyzd"
                print xyzd.T
            assert len(xyzd)==3*self.natoms,"xyzd is not N3 dimensional"
            xyzd = np.reshape(xyzd,(self.natoms,3))

            # => Frozen <= #
            if self.FZN_ATOMS is not None:
                for a in [(i-1) for i in self.FZN_ATOMS]:
                    xyzd[a,:]=0.

            # => Add Change in Coords <= #
            xyz1 = self.coords + xyzd/SCALEBT 

            # => Calc Mag <= #
            mag=np.dot(np.ndarray.flatten(xyzd),np.ndarray.flatten(xyzd))
            magall.append(mag)
            xyzall.append(xyz1)

            # update coords
            xyzp = np.copy(self.coords) 
            self.coords = xyz1

            self.update_ics()
            self.bmatp=self.bmatp_create()
            self.bmat_create()
            if self.print_level==3:
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

            qprim_current = self.primitive_internal_values()
            #self.dqprim = self.primitive_internal_difference(qprim_current,qprim)

        #write convergence geoms to file 
        largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        for mol in opt_molecules:
            largeXyzFile.write(pb.readstring("xyz",mol))
        if self.print_level>0:
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
        return np.diag(Hdiagp)

        #self.Hintp=np.diag(Hdiagp)
        #Hdiagp=np.asarray(Hdiagp)
        #Hdiagp=np.reshape(Hdiagp,(self.num_ics,1))
        #tmp = np.zeros((self.nicd,self.num_ics),dtype=float)
        #for i in range(self.nicd): 
        #    for k in range(self.num_ics):
        #        tmp[i,k] = self.Ut[i,k]*Hdiagp[k]
        ##self.Hint = np.matmul(tmp,np.transpose(self.Ut))


    # TODO make  opt_type the name variable in params
    def update_DLC(self,opt_type,ictan):
        if opt_type=='UCONSTRAINED':
            self.form_unconstrained_DLC()
        elif opt_type in ["ICTAN", "CLIMB"]:
            self.form_constrained_DLC(ictan)
        elif opt_type in ['MECI']:
            self.form_CI_DLC()
        elif opt_type in ['SEAM','TS-SEAM']:
            self.form_constrained_CI_DLC(constraints=ictan)

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

    def form_unconstrained_DLC(self):
        self.bmatp = self.bmatp_create()
        self.bmatp_to_U()
        self.bmat_create()

    def proc_evaluate(self,q,n):
        ''' 
        Evaluates the energy and gradient at a point q.
        But resets the coords back to q after updating.
        returns only the gradqprim and qprim in the 
        non-constraint region (n).
        '''

        if (self.q!=q).any():
            # stash q
            coordp = self.coords.copy()
            dq = q-self.q
            # this updates the geom 
            self.ic_to_xyz_opt(dq)
        fx =self.PES.get_energy(self.geom)
        grad = self.PES.get_gradient(self.geom)
        gradq = self.grad_to_q(grad)

        # primitive values --  why doesn't this work?
        #qprim1 = np.dot(np.transpose(self.Ut[:n]),self.q[:n])
        #qprim1[self.BObj.nbonds:] *= 180./np.pi
        #print "qprim1"
        #print qprim1[:self.BObj.nbonds].T
        #print qprim1[self.BObj.nbonds:self.AObj.nangles].T
        #print qprim1[self.BObj.nbonds+self.AObj.nangles:].T

        gradqprim = np.dot(np.transpose(self.Ut[:n]),gradq[:n])
        qprim = self.primitive_internal_values()
        qprim = np.reshape(qprim,(self.num_ics,1))
        #print "qprim"
        ##print qprim.T
        #print qprim[:self.BObj.nbonds].T
        #print qprim[self.BObj.nbonds:self.AObj.nangles].T
        #print qprim[self.BObj.nbonds+self.AObj.nangles:].T

        result = { 'fx': fx, 'g':gradq, 'qprim':qprim,'gradqprim':gradqprim}

        # reset q 
        if (self.q!=q).any():
            self.coords = coordp
            self.update_ics()
            self.bmatp=self.bmatp_create()
            self.bmat_create()
            self.PES.lot.hasRanForCurrentCoords=True

        return result

    def convert_primitive_to_DLC(self,prim):
        return np.dot(self.Ut,prim) # (nicd,numic)(num_ic,1)

    def append_data(self,x,proc_results,xnorm,gnorm,step):
        self.xs.append(x)
        dq = x - self.q

        # => update geometry <= #
        self.ic_to_xyz_opt(dq,quiet=True)
        self.PES.lot.hasRanForCurrentCoords=True

        self.geoms.append(self.geom)
        self.fx.append(proc_results['fx'])
        self.xnorm.append(xnorm)
        self.gnorm.append(gnorm)
        self.step.append(step)
        self.gradrms = gnorm
        return

    def grad_to_q(self,grad):
        if self.FZN_ATOMS is not None:
            for a in [3*(i-1) for i in self.FZN_ATOMS]:
                grad[a:a+3]=0.
        gradq = np.dot(self.bmatti,grad)
        return gradq

    # move to base
    def Hintp_to_Hint(self):
        tmp = np.dot(self.Ut,self.Hintp) #(nicd,numic)(num_ic,num_ic)
        return np.matmul(tmp,np.transpose(self.Ut)) #(nicd,numic)(numic,numic)

    # base
    def diagonalize_G(self,G):
        SVD=False
        if SVD:
            v_temp,e,vh  = np.linalg.svd(G)
        else:
            e,v_temp = np.linalg.eigh(G)
            idx = e.argsort()[::-1]
            e = e[idx]
            v_temp = v_temp[:,idx]
        v = np.transpose(v_temp)
        return e,v

    def q_create(self):  
        """Determines the scalars in delocalized internal coordinates"""

        #print(" Determining q in ICs")
        N3=3*self.natoms
        q = np.zeros((self.nicd,1),dtype=float)

        dists=[self.distance(bond[0],bond[1]) for bond in self.BObj.bonds ]
        angles=[self.get_angle(angle[0],angle[1],angle[2])*np.pi/180. for angle in self.AObj.angles ]
        tmp =[self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]) for torsion in self.TObj.torsions]
        torsions=[]
        for i,j in zip(self.torv0,tmp):
            tordiff = i-j
            if tordiff>180.:
                torfix=360.
            elif tordiff<-180.:
                torfix=-360.
            else:
                torfix=0.
            torsions.append((j+torfix)*np.pi/180.)

        for i in range(self.nicd):
            q[i] = np.dot(self.Ut[i,0:self.BObj.nbonds],dists) + \
                    np.dot(self.Ut[i,self.BObj.nbonds:self.AObj.nangles+self.BObj.nbonds],angles) \
                    + np.dot(self.Ut[i,self.BObj.nbonds+self.AObj.nangles:self.num_ics_p],torsions)

        #print("Printing q")
        #print np.transpose(q)
        return q

    def fromDLC_to_ICbasis(self,vecq):
        """
        This function takes a matrix of vectors wrtiten in the basis of U.
        The components in this basis are called q.
        """
        vec_U = np.zeros((self.num_ics,1),dtype=float)
        assert np.shape(vecq) == (self.nicd,1), "vecq is not nicd long"
        vec_U = np.dot(self.Ut.T,vecq)
        return vec_U/np.linalg.norm(vec_U)


