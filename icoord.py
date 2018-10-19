import numpy as np
import openbabel as ob
import pybel as pb
import options
import elements 
import os
from units import *
import itertools

from _icoord import Mixin

class ICoord(Mixin):

    @staticmethod
    def default_options():
        """ ICoord default options. """

        if hasattr(ICoord, '_default_options'): return ICoord._default_options.copy()
        opt = options.Options() 
        opt.add_option(
            key='isOpt',
            value=1,
            required=False,
            allowed_types=[int],
            doc='Something to do with how coordinates are setup? Ask Paul')

        opt.add_option(
            key='MAX_FRAG_DIST',
            value=12.0,
            required=False,
            allowed_types=[float],
            doc='Maximum fragment distance considered for making fragments')

        opt.add_option(
            key='OPTTHRESH',
            value=0.0005,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
                key="mol",
                required=False,
                allowed_types=[pb.Molecule],
                doc='Pybel molecule object (not OB.Mol)')

        opt.add_option(
                key="lot",
                required=True,
                doc='level of theory object')

        opt.add_option(
                key="bonds"
                )

        opt.add_option(
                key="angles"
                )

        opt.add_option(
                key="torsions"
                )

        ICoord._default_options = opt
        return ICoord._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return ICoord(ICoord.default_options().set_values(kwargs))

    @staticmethod
    def union_ic(
            icoordA,
            icoordB,
            ):
        """ return union ICoord of two ICoord Objects """
        unionBonds    = list(set(icoordA.bonds) | set(icoordB.bonds))
        #b = []
        #seen=set()
        #for bond in unionBonds:
        #    s=frozenset(bond)
        #    if s not in seen:
        #        seen.add(s)
        #        b.append(bond)
        #print b

        unionAngles   = list(set(icoordA.angles) | set(icoordB.angles))
        unionTorsions = list(set(icoordA.torsions) | set(icoordB.torsions))

        print icoordA.bonds
        print icoordB.bonds
        print unionBonds
        #print unionAngles
        #print unionTorsions
        #print "Saving bond union"
        bonds = []
        angles = []
        torsions = []
        #for bond in b:
        for bond in unionBonds:
            if bond not in icoordA.bonds:
                print "need to add bond to %s" % (bond,)
                isOkay = icoordA.mol.OBMol.AddBond(bond[0]+1,bond[1]+1,1)
                print "Bond: %s added okay? %r" % (bond,isOkay)
                bonds.append(bond)
        #for angle in unionAngles:
        #    #print angle
        #    if bond not in icoordA.angles or bond not in icoordB.angles:
        #        pass
        #        #print "need to add angle to %i" % angle[0]
        #    angles.append(angle)
        #for torsion in unionTorsions:
        #    torsions.append(torsion)

        ic3 = ICoord(icoordA.options.copy().set_values({
            'bonds' : bonds,
             'angles': angles,
             'torsions': torsions,
            }))

        return ic3

    
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
        self.lot = self.options['lot']
        self.OPTTHRESH = self.options['OPTTHRESH']


        #self.print_xyz()
        self.Elements = elements.ElementData()
        self.ic_create()
        self.bmatp_create()
        self.bmatp_to_U()
        self.bmat_create()

        self.make_Hint()  
        self.pgradqprim = np.zeros((self.num_ics,1),dtype=float)
        self.gradqprim = np.zeros((self.num_ics,1),dtype=float)
        self.gradq = np.zeros((self.nicd,1),dtype=float)
        self.pgradq = np.zeros((self.nicd,1),dtype=float)
        self.gradrms = 1000.
        self.SCALEQN = 1.0
        self.MAXAD = 0.075
        self.DMAX = 0.1
        self.ixflag = 0
        self.energy = 0.
        self.DMIN0 =self.DMAX/50.
        self.lot.coords = np.zeros((len(self.mol.atoms),3))
        for i,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
            self.lot.coords[i,0] = a.GetX()
            self.lot.coords[i,1] = a.GetY()
            self.lot.coords[i,2] = a.GetZ()


        self.nretry = 0 
        
    def ic_create(self):
        self.natoms= len(self.mol.atoms)
        #print self.natoms
        self.make_bonds()
        if self.isOpt>0:
            print(" isOpt: %i" %self.isOpt)
            self.make_frags()
            self.bond_frags()
        self.coord_num()
        self.make_angles()
        self.make_torsions()
        self.make_imptor()
        if self.isOpt==1:
            self.linear_ties()
        self.make_nonbond() 

    def make_bonds(self):
        self.nbonds=0
        self.bonds=[]
        self.bondd=[]
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            self.nbonds+=1
            a=bond.GetBeginAtomIdx()-1
            b=bond.GetEndAtomIdx()-1
            if a>b:
                self.bonds.append((a,b))
            else:
                self.bonds.append((b,a))
            #self.bonds.append((bond.GetBeginAtomIdx()-1,bond.GetEndAtomIdx()-1))
            self.bondd.append(bond.GetLength())
        print "number of bonds is %i" %self.nbonds
        print "printing bonds"
        for n,bond in enumerate(self.bonds):
            print "%s: %1.2f" %(bond, self.bondd[n])

    def coord_num(self):
        self.coordn=[]
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            count=0
            for nbr in ob.OBAtomAtomIter(a):
                count+=1
            self.coordn.append(count)
        #print self.coordn

    def make_angles(self):
        self.nangles=0
        self.angles=[]
        self.anglev=[]
        for angle in ob.OBMolAngleIter(self.mol.OBMol):
            self.nangles+=1
            self.angles.append(angle)
            self.anglev.append(self.get_angle(angle[0],angle[1],angle[2]))
        print "number of angles is %i" %self.nangles
        print "printing angles"
        for n,angle in enumerate(self.angles):
            print "%s: %1.2f" %(angle, self.anglev[n])


    def make_torsions(self):
        self.ntor=0
        self.torsions=[]
        self.torv=[]
        for torsion in ob.OBMolTorsionIter(self.mol.OBMol):
            self.ntor+=1
            self.torsions.append(torsion)
            self.torv.append(self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]))
        print "number of torsions is %i" %self.ntor
        print "printing torsions"
        for n,torsion in enumerate(self.torsions):
            print "%s: %1.2f" %(torsion, self.torv[n])


    def make_nonbond(self):
        """ anything not connected by bond or angle """
        self.nonbond=[]
        for i in range(self.natoms):
            for j in range(i):
                found=False
                for k in range(self.nbonds):
                    if found==True:
                        break
                    if (self.bonds[k][0]==i and self.bonds[k][1]==j) or (self.bonds[k][0]==j and self.bonds[k][1]==i):
                        found=True
                for k in range(self.nangles):
                    if found==True:
                        break
                    if self.angles[k][0]==i:
                        if self.angles[k][1]==j:
                            found=True
                        elif self.angles[k][2]==j:
                            found=True
                    elif self.angles[k][1]==i:
                        if self.angles[k][0]==j:
                            found=True
                        elif self.angles[k][2]==j:
                            found=True
                    elif self.angles[k][2]==i:
                        if self.angles[k][0]==j:
                            found=True
                        elif self.angles[k][1]==j:
                            found=True
                if found==False:
                   self.nonbond.append(self.distance(i,j))
        #print self.nonbond

    """ Is this function even used? """
    def make_imptor(self):
        self.imptor=[]
        self.nimptor=0
        self.imptorv=[]
        count=0
        for i in self.angles:
            #print i
            try:
                for j in self.angles[0:count]:
                    found=False
                    a1=i[0]
                    m1=i[1]
                    c1=i[2]
                    a2=j[0]
                    m2=j[1]
                    c2=j[2]
                    #print(" angle: %i %i %i angle2: %i %i %i" % (a1,m1,c1,a2,m2,c2))
                    if m1==m2:
                        if a1==a2:
                            found=True
                            d=self.mol.OBMol.GetAtom(c2+1)
                        elif a1==c2:
                            found=True
                            d=self.mol.OBMol.GetAtom(a2+1)
                        elif c1==c2:
                            found=True
                            d=self.mol.OBMol.GetAtom(a2+1)
                        elif c1==a2:
                            found=True
                            d=self.mol.OBMol.GetAtom(c2+1)
                    if found==True:
                        a=self.mol.OBMol.GetAtom(c1+1)
                        b=self.mol.OBMol.GetAtom(a1+1)
                        c=self.mol.OBMol.GetAtom(m1+1)
                        imptorvt=self.mol.OBMol.GetTorsion(a,b,c,d)
                        #print imptorvt
                        if abs(imptorvt)>12.0 and abs(imptorvt-180.)>12.0:
                            found=False
                        else:
                            self.imptorv.append(imptorvt)
                            self.imptor.append((a.GetIndex(),b.GetIndex(),c.GetIndex(),d.GetIndex()))
                            self.nimptor+=1
            except Exception as e: print(e)
            count+=1
            return

    def update_ics(self):
        self.update_xyz()
        self.update_bonds()
        self.update_angles()
        self.update_torsions()

    def update_xyz(self):
        """ Updates the mol.OBMol object coords: Important for ICs"""
        for i,xyz in enumerate(self.lot.coords):
            self.mol.OBMol.GetAtom(i+1).SetVector(xyz[0],xyz[1],xyz[2])

    def update_bonds(self):
        self.bondd=[]
        for bond in self.bonds:
            self.bondd.append(self.distance(bond[0],bond[1]))

    def update_angles(self):
        self.anglev=[]
        for angle in self.angles:
            self.anglev.append(self.get_angle(angle[0],angle[1],angle[2]))

    def update_torsions(self):
        self.torv=[]
        for torsion in self.torsions:
            self.torv.append(self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]))

    def linear_ties(self):
        maxsize=0
        for anglev in self.anglev:
            if anglev>160.:
                maxsize+=1
        blist=[]
        n=0
        for anglev,angle in zip(self.anglev,self.angles):
            if anglev>160.:
                blist.append(angle)
                print(" linear angle %i of %i: %s (%4.2f)",n+1,maxsize,angle,anglev)
                n+=1

        # atoms attached to linear atoms
        clist=[[]]
        m=np.zeros((n)) #number of nbr atoms?
        for i in range(n):
            # a is the vertex and not included
            b=self.mol.OBMol.GetAtom(blist[i][1])
            c=self.mol.OBMol.GetAtom(blist[i][2])
            for nbr in ob.OBAtomAtomIter(b):
                if nbr.GetIndex() != c.GetIndex():
                    clist[i].append(nbr.GetIndex())
                    m[i]+=1
                    
            for nbr in ob.OBAtomAtomIter(c):
                if nbr.GetIndex() != b.GetIndex():
                    clist[i].append(nbr.GetIndex())
                    m[i]+=1

        # cross linking 
        for i in range(n):
            a1=blist[i][1]
            a2=blist[i][2] # not vertices
            bond=(a1,a2)
            if bond_exists(bond) == False:
                print(" adding bond via linear ties %s" % bond)
                self.bonds.append(bond)
            for j in range(m[i]):
                for k in range(j):
                    b1=clist[i][j]
                    b2=clist[i][k]
                    found=False
                    for angle in self.angles:
                        if b1==angle[1] and b2==angle[2]: #0 is the vertex and don't want?
                            found=True
                        elif b2==angle[1] and b1==angle[2]:
                            found=True
                    if found==False:
                        if bond_exists((b1,a1))==True:
                            c1=b1
                        if bond_exists(b2,a1)==True:
                            c1=b2
                        if bond_exists(b1,a2)==True:
                            c2=b1
                        if bond_exists(b2,a2)==True:
                            c2=b2
                        torsion= (c1,a1,a2,c2)
                        print(" adding torsion via linear ties %s" %torsion)
                        self.torsions.append(torsion)

    def make_frags(self):
        """ Currently only works for two fragments """

        print("making frags")
        nfrags=0
        merged=0
        self.frags=[]
        frag1=[]
        frag2=[]
        for n,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
            found=False
            if n==0:
                frag1.append((0,n))
            else:
                found=False
                for nbr in ob.OBAtomAtomIter(a):
                    if (0,nbr.GetIndex()) in frag1:
                        found=True
                if found==True:
                    frag1.append((0,a.GetIndex()))
                if found==False:
                    frag2.append((1,a.GetIndex()))

        if not frag2:
            self.nfrags=1
        else:
            self.nfrags=2
        self.frags=frag1+frag2
        for i in self.frags:
            print(" atom[%i]: %i " % (i[1],i[0]))

        print(" nfrags: %i" % (self.nfrags))



    def bond_frags(self):
        if self.nfrags<2:
            return
        found=0
        found2=0
        found3=0
        found4=0

        frags= [i[0] for i in self.frags]
        for n1 in range(self.nfrags):
            for n2 in range(n1):
                print(" Connecting frag %i to %i" %(n1,n2))
                found=0
                found2=0
                found3=0
                found4=0
                close=0.
                mclose=1000.
                a1=-1
                a2=-1
                b1=-1
                b2=-1
                mclose2=1000.
                c1=-1
                c2=-1
                mclose3=1000.
                d1 = -1;
                d2 = -1
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
                    if (self.getIndex(comb[0][1]) > 1 or self.getIndex(comb[1][1])>1) and dist21 > 4. and dist22 >4. and close<mclose2 and close < self.MAX_FRAG_DIST: 
                        mclose2 = close
                        b1=i
                        b2=j
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
                if found>0 and self.bond_exists(bond1)==False:
                    print("bond pair1 added : %s" % (bond1,))
                    self.bonds.append(bond1)
                    self.nbonds+=1
                    self.bondd.append(mclose)
                    print "bond dist: %1.4f" % mclose
                bond2=(b1,b2)
                if found2>0 and self.bond_exists(bond2)==False:
                    self.bonds.append(bond2)
                    print("bond pair2 added : %s" % (bond2,))
                bond3=(c1,c2)
                if found3>0 and self.bond_exists(bond3)==False:
                    self.bonds.append(bond3)
                    print("bond pair2 added : %s" % (bond3,))
                bond4=(d1,d2)
                if found4>0 and self.bond_exists(bond4)==False:
                    self.bonds.append(bond4)
                    print("bond pair2 added : %s" % (bond24,))

                isOkay = self.mol.OBMol.AddBond(bond1[0]+1,bond1[1]+1,1)
                print "Bond added okay? %r" % isOkay

                if self.isOpt==2:
                    print("Checking for linear angles in newly added bond")
                    #TODO

    def bmatp_create(self):
        self.num_ics = self.nbonds + self.nangles + self.ntor
        N3 = 3*self.natoms
        #print "Number of internal coordinates is %i " % self.num_ics
        self.bmatp=np.zeros((self.num_ics,N3),dtype=float)
        i=0
        for bond in self.bonds:
            a1=bond[0]
            a2=bond[1]
            dqbdx = self.bmatp_dqbdx(a1,a2)
            self.bmatp[i,3*a1+0] = dqbdx[0]
            self.bmatp[i,3*a1+1] = dqbdx[1]
            self.bmatp[i,3*a1+2] = dqbdx[2]
            self.bmatp[i,3*a2+0] = dqbdx[3]
            self.bmatp[i,3*a2+1] = dqbdx[4]
            self.bmatp[i,3*a2+2] = dqbdx[5]
            i+=1
            #print "%s" % ((a1,a2),)

        for angle in self.angles:
            a1=angle[1]
            a2=angle[0] #vertex
            a3=angle[2]
            dqadx = self.bmatp_dqadx(a1,a2,a3)
            self.bmatp[i,3*a1+0] = dqadx[0]
            self.bmatp[i,3*a1+1] = dqadx[1]
            self.bmatp[i,3*a1+2] = dqadx[2]
            self.bmatp[i,3*a2+0] = dqadx[3]
            self.bmatp[i,3*a2+1] = dqadx[4]
            self.bmatp[i,3*a2+2] = dqadx[5]
            self.bmatp[i,3*a3+0] = dqadx[6]
            self.bmatp[i,3*a3+1] = dqadx[7]
            self.bmatp[i,3*a3+2] = dqadx[8]
            i+=1
            #print i
            #print "%s" % ((a1,a2,a3),)

        for torsion in self.torsions:
            a1=torsion[0]
            a2=torsion[1]
            a3=torsion[2]
            a4=torsion[3]
            #print "%s" % ((a1,a2,a3,a4),)
            dqtdx = self.bmatp_dqtdx(a1,a2,a3,a4)
            self.bmatp[i,3*a1+0] = dqtdx[0]
            self.bmatp[i,3*a1+1] = dqtdx[1]
            self.bmatp[i,3*a1+2] = dqtdx[2]
            self.bmatp[i,3*a2+0] = dqtdx[3]
            self.bmatp[i,3*a2+1] = dqtdx[4]
            self.bmatp[i,3*a2+2] = dqtdx[5]
            self.bmatp[i,3*a3+0] = dqtdx[6]
            self.bmatp[i,3*a3+1] = dqtdx[7]
            self.bmatp[i,3*a3+2] = dqtdx[8]
            self.bmatp[i,3*a4+0] = dqtdx[9]
            self.bmatp[i,3*a4+1] = dqtdx[10]
            self.bmatp[i,3*a4+2] = dqtdx[11]
            i+=1

        #print "printing bmatp"
        #print self.bmatp
        #print "\n"
        #print "shape of bmatp is %s" %(np.shape(self.bmatp),)

        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)

        #print self.bmatp

    def bmatp_to_U(self):
        N3=3*self.natoms
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        G=np.matmul(self.bmatp,np.transpose(self.bmatp))

        # Singular value decomposition
        v_temp,e,vh  = np.linalg.svd(G)
        v = np.transpose(v_temp)
        #print e
        #print v
        
        lowev=0
        self.nicd=N3-6
        for eig in e[self.nicd:0:-1]:
            if eig<0.001:
                lowev+=1
        if lowev>0:
            print("!!!!! lowev: %i" % lowev)

        self.nicd -= lowev
        if lowev>3:
            print(" Error: optimization space less than 3N-6 DOF")
            exit(-1)

        #print(" Number of internal coordinate dimensions %i" %self.nicd)
        redset = self.num_ics - self.nicd
        #print "\nU matrix  i.e. diag(BB^T)"
        self.Ut=v[0:self.nicd,:]
        #print("U eigenvalues")
        #print e
        #print("printing non-redundant vectors of U")
        #print self.Ut
        #print "Shape of U is %s\n" % (np.shape(self.Ut),)

        self.torv0 = list(self.torv)
        

    def q_create(self):  
        """Determines the scalars in delocalized internal coordinates"""

        #print(" Determining q in ICs")
        N3=3*self.natoms
        self.q = np.zeros(self.nicd)
        #print "Number of ICs %i" % self.num_ics
        #print "Number of IC dimensions %i" %self.nicd
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)

        dists=[self.distance(bond[0],bond[1]) for bond in self.bonds ]
        angles=[self.get_angle(angle[0],angle[1],angle[2])*np.pi/180. for angle in self.angles ]
        tmp =[self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]) for torsion in self.torsions]
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
            n=+1


        for i,row in enumerate(self.Ut):
            Ubond = row[0:self.nbonds]
            Uangle =row[self.nbonds:self.nangles+self.nbonds]
            Utorsion = row[self.nbonds+self.nangles:self.nbonds+self.nangles+self.ntor]
            self.q[i] = np.dot(Ubond,dists) + np.dot(Uangle,angles) + np.dot(Utorsion,torsions)

        #print("Printing q")
        #print self.q


    def bmat_create(self):

        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)

        #print(" In bmat create")
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        self.q_create()

        bmat = np.matmul(self.Ut,self.bmatp)
        """
        print("printing bmat")
        print bmat
        print(" Shape of bmat %s" %(np.shape(bmat),))
        """

        bbt = np.matmul(bmat,np.transpose(bmat))
        #print(" Shape of bbt %s" %(np.shape(bbt),))
        bbti = np.linalg.inv(bbt)
        #print("bmatti formation")
        self.bmatti = np.matmul(bbti,bmat)
        #print self.bmatti
        #print(" Shape of bmatti %s" %(np.shape(self.bmatti),))

    def ic_to_xyz(self,dq):
        """ Transforms ic to xyz, used by addNode"""

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        self.update_ics()
        self.bmatp_create()
        self.bmat_create()

        SCALEBT = 1.5
        #print("q at beginning is")
        #print self.q
        N3=self.natoms*3
        qn = self.q + dq  #target IC values
        #print(" qn is ")
        #print qn
        #print("dq at start is")
        #print dq
        #print("\n")
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
            xyzd=np.matmul(btit,dq)
            xyzd = np.reshape(xyzd,(self.natoms,3))

            #TODO Frozen

            # => Calc Mag <= #
            mag=np.dot(np.ndarray.flatten(xyzd),np.ndarray.flatten(xyzd))
            magall.append(mag)

            if mag>magp:
                SCALEBT *=1.5
            magp=mag

            # update coords
            xyz1 = self.lot.coords + xyzd/SCALEBT 
            xyzall.append(xyz1)
            self.lot.coords = xyz1
            self.update_ics()
            self.bmatp_create()
            self.bmat_create()

            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            dq = qn - self.q
            #print dq
            #dqmag = np.linalg.norm(dq)
            #print dqmag

            if mag<0.00005: break
        print("\n magall ")
        print magall
        #print("\n xyzall")
        #print xyzall

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
        for i in range(self.natoms):
            tmpvec = self.getCoords(i)
            self.lot.coords[i,0] = tmpvec[0]
            self.lot.coords[i,1] = tmpvec[1]
            self.lot.coords[i,2] = tmpvec[2]
        xyzall.append(self.lot.coords)

        magp=100
        dqmagp=100.

        dq = dq0
        #target IC values
        qn = self.q + dq 

        #primitive internal values
        qprim = np.concatenate((self.bondd,self.anglev,self.torv))

        opt_molecules=[]
        xyzfile=os.getcwd()+"/ic_to_xyz.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

        # => Calc Change in Coords <= #
        for n in range(MAX_STEPS):
            #print("iteration step %i" %n)
            btit = np.transpose(self.bmatti)
            xyzd=np.matmul(btit,dq)
            xyzd = np.reshape(xyzd,(self.natoms,3))

            #TODO frozen

            # => Add Change in Coords <= #
            xyz1 = self.lot.coords + xyzd/SCALEBT 

            # => Calc Mag <= #
            mag=np.dot(np.ndarray.flatten(xyzd),np.ndarray.flatten(xyzd))
            magall.append(mag)
            xyzall.append(xyz1)

            # update coords
            xyzp = np.copy(self.lot.coords) # note that when we modify coords, xyzp will not change
            self.lot.coords = xyz1

            self.update_ics()
            self.bmatp_create()
            self.bmat_create()

            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            #calc new dq
            dq = qn - self.q

            dqmag = np.linalg.norm(dq)
            dqmagall.append(dqmag)
            if dqmag<0.0001: break

            if dqmag>dqmagp*10.:
                print(" Q%i" % n)
                SCALEBT *= 2.0
                self.lot.coords = np.copy(xyzp)
                self.update_ics()
                self.bmatp_create()
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
            for mag in magall:
                if mag<maglow:
                    maglow=mag
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
        elif self.ixflag>0:
            self.ixflag = 0

        if retry==False:
            self.update_ics()
            torsions=[]
            for i,j in zip(self.torv,qprim[self.nbonds+self.nangles:]):
                tordiff = i-j
                if tordiff>180.:
                    torfix=-360.
                elif tordiff<-180.:
                    torfix=360.
                else:
                    torfix=0.
                torsions.append((i+torfix))

            qprim_cur = np.concatenate((self.bondd,self.anglev,torsions))
            self.dqprim = qprim_cur - qprim
            self.dqprim[self.nbonds:] *= np.pi/180.
            self.dqprim = np.reshape(self.dqprim,(self.num_ics,1))

        #write convergence geoms to file 
        #largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        #for mol in opt_molecules:
        #    largeXyzFile.write(pb.readstring("xyz",mol))
       
        #print(" \n magall")
        #print magall
        #print(" \n dmagall")
        #print dqmagall
        #print "\n"
        if retry==True:
            self.ic_to_xyz_opt(dq0)
        else:
            return rflag

    def make_Hint(self):
        self.newHess = 5
        Hdiagp = []
        for bond in self.bonds:
            Hdiagp.append(0.35*self.close_bond(bond))
        for angle in self.angles:
            Hdiagp.append(0.2)
        for tor in self.torsions:
            Hdiagp.append(0.035)

        self.Hintp=np.diag(Hdiagp)
        #print(" Hdiagp elements")
        #for i in Hdiagp:
        #    print i
        
        tmp = np.matmul(self.Ut,self.Hintp)
        self.Hint = np.matmul(self.Ut,np.transpose(tmp))
        self.Hinv = np.linalg.inv(self.Hint)

        #print("Hint elements")
        #print Hint
        print("Shape of Hint is %s" % (np.shape(self.Hint),))

        #if self.optCG==False or self.isTSNode==False:
        #    print "Not implemented"

    def Hintp_to_Hint(self):
        tmp = np.matmul(self.Ut,self.Hintp)
        Hint = np.matmul(self.Ut,np.transpose(tmp))

    def optimize(self,nsteps):
        xyzfile=os.getcwd()+"/xyzfile.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules=[]
        opt_molecules.append(obconversion.WriteString(self.mol.OBMol))
        self.energy = self.lot.getEnergy()

        print "Initial energy is %1.4f\n" % self.energy

        for step in range(nsteps):
            print("iteration step %i" %step)

            # => Opt step <= #
            self.opt_step()
            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            #write convergence
            largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
            for mol in opt_molecules:
                largeXyzFile.write(pb.readstring("xyz",mol))

            if self.gradrms<self.OPTTHRESH:
                break


    def opt_step(self):
        energy=0.

        energyp = self.energy
        grad = self.lot.getGrad()
        self.bmatp_create()
        self.bmat_create()

        # grad in ics
        self.pgradq = self.gradq
        self.gradq = self.grad_to_q(grad)
        pgradrms = self.gradrms
        self.gradrms = np.linalg.norm(self.gradq)
        print("gradrms = %1.4f" % self.gradrms)

        # For Hessian update
        self.pgradqprim=self.gradqprim
        self.gradqprim = np.matmul(np.transpose(self.Ut),self.gradq)

        # => Take Eigenvector Step <=#
        dq = self.update_ic_eigen(self.gradq)

        # regulate max overall step
        smag = np.linalg.norm(dq)
        print(" ss: %1.3f (DMAX: %1.3f)" %(smag,self.DMAX))
        if smag>self.DMAX:
            dq = np.fromiter(( xi*self.DMAX/smag for xi in dq), dq.dtype)

        print dq

        # => update geometry <=#
        rflag = self.ic_to_xyz_opt(dq)
        #TODO if rflag and ixflag
      
        # => Update Hessian <= #
        self.update_bfgsp(dq)

        # => calc energyat new position <= #
        self.energy = self.lot.getEnergy()
        print self.energy

        # check goodness of step
        dEstep = self.energy - energyp
        dEpre = self.compute_predE(dq)
        ratio = dEstep/dEpre
        print "ratio is %1.4f" % ratio

        # => step controller  <= #
        if dEstep>0.001:
            print("decreasing DMAX")
            if smag <self.DMAX:
                self.DMAX = smag/1.5
            else: 
                self.DMAX = self.DMAX/1.5
        elif (ratio<0.2 or ratio>1.5) and abs(dEpre)>0.05:
            print("decreasing DMAX")
            if smag<self.DMAX:
                self.DMAX = smag/1.1
            else:
                self.DMAX = smag/1.2
        elif (ratio>0.75 and ratio<1.25) and self.gradrms<pgradrms*1.35:
            print("increasing DMAX")
            self.DMAX=self.DMAX*1.1
            if self.DMAX>0.25:
                self.DMAX=0.25
        if self.DMAX<self.DMIN0:
            self.DMAX=self.DMIN0

        return 


    def update_bfgsp(self,dq):
        print("In update bfgsp")
        self.newHess-=1

        dx = self.dqprim
        #print "shape of dx is %s" %(np.shape(dx),)
        dg = self.gradqprim - self.pgradqprim
        #print "shape of dg is %s" %(np.shape(dg),)
        Hdx = np.matmul(self.Hintp,dx)
        #print "shape of Hdx is %s" %(np.shape(Hdx),)
        dxHdx = np.dot(np.transpose(dx),Hdx)
        #print "shape of dxHdx is %s" %(np.shape(dxHdx),)
        dgdg = np.outer(dg,dg)
        #print "shape of dgdg is %s" %(np.shape(dgdg),)
        dgtdx = np.dot(np.transpose(dg),dx)

        if dgtdx>0.:
            if dgtdx<0.001: dgtdx=0.001
            self.Hintp += dgdg/dgtdx
        if dxHdx>0.:
            if dxHdx<0.001: dxHdx=0.001
            self.Hintp -= np.outer(Hdx,Hdx)/dxHdx

        self.Hintp_to_Hint()


if __name__ == '__main__':
    from pytc import *
    

    if 0:
        # fragment example -- not working properly -- 
        filepath1="tests/SiH4.xyz"
        filepath2="tests/SiH2H2.xyz"

        # LOT object
        nocc=11
        nactive=2
        lot1=PyTC.from_options(calc_states=[(0,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot2 = PyTC(lot1.options.copy())

        # ICoord object
        mol1=pb.readfile("xyz",filepath1).next()
        mol2=pb.readfile("xyz",filepath2).next()
        #print "ic1"
        #ic1=ICoord.from_options(mol=mol1,lot=lot1)
        print "ic2"
        ic2=ICoord.from_options(mol=mol2,lot=lot2)
        #ic3= ICoord.union_ic(ic2,ic1)
        #ic3.bmatp_create()
        #ic3.bmatp_to_U()
        #ic3.bmat_create()

    if 1:
        #optimize example 
        filepath="tests/twisted_fluoroethene.xyz"
        nocc=11
        nactive=2
        lot1=PyTC.from_options(calc_states=[(0,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot1.cas_from_file(filepath)
        print "ic1"
        mol1=pb.readfile("xyz",filepath).next()
        ic1=ICoord.from_options(mol=mol1,lot=lot1)
        ic1.optimize(1)
