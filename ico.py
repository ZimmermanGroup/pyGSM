import numpy as np
import openbabel as ob
import pybel as pb
import options


class ICoord(object):

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
                key="mol",
                required=False,
                allowed_types=[pb.Molecule],
                doc='Pybel molecule object (not OB.Mol)')

        ICoord._default_options = opt
        return ICoord._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return ICoord(ICoord.default_options().set_values(kwargs))
    
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
        #self.print_xyz()
        
    def print_xyz(self):
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            print(" %1.4f %1.4f %1.4f" %(a.GetX(), a.GetY(), a.GetZ()) )

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
        MAX_BOND_DIST=0.
        self.nbonds=0
        self.bonds=[]
        self.bondd=[]
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            self.nbonds+=1
            self.bonds.append((bond.GetBeginAtomIdx()-1,bond.GetEndAtomIdx()-1))
            self.bondd.append(bond.GetLength())
        print "number of bonds is %i" %self.nbonds

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
        #print self.angles
        #print self.anglev
        print "number of angles is %i" %self.nangles


    def make_torsions(self):
        self.ntor=0
        self.torsions=[]
        self.torv=[]
        for torsion in ob.OBMolTorsionIter(self.mol.OBMol):
            self.ntor+=1
            self.torsions.append(torsion)
            self.torv.append(self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]))
        #print self.torsions
        #print self.torv
        print "number of torsions is %i" %self.ntor


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
        self.update_bonds()
        self.update_angles()
        self.update_torsions()

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

    def union_ic(
            self,
            icoordA,
            icoordB,
            ):
        """ return the union of two lists """
        unionBonds    = list(set(icoordA.bonds) | set(icoordB.bonds))
        unionAngles   = list(set(icoordA.angles) | set(icoordB.angles))
        unionTorsions = list(set(icoordA.torsions) | set(icoordB.torsions))
        print "Saving bond union"
        self.bonds = []
        self.angles = []
        self.torsions = []
        for bond in unionBonds:
            self.bonds.append(bond)
        for angle in unionAngles:
            self.angles.append(angle)
        for torsion in unionTorsions:
            self.torsions.append(torsion)

    def bond_exists(self,bond):
        if bond in self.bonds:
            return True
        else:
            return False

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
                n=+1

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

    def distance(self,i,j):
        """ for some reason openbabel has this one based """
        a1=self.mol.OBMol.GetAtom(i+1)
        a2=self.mol.OBMol.GetAtom(j+1)
        return a1.GetDistance(a2)

    def get_angle(self,i,j,k):
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1)
        c=self.mol.OBMol.GetAtom(k+1)
        return self.mol.OBMol.GetAngle(b,a,c) #a is the vertex

    def get_torsion(self,i,j,k,l):
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1)
        c=self.mol.OBMol.GetAtom(k+1)
        d=self.mol.OBMol.GetAtom(l+1)
        return self.mol.OBMol.GetTorsion(a,b,c,d)


    def getIndex(self,i):
        return self.mol.OBMol.GetAtom(i+1).GetIndex()

    def isTM(self,i):
        anum= self.getIndex(i)
        if anum>20:
            if anum<31:
                return True
            elif anum >38 and anum < 49:
                return True
            elif anum >71 and anum <81:
                return True


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

                for i in range(self.natoms):
                    for j in range(self.natoms):
                        if frags[i]==n1 and frags[j]==n2:
                            close=self.distance(i,j)
                            #connect whatever is closest
                            if close < mclose and close < self.MAX_FRAG_DIST:
                                mclose = close
                                a1=i
                                a2=j
                                found=1

                for i in range(self.natoms):
                    for j in range(self.natoms):
                        if frags[i]==n1 and frags[j]==n2:
                            close=self.distance(i,j)
                            #connect second pair heavies or H-Bond only, away from first pair
                            dia1 = self.distance(i,a1)
                            dja1 = self.distance(j,a1)
                            dia2 = self.distance(i,a2)
                            dja2 = self.distance(j,a2)
                            dist21 = (dia1+dja1)/2.
                            dist22 = (dia2+dja2)/2.

                            #TODO changed from 4.5 to 4
                            if (self.getIndex(i) > 1 or self.getIndex(j)>1) and dist21 > 4. and dist22 >4. and close<mclose2 and close < self.MAX_FRAG_DIST: 
                                mclose2 = close
                                b1=i
                                b2=j
                                found2=1
                
                for i in range(self.natoms):
                    for j in range(self.natoms):
                        if frags[i]==n1 and frags[j]==n2 and b1>0 and b2>0:
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
                        if frags[i]==n1 and frags[j]==n2 and self.isOpt==2:
                            #connect fourth pair, TM only, away from first pair
                            if c1!=i and c2!=i and c1!=j and c2!=j: #don't repeat 
                                if self.isTM(i) or self.isTM(j):
                                    close=self.distance(i,j)
                                    if close<mclose4 and close<self.MAX_FRAG_DIST:
                                        mclose4=close
                                        d1=i
                                        d2=j
                                        found4=1

                bond1=(a1,a2)
                if found>0 and self.bond_exists(bond1)==False:
                    print("bond pair1 added : %s" % (bond1,))
                    self.bonds.append(bond1)
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

                if self.isOpt==2:
                    print("Checking for linear angles in newly added bond")
                    #TODO

    def bmatp_dqbdx(self,i,j):
        u = np.zeros(3,dtype=float)
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        u=np.subtract(coora,coorb)
        norm= np.linalg.norm(u)
        u = u/norm
        dqbdx = np.zeros(6,dtype=float)
        dqbdx[0] = u[0]
        dqbdx[1] = u[1]
        dqbdx[2] = u[2]
        dqbdx[3] = -u[0]
        dqbdx[4] = -u[1]
        dqbdx[5] = -u[2]
        return dqbdx

    def bmatp_dqadx(self,i,j,k):
        u = np.zeros(3,dtype=float)
        v = np.zeros(3,dtype=float)
        w = np.zeros(3,dtype=float)
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1) #vertex
        c=self.mol.OBMol.GetAtom(k+1)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        coorc=np.array([c.GetX(),c.GetY(),c.GetZ()])
        u=np.subtract(coora,coorb)
        v=np.subtract(coorc,coorb)
        n1=self.distance(i,j)
        n2=self.distance(j,k)
        u=u/n1
        v=v/n2

        w=np.cross(u,v)
        nw = np.linalg.norm(w)
        if nw < 1e-3:
            print(" linear angle detected")
            vn = np.zeros(3,dtype=float)
            vn[2]=1.
            w=np.cross(u,vn)
            nw = np.linalg.norm(w)
            if nw < 1e-3:
                vn[2]=0.
                vn[1]=1.
                w=np.cross(u,vn)

        n3=np.linalg.norm(w)
        w=w/n3
        uw=np.cross(u,w)
        wv=np.cross(w,v)
        dqadx = np.zeros(9,dtype=float)
        dqadx[0] = uw[0]/n1
        dqadx[1] = uw[1]/n1
        dqadx[2] = uw[2]/n1
        dqadx[3] = -uw[0]/n1 + -wv[0]/n2
        dqadx[4] = -uw[1]/n1 + -wv[1]/n2
        dqadx[5] = -uw[2]/n1 + -wv[2]/n2
        dqadx[6] = wv[0]/n2
        dqadx[7] = wv[1]/n2
        dqadx[8] = wv[2]/n2

        return dqadx

    def bmatp_dqtdx(self,i,j,k,l):
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1) 
        c=self.mol.OBMol.GetAtom(k+1)
        d=self.mol.OBMol.GetAtom(l+1)

        angle1=self.mol.OBMol.GetAngle(a,b,c)*np.pi/180.
        angle2=self.mol.OBMol.GetAngle(b,c,d)*np.pi/180.
        if angle1>3.0 or angle2>3.0:
            print(" near-linear angle")
            return
        u = np.zeros(3,dtype=float)
        v = np.zeros(3,dtype=float)
        w = np.zeros(3,dtype=float)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        coorc=np.array([c.GetX(),c.GetY(),c.GetZ()])
        coord=np.array([d.GetX(),d.GetY(),d.GetZ()])
        u=np.subtract(coora,coorb)
        w=np.subtract(coorc,coorb)
        v=np.subtract(coord,coorc)
        
        n1=self.distance(i,j)
        n2=self.distance(j,k)
        n3=self.distance(k,l)

        u=u/n1
        v=v/n1
        w=w/n1

        uw=np.cross(u,w)
        vw=np.cross(v,w)

        cosphiu = np.dot(u,w)
        cosphiv = -1*np.dot(v,w)
        sin2phiu = 1.-cosphiu*cosphiu
        sin2phiv = 1.-cosphiv*cosphiv

        if sin2phiu < 1e-3 or sin2phiv <1e-3:
            return

        #CPMZ possible error in uw calc
        dqtdx = np.zeros(12,dtype=float)
        dqtdx[0]  = uw[0]/(n1*sin2phiu);
        dqtdx[1]  = uw[1]/(n1*sin2phiu);
        dqtdx[2]  = uw[2]/(n1*sin2phiu);
        dqtdx[3]   = -uw[0]/(n1*sin2phiu) + ( uw[0]*cosphiu/(n2*sin2phiu) + vw[0]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[4]   = -uw[1]/(n1*sin2phiu) + ( uw[1]*cosphiu/(n2*sin2phiu) + vw[1]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[5]   = -uw[2]/(n1*sin2phiu) + ( uw[2]*cosphiu/(n2*sin2phiu) + vw[2]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[6]   =  vw[0]/(n3*sin2phiv) - ( uw[0]*cosphiu/(n2*sin2phiu) + vw[0]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[7]   =  vw[1]/(n3*sin2phiv) - ( uw[1]*cosphiu/(n2*sin2phiu) + vw[1]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[8]   =  vw[2]/(n3*sin2phiv) - ( uw[2]*cosphiu/(n2*sin2phiu) + vw[2]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[9]   = -vw[0]/(n3*sin2phiv)                                                                                  
        dqtdx[10]  = -vw[1]/(n3*sin2phiv)                                                                                  
        dqtdx[11]  = -vw[2]/(n3*sin2phiv)

        return dqtdx


    def bmatp_create(self):
        self.num_ics = self.nbonds + self.nangles + self.ntor
        N3 = 3*self.natoms
        print "Number of internal coordinates is %i " % self.num_ics
        self.bmatp=np.zeros((self.num_ics,N3),dtype=float)
        #TODO Not sure to make this rows or columns
        i=0
        for bond in self.bonds:
            a1=bond[0]
            a2=bond[1]
            dqbdx = self.bmatp_dqbdx(a1,a2)
            self.bmatp[i][3*a1+0] = dqbdx[0]
            self.bmatp[i][3*a1+1] = dqbdx[1]
            self.bmatp[i][3*a1+2] = dqbdx[2]
            self.bmatp[i][3*a2+0] = dqbdx[3]
            self.bmatp[i][3*a2+1] = dqbdx[4]
            self.bmatp[i][3*a2+2] = dqbdx[5]
            i+=1
            #print "%s" % ((a1,a2),)

        for angle in self.angles:
            a1=angle[1]
            a2=angle[0] #vertex
            a3=angle[2]
            dqadx = self.bmatp_dqadx(a1,a2,a3)
            self.bmatp[i][3*a1+0] = dqadx[0]
            self.bmatp[i][3*a1+1] = dqadx[1]
            self.bmatp[i][3*a1+2] = dqadx[2]
            self.bmatp[i][3*a2+0] = dqadx[3]
            self.bmatp[i][3*a2+1] = dqadx[4]
            self.bmatp[i][3*a2+2] = dqadx[5]
            self.bmatp[i][3*a3+0] = dqadx[6]
            self.bmatp[i][3*a3+1] = dqadx[7]
            self.bmatp[i][3*a3+2] = dqadx[8]
            i+=1
            #print i
            #print "%s" % ((a1,a2,a3),)

        for torsion in self.torsions:
            a1=torsion[0]
            a2=torsion[1]
            a3=torsion[2]
            a4=torsion[3]
            dqtdx = self.bmatp_dqtdx(a1,a2,a3,a4)
            self.bmatp[i][3*a1+0] = dqtdx[0]
            self.bmatp[i][3*a1+1] = dqtdx[1]
            self.bmatp[i][3*a1+2] = dqtdx[2]
            self.bmatp[i][3*a2+0] = dqtdx[3]
            self.bmatp[i][3*a2+1] = dqtdx[4]
            self.bmatp[i][3*a2+2] = dqtdx[5]
            self.bmatp[i][3*a3+0] = dqtdx[6]
            self.bmatp[i][3*a3+1] = dqtdx[7]
            self.bmatp[i][3*a3+2] = dqtdx[8]
            self.bmatp[i][3*a4+0] = dqtdx[9]
            self.bmatp[i][3*a4+1] = dqtdx[10]
            self.bmatp[i][3*a4+2] = dqtdx[11]
            i+=1
            #print i
            #print dqtdx
            #print "%s" % ((a1,a2,a3,a4),)

        #print self.bmatp

    def bmatp_to_U(self):
        N3=3*self.natoms
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        print "printing bmatp"
        print self.bmatp

        print "\n"
        print "shape of bmatp is %s" %(np.shape(self.bmatp),)

        G=np.matmul(self.bmatp,np.transpose(self.bmatp))
        #print G
        print "Shape of G is %s" % (np.shape(G),)
        e,v = np.linalg.eig(G)
        e = np.real(e)
        v= np.real(v)
        print "eigenvalues of BB^T" 
        print e
        print "\n"
        lowev=[]

        self.nicd=self.num_ics
        #TODO this is a hack
        for i in e:
            if np.real(i)<0.001:
                lowev.append(i)
                self.nicd -=1
        if self.nicd<N3-6:
            print(" Error: optimization space less than 3N-6 DOF")
            print len(lowev)
            exit(-1)

        print(" Number of internal coordinate dimensions %i" %self.nicd)
        print(" Number of lowev %i" %len(lowev))

        print "diag(BB^T)"
        print v
        print "\n"

        redset = N3 - self.nicd
        #self.U = v[0:self.nicd, :]
        self.U = v
        #print "Shape of U is %s" % (np.shape(self.U),)


    def q_create(self):  
        """Determines the scalars in delocalized internal coordinates"""

        print(" Determining q in ICs")
        N3=3*self.natoms
        self.q = np.zeros(self.nicd)
        print "Number of ICs %i" % self.num_ics
        print "Number of IC dimensions %i" %self.nicd
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)

        dists=[self.distance(bond[0],bond[1]) for bond in self.bonds ]
        angles=[self.get_angle(angle[0],angle[1],angle[2])*np.pi/180. for angle in self.angles ]
        torsions =[self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3])*np.pi/180. for torsion in self.torsions]

        for i in range(self.nicd):
            Ubond=self.U[i][0:self.nbonds]
            Uangle=self.U[i][self.nbonds:self.nangles+self.nbonds]
            Utorsion=self.U[i][self.nangles+self.nbonds:self.nangles+self.nbonds+self.ntor]
            self.q[i] = np.dot(Ubond,dists) + np.dot(Uangle,angles) + np.dot(Utorsion,torsions)


    def bmat_create(self):

        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)

        print(" In bmat create")
        np.set_printoptions(precision=4)
        np.set_printoptions(suppress=True)
        self.q_create()

        print(" now making bmat in delocalized ICs")
        bmat = np.matmul(np.transpose(self.U),self.bmatp)
        print("printing bmat")
        print bmat
        print(" Shape of bmat %s" %(np.shape(bmat),))

        bbt = np.matmul(bmat,np.transpose(bmat))
        print(" Shape of bbt %s" %(np.shape(bbt),))

        bbti = np.linalg.inv(bbt)
        print("bmatti formation")
        self.bmatti = np.matmul(bbti,bmat)
        print self.bmatti

        print(" Shape of bmatti %s" %(np.shape(self.bmatti),))


if __name__ == '__main__':

    from obutils import *

    """
    # => bimolecular example to test make frags <= #
    filepath="bimol.xyz"
    molecules=read_molecules(filepath,single=False)
    mol1 = pb.Molecule(molecules[0])
    mol2 = pb.Molecule(molecules[1])
    ic1=ICoord.from_options(mol=mol1)
    print ic1.isOpt
    print ic1.MAX_FRAG_DIST
    ic1.ic_create()
    """
    
    filepath="fluoroethene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    ic1=ICoord.from_options(mol=mol)
    ic1.ic_create()
    ic1.bmatp_create()
    ic1.bmatp_to_U()
    ic1.bmat_create()
    
    #ic1.union_ic(ic1,ic2)
    #ic1.update_ics()
