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
        #self.print_xyz()
        
    def print_xyz(self):
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            print(" %1.4f %1.4f %1.4f" %(a.GetX(), a.GetY(), a.GetZ()) )

    def ic_create(self):
        self.make_bonds()
        self.coord_num()
        self.make_angles()
        self.make_torsions()
        self.make_imptor()
        self.linear_ties()
        self.make_nonbond()

    def make_bonds(self):
        MAX_BOND_DIST=0.
        self.nbonds=0
        self.bonds=[]
        self.bondd=[]
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            self.nbonds+=1
            self.bonds.append((bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()))
            self.bondd.append(bond.GetLength())
        #print self.nbonds
        #print self.bonds
        #print self.bondd
        print "number of nbonds is %i" %self.nbonds

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
            #a is the vertex
            a=self.mol.OBMol.GetAtom(angle[0]+1)
            b=self.mol.OBMol.GetAtom(angle[1]+1)
            c=self.mol.OBMol.GetAtom(angle[2]+1)
            self.anglev.append(self.mol.OBMol.GetAngle(b,a,c))
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
            a=self.mol.OBMol.GetAtom(torsion[0]+1)
            b=self.mol.OBMol.GetAtom(torsion[1]+1)
            c=self.mol.OBMol.GetAtom(torsion[2]+1)
            d=self.mol.OBMol.GetAtom(torsion[3]+1)
            self.torv.append(self.mol.OBMol.GetTorsion(a,b,c,d))
        #print self.torsions
        #print self.torv
        print "number of torsions is %i" %self.ntor


    def make_nonbond(self):
        self.nonbond=[]
        for i in range(len(self.mol.atoms)):
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
                   a1=self.mol.OBMol.GetAtom(i+1)
                   a2=self.mol.OBMol.GetAtom(j+1)
                   self.nonbond.append(a1.GetDistance(a2))
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
                        print imptorvt
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
            a1=self.mol.OBMol.GetAtom(bond[0])
            a2=self.mol.OBMol.GetAtom(bond[1])
            self.bondd.append(a1.GetDistance(a2))

    def update_angles(self):
        self.anglev=[]
        for angle in self.angles:
            a=self.mol.OBMol.GetAtom(angle[0]+1)
            b=self.mol.OBMol.GetAtom(angle[1]+1)
            c=self.mol.OBMol.GetAtom(angle[2]+1)
            # a is the vertex
            self.anglev.append(self.mol.OBMol.GetAngle(b,a,c))

    def update_torsions(self):
        self.torv=[]
        for torsion in self.torsions:
            a=self.mol.OBMol.GetAtom(torsion[0]+1)
            b=self.mol.OBMol.GetAtom(torsion[1]+1)
            c=self.mol.OBMol.GetAtom(torsion[2]+1)
            d=self.mol.OBMol.GetAtom(torsion[3]+1)
            self.torv.append(self.mol.OBMol.GetTorsion(a,b,c,d))

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
        print maxsize
        #blist = np.full((3*maxsize+1,1),-1,dtype=int)
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

filepath="initial0000.xyz"
mol=pb.readfile("xyz",filepath).next()
ic1=ICoord.from_options(mol=mol)
ic1.ic_create()

#filepath="2.xyz"
#mol=pb.readfile("xyz",filepath).next()
#ic2=ICoord.from_options(mol=mol)
#ic2.ic_create()

#ic1.union_ic(ic1,ic2)
#ic1.update_ics()
