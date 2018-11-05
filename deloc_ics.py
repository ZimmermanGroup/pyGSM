import numpy as np
import openbabel as ob
import pybel as pb
import options
import elements 
import os
from units import *
import itertools
from copy import deepcopy
import manage_xyz

from _icoord import ICoords
from _icoord import Bond_obj
from _icoord import Ang_obj
from _icoord import Tor_obj
from _bmat import Bmat
from _obutils import Utils
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class DLC(ICoords,Bmat,Utils):

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
            key='OPTTHRESH',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

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
        self.OPTTHRESH = self.options['OPTTHRESH']
        bonds = self.options['bonds']
        angles = self.options['angles']
        torsions = self.options['torsions']
        if bonds is not None:
            self.BObj = Bond_obj(None,bonds,None)
            self.BObj.update(self.mol)
            self.madeBonds = True
            self.AObj = Ang_obj(None,angles,None)
            self.AObj.update(self.mol)
            self.TObj = Tor_obj(None,torsions,None)
            self.TObj.update(self.mol)
        else:
            self.madeBonds =False
        self.resetopt = self.options['resetopt']
        self.print_level = self.options['print_level']

        #self.print_xyz()
        self.Elements = elements.ElementData()

        if self.isOpt>0:
            self.ic_create()
            self.bmatp=self.bmatp_create()
            self.bmatp_to_U()
            self.bmatti=self.bmat_create()
            self.make_Hint()  
            self.pgradqprim = np.zeros((self.num_ics,1),dtype=float)
            self.gradqprim = np.zeros((self.num_ics,1),dtype=float)
            self.gradq = np.zeros((self.nicd,1),dtype=float)
            self.pgradq = np.zeros((self.nicd,1),dtype=float)
            self.gradrms = 1000.
            self.SCALEQN = 1.0
            self.MAXAD = 0.075
            self.ixflag = 0
            self.energy = 0.
            self.DMAX = 0.1
            self.DMIN0 =self.DMAX/10.
            self.coords = np.zeros((len(self.mol.atoms),3))
            for i,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
                self.coords[i,0] = a.GetX()
                self.coords[i,1] = a.GetY()
                self.coords[i,2] = a.GetZ()

        # TODO might be a Pybel way to do 
        atomic_nums = self.getAtomicNums()
        myelements = [ self.Elements.from_atomic_number(i) for i in atomic_nums]
        atomic_symbols = [ele.symbol for ele in myelements]
        self.geom=manage_xyz.combine_atom_xyz(atomic_symbols,self.coords)

        self.nretry = 0 

    @staticmethod
    def union_ic(
            icoordA,
            icoordB,
            ):
        """ return union DLC of two DLC Objects """
        unionBonds    = list(set(icoordA.BObj.bonds) | set(icoordB.BObj.bonds))
        unionAngles   = list(set(icoordA.AObj.angles) | set(icoordB.AObj.angles))
        unionTorsions = list(set(icoordA.TObj.torsions) | set(icoordB.TObj.torsions))

        bonds = []
        angles = []
        torsions = []
        for bond in unionBonds:
            bonds.append(bond)
        for angle in unionAngles:
            angles.append(angle)
        for torsion in unionTorsions:
            torsions.append(torsion)

        icoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1=pb.readfile('xyz','tmp1.xyz').next()
        pes1 = deepcopy(icoordA.PES)

        return DLC(icoordA.options.copy().set_values({
                'bonds' : bonds,
                'angles': angles,
                'torsions': torsions,
                'mol' : mol1,
                'PES' : pes1,
                }))

    @staticmethod
    def add_node(ICoordA,ICoordB):
        dq0 = np.zeros((ICoordA.nicd,1))

        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        PES1 = deepcopy(ICoordA.PES)
        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol" : mol1,
            "bonds" : ICoordA.BObj.bonds,
            "PES" : PES1
            }))

        ictan = DLC.tangent_1(ICoordA,ICoordB)
        ICoordC.opt_constraint(ictan)
        dqmag = np.dot(ICoordC.Ut[-1,:],ictan)
        print " dqmag: %1.3f"%dqmag
        ICoordC.bmatp_create()
        ICoordC.bmat_create()
        #if self.nnodes-self.nn != 1:
        if 1:
            dq0[ICoordC.nicd-1] = -dqmag/7.
            #dq0[newic.nicd-1] = -dqmag/float(self.nnodes-self.nn)
        else:
            dq0[ICoordC.nicd-1] = -dqmag/2.0;
        
        print " dq0[constraint]: %1.3f" % dq0[ICoordC.nicd-1]
        ICoordC.ic_to_xyz(dq0)
        ICoordC.update_ics()
        ICoordC.dqmag = dqmag

        return ICoordC
        
    def ic_create(self):
        self.natoms= len(self.mol.atoms)
        self.coordn = self.coord_num()

        if self.madeBonds==False:
            print "making bonds"
            self.BObj = self.make_bonds()
            #TODO 
            if self.isOpt>0:
                print(" isOpt: %i" %self.isOpt)
                self.nfrags,self.frags = self.make_frags()
                self.bond_frags()
            self.AObj = self.make_angles()
            self.TObj = self.make_torsions()
        else:
            self.BObj.update(self.mol)
            self.AObj.update(self.mol)
            self.TObj.update(self.mol)

        #self.make_imptor()
        if self.isOpt==1:
            self.linear_ties()
        #self.make_nonbond() 


    def update_ics(self):
        self.update_xyz()
        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords)
        self.PES.lot.hasRanForCurrentCoords= False
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

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
        clist=[[]]
        m =[]
        for i in range(n):
            # b is the vertex 
            a=self.mol.OBMol.GetAtom(blist[i][0])
            b=self.mol.OBMol.GetAtom(blist[i][1])
            c=self.mol.OBMol.GetAtom(blist[i][2])
            tmp=0
            for nbr in ob.OBAtomAtomIter(a):
                if nbr.GetIndex() != b.GetIndex():
                    clist[i].append(nbr.GetIndex()+1)
                    tmp+=1
            for nbr in ob.OBAtomAtomIter(c):
                if nbr.GetIndex() != b.GetIndex():
                    clist[i].append(nbr.GetIndex()+1)
                    tmp+=1
            m.append(tmp)

        # cross linking 
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
                        torsion= (c1,a1,a2,c2)
                        print(" adding torsion via linear ties %s" %torsion)
                        self.TObj.torsions.append(torsion)
                        self.TObj.ntor +=1


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
                    self.BObj.bonds.append(bond1)
                    self.BObj.nbonds+=1
                    self.BObj.bondd.append(mclose)
                    print "bond dist: %1.4f" % mclose
                    isOkay = self.mol.OBMol.AddBond(bond1[0]+1,bond1[1]+1,1)
                    print "Bond added okay? %r" % isOkay
                bond2=(b1,b2)
                if found2>0 and self.bond_exists(bond2)==False:
                    self.BObj.bonds.append(bond2)
                    print("bond pair2 added : %s" % (bond2,))
                bond3=(c1,c2)
                if found3>0 and self.bond_exists(bond3)==False:
                    self.BObj.bonds.append(bond3)
                    print("bond pair2 added : %s" % (bond3,))
                bond4=(d1,d2)
                if found4>0 and self.bond_exists(bond4)==False:
                    self.BObj.bonds.append(bond4)
                    print("bond pair2 added : %s" % (bond24,))


                if self.isOpt==2:
                    print("Checking for linear angles in newly added bond")
                    #TODO
        return isOkay

    def bmatp_to_U(self):
        N3=3*self.natoms
        G=np.matmul(self.bmatp,np.transpose(self.bmatp))

        # Singular value decomposition
        v_temp,e,vh  = np.linalg.svd(G)
        v = np.transpose(v_temp)
        
        lowev=0
        self.nicd=N3-6
        for eig in e[self.nicd-1:0:-1]:
            if eig<0.001:
                lowev+=1

        self.nicd -= lowev
        if lowev>3:
            print(" Error: optimization space less than 3N-6 DOF")
            exit(-1)

        #print(" Number of internal coordinate dimensions %i" %self.nicd)
        redset = self.num_ics - self.nicd
        idx = e.argsort()[::-1]
        v = v[idx[::-1]]
        self.Ut=v[redset:,:]

        self.torv0 = list(self.TObj.torv)
        

    def q_create(self):  
        """Determines the scalars in delocalized internal coordinates"""

        #print(" Determining q in ICs")
        N3=3*self.natoms
        self.q = np.zeros((self.nicd,1),dtype=float)

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
            self.q[i] = np.dot(self.Ut[i,0:self.BObj.nbonds],dists) + \
                    np.dot(self.Ut[i,self.BObj.nbonds:self.AObj.nangles+self.BObj.nbonds],angles) \
                    + np.dot(self.Ut[i,self.BObj.nbonds+self.AObj.nangles:],torsions)

        #print("Printing q")
        #print np.transpose(self.q)


    def ic_to_xyz(self,dq):
        """ Transforms ic to xyz, used by addNode"""

        self.update_ics()
        self.bmatp=self.bmatp_create()
        self.bmatti=self.bmat_create()

        SCALEBT = 1.5
        N3=self.natoms*3
        print np.transpose(dq)
        qn = self.q + dq  #target IC values
        print "target IC values"
        print np.transpose(qn)
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

            #TODO Frozen

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
            self.bmatti=self.bmat_create()
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
            xyzd=np.matmul(btit,dq)
            assert len(xyzd)==3*self.natoms,"xyzd is not N3 dimensional"
            xyzd = np.reshape(xyzd,(self.natoms,3))

            #TODO frozen

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
            self.bmatti=self.bmat_create()

            opt_molecules.append(obconversion.WriteString(self.mol.OBMol))

            #calc new dq
            dq = qn - self.q

            dqmag = np.linalg.norm(dq)
            dqmagall.append(dqmag)
            if dqmag<0.0001: break

            if dqmag>dqmagp*10.:
                print(" Q%i" % n)
                SCALEBT *= 2.0
                self.coords = np.copy(xyzp)
                self.update_ics()
                self.bmatp=self.bmatp_create()
                self.bmatti=self.bmat_create()
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
                torsion_diff.append((i+torfix))

            bond_diff = self.BObj.bondd - qprim[:self.BObj.nbonds]
            angle_diff = self.AObj.anglev - qprim[self.BObj.nbonds:self.AObj.nangles+self.BObj.nbonds]
            self.dqprim = np.concatenate((bond_diff,angle_diff,torsion_diff))
            self.dqprim[self.BObj.nbonds:] *= np.pi/180.
            self.dqprim = np.reshape(self.dqprim,(self.num_ics,1))

        #write convergence geoms to file 
        #largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        #for mol in opt_molecules:
        #    largeXyzFile.write(pb.readstring("xyz",mol))
       
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

        self.Hintp=np.diag(Hdiagp)
        Hdiagp=np.asarray(Hdiagp)
        Hdiagp=np.reshape(Hdiagp,(self.num_ics,1))

        tmp = np.zeros((self.nicd,self.num_ics),dtype=float)
        for i in range(self.nicd): 
            for k in range(self.num_ics):
                tmp[i,k] = self.Ut[i,k]*Hdiagp[k]

        self.Hint = np.matmul(tmp,np.transpose(self.Ut))
        self.Hinv = np.linalg.inv(self.Hint)

        #if self.optCG==False or self.isTSNode==False:
        #    print "Not implemented"


    def opt_step(self,nconstraints):
        energy=0.

        #print "in opt step: coordinates at current step are"
        #print self.coords
        energyp = self.energy
        grad = self.PES.get_gradient(self.geom)
        self.bmatp=self.bmatp_create()
        self.bmatti=self.bmat_create()
        coorp = np.copy(self.coords)

        # grad in ics
        self.pgradq = self.gradq
        self.gradq = self.grad_to_q(grad)
        pgradrms = self.gradrms
        self.gradrms = np.linalg.norm(self.gradq)*1./np.sqrt(self.nicd)
        if self.print_level==1:
            print("gradrms = %1.5f" % self.gradrms),
        self.buf.write(" gRMS=%1.5f" %(self.gradrms))
        if self.gradrms<self.OPTTHRESH:
            return

        # For Hessian update
        self.pgradqprim=self.gradqprim
        self.gradqprim = np.dot(np.transpose(self.Ut),self.gradq)

        # => Update Hessian <= #
        if self.do_bfgs == True:
            self.update_Hessian()
        self.do_bfgs = True

        # => Take Eigenvector Step <=#
        dq = self.update_ic_eigen(self.gradq,nconstraints)

        # regulate max overall step
        smag = np.linalg.norm(dq)
        self.buf.write(" ss: %1.5f (DMAX: %1.3f" %(smag,self.DMAX))
        if self.print_level==1:
            print(" ss: %1.5f (DMAX: %1.3f)" %(smag,self.DMAX)),

        if smag>self.DMAX:
            dq = np.fromiter(( xi*self.DMAX/smag for xi in dq), dq.dtype)
        dq= np.asarray(dq).reshape(self.nicd,1)

        # => update geometry <=#
        rflag = self.ic_to_xyz_opt(dq)

        #TODO if rflag and ixflag
        if rflag==True:
            print "rflag" 
            self.DMAX=self.DMAX/1.6
            dq=self.update_ic_eigen(self.gradq,nconstraints)
            self.ic_to_xyz_opt(dq)
            self.do_bfgs=False

        ## => update ICs <= #
        self.update_ics()
        self.bmatp=self.bmatp_create()
        self.bmatp_to_U()
        self.bmatti=self.bmat_create()
        self.Hint=self.Hintp_to_Hint()
     
        # => calc energyat new position <= #
        self.energy = self.PES.get_energy(self.geom) - self.V0
        self.buf.write(" E(M): %4.5f" %(self.energy))
        if self.print_level==1:
            print "E(M): %4.5f" % self.energy,

        # check goodness of step
        dEstep = self.energy - energyp
        dEpre = self.compute_predE(dq)

        ratio = dEstep/dEpre
        self.buf.write(" ratio: %1.4f" %(ratio))
        if self.print_level==1:
            print "ratio is %1.4f" % ratio,

        # => step controller  <= #
        if dEstep>0.01:
            if self.print_level==1:
                print("decreasing DMAX"),
            self.buf.write(" decreasing DMAX")
            if smag <self.DMAX:
                self.DMAX = smag/1.5
            else: 
                self.DMAX = self.DMAX/1.5
            if dEstep > 2.0 and self.resetopt==True:
                print "resetting coords to coorp"
                self.coords = coorp
                self.energy = self.PES.get_energy(self.geom) - self.V0
                self.update_ics()
                self.bmatp=self.bmatp_create()
                self.bmatp_to_U()
                self.bmatti=self.bmat_create()
                self.Hint=self.Hintp_to_Hint()
                self.do_bfgs=False
        elif ratio<0.25:
            if self.print_level==1:
                print("decreasing DMAX"),
            self.buf.write(" decreasing DMAX")
            if smag<self.DMAX:
                self.DMAX = smag/1.1
            else:
                self.DMAX = self.DMAX/1.2
            self.make_Hint()
        elif (ratio>0.75 and ratio<1.25) and smag > self.DMAX and self.gradrms<pgradrms*1.35:
            if self.print_level==1:
                print("increasing DMAX"),
            self.buf.write(" increasing DMAX")
            self.DMAX=self.DMAX*1.1 + 0.01
            if self.DMAX>0.25:
                self.DMAX=0.25
        if self.DMAX<self.DMIN0:
            self.DMAX=self.DMIN0

        return  smag


    def update_Hessian(self):
        #print("In update bfgsp")
        self.newHess-=1
        change = self.update_bfgsp()
        self.Hintp += change
        self.Hint=self.Hintp_to_Hint()

    def opt_constraint(self,ictan):
        norm = np.linalg.norm(ictan)
        ictan = ictan/norm

        dots = np.matmul(self.Ut,ictan)
        Cn = np.matmul(np.transpose(self.Ut),dots)
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm
        basis=np.zeros((self.nicd,self.num_ics),dtype=float)
        basis[-1,:] = list(Cn)
        for i,v in enumerate(self.Ut):
            w = v - np.sum( np.dot(v,b)*b  for b in basis )
            tmp = w/np.linalg.norm(w)
            if (w > 1e-10).any():  
                basis[i,:] =tmp
        self.Ut = np.array(basis)
        #print "Check if Ut is orthonormal"
        #dots = np.matmul(self.Ut,np.transpose(self.Ut))



