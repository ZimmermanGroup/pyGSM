import numpy as np
import openbabel as ob
import pybel as pb
import options
import os
from units import *
import itertools
from copy import deepcopy
import manage_xyz
from _icoord import ICoords
from _bmat import Bmat
from base_dlc import *
from sklearn import preprocessing
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

class DLC(Base_DLC,Bmat,Utils):

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return DLC(DLC.default_options().set_values(kwargs))

    def setup(self):
        #if self.isOpt>0:
        if True:
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
            self.nretry = 0 
            self.DMIN0 =self.DMAX/10.
            self.coords = np.zeros((len(self.mol.atoms),3))
            for i,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
                self.coords[i,0] = a.GetX()
                self.coords[i,1] = a.GetY()
                self.coords[i,2] = a.GetZ()

        # TODO might be a Pybel way to do 
        atomic_nums = self.getAtomicNums()
        Elements = elements.ElementData()
        myelements = [ Elements.from_atomic_number(i) for i in atomic_nums]
        atomic_symbols = [ele.symbol for ele in myelements]
        self.geom=manage_xyz.combine_atom_xyz(atomic_symbols,self.coords)


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
        permAngle = list(itertools.permutations([0,1,2]))
        permTor = list(itertools.permutations([0,1,2,3]))
        for angle in angleB:
            foundA=False
            for perm in permAngle:
                if (angle[perm[0]],angle[perm[1]],angle[perm[2]]) in angleA:
                    foundA=True
                    break
            if foundA==False:
                angleA.append(angle)

        angleA.append((1,4,3))
        for torsion in torsionB:
            foundA=False
            for perm in permTor:
                if (torsion[perm[0]],torsion[perm[1]],torsion[perm[2]],torsion[perm[3]]) in torsionA:
                    foundA=True
                    break
            if foundA==False:
                torsionA.append(torsion)


        #print "icoordA,icoordB bonds"
        #print icoordA.BObj.bonds
        #print icoordB.BObj.bonds
        #unionBonds    = list(set(icoordA.BObj.bonds) | set(icoordB.BObj.bonds))
        #unionBonds.sort()
        #print "printing unionBonds"
        #print unionBonds
        #unionAngles   = list(set(icoordA.AObj.angles) | set(icoordB.AObj.angles))
#       # print "printing unionANgles"
#       # print unionAngles
        #unionAngles = map(tuple,set(map(frozenset,unionAngles)))
#       # print unionAngles
        #unionTorsions = list(set(icoordA.TObj.torsions) | set(icoordB.TObj.torsions))
#       # print "printing unionTorsions"
#       # print unionTorsions
        #unionTorsions = map(tuple,set(map(frozenset,unionTorsions)))
#       # print unionTorsions
        #bonds = []
        #angles = []
        #torsions = []
        #for bond in unionBonds:
        #    bonds.append(bond)
        #for angle in unionAngles:
        #    angles.append(angle)
        #for torsion in unionTorsions:
        #    torsions.append(torsion)
        print "printing Union ICs"
        print bondA
        print angleA
        print torsionA
        icoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1=pb.readfile('xyz','tmp1.xyz').next()
        pes1 = deepcopy(icoordA.PES)
        return DLC.from_options(
                bonds= bondA,
                angles= angleA,
                torsions= torsionA,
                mol = mol1,
                PES = pes1,
                nicd= icoordA.nicd
                )

    @staticmethod
    def add_node(ICoordA,ICoordB,nmax,ncurr):
        dq0 = np.zeros((ICoordA.nicd,1))

        ICoordA.mol.write('xyz','tmp1.xyz',overwrite=True)
        mol1 = pb.readfile('xyz','tmp1.xyz').next()
        PES1 = deepcopy(ICoordA.PES)
        ICoordC = DLC(ICoordA.options.copy().set_values({
            "mol" : mol1,
            "bonds" : ICoordA.BObj.bonds,
            "angles" : ICoordA.AObj.angles,
            "torsions" : ICoordA.TObj.torsions,
            "PES" : PES1
            }))

        ICoordC.setup()
        ictan = DLC.tangent_1(ICoordB,ICoordA)
        #print ictan.T
        ICoordC.opt_constraint(ictan)
        dqmag = np.dot(ICoordC.Ut[-1,:],ictan)
        print " dqmag: %1.3f"%dqmag
        ICoordC.bmatp_create()
        ICoordC.bmat_create()
        if nmax-ncurr > 1:
            dq0[ICoordC.nicd-1] = dqmag/float(nmax-ncurr)
        else:
            dq0[ICoordC.nicd-1] = dqmag/2.0;

        print " dq0[constraint]: %1.3f" % dq0[ICoordC.nicd-1]
        ICoordC.ic_to_xyz(dq0)
        ICoordC.update_ics()
        ICoordC.bmatp_create()
        ICoordC.bmatp_to_U()
        ICoordC.bmat_create()
        ictan = DLC.tangent_1(ICoordC,ICoordA)
        
        #ICoordC.dqmag = dqmag

        return ICoordC


    def ic_create(self):

        self.natoms= len(self.mol.atoms)
        self.coordn = self.coord_num()

        if self.madeBonds==False:
            print " making bonds"
            self.BObj = self.make_bonds()
            #TODO 
            if self.isOpt>0:
                print(" isOpt: %i" %self.isOpt)
                self.nfrags,self.frags = self.make_frags()
                self.BObj.update(self.mol)
                self.bond_frags()
                self.AObj = self.make_angles()
                self.TObj = self.make_torsions()
                self.AObj.update(self.mol)
                self.TObj.update(self.mol)
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
                    print(" bond pair1 added : %s" % (bond1,))
                    self.BObj.bonds.append(bond1)
                    self.BObj.nbonds+=1
                    self.BObj.bondd.append(mclose)
                    print " bond dist: %1.4f" % mclose
                    isOkay = self.mol.OBMol.AddBond(bond1[0]+1,bond1[1]+1,1)
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

    def ic_to_xyz(self,dq):
        """ Transforms ic to xyz, used by addNode"""
        self.update_ics()
        self.bmatp=self.bmatp_create()
        self.bmatti=self.bmat_create()
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

    def grad_to_q(self,grad):
        gradq = np.dot(self.bmatti,grad)
        return gradq

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

    def combined_step(self,nconstraints):
        printf(" taking step along x and orthogonal vector (besides y) ")
        return


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
        self.gradrms = np.linalg.norm(self.gradq)*1./np.sqrt(self.nicd-nconstraints)
        if self.print_level==1:
            print("gradrms = %1.5f" % self.gradrms),
        self.buf.write(" gRMS=%1.5f" %(self.gradrms))
        if self.gradrms < self.OPTTHRESH:
            return 0.

        # For Hessian update
        self.pgradqprim=self.gradqprim
        self.gradqprim = np.dot(np.transpose(self.Ut),self.gradq)

        # => Update Hessian <= #
        if self.do_bfgs == True:
            self.update_Hessian()
        self.do_bfgs = True

        # => Take Eigenvector Step <=#
        dq = self.update_ic_eigen(self.gradq,nconstraints)
        print " printing dq:",np.transpose(dq)
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
        #self.bmatp=self.bmatp_create()
        #self.bmatp_to_U()
        #self.bmatti=self.bmat_create()
        #self.Hint=self.Hintp_to_Hint()
     
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

    def combined_step(self):
        return

    def update_Hessian(self):
        #print("In update bfgsp")
        self.newHess-=1
        change = self.update_bfgsp()
        self.Hintp += change
        self.Hint=self.Hintp_to_Hint()

    def opt_constraint(self,ictan):
        norm = np.linalg.norm(ictan)
        C = ictan/norm
        dots = np.matmul(self.Ut,ictan)
        # Cn is C in basis
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
        #print "orthogonal basis"
        #for i in range(self.nicd):
        #    for j in range(self.num_ics):
        #        print "%1.3f"% self.Ut[i,j],
        #    print ""

    def orthogonalize(self,vecs):
        basis=np.zeros_like(vecs)
        basis[-1,:] = vecs[-1,:] # orthogonalizes with respect to the last
        for i,v in enumerate(vecs):
            w = v - np.sum( np.dot(v,b)*b  for b in basis)
            tmp = w/np.linalg.norm(w)
            if (w > 1e-10).any():  
                basis[i,:]=tmp
        #dots = np.matmul(basis,np.transpose(basis))
        #print "orthogonal basis"
        #for i in range(len(basis)):
        #    for j in range(self.num_ics):
        #        print "%1.3f"% basis[i,j],
        #    print ""
        return basis

    def fromDLC_to_ICbasis(self,vecq):
        """
        This function takes a matrix of vectors wrtiten in the basis of U.
        The components in this basis are called q.
        """
        vec_U = np.zeros((self.num_ics,1),dtype=float)
        assert np.shape(vecq) == (self.nicd,1), "vecq is not nicd long"
        vec_U = np.dot(self.Ut.T,vecq)
        return vec_U/np.linalg.norm(vec_U)

    def opt_constraint2(self,C):
        """
        This function takes a matrix of vectors wrtiten in the basis of ICs
        same as U vectors, and returns a new normalized Ut with those vectors as 
        basis vectors.
        """
        
        # normalize all constraints
        Cn = preprocessing.normalize(C.T,norm='l2')
        dots = np.matmul(Cn,Cn.T)

        # orthogonalize
        Cn = self.orthogonalize(Cn) 
        #print "shape of Cn is %s" %(np.shape(Cn),)

        # write Cn in terms of C_U?
        dots = np.matmul(self.Ut,Cn.T)
        C_U = np.matmul(self.Ut.T,dots)

        # normalize C_U
        C_U = preprocessing.normalize(C_U.T,norm='l2')
        #print "shape of overlaps is %s, shape of Ut is %s, shape of C_U is %s" %(np.shape(dots),np.shape(self.Ut),np.shape(C_U))

        basis=np.zeros((self.nicd,self.num_ics),dtype=float)
        for n,row in enumerate(C_U):
            basis[self.nicd-len(C_U)+n,:] =row 
        count=0
        for v in self.Ut:
            w = v - np.sum( np.dot(v,b)*b  for b in basis )
            tmp = w/np.linalg.norm(w)
            if (w > 1e-4).any():  
                basis[count,:] =tmp
                count +=1
        self.Ut = np.array(basis)
        #print "printing Ut"
        #print self.Ut
        #print "Check if Ut is orthonormal"
        #dots = np.matmul(self.Ut,np.transpose(self.Ut))
        #print dots
