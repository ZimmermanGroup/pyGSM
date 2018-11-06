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
from _icoord import *
import elements 

class Base_DLC(Utils,ICoords):

    @staticmethod
    def default_options():
        """ Base_DLC default options. """

        if hasattr(Base_DLC, '_default_options'): return Base_DLC._default_options.copy()
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
                key="nicd",
                value=None,
                required=False,
                )

        opt.add_option(
            key='OPTTHRESH',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        Base_DLC._default_options = opt
        return Base_DLC._default_options.copy()

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
        self.nicd=self.options['nicd']
        self.OPTTHRESH=self.options['OPTTHRESH']
        self.madeBonds = False
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
        raise NotImplementedError()

    # can be inherited and modified for hybrid

    def ic_create(self):
        raise NotImplementedError()
    def update_ics(self):
        raise NotImplementedError()
    def grad_to_q(self,grad):
        raise NotImplementedError()
    def make_Hint():
        raise NotImplementedError()
    def opt_step(self,nconstraints):
        raise NotImplementedError()
    def ic_to_xyz(self,dq):
        raise NotImplementedError()
    def ic_to_xyz_opt(self,dq0):
        raise NotImplementedError()

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
        raise NotImplementedError()

    # => bmatp creation is handled in _bmat or _hbmat

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

    def Hintp_to_Hint(self):
        tmp = np.matmul(self.Ut,np.transpose(self.Hintp))
        return np.matmul(tmp,np.transpose(self.Ut))

    # put where?
    def update_bfgsp(self):
        #print("In update bfgsp")
        dx = self.dqprim
        dg = self.gradqprim - self.pgradqprim
        Hdx = np.dot(self.Hintp,dx)
        dxHdx = np.dot(np.transpose(dx),Hdx)
        dgdg = np.outer(dg,dg)
        dgtdx = np.dot(np.transpose(dg),dx)
        change = np.zeros_like(self.Hintp)
        if dgtdx>0.:
            if dgtdx<0.001: dgtdx=0.001
            change += dgdg/dgtdx
        if dxHdx>0.:
            if dxHdx<0.001: dxHdx=0.001
            change -= np.outer(Hdx,Hdx)/dxHdx
        return change

    def compute_predE(self,dq0):
        # compute predicted change in energy 
        assert np.shape(dq0)==(self.nicd,1), "dq0 not (nicd,1) "
        assert np.shape(self.gradq)==(self.nicd,1), "gradq not (nicd,1) "
        assert np.shape(self.Hint)==(self.nicd,self.nicd), "Hint not (nicd,nicd) "
        dEtemp = np.dot(self.Hint,dq0)
        dEpre = np.dot(np.transpose(dq0),self.gradq) + 0.5*np.dot(np.transpose(dEtemp),dq0)
        dEpre *=KCAL_MOL_PER_AU
        if abs(dEpre)<0.005: dEpre = np.sign(dEpre)*0.005
        print( "predE: %1.4f " % dEpre),
        return dEpre

    def update_ic_eigen(self,gradq,nconstraints=0):
        SCALE =self.SCALEQN
        if self.newHess>0: SCALE = self.SCALEQN*self.newHess
        if self.SCALEQN>10.0: SCALE=10.0
        lambda1 = 0.0

        nicd_c = self.nicd-nconstraints
        temph = self.Hint[:nicd_c,:nicd_c]
        e,v_temp = np.linalg.eigh(temph)

        v = np.transpose(v_temp)
        leig = e[0]

        if leig < 0:
            lambda1 = -leig+0.015
        else:
            lambda1 = 0.005
        if abs(lambda1)<0.005: lambda1 = 0.005

        # => grad in eigenvector basis <= #
        gradq = gradq[:nicd_c,0]
        gqe = np.dot(v,gradq)

        dqe0 = np.divide(-gqe,e+lambda1)/SCALE
        dqe0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dqe0 ]

        dq0 = np.dot(v_temp,dqe0)
        dq0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dq0 ]
        #print "dq0"
        #print ["{0:0.5f}".format(i) for i in dq0]
        dq_c = np.zeros((self.nicd,1))
        for i in range(nicd_c):
            dq_c[i,0] = dq0[i]
        return dq_c
