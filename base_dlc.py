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
from sklearn import preprocessing
import StringIO

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
        self.isTSnode = False
        self.use_constraint = False
        #self.stage1opt = False
        self.update_hess=False
        self.buf = StringIO.StringIO()
        self.HESS_TANG_TOL_TS=0.35
        self.path_overlap=0.0

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
        #print 'printing m'
        #print m
        #print 'printing clist:'
        #print clist
        #print 'printing blist:'
        #print blist
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
                        if not self.torsion_exists(torsion) and len(set(torsion))==4:
                            print(" adding torsion via linear ties %s" %(torsion,))
                            self.TObj.torsions.append(torsion)
                            self.TObj.ntor +=1
        self.BObj.update(self.mol)
        self.AObj.update(self.mol)
        self.TObj.update(self.mol)

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

    def bmat_create(self):
        #print(" In bmat create")
        self.q_create()
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

    def Hintp_to_Hint(self):
        #tmp = np.matmul(self.Ut,np.transpose(self.Hintp))
        tmp = np.dot(self.Ut,self.Hintp) #(nicd,numic)(num_ic,num_ic)
        return np.matmul(tmp,np.transpose(self.Ut)) #(nicd,numic)(numic,numic)

    # put where?
    def update_bfgsp(self):
        #print("In update bfgsp")
        dx = self.dqprim
        dg = self.gradqprim - self.pgradqprim
        Hdx = np.dot(self.Hintp,dx)
        if self.print_level==2:
            print "dg:", dg.T
            print "dx:", dx.T
            print "Hdx"
            print Hdx.T
        dxHdx = np.dot(np.transpose(dx),Hdx)
        dgdg = np.outer(dg,dg)
        dgtdx = np.dot(np.transpose(dg),dx)
        change = np.zeros_like(self.Hintp)

        if self.print_level==2:
            print "dgtdx: %1.3f dxHdx: %1.3f dgdg" % (dgtdx,dxHdx)
            print dgdg

        if dgtdx>0.:
            if dgtdx<0.001: dgtdx=0.001
            change += dgdg/dgtdx
        if dxHdx>0.:
            if dxHdx<0.001: dxHdx=0.001
            change -= np.outer(Hdx,Hdx)/dxHdx

        return change

    def update_bofill(self):

        #print "in update bofill"
        #unit test
        if False:
            self.Hint = np.loadtxt("hint.txt")
            self.Ut = np.loadtxt("Ut.txt")
            dx = np.loadtxt("dx.txt")
            dg = np.loadtxt("dg.txt")
            dx = np.reshape(dx,(self.nicd,1))
            dg = np.reshape(dg,(self.nicd,1))
        else:
            dx = np.copy(self.dq) #nicd,1
            dg = self.gradq - self.pgradq #nicd,1
        #print "dg" 
        #print dg.T
        #print "dx" 
        #print dx.T
        G = np.copy(self.Hint) #nicd,nicd
        Gdx = np.dot(G,dx) #(nicd,nicd)(nicd,1) = (nicd,1)
        dgmGdx = dg - Gdx # (nicd,1)

        # MS
        dgmGdxtdx = np.dot(dgmGdx.T,dx) #(1,nicd)(nicd,1)
        Gms = np.outer(dgmGdx,dgmGdx)/dgmGdxtdx

        #PSB
        dxdx = np.outer(dx,dx)
        dxtdx = np.dot(dx.T,dx)
        dxtdg = np.dot(dx.T,dg)
        dxtGdx = np.dot(dx.T,Gdx)
        dxtdx2 = dxtdx*dxtdx
        dxtdgmdxtGdx = dxtdg - dxtGdx 
        Gpsb = np.outer(dgmGdx,dx)/dxtdx + np.outer(dx,dgmGdx)/dxtdx - dxtdgmdxtGdx*dxdx/dxtdx

        # Bofill mixing 
        dxtE = np.dot(dx.T,dgmGdx) #(1,nicd)(nicd,1)
        EtE = np.dot(dgmGdx.T,dgmGdx)  #E is dgmGdx
        phi = 1. - dxtE*dxtE/(dxtdx*EtE)

        self.Hint += (1.-phi)*Gms + phi*Gpsb
        #print "dxtdx" 
        #print dxtdx
        #print "dxtdx" 
        #print "dgmGdxtdx" 
        #print dgmGdxtdx
        #print dxtdg
        #print "dxtGdx" 
        #print dxtGdx
        #print "dxtE" 
        #print dxtE
        #print "EtE" 
        #print EtE
        #print "phi" 
        #print phi
        #print "Hint after bofill"
        #with np.printoptions(threshold=np.inf):
        #    print self.Hint
        self.Hinv = np.linalg.inv(self.Hint)


    def compute_predE(self,dq0,nconstraints):
        # compute predicted change in energy 
        assert np.shape(dq0)==(self.nicd,1), "dq0 not (nicd,1) "
        assert np.shape(self.gradq)==(self.nicd,1), "gradq not (nicd,1) "
        assert np.shape(self.Hint)==(self.nicd,self.nicd), "Hint not (nicd,nicd) "
        dEtemp = np.dot(self.Hint[:self.nicd-nconstraints,:self.nicd-nconstraints],dq0[:self.nicd-nconstraints])
        dEpre = np.dot(np.transpose(dq0[:self.nicd-nconstraints]),self.gradq[:self.nicd-nconstraints]) + 0.5*np.dot(np.transpose(dEtemp),dq0[:self.nicd-nconstraints])
        dEpre *=KCAL_MOL_PER_AU
        if abs(dEpre)<0.05: dEpre = np.sign(dEpre)*0.05
        if self.print_level>1:
            print( "predE: %1.4f " % dEpre),
        return dEpre

    def update_ic_eigen(self,nconstraints=0):
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
        gradq = self.gradq[:nicd_c,0]
        gqe = np.dot(v,gradq)

        dqe0 = np.divide(-gqe,e+lambda1)/SCALE
        dqe0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dqe0 ]

        # => Convert step back to DLC basis <= #
        dq0 = np.dot(v_temp,dqe0)
        dq0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dq0 ]
        if self.print_level==2:
            print "eigen opt Hint ev:"
            print e
            print "gqe"
            print gqe.T
            print "dqe0"
            print dqe0
            print "dq0"
            print ["{0:0.5f}".format(i) for i in dq0]
        dq_c = np.zeros((self.nicd,1))
        for i in range(nicd_c):
            dq_c[i,0] = dq0[i]
        return dq_c

    def walk_up(self,n):
        """ walk up the n'th DLC"""
        print "walking up the %i coordinate" % n
        #print "print gradq[n]", self.gradq[n]
        #print "type", type(self.gradq[n])
        assert isinstance(self.gradq[n,0],float), "gradq[n] is not float!"
        if self.print_level>0:
            print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = self.SCALEQN*1.0 
       
        dq = self.gradq[n,0]/SCALE
        if abs(dq) > self.MAXAD/SCALEW:
            dq = np.sign(dq)*self.MAXAD/SCALE

        return dq

    def dgrad_step(self):
        """ takes a linear step along dgrad"""
        dgrad = self.PES.get_dgrad(self.geom)
        dgradq = self.grad_to_q(dgrad)
        norm_dg = np.linalg.norm(dgradq)
        if self.print_level>0:
            print " norm_dg is %1.4f" % norm_dg,
            print " dE is %1.4f" % self.PES.dE,

        dq = -self.PES.dE/KCAL_MOL_PER_AU/norm_dg 
        if dq<-0.075:
            dq=-0.075

        return dq

    def update_ic_eigen_ts(self,ictan,nconstraints=1):
        """ this method follows the overlap with reaction tangent"""
        lambda1 = 0.
        SCALE = self.SCALEQN
        if SCALE > 10:
            SCALE = 10.

        self.form_unconstrained_DLC()

        norm = np.linalg.norm(ictan)
        C = ictan/norm
        #print "C"
        #for i in C:
        #    print " %1.3f" %i,
        #print

        #print "Ut"
        #with np.printoptions(threshold=np.inf):
        #    print(self.Ut)

        dots = np.dot(self.Ut,C) #(nicd,numic)(numic,1)
        #print "dots"
        #for i in dots:
        #    print " %1.3f" %i,
        #print
        Cn = np.dot(self.Ut.T,dots) #(numic,nicd)(nicd,1) = numic,1
        #print "shape of Cn is %s" %(np.shape(Cn),)
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm
        #print "Cn"
        #for i in Cn:
        #    print " %1.3f" %i,
        #print


        #TODO should we diagonalize the full matrix? Full matrix done in GSM
        #print "Hint"
        #with np.printoptions(threshold=np.inf):
        #    print self.Hint

        eigen,tmph = np.linalg.eigh(self.Hint) #nicd,nicd
        tmph = tmph.T
        #tmph,eigen,vh  = np.linalg.svd(self.Hint)
        #print "eigenvalues"
        #print eigen
        #print "vectors"
        #with np.printoptions(threshold=np.inf):
        #    print tmph
      
        #print "checking Ut"
        #with np.printoptions(threshold=np.inf):
        #    print self.Ut

        #TODO nneg should be self and checked
        nneg = 0
        for i in range(self.nicd):
            if eigen[i] < -0.01:
                nneg += 1

        #=> Overlap metric <= #
        overlap = np.dot(np.matmul(tmph,self.Ut),Cn) #(nicd,nicd)(nicd,num_ic)(num_ic,1) = (nicd,1)
    
        #print "printing overlaps"
        #print overlap.T

        # Max overlap metrics
        absoverlap = np.abs(overlap)
        maxol = np.max(absoverlap)
        maxoln = np.argmax(absoverlap)
        maxols = overlap[maxoln]
        self.path_overlap = maxol
        self.path_overlap_n = maxoln

        if self.print_level>-1:
            print "t/ol %i: %3.2f" % (maxoln,maxol)
        self.buf.write("t/ol %i: %3.2f" % (maxoln,maxol))

        # => if overlap is small use Cn as Constraint <= #
        if maxol < self.HESS_TANG_TOL_TS or self.gradrms > self.OPTTHRESH*20.: 
            print "doing normal step"
            self.form_constrained_DLC(ictan)
            self.Hintp_to_Hint()
            #self.use_constraint = 1 #TODO would this triger something in regular GSM?
            dq = self.update_ic_eigen(nconstraints)
            dq[-1]=self.walk_up(self.nicd-1)
            return dq

        leig = eigen[1]
        if maxoln!=0:
            leig = eigen[0]
        if leig < 0. and maxoln==0:
            lambda1 = -leig
        else:
            lambda1 = 0.01

        if abs(lambda1)<0.005:
            lambda1 = 0.005

        # => grad in eigenvector basis <= #
        gqe = np.dot(tmph,self.gradq)

        print "printing eigenvalues"
        for i in eigen:
            print "%1.3f" % i,
        print
        print "printing gqe"
        for i in gqe:
            print "%1.3f" % i,
        print
    
        #if not self.isTSnode:
        #    dqe0[maxoln] = 0
        #else:
        path_overlap_e_g = gqe[maxoln]
        print ' gtse: {:1.4f} '.format(gqe[maxoln,0])

        #print "SCALE=%1.1f lambda=%1.3f" %(SCALE,lambda1)
        dqe0 = np.zeros(self.nicd)
        lambda0 = 0.0025
        dqe0[maxoln] = gqe[maxoln] / (abs(eigen[maxoln]) + lambda0)/SCALE
        for i in range(self.nicd):
            if i != maxoln:
                dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
           
        #print "printing dqe0"
        #print dqe0.T

        # => Convert step back to DLC basis <= #
        dq0 = np.dot(tmph,dqe0)
        dq0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dq0 ]
        dq = np.zeros((self.nicd,1))
        for i in range(self.nicd):
            dq[i,0] = dq0[i]
        
        #TODO can do something special here for seams by setting the second to last two values to zero
        #print "printing dq"
        #print dq.T

        return dq
