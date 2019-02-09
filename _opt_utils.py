import numpy as np
from units import *


class OStep_utils:
    def grad_to_q(self,grad):
        if self.FZN_ATOMS is not None:
            for a in [3*i for i in self.FZN_ATOMS]:
                grad[a:a+3]=0.
        gradq = np.dot(self.bmatti,grad)
        return gradq

    def check_overlap_good(self,opt_type=4):
        if (self.path_overlap < self.HESS_TANG_TOL_TS or self.gradrms > self.OPTTHRESH*20.) and opt_type==4: 
            return  False
        elif (self.path_overlap < self.HESS_TANG_TOL or self.gradrms > self.OPTTHRESH*20.) and opt_type==3: 
            return False
        else:
            return True

    def eigenvector_step(self,opt_type,ictan):
        # => Take Eigenvector Step <=#
        if opt_type in [0,1,2,5,6,7]:
            dq,opt_type = self.update_ic_eigen(opt_type)
        elif opt_type ==3:
            dq,opt_type = self.update_ic_eigen_h(ictan)
        elif opt_type==4:
            dq,opt_type = self.update_ic_eigen_ts(ictan)

        # regulate max overall step
        #TODO should this be after adding constraint step?
        self.smag = np.linalg.norm(dq)
        self.buf.write(" ss: %1.5f (DMAX: %1.3f)" %(self.smag,self.DMAX))
        if self.print_level>0:
            print(" ss: %1.5f (DMAX: %1.3f)" %(self.smag,self.DMAX)),
        if self.smag>self.DMAX:
            dq = np.fromiter(( xi*self.DMAX/self.smag for xi in dq), dq.dtype)
        dq= np.asarray(dq).reshape(self.nicd,1)

        return dq,opt_type

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


    def set_lambda1(self,eigen,opt_type):

        #TODO use opt_type 
        if opt_type in [3,4]:
            leig = eigen[1]  #! this is eigen[0] if update_ic_eigen() ### also diff values 
            if self.path_overlap_n!=0:
                leig = eigen[0]
            if leig < 0. and self.path_overlap_n==0:
                lambda1 = -leig
            else:
                lambda1 = 0.01
        else:
            leig = eigen[0]
            if leig < 0:
                lambda1 = -leig+0.015
            else:
                lambda1 = 0.005
        if abs(lambda1)<0.005: lambda1 = 0.005

        return lambda1

    def eigenvector_follow_step(self,SCALE,lambda1,gqe,eigen,opt_type):
        #assert self.opt_step in [3,4],"not implemented for other types"
        if self.print_level>0:
            print " SCALE=%1.1f lambda1=%1.3f" %(SCALE,lambda1)
        dqe0 = np.zeros(self.nicd)
        for i in range(self.nicd):
            if i != self.path_overlap_n:
                dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
       
        if opt_type==4:
            lambda0 = 0.0025
            dqe0[self.path_overlap_n] = gqe[self.path_overlap_n] / (abs(eigen[self.path_overlap_n]) + lambda0)/SCALE

        return dqe0

    def convert_dqe0_to_dq(self,dqe0,tmph):
        # => Convert step back to DLC basis <= #
        dq0 = np.dot(tmph.T,dqe0)

        #regulate max step size
        dq0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dq0 ]
        dq = np.reshape(dq0,(-1,1))
        return dq

    def print_values(self):
        #print "C"
        #for i in C:
        #    print " %1.3f" %i,
        #print
        #print "Ut"
        #with np.printoptions(threshold=np.inf):
        #    print(self.Ut)
        #print "dots"
        #for i in dots:
        #    print " %1.3f" %i,
        #print
        #print "Cn"
        #for i in Cn:
        #    print " %1.3f" %i,
        #print
        #print "Hint"
        #with np.printoptions(threshold=np.inf):
        #    print self.Hint

        #print "printing eigenvalues"
        #for i in eigen:
        #    print "%1.3f" % i,
        #print
        #print "tmph"
        #with np.printoptions(threshold=np.inf):
        #    print tmph
      
        #print "checking Ut"
        #with np.printoptions(threshold=np.inf):
        #    print self.Ut

        #print "printing gqe"
        #for i in gqe:
        #    print "%1.3f" % i,
        #print

        #print "printing dqe0"
        #print dqe0.T
        if self.print_level==2:
            print "eigen opt Hint ev:"
            print e
            print "gqe"
            print gqe.T
            print "dqe0"
            print dqe0
            print "dq0"
            print ["{0:0.5f}".format(i) for i in dq0]

        return

    def update_ic_eigen_h(self,ictan):
        raise RuntimeError, "Not using this anymore"
        SCALE = self.SCALEQN
        if self.newHess>0: SCALE = self.SCALEQN*self.newHess
        if SCALE > 10:
            SCALE = 10.
        # => get eigensolution of Hessian <= 
        eigen,tmph = np.linalg.eigh(self.Hint) #nicd,nicd
        tmph = tmph.T

        #TODO nneg should be self and checked
        nneg = 0
        for i in range(self.nicd):
            if eigen[i] < -0.01:
                nneg += 1

        norm = np.linalg.norm(ictan)
        C = ictan/norm
        dots = np.dot(self.Ut,C) #(nicd,numic)(numic,1)
        Cn = np.dot(self.Ut.T,dots) #(numic,nicd)(nicd,1) = numic,1
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm

        #=> Overlap metric <= #
        overlap = np.dot(np.dot(tmph,self.Ut),Cn) #(nicd,nicd)(nicd,num_ic)(num_ic,1) = (nicd,1)

        # Max overlap metrics
        self.maxol_w_Hess(overlap[0:4])

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1(eigen,3)

        # => if overlap is small use Cn as Constraint <= #
        #dq = self.check_overlap(ictan)
        if self.check_overlap_good(ictan):
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph,self.gradq)
            path_overlap_e_g = gqe[self.path_overlap_n]
            # => calculate eigenvector step <=#
            dqe0 = self.eigenvector_follow_step(SCALE,lambda1,gqe,eigen,3)
            # => Convert step back to DLC basis <= #
            dq = self.convert_dqe0_to_dq(dqe0,tmph)
        else:
            self.form_constrained_DLC(ictan)
            self.Hint = self.Hintp_to_Hint()
            dq = self.update_ic_eigen(1)

        return dq

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

        #self.Hint += (1.-phi)*Gms + phi*Gpsb
        change = (1.-phi)*Gms + phi*Gpsb
        #self.Hinv = np.linalg.inv(self.Hint)
        return change


    def compute_predE(self,dq0,nconstraints):
        # compute predicted change in energy 
        assert np.shape(dq0)==(self.nicd,1), "dq0 not (nicd,1)={}, dq0 is {}".format((self.nicd,1),np.shape(dq0))
        assert np.shape(self.gradq)==(self.nicd,1), "gradq not (nicd,1) "
        assert np.shape(self.Hint)==(self.nicd,self.nicd), "Hint not (nicd,nicd) "
        dEtemp = np.dot(self.Hint[:self.nicd-nconstraints,:self.nicd-nconstraints],dq0[:self.nicd-nconstraints])
        dEpre = np.dot(np.transpose(dq0[:self.nicd-nconstraints]),self.gradq[:self.nicd-nconstraints]) + 0.5*np.dot(np.transpose(dEtemp),dq0[:self.nicd-nconstraints])
        dEpre *=KCAL_MOL_PER_AU
        #if abs(dEpre)<0.05: dEpre = np.sign(dEpre)*0.05
        if self.print_level>1:
            print( "predE: %1.4f " % dEpre),
        return dEpre

    def update_ic_eigen(self,opt_type):
        assert opt_type not in [3,4],"use different updater."
        nconstraints=self.get_nconstraints(opt_type)
        SCALE =self.SCALEQN
        if self.newHess>0: SCALE = self.SCALEQN*self.newHess
        if self.SCALEQN>10.0: SCALE=10.0

        nicd_c = self.nicd-nconstraints
        temph = self.Hint[:nicd_c,:nicd_c]
        e,v_temp = np.linalg.eigh(temph)
        v_temp = v_temp.T
        
        # => get lambda1
        lambda1 = self.set_lambda1(e,opt_type)

        # => grad in eigenvector basis <= #
        gradq = self.gradq[:nicd_c,0]
        gqe = np.dot(v_temp,gradq)

        dqe0 = np.divide(-gqe,e+lambda1)/SCALE
        dqe0 = [ np.sign(i)*self.MAXAD if abs(i)>self.MAXAD else i for i in dqe0 ]

        # => Convert step back to DLC basis <= #
        dq = self.convert_dqe0_to_dq(dqe0,v_temp)
        dq_c = np.zeros((self.nicd,1))
        for i in range(nicd_c): dq_c[i,0] = dq[i]
        
        return dq_c,opt_type

    def walk_up(self,n):
        """ walk up the n'th DLC"""
        print " walking up the %i coordinate" % n
        #print "print gradq[n]", self.gradq[n]
        #print "type", type(self.gradq[n])
        assert isinstance(self.gradq[n,0],float), "gradq[n] is not float!"
        if self.print_level>0:
            print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        #self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = self.SCALEQN*1.0 
       
        dq = self.gradq[n,0]/SCALE
        if abs(dq) > self.MAXAD/SCALEW:
            dq = np.sign(dq)*self.MAXAD/SCALE

        return dq

    def get_nconstraints(self,opt_type):
        nconstraints=0
        if opt_type in [1,2]:
            nconstraints=1
        elif opt_type==5:
            nconstraints=2
        elif opt_type in [6,7]:
            nconstraints=3
        return nconstraints
