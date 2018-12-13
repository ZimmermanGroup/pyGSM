import numpy as np


class OStep_utils:

    def check_overlap(self,ictan):
        if (self.path_overlap < self.HESS_TANG_TOL_TS or self.gradrms > self.OPTTHRESH*20.) and self.opt_type==4: 
            self.form_constrained_DLC(ictan) #not sure if this can be in add-in?
            self.Hintp_to_Hint()
            dq = self.update_ic_eigen(1)
            dq[-1]=self.walk_up(self.nicd-1)
            return dq
        elif (self.path_overlap < self.HESS_TANG_TOL or self.gradrms > self.OPTTHRESH*20.) and self.opt_type==3: 
            self.form_constrained_DLC(ictan)
            self.Hintp_to_Hint()
            dq = self.update_ic_eigen(1)
            return dq
        else:
            return None


    def set_lambda1(self,eigen):

        #TODO use opt_type 
        if self.opt_type in [3,4]:
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

    def eigenvector_follow_step(self,SCALE,lambda1,gqe,eigen):
        #assert self.opt_step in [3,4],"not implemented for other types"
        if self.print_level>0:
            print "SCALE=%1.1f lambda1=%1.3f" %(SCALE,lambda1)
        dqe0 = np.zeros(self.nicd)
        for i in range(self.nicd):
            if i != self.path_overlap_n:
                dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
       
        if self.opt_type==4:
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
