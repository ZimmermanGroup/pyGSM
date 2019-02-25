import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack
from opt_parameters import parameters
from dlc import DLC
from cartesian import Cartesian



class base_optimizer(object):
    ''' some common functions that the children can use (ef, cg, hybrid ef/cg, etc). 
    e.g. walk_up, dgrad_step, what else?
    '''

    def __init__(self):
        return

    def optimize(self,c_obj,params,opt_steps=3,ictan=None):
        raise NotImplementedError

    def get_nconstraints(self,opt_type):
        if opt_type in ["ICTAN", "CLIMB"]:
            nconstraints = 1
        elif opt_type in ['MECI']:
            nconstraints=2
        elif opt_type in ['SEAM','TS-SEAM']:
            nconstraints=3
        else:
            nconstraints=0
        return nconstraints

    def check_inputs(self,c_obj,opt_type,ictan):
        if opt_type in ['MECI','SEAM','TS-SEAM']:
            assert c_obj.PES.lot.do_coupling==True,"Turn do_coupling on."                   
        elif opt_type not in ['MECI','SEAM','TS-SEAM']: 
            assert c_obj.PES.lot.do_coupling==False,"Turn do_coupling off."                 
        if opt_type=="UCONSTRAINED":  
            assert ictan==None
        if opt_type in ['ICTAN','CLIMB','TS', 'SEAM','TS-SEAM']  and ictan.any()==None:
            raise RuntimeError, "Need ictan"
        if opt_type in ['TS','TS-SEAM']:
            assert c_obj.isTSnode,"only run climb and eigenvector follow on TSnode."  

    def converged(self,g,n,params):
        # check if finished
        gradrms = np.sqrt(np.dot(g[:n].T,g[:n])/n)
        #print "current gradrms= %r au" % gradrms
        #print "gnorm =",gnorm
        
        gmax = np.max(g[:n])/ANGSTROM_TO_AU
        #print "maximum gradient component (au)", gmax

        if gradrms <= params.conv_grms  or \
            (self.disp <= params.conv_disp and self.Ediff <= params.conv_Ediff) or \
            (gmax <= params.conv_gmax and self.Ediff <= params.conv_Ediff):
            #print '[INFO] converged'
            return True
        return False

    def set_lambda1(self,opt_type,eigen):
        if opt_type == 'TS':
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

    def get_constraint_steps(self,c_obj,opt_type,g):
        nconstraints=self.get_nconstraints(opt_type)
        n=len(g)

        # Need nicd_DLC defined ... what  to do for Cartesian? TODO
        end = c_obj.nicd_DLC

        #return constraint_steps
        constraint_steps = np.zeros((n,1))
        # => ictan climb
        if opt_type=="ICTAN": 
            constraint_steps[end-1]=self.walk_up(g,end-1)
        # => MECI
        elif opt_type=='MECI': 
            constraint_steps[end-1] = self.dgrad_step(c_obj) #last vector is x
        elif opt_type=='SEAM':
            constraint_steps[end-2]=self.dgrad_step(c_obj)
        # => seam climb
        elif opt_type=='TS-SEAM':
            constraint_steps[end-2]=self.dgrad_step(c_obj)
            constraint_steps[end-1]=self.walk_up(g,end-1)

        return constraint_steps

    def dgrad_step(self,c_obj):
        """ takes a linear step along dgrad"""
        dgrad = c_obj.PES.get_dgrad(self.geom)
        dgradq = c_obj.grad_to_q(dgrad)
        norm_dg = np.linalg.norm(dgradq)
        #if self.print_level>0:
        #    print " norm_dg is %1.4f" % norm_dg,
        #    print " dE is %1.4f" % self.PES.dE,

        dq = -c_obj.PES.dE/KCAL_MOL_PER_AU/norm_dg 
        if dq<-0.075:
            dq=-0.075

        return dq

    def walk_up(self,g,n):
        """ walk up the n'th DLC"""
        print " walking up the %i coordinate" % n
        assert isinstance(g,float), "gradq[n] is not float!"
        #if self.print_level>0:
        #    print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        #self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = self.SCALEQN*1.0 
        dq = g[n,0]/SCALE
        if abs(dq) > self.options['MAXAD']/SCALEW:
            dq = np.sign(dq)*self.options['MAXAD']/SCALE

        return dq

