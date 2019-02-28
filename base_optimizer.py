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

    def set_lambda1(self,opt_type,eigen,maxoln=None):
        if opt_type == 'TS':
            leig = eigen[1]  #! this is eigen[0] if update_ic_eigen() ### also diff values 
            if maxoln!=0:
                leig = eigen[0]
            if leig < 0. and maxoln==0:
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

    def get_constraint_steps(self,c_obj,opt_type,g,params):
        nconstraints=self.get_nconstraints(opt_type)
        n=len(g)

        # Need nicd_DLC defined ... what  to do for Cartesian? TODO
        end = c_obj.nicd_DLC

        #return constraint_steps
        constraint_steps = np.zeros((n,1))
        # => ictan climb
        if opt_type=="CLIMB": 
            constraint_steps[end-1]=self.walk_up(g,end-1,params)
        # => MECI
        elif opt_type=='MECI': 
            constraint_steps[end-1] = self.dgrad_step(c_obj) #last vector is x
        elif opt_type=='SEAM':
            constraint_steps[end-2]=self.dgrad_step(c_obj)
        # => seam climb
        elif opt_type=='TS-SEAM':
            constraint_steps[end-2]=self.dgrad_step(c_obj)
            constraint_steps[end-1]=self.walk_up(g,end-1,params)

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

    def walk_up(self,g,n,params):
        """ walk up the n'th DLC"""
        #assert isinstance(g[n],float), "gradq[n] is not float!"
        #if self.print_level>0:
        #    print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        #self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = params.options['SCALEQN']
        dq = g[n,0]/SCALE
        print " walking up the %i coordinate = %1.4f" % (n,dq)
        if abs(dq) > params.options['MAXAD']/SCALEW:
            dq = np.sign(dq)*params.options['MAXAD']/SCALE

        return dq

    def step_controller(self,step,ratio,gradrms,pgradrms,params):
        # => step controller controls DMAX/DMIN <= #
        params.options['DMAX'] = step
        if (ratio<0.25 or ratio>1.5): #can also check that dEpre >0.05?
            if step<params.options['DMAX']:
                params.options['DMAX'] = step/1.2
            else:
                params.options['DMAX'] = params.options['DMAX']/1.2
        elif ratio>0.75 and step > params.options['DMAX'] and gradrms<(pgradrms*1.35):
            #and ratio<1.25 and  
            #if self.print_level>0:
            if True:
                print("increasing DMAX"),
            #self.buf.write(" increasing DMAX")
            if step > params.options['DMAX']:
                if True:
                    print " wolf criterion increased stepsize... ",
                    print " setting DMAX to wolf  condition...?"
                    params.options['DMAX']=step
            params.options['DMAX']=params.options['DMAX']*1.2 + 0.01
            if params.options['DMAX']>0.25:
                params.options['DMAX']=0.25
        
        if params.options['DMAX']<params.DMIN:
            params.options['DMAX']=params.DMIN
        print " DMAX", params.options['DMAX']

    def eigenvector_step(self,c_obj,params,g,n):

        SCALE =params.options['SCALEQN']
        if c_obj.newHess>0: SCALE = params.options['SCALEQN']*c_obj.newHess
        if params.options['SCALEQN']>10.0: SCALE=10.0
        
        # convert to eigenvector basis
        temph = c_obj.Hint[:n,:n]
        e,v_temp = np.linalg.eigh(temph)
        v_temp = v_temp.T
        gqe = np.dot(v_temp,g[:n])
        
        lambda1 = self.set_lambda1('NOT-TS',e) 

        dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
        dqe0 = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dqe0 ]
        
        # => Convert step back to DLC basis <= #
        dq_tmp = np.dot(v_temp.T,dqe0)
        dq_tmp = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dq_tmp ]
        dq = np.zeros((c_obj.nicd_DLC,1))
        for i in range(n): dq[i,0] = dq_tmp[i]

        return dq

    # need to modify this only for the DLC region
    def TS_eigenvector_step(self,c_obj,params,g,ictan):
        '''
        Takes an eigenvector step using the Bofill updated Hessian ~1 negative eigenvalue in the
        direction of the reaction path.

        '''
        SCALE =params.options['SCALEQN']
        if c_obj.newHess>0: SCALE = params.options['SCALEQN']*c_obj.newHess
        if params.options['SCALEQN']>10.0: SCALE=10.0

        # constraint vector
        norm = np.linalg.norm(ictan)
        C = ictan/norm
        dots = np.dot(c_obj.Ut,C) #(nicd,numic)(numic,1)
        Cn = np.dot(c_obj.Ut.T,dots) #(numic,nicd)(nicd,1) = numic,1
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm

        # => get eigensolution of Hessian <= 
        eigen,tmph = np.linalg.eigh(c_obj.Hint) #nicd,nicd 
        tmph = tmph.T

        #TODO nneg should be self and checked
        params.nneg = 0
        for i in range(c_obj.nicd_DLC):
            if eigen[i] < -0.01:
                params.nneg += 1

        #=> Overlap metric <= #
        overlap = np.dot(np.dot(tmph,c_obj.Ut),Cn) 

        print "overlap", overlap[:4]
        # Max overlap metrics
        path_overlap,maxoln = self.maxol_w_Hess(overlap[0:4])
        print " t/ol %i: %3.2f" % (maxoln,path_overlap)

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1('TS',eigen,maxoln)

        maxol_good=True
        if path_overlap < params.options['HESS_TANG_TOL_TS']:
            maxol_good = False

        if maxol_good:
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph,g)
            path_overlap_e_g = gqe[maxoln]
            print ' gtse: {:1.4f} '.format(path_overlap_e_g[0])
            # => calculate eigenvector step <=#
            dqe0 = np.zeros((c_obj.nicd_DLC,1))
            for i in range(c_obj.nicd_DLC):
                if i != maxoln:
                    dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
            lambda0 = 0.0025
            dqe0[maxoln] = gqe[maxoln] / (abs(eigen[maxoln]) + lambda0)/SCALE

            # => Convert step back to DLC basis <= #
            dq = np.dot(tmph.T,dqe0)  # should it be transposed?
            dq = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dq ]
        else:
            # => if overlap is small use Cn as Constraint <= #
            c_obj.form_constrained_DLC(ictan) 
            c_obj.Hint = c_obj.Hintp_to_Hint()
            dq = self.eigenvector_step(c_obj,params,g,c_obj.nicd_DLC-1)  #hard coded!

        return dq,maxol_good

    def maxol_w_Hess(self,overlap):
        # Max overlap metrics
        absoverlap = np.abs(overlap)
        path_overlap = np.max(absoverlap)
        path_overlap_n = np.argmax(absoverlap)
        return path_overlap,path_overlap_n
