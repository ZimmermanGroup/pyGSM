import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack
from opt_parameters import parameters
from dlc import DLC
from cartesian import Cartesian
from base_optimizer import base_optimizer


class hybrid_ef_cg(base_optimizer):

    ''' Do eigenvector following in the IC region. Do  conjugate gradient in the 
    Cartesian region'''

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return hybrid_ef_cg(hybrid_ef_cg.default_options().set_values(kwargs))

    def optimize(self,c_obj,params,opt_steps=3,micro_iterations=1,ictan=None):
        self.disp = 1000.
        self.Ediff = 1000.
        
        opt_type=params.options['opt_type']
        self.check_inputs(c_obj,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)

        # copy of coordinates
        x = np.copy(c_obj.q)
        
        # the number of coordinates
        n = len(x) # total number of coords
        nicd_DLC =  c_obj.nicd_DLC-nconstraints  # number of internal coords  minus constraints
        nicd_CART = n - nicd_DLC                 # number of internal coordinates in the CART region
        CSTART=nicd_DLC+nconstraints

        Linesearch=params.options['Linesearch']

        update_hess=False
        for i in range(opt_steps):
            print " On opt step ",i
        
            # update DLC  --> this changes q in DLC only
            if not c_obj.__class__.__name__=='Cartesian':
                if opt_type != 'TS':
                    c_obj.update_DLC(opt_type,ictan)
                else:
                    c_obj.bmatp = c_obj.bmatp_create()
       

            # update Bfgs only happens in DLC
            if update_hess == True:
                c_obj.update_Hessian('BFGS')
            update_hess = True
        
            ## Evaluate the function value and its gradient. NEED API to do this efficiently.
            fx_qm, g_qm = c_obj.QM_eNg_evaluate(x[:nicd_DLC])
            fx_mm, g_mm = c_obj.MM_eNg_evaluate(x[CSTART:])
           
        
            ################################################# 
            ######   STEP 1: Is MM CG optimization   ########
            ###### ====> Conjugate Gradient <===== ##########
            ################################################# 

            for i in xrange(microiterations):
                SCALE =params.options['SCALEQN']
                if c_obj.newHess>0: SCALE = params.options['SCALEQN']*c_obj.newHess
                if params.options['SCALEQN']>10.0: SCALE=10.0
        
                if self.initial_step==True:
                    # d: store the negative gradient of the object function on point x.
                    d = -g_mm
                    # compute the initial step
                    self.mm_step = 1.0 / np.sqrt(np.dot(d.T, d))
                    # set initial step to false
                    self.initial_step=False
                else:   
                    # Fletcher-Reeves formula for Beta
                    # http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method       
                    dnew = -g_mm
                    deltanew = np.dot(dnew.T,dnew)
                    deltaold=np.dot(d.T,d)
                    beta = deltanew/deltaold
                    d = dnew + beta*dnew
                    # not using this step size, using last iterations step size from linesearch

                # store
                xp = x.copy()
                gp_mm = g_mm.copy()
                fxp_mm = fx_mm

                print " ### Starting  line search for MM region ###"
                ls = Linesearch(nicd_CART, x[CSTART:], fx_mm, g_mm, d, self.mm_step, xp[CSTART], gp_mm,0, params,MM_eNg_evaluate)
                print "done line search"

                # revert to the privious point
                if ls['status'] < 0:
                    self.x = xp.copy()
                    print '[ERROR] the point return to the privious point'
                    return ls['status']
                fx_mm = ls['fx']
                self.mm_step = ls['step']
                x[CSTART:] = ls['x']
                g_mm = ls['g']

                self.disp = np.max(x - xp)/ANGSTROM_TO_AU
                self.Ediff = fx_mm -fxp_mm / KCAL_MOL_PER_AU

            # TODO need to re-evaluate the polarization according to Schlegel ?
            # TODO special linesearch algorithm designed by Schlegel ?


            ################################################# 
            ########### =====>  Step 2:   <====== ###########
            ######  ====> Eigenvector Following <===== ######
            ################################################# 
            # convert to eigenvector basis
            temph = c_obj.Hint[:nicd_DLC,:nicd_DLC]
            e,v_temp = np.linalg.eigh(temph)
            v_temp = v_temp.T
            gqe = np.dot(v_temp,g[:nicd_DLC])
            lambda1 = self.set_lambda1(opt_type,e)
            dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
            dqe0 = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dqe0 ]
            # Convert step back to DLC basis #
            dq_tmp = np.dot(v_temp.T,dqe0)
            #regulate max step size and shape
            dq_tmp = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dq_tmp ]
            dq = np.zeros((CSTART,1))
            for i in range(CSTART): dq[i,0] = dq_tmp[i]

            print "step=",step
            if step>params.options['DMAX']:
                step=params.options['DMAX']
                print " reducing step, new step =",step

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(c_obj,opt_type,g)
            # => add constraint_step to step <= #
            dq += constraint_steps

            # calculate predicted value
            dEtemp = np.dot(c_obj.Hint[:nicd_DLC,:nicd_DLC],dq[:nicd_DLC]*step)
            dEpre = np.dot(np.transpose(dq[:nicd_DLC]*step),g[:nicd_DLC]) + 0.5*np.dot(np.transpose(dEtemp),dq[:nicd_DLC]*step)
            dEpre *=KCAL_MOL_PER_AU
            # constraint contribution
            for nc in range(nconstraints):
                dEpre += g[nicd_DLC-nc-1]*dq[nicd_DLC-nc-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq

            # store
            xp = x.copy()
            gp_qm = g_qm.copy()
            fxp_qm = fx_qm

            print " ### Starting  line search for QM / IC_REGION###"
            ls = Linesearch(nicd_DLC, x[:nicd_DLC], fx_qm, g_qm, dq, step, xp[:nicd_DLC], gp_qm,constraint_steps, params,c_obj.QMM_eNg_evaulate)
            print " ## Done line search"
   
            # revert to the privious point
            if ls['status'] < 0:
                x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']

            # get values at new point
            fx_qmm = ls['fx']
            step = ls['step']
            x[:CSTART] = ls['x']
            g_qmm = ls['g']
            gradrms = np.sqrt(np.dot(g_qmm.T,g_mm)/nicd_DLC)
        
            # check goodness of step
            dEstep = fx_qmm - fxp_qmm
            ratio = dEstep/dEpre
        
            print "ratio is =",ratio
            print "dEpre is =",dEpre
            print "dEstep is =",dEstep
        
            # => step controller controls DMAX/DMIN <= #
            params.options['DMAX'] = step
            if (ratio<0.25 or ratio>1.5): #can also check that dEpre >0.05?
                if step<params.options['DMAX']:
                    params.options['DMAX'] = step/1.2
                else:
                    params.options['DMAX'] = params.options['DMAX']/1.2
            elif ratio>0.75 and ratio<1.25 and step > params.options['DMAX'] and gradrms<(pgradrms*1.35):
                #if self.print_level>0:
                #    print("increasing DMAX"),
                #self.buf.write(" increasing DMAX")
                params.options['DMAX']=params.options['DMAX']*1.2 + 0.01
                if params.options['DMAX']>0.25:
                    params.options['DMAX']=0.25
        
            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = (fx_qmm -fxp_qmm) / KCAL_MOL_PER_AU
            print "maximum displacement component (au)", self.disp
            print " Ediff (au)", self.Ediff

