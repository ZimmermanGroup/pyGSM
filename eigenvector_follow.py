import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack
from opt_parameters import parameters
from dlc import DLC
from cartesian import Cartesian
from base_optimizer import base_optimizer
import StringIO

class eigenvector_follow(base_optimizer):

    def __init__(self):
        return


    def optimize(self,c_obj,params,refE,opt_steps=3,ictan=None):
        self.disp = 1000.
        self.Ediff = 1000.
        
        opt_type=params.options['opt_type']
        self.check_inputs(c_obj,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)

        self.buf = StringIO.StringIO()

        # the number of coordinates
        if c_obj.__class__.__name__=='Cartesian':
            n = len(x) - nconstraints
        else:
            # number of internal coordinate dimensions in the IC_region 
            # WARNING!!! -- don't do eigenvector opt in the CART_REGION
            n =  c_obj.nicd_DLC-nconstraints 
        
        Linesearch=params.options['Linesearch']
        
        update_hess=False
        for ostep in range(opt_steps):

            print " On opt step ",ostep+1
       
            # update DLC  --> this changes q
            if not c_obj.__class__.__name__=='Cartesian':
                if opt_type != 'TS':
                    c_obj.update_DLC(opt_type,ictan)
                else:
                    c_obj.bmatp = c_obj.bmatp_create()
        
            if update_hess == True:
                if opt_type != 'TS':
                    c_obj.update_Hessian('BFGS')
                else:
                    c_obj.update_Hessian('BOFILL')
            elif opt_type!='TS':
                c_obj.Hint = c_obj.Hintp_to_Hint()

            update_hess = True

            # copy of coordinates (after update)
            x = np.copy(c_obj.q)
        
            # Evaluate the function value and its gradient.
            #fx,g = c_obj.proc_evaluate(x,nconstraints)
            proc_result = c_obj.proc_evaluate(x,n)
            fx=proc_result['fx']
            g = proc_result['g']
        
            # normalize 
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0
   
            if self.converged(g,n,params):
                break
        
            # => Form eigenvector step <= #
            if isinstance(c_obj,Cartesian):
                raise NotImplementedError
            else:
                if params.options['opt_type']!='TS':
                    dq = self.eigenvector_step(c_obj,params,g,n)
                else:
                    dq,maxol_good = self.TS_eigenvector_step(c_obj,params,g,ictan)
                    if not maxol_good:
                        nconstraints=1
                        opt_type='CLIMB'
            
            step = np.linalg.norm(dq)
            dq = dq/step #normalize

            if step>params.options['DMAX']:
                step=params.options['DMAX']
                print " reducing step, new step =",step
        
            # store
            xp = x.copy()
            gp = g.copy()
            fxp = fx
            pgradrms = np.sqrt(np.dot(g.T,g)/n)

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(c_obj,opt_type,g,params)
        
            print " ### Starting  line search ###"
            ls = Linesearch(n, x, fx, g, dq, step, xp, gp,constraint_steps, params,c_obj.proc_evaluate)
            print " ## Done line search"
   
            # revert to the privious point
            if ls['status'] < 0:
                x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']
        
            # get values at new point
            step = ls['step']
            x = ls['x']
            #fx = ls['fx']
            #g = ls['g']
            proc_results=ls['proc_results']
            fx=proc_results['fx']
            g = proc_results['g']

            # calculate gradrms
            gradrms = np.sqrt(np.dot(g.T,g)/n)

            #assert np.linalg.norm(dq) == 1.0, " not normalized"
            print "normalization ", np.linalg.norm(dq)

            # calculate predicted value
            dEtemp = np.dot(c_obj.Hint[:n,:n],dq[:n]*step)
            dEpre = np.dot(np.transpose(dq[:n]*step),g[:n]) + 0.5*np.dot(np.transpose(dEtemp),dq[:n]*step)
            dEpre *=KCAL_MOL_PER_AU
            # constraint contribution
            for nc in range(nconstraints):
                dEpre += g[-nc-1]*constraint_steps[-nc-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq WARNING THIS IS ONLY GOOD FOR DLC
        
            # check goodness of step
            dEstep = fx - fxp
            ratio = dEstep/dEpre
        
            # control step size 
            self.step_controller(step,ratio,gradrms,pgradrms,params)

            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = (fx -fxp) / KCAL_MOL_PER_AU
            #print " maximum displacement component (au)", self.disp
            #print " Ediff (au)", self.Ediff
        
            # report the progress 
            #-- important for DLC since it updates the variables 
            # ... might be better to make separate function for that ...
            c_obj.append_data(x,proc_results, xnorm, gradrms, step)

            self.buf.write(' Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,dEpre,ratio,gradrms,step,params.options['DMAX']))

        
        print(self.buf.getvalue())
        return fx



if __name__=='__main__':
    from qchem import QChem
    import pybel as pb
    from pes import PES
    from dlc import DLC

    basis="sto-3g"
    nproc=1

    filepath="examples/tests/bent_benzene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    
    # => DLC constructor <= #
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
    ic1.form_unconstrained_DLC()
    ic1.make_Hint()
    ic1.Hint = ic1.Hintp_to_Hint()

    param = parameters.from_options(opt_type='UNCONSTRAINED')
    ef = eigenvector_follow(c_obj=ic1,params=param,opt_steps=20)

    print ic1.fx
    manage_xyz.write_xyzs('prc.xyz',ic1.geoms,scale=1.0)

