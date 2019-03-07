import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack,NoLineSearch
from opt_parameters import parameters
from dlc import DLC
from cartesian import Cartesian
from base_optimizer import base_optimizer
import StringIO

class eigenvector_follow(base_optimizer):

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return eigenvector_follow(eigenvector_follow.default_options().set_values(kwargs))

    def optimize(self,c_obj,refE,opt_steps=3,ictan=None):

        # stash/initialize some useful attributes
        self.disp = 1000.
        self.Ediff = 1000.
        opt_type=self.options['opt_type']
        self.check_inputs(c_obj,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        x = np.copy(c_obj.q)
        Linesearch=self.options['Linesearch']
        if opt_type=='TS':
            Linesearch=NoLineSearch
        print "opt_type ",opt_type
        self.buf = StringIO.StringIO()
        if c_obj.__class__.__name__=='Cartesian':
            n = len(x) - nconstraints
        else:
            # number of internal coordinate dimensions in the IC_region 
            # WARNING!!! -- don't do eigenvector opt in the CART_REGION
            n =  c_obj.nicd_DLC-nconstraints 
            self.x_prim=np.zeros((c_obj.num_ics,1),dtype=float)
            self.g_prim=np.zeros((c_obj.num_ics,1),dtype=float)

            self.g_prim_c=np.zeros((c_obj.num_ics,1),dtype=float)
        
        # Evaluate the function value and its gradient.
        proc_result = c_obj.proc_evaluate(x,n)
        fx=proc_result['fx']
        g = proc_result['g']
        update_hess=False
        if self.converged(g,n):
            print " already at min"
            return fx

        for ostep in range(opt_steps):
            print " On opt step ",ostep+1
       
            # update Hess
            if update_hess:
                if opt_type!='TS':
                    self.update_Hessian(c_obj,'BFGS')
                else:
                    self.update_Hessian(c_obj,'BOFILL')
            else:
                if not c_obj.__class__.__name__=='Cartesian' and opt_type!='TS':
                    self.Hint=c_obj.Hintp_to_Hint()
            update_hess = True

            # => Form eigenvector step <= #
            if isinstance(c_obj,Cartesian):
                raise NotImplementedError
            else:
                if self.options['opt_type']!='TS':
                    dq = self.eigenvector_step(c_obj,g,n)
                else:
                    dq,maxol_good = self.TS_eigenvector_step(c_obj,g,ictan)
                    if not maxol_good:
                        nconstraints=1
                        opt_type='CLIMB'
            step = np.linalg.norm(dq)
            dq = dq/step #normalize
            if step>self.options['DMAX']:
                step=self.options['DMAX']
                print " reducing step, new step =",step
        
            # store values
            xp = x.copy()
            gp = g.copy()
            fxp = fx
            pgradrms = np.sqrt(np.dot(g.T,g)/n)
            if not c_obj.__class__.__name__=='Cartesian':
                xp_prim = self.x_prim.copy()
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(c_obj,opt_type,g)
        
            print " ### Starting  line search ###"
            ls = Linesearch(n, x, fx, g, dq, step, xp, gp,constraint_steps,self.linesearch_parameters,c_obj.proc_evaluate)
            if ls['status'] ==-2:
                x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']
            print " ## Done line search"
        
            # get values from linesearch
            step = ls['step']
            x = ls['x']
            proc_results=ls['proc_results']
            fx=proc_results['fx']
            g = proc_results['g']

            # calculate predicted value from old g using step size from linesearch
            dEtemp = np.dot(self.Hint[:n,:n],dq[:n]*step)
            dEpre = np.dot(np.transpose(dq[:n]*step),gp[:n]) + 0.5*np.dot(np.transpose(dEtemp),dq[:n]*step)
            dEpre *=KCAL_MOL_PER_AU
            for nc in range(nconstraints):
                dEpre += gp[-nc-1]*constraint_steps[-nc-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq WARNING THIS IS ONLY GOOD FOR DLC

            # control step size 
            dEstep = fx - fxp
            ratio = dEstep/dEpre
            gradrms = np.sqrt(np.dot(g.T,g)/n)
            self.step_controller(step,ratio,gradrms,pgradrms)
        
            # report the progress 
            #-- important for DLC since it updates the variables 
            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = (fx -fxp) / KCAL_MOL_PER_AU
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0
            c_obj.append_data(x,proc_results, xnorm, gradrms, step)

            # save variables for update Hessian! 
            if not c_obj.__class__.__name__=='Cartesian':
                self.g_prim = proc_results['gradqprim']
                self.x_prim = proc_results['qprim']
                self.dx = x-xp
                self.dg = g - gp
                #self.dx_prim = c_obj.primitive_internal_difference(self.x_prim,xp_prim)
                self.dx_prim = np.dot(np.transpose(c_obj.Ut),dq*step)  #not exactly correct but very close
                self.dg_prim = self.g_prim - gp_prim
                #self.dx_constraint = np.dot(c_obj.Ut,constraint_steps)
                #self.dg_consraint = self.g_prim_c - gp_prim_c
            else:
                raise NotImplementedError

            self.buf.write(' Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,dEpre,ratio,gradrms,step,self.options['DMAX']))

            if self.converged(g,n):
                break

            #update DLC  --> this changes q, g, Hint
            if not c_obj.__class__.__name__=='Cartesian':
                if opt_type != 'TS':
                    c_obj.update_DLC(opt_type,ictan)
                    x = np.copy(c_obj.q)
                    gx = c_obj.PES.get_gradient(c_obj.geom)
                    g = c_obj.grad_to_q(gx)
                    self.Hint=c_obj.Hintp_to_Hint()
                else:
                    c_obj.bmatp = c_obj.bmatp_create()
        
        print(self.buf.getvalue())
        return fx



if __name__=='__main__':
    from qchem import QChem
    import pybel as pb
    from pes import PES
    from dlc import DLC

    basis="6-31G*"
    nproc=8

    filepath="examples/tests/bent_benzene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    
    # => DLC constructor <= #
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
    ic1.form_unconstrained_DLC()
    ic1.make_Hint()

    param = parameters.from_options(opt_type='UNCONSTRAINED')
    ef = eigenvector_follow.from_options()
    ef.optimize(c_obj=ic1,refE=0.,opt_steps=2)

    print ic1.fx
    manage_xyz.write_xyzs('prc.xyz',ic1.geoms,scale=1.0)

