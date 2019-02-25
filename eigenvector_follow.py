import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack
from opt_parameters import parameters
from dlc import DLC
from cartesian import Cartesian
from base_optimizer import base_optimizer

class eigenvector_follow(base_optimizer):

    def __init__(self):
        return


    #def maxol_w_Hess(self,overlap):
    #    # Max overlap metrics
    #    absoverlap = np.abs(overlap)
    #    self.path_overlap = np.max(absoverlap)
    #    self.path_overlap_n = np.argmax(absoverlap)
    #    if self.print_level>-1:
    #        print " t/ol %i: %3.2f" % (self.path_overlap_n,self.path_overlap)
    #    self.buf.write(" t/ol %i: %3.2f" % (self.path_overlap_n,self.path_overlap))

    #def update_ic_eigen_ts(self,ictan):
    #    """ this method follows the overlap with reaction tangent"""
    #    opt_type=3
    #    lambda1 = 0.
    #    SCALE = self.SCALEQN
    #    if self.newHess>0: SCALE = self.SCALEQN*self.newHess
    #    if SCALE > 10:
    #        SCALE = 10.
    #    #TODO buf print SCALE

    #    #testing
    #    unit_test=False
    #    if unit_test:
    #        self.prepare_unit_test()
    #    else:
    #        norm = np.linalg.norm(ictan)
    #        C = ictan/norm
    #        dots = np.dot(self.Ut,C) #(nicd,numic)(numic,1)
    #        Cn = np.dot(self.Ut.T,dots) #(numic,nicd)(nicd,1) = numic,1
    #        norm = np.linalg.norm(Cn)
    #        Cn = Cn/norm
    #   
    #    # => get eigensolution of Hessian <= 
    #    eigen,tmph = np.linalg.eigh(self.Hint) #nicd,nicd
    #    tmph = tmph.T

    #    #TODO nneg should be self and checked
    #    nneg = 0
    #    for i in range(self.nicd):
    #        if eigen[i] < -0.01:
    #            nneg += 1

    #    #=> Overlap metric <= #
    #    overlap = np.dot(np.dot(tmph,self.Ut),Cn) #(nicd,nicd)(nicd,num_ic)(num_ic,1) = (nicd,1)
    #    #print " printing overlaps ", overlap[:4].T

    #    # Max overlap metrics
    #    self.maxol_w_Hess(overlap[0:4])

    #    # => set lamda1 scale factor <=#
    #    lambda1 = self.set_lambda1(eigen,4)

    #    # => if overlap is small use Cn as Constraint <= #
    #    if self.check_overlap_good(opt_type=4):
    #        # => grad in eigenvector basis <= #
    #        gqe = np.dot(tmph,self.gradq)
    #        path_overlap_e_g = gqe[self.path_overlap_n]
    #        if self.print_level>0:
    #            print ' gtse: {:1.4f} '.format(path_overlap_e_g[0])
    #        self.buf.write(' gtse: {:1.4f}'.format(path_overlap_e_g[0]))
    #        # => calculate eigenvector step <=#
    #        dqe0 = self.eigenvector_follow_step(SCALE,lambda1,gqe,eigen,4)
    #        # => Convert step back to DLC basis <= #
    #        dq = self.convert_dqe0_to_dq(dqe0,tmph)
    #    else:
    #        self.form_constrained_DLC(ictan) 
    #        self.Hint = self.Hintp_to_Hint()
    #        dq,tmp = self.update_ic_eigen(1)
    #        opt_type=2

    #    return dq,opt_type

    #
    ## move to eigenvector_follow
    #def check_overlap_good(self):
    #    if self.path_overlap < self.HESS_TANG_TOL_TS:            
    #        return  False
    #    elif (self.path_overlap < self.HESS_TANG_TOL or self.gradrms > self.OPTTHRESH*20.) and opt_type==3: 
    #        return False
    #    else:
    #        return True


    def optimize(self,c_obj,params,opt_steps=3,ictan=None):
        self.disp = 1000.
        self.Ediff = 1000.
        
        opt_type=params.options['opt_type']
        self.check_inputs(c_obj,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)

        # copy of coordinates
        x = np.copy(c_obj.q)
        
        # the number of coordinates
        if c_obj.__class__.__name__=='Cartesian':
            n = len(x) - nconstraints
        else:
            # number of internal coordinate dimensions in the IC_region 
            #-- don't do eigenvector opt in the CART_REGION
            n =  c_obj.nicd_DLC-nconstraints 
        
        Linesearch=params.options['Linesearch']
        
        update_hess=False
        for i in range(opt_steps):
            print " On opt step ",i
        
            # update DLC  --> this changes q
            if not c_obj.__class__.__name__=='Cartesian':
                if opt_type != 'TS':
                    c_obj.update_DLC(opt_type,ictan)
                else:
                    c_obj.bmatp = c_obj.bmatp_create()
        
            if update_hess == True:
                c_obj.update_Hessian('BFGS')
            update_hess = True
        
            # Evaluate the function value and its gradient.
            fx,g = c_obj.proc_evaluate(x)
        
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
                SCALE =params.options['SCALEQN']
                if c_obj.newHess>0: SCALE = params.options['SCALEQN']*c_obj.newHess
                if params.options['SCALEQN']>10.0: SCALE=10.0
        
                # convert to eigenvector basis
                temph = c_obj.Hint[:n,:n]
                e,v_temp = np.linalg.eigh(temph)
                v_temp = v_temp.T
                gqe = np.dot(v_temp,g[:n])
        
                lambda1 = self.set_lambda1(opt_type,e)

                dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
                dqe0 = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dqe0 ]
        
                # => Convert step back to DLC basis <= #
                dq_tmp = np.dot(v_temp.T,dqe0)
                #regulate max step size and shape
                dq_tmp = [ np.sign(i)*params.options['MAXAD'] if abs(i)>params.options['MAXAD'] else i for i in dq_tmp ]
                dq = np.zeros((n,1))
                for i in range(n): dq[i,0] = dq_tmp[i]
        
            step = np.linalg.norm(dq)
            dq = dq/step #normalize
        
            print "step=",step
            if step>params.options['DMAX']:
                step=params.options['DMAX']
                print " reducing step, new step =",step
        
            # store
            xp = x.copy()
            gp = g.copy()
            fxp = fx
            pgradrms = np.sqrt(np.dot(g.T,g)/n)

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(c_obj,opt_type,g)
            # => add constraint_step to step <= #
            dq += constraint_steps
        
            print " ### Starting  line search ###"
            ls = Linesearch(n, x, fx, g, dq, step, xp, gp,constraint_steps, params,c_obj.proc_evaluate)
            print " ## Done line search"
   
            # revert to the privious point
            if ls['status'] < 0:
                x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']
        
            # calculate predicted value
            dEtemp = np.dot(c_obj.Hint[:nicd_DLC,:nicd_DLC],dq[:nicd_DLC]*step)
            dEpre = np.dot(np.transpose(dq[:nicd_DLC]*step),g[:nicd_DLC]) + 0.5*np.dot(np.transpose(dEtemp),dq[:nicd_DLC]*step)
            dEpre *=KCAL_MOL_PER_AU
            # constraint contribution
            for nc in range(nconstraints):
                dEpre += g[nicd_DLC-nc-1]*dq[nicd_DLC-nc-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq
            # get values at new point
            fx = ls['fx']
            step = ls['step']
            x = ls['x']
            g = ls['g']
            gradrms = np.sqrt(np.dot(g.T,g)/n)
        
            # check goodness of step
            dEstep = fx - fxp
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
            self.Ediff = (fx -fxp) / KCAL_MOL_PER_AU
            print "maximum displacement component (au)", self.disp
            print " Ediff (au)", self.Ediff
        
            # report the progress -- important for DLC since it updates the variables better//Cartesian update should happen as well
            c_obj.append_data(x, g, fx, xnorm, gradrms, step)
        
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

