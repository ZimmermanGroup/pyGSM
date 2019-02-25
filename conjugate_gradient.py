import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack


class conjugate_gradient(object):
    def __init__(self):
        self.initial_step = True
        self.disp = 1000.
        self.Ediff = 1000.
        return

    def optimize(self,c_obj,params,opt_steps=3,ictan=None):
        for i in range(opt_steps):

            print " On opt step ",i

            opt_type=params.options['opt_type']
            if opt_type in ["ICTAN", "CLIMB"]:
                nconstraints = 1
            elif opt_type in ['MECI']:
                nconstraints=2
            elif opt_type in ['SEAM','TS-SEAM']:
                nconstraints=3
            else:
                nconstraints=0

            # copy of coordinates
            x = np.copy(c_obj.q)

            # the number of coordinates
            if c_obj.__class__.__name__=='Cartesian':
                n = len(x) - nconstraints
            else:
                n =  c_obj.nicd_DLC-nconstraints 
                if c_obj.nxyzatoms>0:
                    print "warning you probably don't want the cartesian atoms to be frozen."

            Linesearch=params.options['Linesearch']

            # Evaluate the function value and its gradient.
            print " [INFO] Getting current evaluate"
            fx,g = c_obj.proc_evaluate(x)
            print ' [INFO] fx is %r' %(fx)

            # normalize 
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0

            # check if finished
            gradrms = np.sqrt(np.dot(g.T,g)/n)/(KCAL_MOL_PER_AU**2)/(ANGSTROM_TO_AU**2)
            #gradrms = np.sqrt(np.dot(g.T,g)/self.n)
            #gradrms = np.sqrt(np.dot(g.T,g)/self.n)//(ANGSTROM_TO_AU**2)
            print "current gradrms= %r au" % gradrms
            print "gnorm =",gnorm

            gmax = np.max(g)/ANGSTROM_TO_AU/KCAL_MOL_PER_AU
            #gmax = np.max(g)/ANGSTROM_TO_AU
            print "maximum gradient component (au)", gmax
            print "maximum displacement component (au)", self.disp

            if gradrms <= params.conv_grms  or \
                (self.disp <= params.conv_disp and self.Ediff <= params.conv_Ediff) or \
                (gmax <= params.conv_gmax and abs(self.Ediff) <= params.conv_Ediff):
                print '[INFO] converged'
                print gradrms
                print self.Ediff
                print self.disp
                break
                    
            if self.initial_step==True:
                # d: store the negative gradient of the object function on point x.
                d = -g
                # compute the initial step
                self.step = 1.0 / np.sqrt(np.dot(d.T, d))
                # set initial step to false
                self.initial_step=False

            else:   
                # Fletcher-Reeves formula for Beta
                # http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method       
                dnew = -g
                deltanew = np.dot(dnew.T,dnew)
                deltaold=np.dot(d.T,d)
                beta = deltanew/deltaold
                d = dnew + beta*dnew
                #self.step = 1.0 / np.sqrt(np.dot(self.d.T, self.d)) # ? what to put here?

            # store
            xp = x.copy()
            gp = g.copy()
            fxp = fx

            # line search
            print " ### Starting  line search ###"
            ls = Linesearch(n, x, fx, g, d, self.step, xp, gp, params,c_obj.proc_evaluate)
            print "done line search"
            
            # revert to the privious point
            if ls['status'] < 0:
                self.x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']
            fx = ls['fx']
            self.step = ls['step']
            x = ls['x']
            g = ls['g']

            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = fx -fxp / KCAL_MOL_PER_AU

            # report the progress
            # report the progress -- important for DLC since it updates the variables //Cartesian update should happen as well
            c_obj.append_data(x, g, fx, xnorm, gnorm, self.step)


