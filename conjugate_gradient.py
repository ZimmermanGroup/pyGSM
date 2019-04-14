from __future__ import print_function
import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack,NoLineSearch
from base_optimizer import base_optimizer
from nifty import pmat2d,pvec1d

try:
    from io import StringIO
except:
    from StringIO import StringIO


class conjugate_gradient(base_optimizer):

    def optimize(self,molecule,refE=0.,opt_type='UNCONSTRAINED',opt_steps=3,ictan=None):
        print(" initial E %5.4f" % (molecule.energy - refE))
        if opt_type=='TS':
            raise RuntimeError

        # stash/initialize some useful attributes
        geoms = []
        energies=[]
        geoms.append(molecule.geometry)
        energies.append(molecule.energy-refE)
        self.initial_step = True
        self.disp = 1000.
        self.Ediff = 1000.
        self.check_inputs(molecule,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        self.buf = StringIO()

        # form initial coord basis
        constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        # number of coordinates
        if molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n =  n_actual - nconstraints 
        
        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
        g_prim = np.dot(molecule.coord_basis,g)

        molecule.gradrms = np.sqrt(np.dot(g[nconstraints:].T,g[nconstraints:])/n)
        if molecule.gradrms < self.conv_grms:
            print(" already at min")
            return geoms,energies

        for ostep in range(opt_steps):
            print(" On opt step {} ".format(ostep+1))

            if self.initial_step==True:
                # d: store the negative gradient of the object function on point x.
                # compute the initial step
                d_prim = -g_prim
                # set initial step to false
                self.initial_step=False
            else:   
                # Fletcher-Reeves formula for Beta
                # http://en.wikipedia.org/wiki/Nonlinear_conjugate_gradient_method       
                dnew = -g_prim  #
                deltanew = np.dot(dnew.T,dnew)
                deltaold=np.dot(-gp_prim.T,-gp_prim)
                beta = deltanew/deltaold
                print(" beta = %1.2f" % beta)
                d_prim = dnew + beta*d_prim

            # form in DLC basis (does nothing if cartesian)
            d = np.dot(molecule.coord_basis.T,d_prim)

            # normalize the direction
            actual_step = np.linalg.norm(d)
            print(" actual_step= %1.2f"% actual_step)
            d = d/actual_step #normalize
            if actual_step>self.options['DMAX']:
                step=self.options['DMAX']
                print(" reducing step, new step = %1.2f" %step)
            else:
                step=actual_step

            # store
            xp = x.copy()
            gp = g.copy()
            gp_prim = np.dot(molecule.coord_basis,gp)
            xyzp = xyz.copy()
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)

            # line search
            print(" Linesearch")
            ls = self.Linesearch(n, x, fx, g, d, step, xp, gp,constraint_steps,self.linesearch_parameters,molecule)
            print(" Done linesearch")
            
            # revert to the previous point
            if ls['status'] < 0:
                x = xp.copy()
                print('[ERROR] the point return to the previous point')
                return ls['status']

            # get values from linesearch
            p_step = step
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']
            g_prim = np.dot(molecule.coord_basis,g)

            #control step size
            if step < p_step: 
                self.options['DMAX']  /= 2.
            elif step > p_step:
                self.options['DMAX'] *= 2.
            if self.options['DMAX'] < 0.01:
                self.options['DMAX'] = 0.01
            elif self.options['DMAX'] >0.25:
                self.options['DMAX'] = 0.25

            # dE 
            dEstep = fx - fxp
            print(" dEstep=%5.4f" %dEstep)

            # update molecule xyz
            xyz = molecule.update_xyz(x-xp)
            geoms.append(molecule.geometry)
            energies.append(molecule.energy-refE)

            if self.options['print_level']>0:
                print(" Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (ostep+1,fx-refE,molecule.gradrms,step,self.options['DMAX']))
            self.buf.write(u' Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,molecule.gradrms,step,self.options['DMAX']))

            #gmax = np.max(g)/ANGSTROM_TO_AU/KCAL_MOL_PER_AU
            #print "current gradrms= %r au" % gradrms
            gmax = np.max(g)/ANGSTROM_TO_AU
            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = fx -fxp / KCAL_MOL_PER_AU
            print(" maximum displacement component %1.2f (au)" % self.disp)
            print(" maximum gradient component %1.2f (au)" % gmax)

            # check for convergence TODO
            molecule.gradrms = np.sqrt(np.dot(g[nconstraints:].T,g[nconstraints:])/n)
            if molecule.gradrms < self.conv_grms:
                break

            # check if finished
            #if gradrms <= params.conv_grms  or \
            #    (self.disp <= params.conv_disp and self.Ediff <= params.conv_Ediff) or \
            #    (gmax <= params.conv_gmax and abs(self.Ediff) <= params.conv_Ediff):
            #    print('[INFO] converged')
            #    print(gradrms)
            #    #print self.Ediff
            #    #print self.disp
            #    break

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                fx = molecule.energy
                dE = molecule.difference_energy
                if dE != 1000.:
                    print(" difference energy is %5.4f" % dE)
                g = molecule.gradient.copy()
                molecule.form_Hessian_in_basis()
            print()

        print(" opt-summary")
        print(self.buf.getvalue())
        return geoms,energies
