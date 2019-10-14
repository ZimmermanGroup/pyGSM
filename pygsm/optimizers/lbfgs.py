from __future__ import print_function

# standard library imports
import sys
from os import path
try:
    from io import StringIO
except:
    from StringIO import StringIO

# third party
import numpy as np
from scipy.optimize.lbfgsb import LbfgsInvHessProduct

# local application imports
from ._linesearch import backtrack,NoLineSearch
from .base_optimizer import base_optimizer
from utilities import *

class iterationData:
    """docstring for iterationData"""
    def __init__(self, alpha, s, y):
        self.alpha = alpha
        self.s_prim = s  #step
        self.y_prim = y  #diff in grad 

class lbfgs(base_optimizer):
    """the class of lbfgs method"""
       
    def __init__(self,options):
        super(lbfgs,self).__init__(options)
        self.k = 0
        self.end = 0

    def optimize(
            self,
            molecule,
            refE=0.,
            opt_type='UNCONSTRAINED',
            opt_steps=20,
            maxcor=10,
            ictan=None,
            xyzframerate=4,
            ):

        # stash/initialize some useful attributes
        print(" initial E %5.4f" % (molecule.energy - refE))
        geoms = []
        energies=[]
        geoms.append(molecule.geometry)
        energies.append(molecule.energy-refE)
        self.check_inputs(molecule,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        self.buf = StringIO()

        # form initial coord basis
        constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        if opt_type=='SEAM' or opt_type=='MECI':
            self.opt_cross=True

        # get coordinates 
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)
        x_prim = molecule.primitive_internal_values
        num_coords =  molecule.num_coordinates - nconstraints 
        
        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
    
        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]

        g_prim = block_matrix.dot(molecule.coord_basis,gc)
        molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/num_coords)

        # primitive constraint step
        self.cstep_prim = np.zeros_like(g_prim)
        dE = molecule.difference_energy

        # Not checking convergence here anymore ...
        #if molecule.PES.__class__.__name__!="PES" and self.opt_cross:
        #    if molecule.gradrms < self.conv_grms and abs(dE)<1.0:
        #        print(" converged")
        #        return geoms,energies
        #elif molecule.gradrms < self.conv_grms:
        #    print(" converged")
        #    return geoms,energies
        
        # reset k in principle k does not have to reset but . . . 
        self.k = 0
        self.end=0

        # initialize the iteration data list
        if self.k==0:
            self.lm = []
            for i in range(0, maxcor):
                s_prim = np.zeros_like(g_prim)
                y_prim = np.zeros_like(g_prim)
                self.lm.append(iterationData(0.0, s_prim.flatten(), y_prim.flatten()))
       
        for ostep in range(opt_steps):
            print(" On opt step {} ".format(ostep+1))

            if self.k!=0:
                # update vectors s and y:
                # TODO this doesn't work exactly with constraint steps
                self.lm[self.end].s_prim = molecule.coord_obj.Prims.calcDiff(xyz,self.xyzp) - self.cstep_prim.flatten()
                self.lm[self.end].y_prim = g_prim - self.gp_prim

                self.end = (self.end + 1) % maxcor
                #j = self.end
                bound = min(self.k, maxcor)
                s_prim = np.array([self.lm[i].s_prim.flatten() for i in range(maxcor)])
                y_prim = np.array([self.lm[i].y_prim.flatten() for i in range(maxcor)])
                hess_inv = LbfgsInvHessProduct(s_prim[:bound],y_prim[:bound])
                # compute the negative gradients
                d_prim = -g_prim
                # perform matrix product
                d_prim = hess_inv._matvec(d_prim)
                d_prim = np.reshape(d_prim,(-1,1))
            else:
                # d: store the negative gradient of the object function on point x.
                d_prim = -g_prim
            self.k = self.k + 1

            # form in DLC basis (does nothing if cartesian)
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis),d_prim)

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
            self.xyzp = xyz.copy()
            gp = g.copy()
            self.gp_prim = block_matrix.dot(molecule.coord_basis,gc)
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)
            self.cstep_prim = block_matrix.dot(molecule.coord_basis,constraint_steps) 

            # line search  
            #print(" Linesearch")
            ls = self.Linesearch(nconstraints, x, fx, gc, d, step, xp,constraint_steps,self.linesearch_parameters,molecule,verbose)
            #print(" Done linesearch")
            
            # revert to the privious point
            if ls['status'] < 0:
                x = xp.copy()
                molecule.xyz = self.xyzp
                g = gp.copy()
                print('[ERROR] the point return to the previous point')
                self.lm = []
                for i in range(0, maxcor):
                    s_prim = np.zeros_like(g_prim)
                    y_prim = np.zeros_like(g_prim)
                    self.lm.append(iterationData(0.0, s_prim.flatten(), y_prim.flatten()))
                    self.k = 0
                    self.end =0

            # save new values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']

            if ls['step'] > self.options['DMAX']:
                if ls['step']< 0.4:     # absolute max
                    print(" Increasing DMAX to {}".format(ls['step']))
                    self.options['DMAX'] = ls['step']
            elif ls['step']<step:
                if ls['step']>0.05:     # absolute min
                    print(" Decreasing DMAX to {}".format(ls['step']))
                    self.options['DMAX'] = ls['step']

            dEstep = fx - fxp
            dq = x-xp
            dEpre = np.dot(gc.T,dq)*units.KCAL_MOL_PER_AU
            constraint_energy = np.dot(gp.T,constraint_steps)*units.KCAL_MOL_PER_AU  
            print("constraint_energy: %1.4f" % constraint_energy)
            dEpre += constraint_energy

            if abs(dEpre)<0.05:
                dEpre = np.sign(dEpre)*0.05

            ratio = dEstep/dEpre
        
            print(" dEstep=%5.4f" %dEstep)
            print(" dEpre=%5.4f" %dEpre)
            print(" ratio=%5.4f" %ratio)

            # update molecule xyz
            xyz = molecule.update_xyz(x-xp)

            # project out the constraints
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
            g_prim = block_matrix.dot(molecule.coord_basis,gc)

            dE = molecule.difference_energy
            if dE is not 1000.:
                print(" difference energy is %5.4f" % dE)
            molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/num_coords)

            if ostep % xyzframerate==0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy-refE)

            if self.options['print_level']>0:
                print(" Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.options['DMAX']))
                #print(" Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (ostep+1,fx-refE,molecule.gradrms,step,self.options['DMAX']))
            #self.buf.write(u' Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,molecule.gradrms,step,self.options['DMAX']))
            self.buf.write(u' Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.options['DMAX']))

            if molecule.PES.__class__.__name__!="PES" and self.opt_cross:
                if molecule.gradrms < self.conv_grms and abs(dE)<1.0:
                    print(" converged")
                    if ostep % xyzframerate!=0:
                        geoms.append(molecule.geometry)
                        energies.append(molecule.energy-refE)
                    break
            # Check for  normal convergence
            else:
                gmax = float(np.max(g))
                disp = float(np.linalg.norm((xyz-self.xyzp).flatten()))
                
                try:
                    #print(" gmax {:5.4} disp {:5.4} Ediff {:5.4} gradrms {:5.4}".format(gmax,disp,dEstep,molecule.gradrms))
                    print(" gmax %5.4f disp %5.4f Ediff %5.4f gradrms %5.4f" % (gmax,disp,dEstep,molecule.gradrms))
                except:
                    print(gmax)
                    print(disp)
                if molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                    print(" converged")
                    if ostep % xyzframerate!=0:
                        geoms.append(molecule.geometry)
                        energies.append(molecule.energy-refE)
                        manage_xyz.write_xyzs_w_comments('opt_{}.xyz'.format(molecule.node_id),geoms,energies,scale=1.)
                    break
            #print " ########## DONE WITH TOTAL STEP #########"

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                if opt_type == 'SEAM' or opt_type=="MECI":
                    constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    # project out the constraint
                    gc = g.copy()
                    for c in molecule.constraints.T:
                        gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
                    g_prim = block_matrix.dot(molecule.coord_basis,gc)

        print(" opt-summary")
        print(self.buf.getvalue())
        return geoms,energies

