from __future__ import print_function

# standard library imports
import sys
import os
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
            verbose=False,
            path=os.getcwd(),
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
        num_coords =  molecule.num_coordinates - nconstraints - molecule.num_frozen_atoms*3
        
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
        
        ## reset k in principle k does not have to reset but . . . 
        # TRY Turning off Feb 2020
        if opt_type != 'CLIMB':
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


            SCALE =self.options['SCALEQN']
            if molecule.newHess>0: SCALE = self.options['SCALEQN']*molecule.newHess

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
                d_prim = np.reshape(d_prim,(-1,1))/SCALE
            else:
                # d: store the negative gradient of the object function on point x.
                d_prim = -g_prim/SCALE
            self.k = self.k + 1

            # form in DLC basis (does nothing if cartesian)
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis),d_prim)

            # normalize the direction
            actual_step = np.linalg.norm(d)
            print(" actual_step= %1.5f"% actual_step)
            d = d/actual_step #normalize
            if actual_step>self.DMAX:
                step=self.DMAX
                print(" reducing step, new step = %1.5f" %step)
            else:
                step=actual_step

            # store
            xp = x.copy()
            self.xyzp = xyz.copy()
            gp = g.copy()
            self.gp_prim = block_matrix.dot(molecule.coord_basis,gc)
            fxp = fx
            pgradrms = molecule.gradrms

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)
            self.cstep_prim = block_matrix.dot(molecule.coord_basis,constraint_steps) 

            # line search  
            ls = self.Linesearch(nconstraints, x, fx, gc, d, step, xp,constraint_steps,self.linesearch_parameters,molecule,verbose)
            
            # save new values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']

            dEstep = fx - fxp
            dq = x-xp

            # TODO dEpre is missing second order effects or is it?
            dEpre = np.dot(gc.T,dq)*units.KCAL_MOL_PER_AU
            constraint_energy = np.dot(gp.T,constraint_steps)*units.KCAL_MOL_PER_AU  
            if opt_type not in ['UNCONSTRAINED','ICTAN']:
                print("constraint_energy: %1.4f" % constraint_energy)
            dEpre += constraint_energy

            #if abs(dEpre)<0.05:
            #    dEpre = np.sign(dEpre)*0.05
            ratio = dEstep/dEpre
            print(" dEstep=%5.4f" %dEstep)
            print(" dEpre=%5.4f" %dEpre)
            print(" ratio=%5.4f" %ratio)

            # revert to the privious point
            if ls['status'] < 0 or (ratio<0. and opt_type!='CLIMB'):
                if ratio<0.:
                    ls['status']=-1
                x = xp.copy()
                molecule.xyz = self.xyzp.copy()
                g = gp.copy()
                fx = fxp
                #fx = molecule.energy
                ratio=0.
                dEstep=0.

                print('[ERROR] the point return to the previous point')
                self.lm = []
                for i in range(0, maxcor):
                    s_prim = np.zeros_like(g_prim)
                    y_prim = np.zeros_like(g_prim)
                    self.lm.append(iterationData(0.0, s_prim.flatten(), y_prim.flatten()))
                self.k = 0
                self.end =0
                molecule.newHess=5
                if self.DMAX <= self.DMIN:
                    print(" Reached minimum step,exiting")
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy-refE)
                    break

                self.DMAX = ls['step']/2
                if self.DMAX < self.DMIN:
                    self.DMAX = self.DMIN
            else:
                # update molecule xyz
                xyz = molecule.update_xyz(x-xp)

            # if ratio is less than 0.3 than reduce DMAX
            flag=True
            if ratio<0.3 and ls['status']==0: #and abs(dEpre)>0.05:
                print(" Reducing DMAX")
                self.DMAX /= 1.5
                if self.DMAX < self.DMIN:
                    self.DMAX = self.DMIN
                if molecule.newHess<5:
                    molecule.newHess+=1
                flag=False

                if opt_type=="CLIMB":
                    if self.SCALE_CLIMB<10. and opt_type=='CLIMB':
                        self.SCALE_CLIMB+=1.
                        print('SCALING CLIMB BY {}'.format(self.SCALE_CLIMB))
            elif ratio>0.3:
                molecule.newHess-=1
                if opt_type=="CLIMB":
                    if self.SCALE_CLIMB>1.:
                        self.SCALE_CLIMB -=1.


            # project out the constraints
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
            g_prim = block_matrix.dot(molecule.coord_basis,gc)

            dE = molecule.difference_energy
            if dE < 100.:
                print(" difference energy is %5.4f" % dE)
            molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/num_coords)


            # control step size  NEW FEB 2020
            if flag:
                if ls['status']==0:  # passed
                    dgradrms = molecule.gradrms - pgradrms
                    print("dgradrms ",dgradrms)
                    if ls['step'] > self.DMAX:
                        if ls['step']<= self.options['abs_max_step']:     # absolute max
                            print(" Increasing DMAX to {}".format(ls['step']))
                            self.DMAX = ls['step']
                        else:
                            self.DMAX =self.options['abs_max_step']
                    elif ls['step']<self.DMAX:
                        if ls['step']>=self.DMIN:     # absolute min
                            print(" Decreasing DMAX to {}".format(ls['step']))
                            self.DMAX = ls['step']
                        elif ls['step']<=self.DMIN:
                            self.DMAX = self.DMIN
                            print(" Decreasing DMAX to {}".format(self.DMIN))
                    elif ratio>0.85 and ratio<1.1 and actual_step>self.DMAX and dgradrms<-0.00005:
                        print(" HERE increasing DMAX")
                        self.DMAX *= 1.1
                        if self.DMAX>self.options['abs_max_step']:
                            self.DMAX=self.options['abs_max_step']
                else:
                    print("status not zero")


            if ostep % xyzframerate==0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy-refE)
                manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path,molecule.node_id),geoms,energies,scale=1.)

            if self.options['print_level']>0:
                print(" Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.DMAX))
            self.buf.write(u' Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.DMAX))

            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz-self.xyzp).flatten()))
            print(" gmax %5.4f disp %5.4f dEstep %5.4f gradrms %5.4f\n" % (gmax,disp,dEstep,molecule.gradrms))
            self.converged=False
            if self.opt_cross and abs(dE)<self.conv_dE and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp and ls['status']==0:

                # TODO Seam Climb
                if opt_type=="TS-SEAM":
                    gts = np.dot(g.T,molecule.constraints[:,0])
                    if abs(gts)<self.conv_grms*5:
                        self.converged=True
                else:
                    self.converged=True

            elif not self.opt_cross and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                if opt_type=="CLIMB":
                    gts = np.dot(g.T,molecule.constraints[:,0])
                    if abs(gts)<self.conv_grms*5:
                        self.converged=True
                else:
                    self.converged=True

            if self.converged:
                print(" converged")
                if ostep % xyzframerate!=0:
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy-refE)
                    manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path,molecule.node_id),geoms,energies,scale=1.)
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
            sys.stdout.flush()

        print(" opt-summary")
        print(self.buf.getvalue())
        return geoms,energies

