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

# local application imports
from ._linesearch import backtrack,NoLineSearch,golden_section
from .base_optimizer import base_optimizer
from utilities import *

class eigenvector_follow(base_optimizer):

    def optimize(
            self,
            molecule,
            refE=0.,
            opt_type='UNCONSTRAINED',
            opt_steps=3,
            ictan=None,
            xyzframerate=4,
            verbose=False,
            path=os.getcwd(),
            ):

        # stash/initialize some useful attributes
        self.check_inputs(molecule,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        self.buf = StringIO()

        #print " refE %5.4f" % refE
        print(" initial E %5.4f" % (molecule.energy - refE))
        print(" CONV_TOL %1.5f" % self.conv_grms)
        geoms = []
        energies=[]
        geoms.append(molecule.geometry)
        energies.append(molecule.energy-refE)
        self.converged=False

        #form initial coord basis
        if opt_type != 'TS':
            constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
            molecule.update_coordinate_basis(constraints=constraints)
            molecule.form_Hessian_in_basis()

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()
        # project out the constraint
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
        gmax = float(np.max(np.absolute(gc)))

        if self.check_only_grad_converged:
            if molecule.gradrms < self.conv_grms and gmax < self.conv_gmax:
                self.converged=True
                return geoms,energies
            else:
                self.check_only_grad_converged=False

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        if opt_type=='TS':
            self.Linesearch=NoLineSearch
        if opt_type=='SEAM' or opt_type=='MECI' or opt_type=="TS-SEAM":
            self.opt_cross=True

        # TODO are these used? -- n is used for gradrms,linesearch
        if molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n =  n_actual - nconstraints 
            self.x_prim=np.zeros((molecule.num_primitives,1),dtype=float)
            self.g_prim=np.zeros((molecule.num_primitives,1),dtype=float)
        

        molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/n)
        dE = molecule.difference_energy
        update_hess=False

        # ====>  Do opt steps <======= #
        for ostep in range(opt_steps):
            print(" On opt step {} for node {}".format(ostep+1,molecule.node_id))

            # update Hess
            if update_hess:
                if opt_type!='TS':
                    change = self.update_Hessian(molecule,'BFGS')
                else:
                    self.update_Hessian(molecule,'BOFILL')
            update_hess = True
        
            # => Form eigenvector step <= #
            if molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                raise NotImplementedError
            else:
                if opt_type !='TS':
                    dq = self.eigenvector_step(molecule,gc)
                else:
                    dq = self.TS_eigenvector_step(molecule,g,ictan)
                    if not self.maxol_good:
                        print(" Switching to climb! Maxol not good!")
                        nconstraints=1
                        opt_type='CLIMB'

            actual_step = np.linalg.norm(dq)
            #print(" actual_step= %1.2f"% actual_step)
            dq = dq/actual_step #normalize
            if actual_step>self.DMAX:
                step=self.DMAX
                #print(" reducing step, new step = %1.2f" %step)
            else:
                step=actual_step
        
            # store values
            xp = x.copy()
            gp = g.copy()
            xyzp = xyz.copy()
            fxp = fx
            pgradrms = molecule.gradrms
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                xp_prim = self.x_prim.copy()
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)
        
            #print(" ### Starting  line search ###")
            ls = self.Linesearch(nconstraints, x, fx, g, dq, step, xp,constraint_steps,self.linesearch_parameters,molecule,verbose)

            # get values from linesearch
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']

            if ls['status'] ==-2:  
                print('[ERROR] the point return to the privious point')
                x = xp.copy()
                molecule.xyz = xyzp
                g = gp.copy()
                fx = fxp
                ratio=0.
                molecule.newHess=5
                #return ls['status']

            if ls['step'] > self.DMAX:
                if ls['step']<= self.options['abs_max_step']:  # absolute max
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

            # calculate predicted value from Hessian, gp is previous constrained gradient
            scaled_dq = dq*step
            dEtemp = np.dot(self.Hessian,scaled_dq)
            dEpre = np.dot(np.transpose(scaled_dq),gc) + 0.5*np.dot(np.transpose(dEtemp),scaled_dq)
            dEpre *=units.KCAL_MOL_PER_AU
            #print(constraint_steps.T)
            constraint_energy = np.dot(gp.T,constraint_steps)*units.KCAL_MOL_PER_AU  
            #print("constraint_energy: %1.4f" % constraint_energy)
            dEpre += constraint_energy
            #if abs(dEpre)<0.01:
            #    dEpre = np.sign(dEpre)*0.01

            # project out the constraint
            gc = g.copy()
            for c in molecule.constraints.T:
                gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]

            # control step size 
            dEstep = fx - fxp
            print(" dEstep=%5.4f" %dEstep)
            ratio = dEstep/dEpre
            molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/n)
            if ls['status'] !=-2:  
                self.step_controller(actual_step,ratio,molecule.gradrms,pgradrms,dEpre,opt_type,dEstep)

            # update molecule xyz
            xyz = molecule.update_xyz(x-xp)
            if ostep % xyzframerate==0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy-refE)
                manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path,molecule.node_id),geoms,energies,scale=1.)

            # save variables for update Hessian! 
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                # only form g_prim for non-constrained 
                self.g_prim = block_matrix.dot(molecule.coord_basis,gc)
                self.dx = x-xp
                self.dg = g - gp

                self.dx_prim_actual = molecule.coord_obj.Prims.calcDiff(xyz,xyzp)
                self.dx_prim_actual = np.reshape(self.dx_prim_actual,(-1,1))
                self.dx_prim = block_matrix.dot(molecule.coord_basis,scaled_dq)
                self.dg_prim = self.g_prim - gp_prim

            else:
                raise NotImplementedError(" ef not implemented for CART")

            if self.options['print_level']>0:
                print(" Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.DMAX))
            self.buf.write(u' Node: %d Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (molecule.node_id,ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.DMAX))

            # check for convergence TODO
            fx = molecule.energy
            dE = molecule.difference_energy
            if dE< 1000.:
                print(" difference energy is %5.4f" % dE)
            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz-xyzp).flatten()))
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0

            print(" gmax %5.4f disp %5.4f Ediff %5.4f gradrms %5.4f\n" % (gmax,disp,dEstep,molecule.gradrms))

            #TODO turn back on conv_DE
            if self.opt_cross and abs(dE)<self.conv_dE and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                if opt_type=="TS-SEAM":
                    gts = np.dot(g.T,molecule.constraints[:,0])
                    print(" gts %1.4f" % gts)
                    if abs(gts)<self.conv_grms*5:
                        self.converged=True
                else:
                    self.converged=True
            elif not self.opt_cross and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                if opt_type=="CLIMB":
                    gts = np.dot(g.T,molecule.constraints[:,0])
                    if abs(gts)<self.conv_grms*5.:
                        self.converged=True
                elif opt_type=="TS":
                    if self.gtse<self.conv_grms*5.:
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

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                if opt_type != 'TS':
                    constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    # project out the constraint
                    gc = g.copy()
                    for c in molecule.constraints.T:
                        gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
            print()
            sys.stdout.flush()
       
        print(" opt-summary {}".format(molecule.node_id))
        print(self.buf.getvalue())
        return geoms,energies



if __name__=='__main__':
    from qchem import QChem
    from pes import PES
    from molecule import Molecule
    from _linesearch import NoLineSearch
    from slots import Distance

    basis="6-31G*"
    nproc=8

    filepath="examples/tests/bent_benzene.xyz"
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc,fnm=filepath)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    M = Molecule.from_options(fnm=filepath,PES=pes,coordinate_type="DLC")
    distance = Distance(5,8)  #Not 1 based!!
    print(distance)
   
    ef = eigenvector_follow.from_options() #Linesearch=NoLineSearch)
    geoms = ef.optimize(molecule=M,refE=M.energy,opt_steps=5)
    #geoms = ef.optimize(molecule=M,refE=M.energy,opt_steps=1)
    print(M.primitive_internal_coordinates)

    manage_xyz.write_xyzs('opt.xyz',geoms,scale=1.) 


