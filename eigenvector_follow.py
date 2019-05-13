from __future__ import print_function
import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack,NoLineSearch
from cartesian import CartesianCoordinates
from base_optimizer import base_optimizer
from nifty import pmat2d,pvec1d
from block_matrix import block_matrix as bm
from sys import exit

try:
    from io import StringIO
except:
    from StringIO import StringIO

class eigenvector_follow(base_optimizer):

    def optimize(self,molecule,refE=0.,opt_type='UNCONSTRAINED',opt_steps=3,ictan=None):

        #print " refE %5.4f" % refE
        print(" initial E %5.4f" % (molecule.energy - refE))
        geoms = []
        energies=[]
        geoms.append(molecule.geometry)
        energies.append(molecule.energy-refE)

        # stash/initialize some useful attributes
        self.disp = 1000.
        self.Ediff = 1000.
        self.check_inputs(molecule,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        self.buf = StringIO()

        #form initial coord basis
        constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        if opt_type=='TS':
            self.Linesearch=NoLineSearch

        # TODO are these used? -- n is used for gradrms,linesearch
        if molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
            n = molecule.num_coordinates
        else:
            n_actual = molecule.num_coordinates
            n =  n_actual - nconstraints 
            self.x_prim=np.zeros((molecule.num_primitives,1),dtype=float)
            self.g_prim=np.zeros((molecule.num_primitives,1),dtype=float)
        
        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()

        # project out the constraint
        gc = g - np.dot(g.T,molecule.constraints)*molecule.constraints

        molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/n)
        if molecule.gradrms < self.conv_grms:
            print(" already at min")
            return geoms,energies

        update_hess=False
        # ====>  Do opt steps <======= #
        for ostep in range(opt_steps):
            print(" On opt step {} ".format(ostep+1))

            # update Hess
            if update_hess:
                if opt_type!='TS':
                    change = self.update_Hessian(molecule,'BFGS')
                else:
                    self.update_Hessian(molecule,'BOFILL')
            update_hess = True

            # => Form eigenvector step <= #
            if isinstance(molecule.coord_obj,CartesianCoordinates):
                raise NotImplementedError
            else:
                if opt_type !='TS':
                    dq = self.eigenvector_step(molecule,gc,nconstraints)
                else:
                    dq,maxol_good = self.TS_eigenvector_step(molecule,g,ictan)
                    if not maxol_good:
                        nconstraints=1
                        opt_type='CLIMB'

            actual_step = np.linalg.norm(dq)
            print(" actual_step= %1.2f"% actual_step)
            dq = dq/actual_step #normalize
            if actual_step>self.options['DMAX']:
                step=self.options['DMAX']
                print(" reducing step, new step = %1.2f" %step)
            else:
                step=actual_step
        
            # store values
            xp = x.copy()
            gp = gc.copy()
            xyzp = xyz.copy()
            fxp = fx
            pgradrms = molecule.gradrms
            print(" pgradrms {:4}".format(pgradrms))
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                xp_prim = self.x_prim.copy()
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)
        
            #print(" ### Starting  line search ###")
            ls = self.Linesearch(nconstraints, x, fx, gc, dq, step, xp, gp,constraint_steps,self.linesearch_parameters,molecule)
            if ls['status'] ==-2:
                x = xp.copy()
                print('[ERROR] the point return to the privious point')
                return ls['status']
            print(" ## Done line search")
        
            # get values from linesearch
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']
            gc = g - np.dot(g.T,molecule.constraints)*molecule.constraints

            # calculate predicted value from Hessian, gp is previous constrained gradient
            scaled_dq = dq*step
            dEtemp = np.dot(molecule.constrained_Hessian,scaled_dq)
            dEpre = np.dot(np.transpose(scaled_dq),gp) + 0.5*np.dot(np.transpose(dEtemp),scaled_dq)
            dEpre *=KCAL_MOL_PER_AU
            dEpre += np.dot(gp.T,constraint_steps)*KCAL_MOL_PER_AU  #linear approximation

            # control step size 
            dEstep = fx - fxp
            print(" dEstep=%5.4f" %dEstep)
            ratio = dEstep/dEpre
            molecule.gradrms = np.sqrt(np.dot(gc.T,gc)/n)
            self.step_controller(actual_step,ratio,molecule.gradrms,pgradrms,dEpre,opt_type,dEstep)
            # report the progress 
            #TODO make these lists and check last element for convergene
            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = (fx -fxp) / KCAL_MOL_PER_AU
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0

            # update molecule xyz
            xyz = molecule.update_xyz(x-xp)
            geoms.append(molecule.geometry)
            energies.append(molecule.energy-refE)

            # save variables for update Hessian! 
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                # only form g_prim for non-constrained 
                #self.g_prim = np.dot(molecule.coord_basis[:,nconstraints:],g[nconstraints:])
                self.g_prim = bm.dot(molecule.coord_basis,gc)
                self.dx = x-xp
                self.dg = g - gp

                self.dx_prim_actual = molecule.coord_obj.Prims.calcDiff(xyz,xyzp)
                self.dx_prim_actual = np.reshape(self.dx_prim_actual,(-1,1))
                #self.dx_prim = np.dot(molecule.coord_basis[:,nconstraints:],scaled_dq[nconstraints:]) 
                self.dx_prim = bm.dot(molecule.coord_basis,scaled_dq)
                self.dg_prim = self.g_prim - gp_prim

            else:
                raise NotImplementedError(" ef not implemented for CART")

            if self.options['print_level']>0:
                print(" Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.options['DMAX']))
            self.buf.write(u' Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.options['DMAX']))


            # check for convergence TODO
            if molecule.gradrms < self.conv_grms:
                break

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                if opt_type != 'TS':
                    constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    fx = molecule.energy
                    dE = molecule.difference_energy
                    if dE is not 1000.:
                        print(" difference energy is %5.4f" % dE)
                    g = molecule.gradient.copy()
                    molecule.form_Hessian_in_basis()
            print()
       
        print(" opt-summary")
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


