import numpy as np
import manage_xyz
from units import *
from _linesearch import backtrack,NoLineSearch
from dlc import DLC
from cartesian import CartesianCoordinates
from base_optimizer import base_optimizer
import StringIO

class eigenvector_follow(base_optimizer):

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return eigenvector_follow(eigenvector_follow.default_options().set_values(kwargs))

    def optimize(self,molecule,refE=0.,opt_steps=3,ictan=None):

        #print " refE %5.4f" % refE
        geoms = []
        geoms.append(molecule.geometry)

        # stash/initialize some useful attributes
        self.disp = 1000.
        self.Ediff = 1000.
        opt_type=self.options['opt_type']
        self.check_inputs(molecule,opt_type,ictan)
        nconstraints=self.get_nconstraints(opt_type)
        self.buf = StringIO.StringIO()

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        if opt_type=='TS':
            self.Linesearch=NoLineSearch
        #print " opt_type ",opt_type

        if molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
            n = len(x)  # constraints not allowed for cartesian
        else:
            n_actual = len(x)
            n =  len(x) - nconstraints 
            self.x_prim=np.zeros((len(molecule.primitive_internal_coordinates),1),dtype=float)
            self.g_prim=np.zeros((len(molecule.primitive_internal_coordinates),1),dtype=float)
            #self.g_prim_c=np.zeros((n_actual,1),dtype=float)
        
        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient.copy()

        update_hess=False
        if self.converged(g,n):
            print " already at min"
            return fx

        # ====>  Do opt steps <======= #
        for ostep in range(opt_steps):
            print " On opt step ",ostep+1
            print " fx %1.4f" % fx

            # update Hess
            if update_hess:
                if opt_type!='TS':
                    self.update_Hessian(molecule,'BFGS')
                else:
                    self.update_Hessian(molecule,'BOFILL')
            update_hess = True

            # => Form eigenvector step <= #
            if isinstance(molecule.coord_obj,CartesianCoordinates):
                raise NotImplementedError
            else:
                if self.options['opt_type']!='TS':
                    dq = self.eigenvector_step(molecule,g,nconstraints)
                else:
                    pass
                    #TODO
                    #dq,maxol_good = self.TS_eigenvector_step(c_obj,g,ictan)
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
            xyzp = xyz.copy()
            fxp = fx
            pgradrms = np.sqrt(np.dot(g.T,g)/n)
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                xp_prim = self.x_prim.copy()
                gp_prim = self.g_prim.copy()

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule,opt_type,g)
        
            print " ### Starting  line search ###"
            ls = self.Linesearch(n, x, fx, g, dq, step, xp, gp,constraint_steps,self.linesearch_parameters,molecule)
            if ls['status'] ==-2:
                x = xp.copy()
                print '[ERROR] the point return to the privious point'
                return ls['status']
            print " ## Done line search"
        
            # get values from linesearch
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g  = ls['g']

            # calculate predicted value from Hessian
            #TODO change limits since constraints are now at beginning
            dq_actual = x-xp
            dEtemp = np.dot(molecule.Hessian[nconstraints:,nconstraints:],dq_actual[nconstraints:])
            dEpre = np.dot(np.transpose(dq_actual[nconstraints:]),gp[nconstraints:]) + 0.5*np.dot(np.transpose(dEtemp),dq_actual[nconstraints:])
            dEpre *=KCAL_MOL_PER_AU
            for nc in range(nconstraints):
                dEpre += gp[-nc-1]*constraint_steps[-nc-1]*KCAL_MOL_PER_AU  # DO this b4 recalc gradq WARNING THIS IS ONLY GOOD FOR DLC

            # control step size 
            dEstep = fx - fxp
            ratio = dEstep/dEpre
            molecule.gradrms = np.sqrt(np.dot(g.T,g)/n)
            self.step_controller(step,ratio,molecule.gradrms,pgradrms)
        
            # report the progress 
            #TODO make these lists and check last element for convergence
            self.disp = np.max(x - xp)/ANGSTROM_TO_AU
            self.Ediff = (fx -fxp) / KCAL_MOL_PER_AU
            xnorm = np.sqrt(np.dot(x.T, x))
            gnorm = np.sqrt(np.dot(g.T, g)) 
            if xnorm < 1.0:
            	xnorm = 1.0

            # update molecule xyz
            xyz = molecule.update_xyz(dq_actual)
            geoms.append(molecule.geometry)

            # save variables for update Hessian! 
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':

                self.g_prim = np.dot(molecule.coord_basis,g)
                #self.x_prim = molecule.primitive_internal_values
                self.dx = x-xp
                self.dg = g - gp

                self.dx_prim = molecule.coord_obj.Prims.calcDiff(xyz,xyzp)
                #self.dx_prim = np.dot(molecule.coord_basis,dq_actual)  #why is this not working anymore?
                #print (self.dx_prim_actual - self.dx_prim).T
                self.dg_prim = self.g_prim - gp_prim

            else:
                raise NotImplementedError

            self.buf.write(' Opt step: %d E: %5.4f predE: %5.4f ratio: %1.3f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1,fx-refE,dEpre,ratio,molecule.gradrms,step,self.options['DMAX']))

            if self.converged(g,n):
                break

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
                if opt_type != 'TS':
                    constraints = self.get_constraint_vectors(molecule,opt_type,ictan)
                    #print "updating coordinate basis"
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    molecule.form_Hessian_in_basis()
                #else:
                #    c_obj.bmatp = c_obj.bmatp_create()
        
        print(self.buf.getvalue())
        return geoms



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
    print distance
   
    ef = eigenvector_follow.from_options() #Linesearch=NoLineSearch)
    geoms = ef.optimize(molecule=M,refE=M.energy,opt_steps=5)
    #geoms = ef.optimize(molecule=M,refE=M.energy,opt_steps=1)
    print M.primitive_internal_coordinates

    manage_xyz.write_xyzs('opt.xyz',geoms,scale=1.) 


