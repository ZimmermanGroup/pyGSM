from __future__ import print_function
# standard library imports
import sys
import os
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from wrappers import Molecule
from .base_gsm import Base_Method

class DE_GSM(Base_Method):

    def __init__(
            self,
            options,
            ):

        super(DE_GSM,self).__init__(options)

        print(" Forming Union of primitive coordinates")
        self.nodes[0].coord_obj = self.nodes[0].coord_obj.make_union_primitives(self.nodes[-1].coord_obj,self.nodes[0].xyz)

        print('coordinates')
        print(self.nodes[0].coord_obj.Prims.Internals[:100])
        self.nodes[0].form_Primitive_Hessian()
        print(" Done forming union")
        self.nodes[-1].PES.lot.node_id = self.nnodes-1
        self.nodes[-1].coord_obj = self.nodes[0].coord_obj.copy(self.nodes[-1].xyz)
        self.nodes[-1].form_Primitive_Hessian()

        # this tests if the primitives are the same
        assert self.nodes[0].coord_obj == self.nodes[-1].coord_obj, "They should be the same."

        #print(" Primitive Internal Coordinates")
        #print(self.nodes[0].primitive_internal_coordinates)
        print(" number of primitives is", self.nodes[0].num_primitives)
        self.set_V0()

    def go_gsm(self,max_iters=50,opt_steps=3,rtype=2):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find, 
        0 turning of climbing image and TS search
        """

        if not self.isRestarted:
            if self.growth_direction==0:
                self.interpolate(2) 
            elif self.growth_direction==1:
                self.interpolateR(1)
            elif self.growth_direction==2:
                self.interpolateP(1)
            oi = self.growth_iters(iters=max_iters,maxopt=opt_steps) 
            nifty.printcool("Done Growing the String!!!")
            self.done_growing = True
            #nifty.printcool("initial ic_reparam")
            self.get_tangents_1()
            self.ic_reparam(ic_reparam_steps=8)
            self.write_xyz_files(iters=1,base='grown_string',nconstraints=1)
            #self.write_xyz_files(iters=1,base='initial_ic_reparam',nconstraints=1)
        else:
            oi=0
            self.get_tangents_1()

        # set convergence for nodes
        if not (self.climber or self.finder):
            factor = 2.
        else: 
            factor = 1.
        for i in range(self.nnodes):
            if self.nodes[i] !=None:
                self.optimizer[i].conv_grms = self.options['CONV_TOL']*factor

        if self.tscontinue==True:
            if max_iters-oi>0:
                opt_iters=max_iters-oi
                self.TSnode = np.argmax(self.energies)
                self.emax = self.energies[self.TSnode]
                self.opt_iters(max_iter=opt_iters,optsteps=opt_steps,rtype=rtype)
        else:
            print("Exiting early")
        print("Finished GSM!") 

        return self.nnodes,self.energies

    def interpolate(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
        sign = -1
        for i in range(newnodes):
            sign *= -1
            if sign == 1:
                self.interpolateR()
            else:
                self.interpolateP()

    def add_node(self,n1,n2,n3):
        print(" adding node: %i between %i %i from %i" %(n2,n1,n3,n1))
        ictan,_ =  self.tangent(n3,n1)
        Vecs = self.nodes[n1].update_coordinate_basis(constraints=ictan)
        constraint = self.nodes[n1].constraints
        prim_constraint = block_matrix.dot(Vecs,constraint)
        dqmag = np.dot(prim_constraint.T,ictan)
        print(" dqmag: %1.3f"%dqmag)
        sign=-1

        if self.nnodes - self.nn > 1:
            dqmag = sign*dqmag/float(self.nnodes-self.nn)
        else:
            dqmag = sign*dqmag/2.0 
        print(" scaled dqmag: %1.3f"%dqmag)

        dq0 = dqmag*constraint
        old_xyz = self.nodes[n1].xyz.copy()
        new_xyz = self.nodes[n1].coord_obj.newCartesian(old_xyz,dq0)
        new_node = Molecule.copy_from_options(MoleculeA=self.nodes[n1],xyz=new_xyz,new_node_id=n2)
        return new_node

    def set_active(self,nR,nP):
        #print(" Here is active:",self.active)
        if nR!=nP and self.growth_direction==0:
            print((" setting active nodes to %i and %i"%(nR,nP)))
        elif self.growth_direction==1:
            print((" setting active node to %i "%nR))
        elif self.growth_direction==2:
            print((" setting active node to %i "%nP))
        else:
            print((" setting active node to %i "%nR))

        for i in range(self.nnodes):
            if self.nodes[i] != None:
                self.optimizer[i].conv_grms = self.options['CONV_TOL']*2.
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction==1:
            self.active[nP]=False
        if self.growth_direction==2:
            self.active[nR]=False
        #print(" Here is new active:",self.active)

    def check_if_grown(self):
        isDone=False
        if self.nn==self.nnodes:
            isDone=True
            if self.growth_direction==1:
                print("need to add last node")
                raise NotImplementedError
                #TODO

        return isDone

    def check_add_node(self):
        success=True 
        if self.nodes[self.nR-1].gradrms < self.gaddmax and self.growth_direction!=2:
            if self.nodes[self.nR] == None:
                self.interpolateR()
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction!=1:
            if self.nodes[-self.nP-1] == None:
                self.interpolateP()
        return success

    def tangent(self,n1,n2):
        if self.print_level>1:
            print(" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2))
        # this could have been done easier but it is nicer to do it this way
        Q1 = self.nodes[n1].primitive_internal_values 
        Q2 = self.nodes[n2].primitive_internal_values 
        PMDiff = Q2-Q1
        #for i in range(len(PMDiff)):
        for k,prim in zip(list(range(len(PMDiff))),self.nodes[n1].primitive_internal_coordinates):
            if prim.isPeriodic:
                Plus2Pi = PMDiff[k] + 2*np.pi
                Minus2Pi = PMDiff[k] - 2*np.pi
                if np.abs(PMDiff[k]) > np.abs(Plus2Pi):
                    PMDiff[k] = Plus2Pi
                if np.abs(PMDiff[k]) > np.abs(Minus2Pi):
                    PMDiff[k] = Minus2Pi
        return np.reshape(PMDiff,(-1,1)),None


    def make_nlist(self):
        ncurrent = 0
        nlist = [0]*(2*self.nnodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1

        for n in range(self.nnodes-self.nP+1,self.nnodes):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n-1
            ncurrent += 1

        nlist[2*ncurrent] = self.nR -1
        nlist[2*ncurrent+1] = self.nnodes - self.nP

        if False:
            nlist[2*ncurrent+1] = self.nR - 2 #for isMAP_SE

        #TODO is this actually used?
        #if self.nR == 0: nlist[2*ncurrent] += 1
        #if self.nP == 0: nlist[2*ncurrent+1] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes -self.nP
        nlist[2*ncurrent+1] = self.nR-1
        ##TODO is this actually used?
        #if self.nR == 0: nlist[2*ncurrent+1] += 1
        #if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        return ncurrent,nlist

    def check_opt(self,totalgrad,fp,rtype):
        isDone=False
        #if rtype==self.stage: 
        # previously checked if rtype equals and 'stage' -- a previuos definition of climb/find were equal
        #if True:
        if (rtype == 2 and self.find) or (rtype==1 and self.climb):
            if self.nodes[self.TSnode].gradrms<self.options['CONV_TOL']: 
                isDone=True
                self.tscontinue=False
            if totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*self.options['CONV_TOL'] and self.dE_iter<0.02: #TODO extra crit here
                isDone=True
                self.tscontinue=False
        return isDone

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy

        #TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction!=1:
            self.nodes[-1].gradrms = 0.
            print(" Energy of the end points are %4.3f, %4.3f" %(self.nodes[0].energy,self.nodes[-1].energy))
            print(" relative E %4.3f, %4.3f" %(0.0,self.nodes[-1].energy-self.nodes[0].energy))
        else:
            print(" Energy of end points are %4.3f " % self.nodes[0].energy)
            #self.nodes[-1].energy = self.nodes[0].energy
            #self.nodes[-1].gradrms = 0.


if __name__=='__main__':
    from qchem import QChem
    from pes import PES
    from dlc_new import DelocalizedInternalCoordinates
    from eigenvector_follow import eigenvector_follow
    from _linesearch import backtrack,NoLineSearch
    from molecule import Molecule


    #basis="sto-3g"
    basis='6-31G'
    nproc=8
    #functional='HF'
    functional='B3LYP'
    filepath1="examples/tests/butadiene_ethene.xyz"
    filepath2="examples/tests/cyclohexene.xyz"
    #filepath1='reactant.xyz'
    #filepath2='product.xyz'

    lot1=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional=functional,nproc=nproc,fnm=filepath1)
    lot2 = QChem(lot1.options.copy().set_values({'fnm':filepath2}))

    pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    pes2 = PES(pes1.options.copy().set_values({'lot':lot2}))

    M1 = Molecule.from_options(fnm=filepath1,PES=pes1,coordinate_type="DLC")
    M2 = Molecule.from_options(fnm=filepath2,PES=pes2,coordinate_type="DLC")

    optimizer=eigenvector_follow.from_options(print_level=1)  #default parameters fine here/opt_type will get set by GSM

    gsm = GSM.from_options(reactant=M1,product=M2,nnodes=9,optimizer=optimizer,print_level=1)
    gsm.go_gsm(rtype=2,opt_steps=3)

