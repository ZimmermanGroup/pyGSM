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
from coordinate_systems import Distance,Angle,Dihedral,OutOfPlane,TranslationX,TranslationY,TranslationZ,RotationA,RotationB,RotationC


class SE_GSM(Base_Method):

    def __init__(
            self,
            options,
            ):
        super(SE_GSM,self).__init__(options)
        self.nn=1

        print(" Assuming the isomers are initialized!")
        #self.isomer_init()

        print(" Done initializing isomer")
        #self.nodes[0].form_Primitive_Hessian()
        print(" Primitive Internal Coordinates")
        print(self.nodes[0].primitive_internal_coordinates[0:50])
        print(" number of primitives is", self.nodes[0].num_primitives)

        sys.stdout.flush()
        
        #self.reference_frag_xyz = []
        self.reference_xyz = None
        for i in self.driving_coords:
            if "ROTATE" in i:
                self.reference_xyz = self.nodes[0].xyz
                break

        # stash bdist for node 0
        ictan,self.nodes[0].bdist = Base_Method.tangent(self.nodes[0],None,driving_coords=self.driving_coords,reference_xyz=self.reference_xyz)
        self.nodes[0].update_coordinate_basis(constraints=ictan)
        self.set_V0()


    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy
        #TODO should be actual gradient
        self.nodes[0].gradrms = 0.

    def isomer_init(self):
        '''
        The purpose of this function is to add to the primitives the driving coordinate prims if 
        they dont exist.
        This is depracated because it's better to build the topology properly before initializing
        GSM. See main.py
        '''

        #TODO ANGLE, TORSION or OOP between fragments will not work if using TRIC with BLOCK LA
        changed_top = False

        #TODO first check if there is any add/break then rebuild topology and makePrimitives

        for i in self.driving_coords:
            if "ADD" in i or "BREAK" in i:
                # order 
                if i[1]<i[2]:
                    bond = Distance(i[1]-1,i[2]-1)
                else:
                    bond = Distance(i[2]-1,i[1]-1)
                self.nodes[0].coord_obj.Prims.add(bond,verbose=True)
                changed_top =True
            if "ANGLE" in i:
                if i[1]<i[3]:
                    angle = Angle(i[1]-1,i[2]-1,i[3]-1)
                else:
                    angle = Angle(i[3]-1,i[2]-1,i[1]-1)
                self.nodes[0].coord_obj.Prims.add(angle,verbose=True)
            if "TORSION" in i:
                if i[1]<i[4]:
                    torsion = Dihedral(i[1]-1,i[2]-1,i[3]-1,i[4]-1)
                else:
                    torsion = Dihedral(i[4]-1,i[3]-1,i[2]-1,i[1]-1)
                self.nodes[0].coord_obj.Prims.add(torsion,verbose=True)
            if "OOP" in i:
                if i[1]<i[4]:
                    oop = OutOfPlane(i[1]-1,i[2]-1,i[3]-1,i[4]-1)
                else:
                    oop = OutOfPlane(i[4]-1,i[3]-1,i[2]-1,i[1]-1)
                self.nodes[0].coord_obj.Prims.add(oop,verbose=True)

        self.nodes[0].coord_obj.Prims.clearCache()
        if changed_top:
            self.nodes[0].coord_obj.Prims.rebuild_topology_from_prim_bonds(self.nodes[0].xyz)
        self.nodes[0].coord_obj.Prims.reorderPrimitives()
        self.nodes[0].update_coordinate_basis()

    def go_gsm(self,max_iters=50,opt_steps=10,rtype=2):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find, 
        0 turning of climbing image and TS search
        """

        if self.isRestarted==False:
            self.nodes[0].gradrms = 0.
            self.nodes[0].V0 = self.nodes[0].energy
            print(" Initial energy is %1.4f" % self.nodes[0].energy)
            self.add_GSM_nodeR()
            self.growth_iters(iters=max_iters,maxopt=opt_steps)
            if self.tscontinue:
                if self.pastts==1: #normal over the hill
                    self.add_GSM_nodeR(1)
                    self.add_last_node(2)
                elif self.pastts==2 or self.pastts==3: #when cgrad is positive
                    self.add_last_node(1)
                    if self.nodes[self.nR-1].gradrms>5.*self.options['CONV_TOL']:
                        self.add_last_node(1)
                elif self.pastts==3: #product detected by bonding
                    self.add_last_node(1)

            self.nnodes=self.nR
            tmp = []
            for n in range(self.nnodes):
                tmp.append(self.energies[n])
            self.energies = np.asarray(tmp)
            self.emax = self.energies[self.TSnode]

            if self.TSnode == self.nR:
                print(" The highest energy node is the last")
                print(" not continuing with TS optimization.")
                self.tscontinue=False

            print(" Number of nodes is ",self.nnodes)
            print(" Warning last node still not optimized fully")
            self.write_xyz_files(iters=1,base='grown_string',nconstraints=1)
            print(" SSM growth phase over")
            self.done_growing=True

            print(" beginning opt phase")
            print("Setting all interior nodes to active")
            for n in range(1,self.nnodes-1):
                self.active[n]=True
            self.active[self.nnodes-1] = False
            self.active[0] = False

        if not self.isRestarted:
            print(" initial ic_reparam")
            self.ic_reparam(25)
            self.store_energies()
            print(" V_profile (after reparam): ", end=' ')
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(self.energies[n])), end=' ')
            print()
            self.write_xyz_files(iters=1,base='grown_string1',nconstraints=1)

        if self.tscontinue:
            self.opt_iters(max_iter=max_iters,optsteps=3,rtype=rtype) #opt steps fixed at 3 for rtype=1 and 2, else set it to be the large number :) muah hahaahah
        else:
            print("Exiting early")

        print("Finished GSM!")  


    def add_last_node(self,rtype):
        assert rtype==1 or rtype==2, "rtype must be 1 or 2"
        samegeom=False
        noptsteps=100
        if self.nodes[self.nR-1].PES.lot.do_coupling:
            opt_type='MECI'
        else:
            opt_type='UNCONSTRAINED'

        if rtype==1:
            print(" copying last node, opting")
            #self.nodes[self.nR] = DLC.copy_node(self.nodes[self.nR-1],self.nR)
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR-1],new_node_id=self.nR)
            print(" Optimizing node %i" % self.nR)
            self.optimizer[self.nR].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR].optimize(
                        molecule=self.nodes[self.nR],
                        refE=self.nodes[0].V0,
                        opt_steps=noptsteps,
                        opt_type=opt_type,
                        )
            self.active[self.nR]=True
            if (self.nodes[self.nR].xyz == self.nodes[self.nR-1].xyz).all():
                print(" Opt did not produce new geometry")
            else:
                self.nR+=1
        elif rtype==2:
            print(" already created node, opting")
            self.optimizer[self.nR-1].conv_grms = self.options['CONV_TOL']
            self.optimizer[self.nR-1].optimize(
                        molecule=self.nodes[self.nR-1],
                        refE=self.nodes[0].V0,
                        opt_steps=noptsteps,
                        opt_type=opt_type,
                        )
        #print(" Aligning")
        #self.nodes[self.nR-1].xyz = self.com_rotate_move(self.nR-2,self.nR,self.nR-1) 
        return

    def check_add_node(self):
        success=True
        #if self.nodes[self.nR-1].gradrms < self.gaddmax:
        #if self.nodes[self.nR-1].gradrms < self.options['ADD_NODE_TOL']:
        if self.optimizer[self.nR-1].converged:
            if self.nR == self.nnodes:
                print(" Ran out of nodes, exiting GSM")
                raise ValueError
            if self.nodes[self.nR] == None:
                success=self.add_GSM_nodeR()
            else:
                self.active[self.nR-1] = False
        return success

    def add_GSM_nodes(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.add_GSM_nodeR()

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        self.get_tangents_1g()
        return

    def set_active(self,nR,nP=None):
        #print(" Here is active:",self.active)
        print((" setting active node to %i "%nR))

        for i in range(self.nnodes):
            if self.nodes[i] != None:
                self.active[i] = False
                self.optimizer[i].conv_grms = self.options['CONV_TOL']
                print(" conv_tol of node %d is %.4f" % (i,self.optimizer[i].conv_grms))
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        print(" conv_tol of node %d is %.4f" % (nR,self.optimizer[nR].conv_grms))
        #self.optimizer[nR].conv_grms = self.options['CONV_TOL']*2
        self.active[nR] = True
        #print(" Here is new active:",self.active)

    def make_nlist(self):
        ncurrent =0
        nlist = [0]*(2*self.nnodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1
        nlist[2*ncurrent+1] = self.nR -1
        nlist[2*ncurrent] = self.nR -1
        ncurrent += 1
        return ncurrent,nlist

    def check_if_grown(self):
        self.pastts = self.past_ts()
        isDone=False
        #TODO break planes
        condition1 = (abs(self.nodes[self.nR-1].bdist) <=(1-self.BDIST_RATIO)*abs(self.nodes[0].bdist))
        print(" bdist %.3f" % self.nodes[self.nR-1].bdist)

        fp = self.find_peaks(1)
        if self.pastts and self.nn>3 and condition1: #TODO extra criterion here
            print(" pastts is ",self.pastts)
            if self.TSnode == self.nR-1:
                print(" The highest energy node is the last")
                print(" not continuing with TS optimization.")
                self.tscontinue=False
            nifty.printcool("Over the hill")
            isDone=True
        elif fp==-1 and self.energies[self.nR-1]>200. and self.nodes[self.nR-1].gradrms>self.options['CONV_TOL']*5:
            print("growth_iters over: all uphill and high energy")
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True
        elif fp==-2:
            print("growth_iters over: all uphill and flattening out")
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True

        # ADD extra criteria here to check if TS is higher energy than product
        return isDone

    def check_opt(self,totalgrad,fp,rtype):
        isDone=False
        added=False
        if self.TSnode == self.nnodes-2 and (self.find or totalgrad<0.2) and fp==1:
            if self.nodes[self.nR-1].gradrms>self.options['CONV_TOL']:
                print("TS node is second to last node, adding one more node")
                self.add_last_node(1)
                self.nnodes=self.nR
                self.active[self.nnodes-1]=False #GSM makes self.active[self.nnodes-1]=True as well
                self.active[self.nnodes-2]=True #GSM makes self.active[self.nnodes-1]=True as well
                added=True
                print("done adding node")
                print("nnodes = ",self.nnodes)
                self.get_tangents_1()
            return isDone

        # => check string profile <= #
        if fp==-1: #total string is uphill
            print("fp == -1, check V_profile")

            #if self.tstype==2:
            #    print "check for total dissociation"
            #    #check break
            #    #TODO
            #    isDone=True
            #if self.tstype!=2:
            #    print "flatland? set new start node to endpoint"
            #    self.tscontinue=0
            #    isDone=True
            print("total dissociation")
            self.tscontinue
            isDone=True

        if fp==-2:
            print("termination due to dissociation")
            self.tscontinue=False
            self.endearly=True #bools
            isDone=True
        if fp==0:
            self.tscontinue=False
            self.endearly=True #bools
            isDone=True

        # check for intermediates
        #if self.stage==1 and fp>0:
        if self.climb and fp>0:
            fp=self.find_peaks(2)
            if fp>1:
                rxnocc,wint = self.check_for_reaction()
            if fp >1 and rxnocc and wint<self.nnodes-1:
                print("Need to trim string")
                #self.tscontinue=False
                #isDone=True
                #return isDone

        # => Convergence Criteria
        dE_iter = abs(self.emaxp - self.emax)
        TS_conv = self.options['CONV_TOL']
        if self.find and self.optimizer[self.TSnode].nneg>1:
            print(" reducing TS convergence because nneg>1")
            TS_conv = self.options['CONV_TOL']/2.
        self.optimizer[self.TSnode].conv_grms = TS_conv

        if (rtype == 2 and self.find ) or (rtype==1 and self.climb):
            if self.nodes[self.TSnode].gradrms< TS_conv:
                self.tscontinue=False
                isDone=True
                #print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)
                return isDone
            if totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*TS_conv and dE_iter < 0.02:
                self.tscontinue=False
                isDone=True
                #print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)
                return isDone

if __name__=='__main__':
    from .qchem import QChem
    from .pes import PES
    from .dlc_new import DelocalizedInternalCoordinates
    from .eigenvector_follow import eigenvector_follow
    from ._linesearch import backtrack,NoLineSearch
    from .molecule import Molecule

    basis='6-31G'
    nproc=8
    functional='B3LYP'
    filepath1="examples/tests/butadiene_ethene.xyz"
    lot1=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional=functional,nproc=nproc,fnm=filepath1)
    pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    M1 = Molecule.from_options(fnm=filepath1,PES=pes1,coordinate_type="DLC")
    optimizer=eigenvector_follow.from_options(print_level=1)  #default parameters fine here/opt_type will get set by GSM

    gsm = SE_GSM.from_options(reactant=M1,nnodes=20,driving_coords=[("ADD",6,4),("ADD",5,1)],optimizer=optimizer,print_level=1)
    gsm.go_gsm()
