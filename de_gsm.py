import numpy as np
import options
import os
from base_gsm import *
#from dlc import *
from pes import *
import pybel as pb
import sys

class GSM(Base_Method):
    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):

        super(GSM,self).__init__(options)
        exit()

        tmp = self.options['product']
        tmp.PES.lot.node_id = self.nnodes-1
        self.nodes[0] = DLC.union_ic(self.nodes[0],tmp)
        self.nodes[0].Hintp = self.nodes[0].make_Hint()
        print "after union"

        if self.growth_direction !=1:
            print "copying lot for end node"
            lot1 = tmp.PES.lot.copy(
                    tmp.PES.lot, 
                    self.nnodes-1)
        else: 
            lot1 = self.nodes[0].PES.lot.copy(
                    self.nodes[0].PES.lot, 
                    self.nnodes-1)

        if self.nodes[0].PES.__class__.__name__=="Avg_PES":
            PES1 = Avg_PES(self.nodes[0].PES.PES1,self.nodes[0].PES.PES2,lot1)
        else:
            PES1 = PES(self.nodes[0].PES.options.copy().set_values({
                "lot": lot1,
                }))
        self.nodes[-1] = DLC(self.nodes[0].options.copy().set_values(dict(
            mol= tmp.mol,
            PES=PES1,
            )))
        self.nodes[-1].Hintp = self.nodes[-1].make_Hint()

        print "print levels at beginning are ",self.nodes[0].print_level
        if self.growth_direction !=1:
            print "print levels at beginning are ",self.nodes[-1].print_level

    def go_gsm(self,max_iters=50,opt_steps=3,rtype=2):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find, 
        0 turning of climbing image and TS search
        """

        self.set_V0()
        if not self.isRestarted:
            if self.growth_direction==0:
                self.interpolate(2) 
            elif self.growth_direction==1:
                self.interpolateR(1)
            elif self.growth_direction==2:
                self.interpolateP(1)
            oi = self.growth_iters(iters=max_iters,maxopt=opt_steps) 
            print("Done Growing the String!!!")
            self.write_xyz_files(iters=1,base='grown_string',nconstraints=1)
            self.done_growing = True
            print " initial ic_reparam"
            self.get_tangents_1()
            self.ic_reparam(ic_reparam_steps=25)
            self.write_xyz_files(iters=1,base='initial_ic_reparam',nconstraints=1)
    
            # reset hessians :(
            print "resetting Hessians"
            for i in range(self.nnodes):
                self.nodes[0].Hintp = self.nodes[i].make_Hint()
        else:
            oi=0
            self.get_tangents_1()
        for i in range(self.nnodes):
            if self.nodes[i] !=0:
                self.optimizer[i].options['OPTTHRESH'] = self.options['CONV_TOL']*2

        if self.tscontinue==True:
            if max_iters-oi>0:
                opt_iters=max_iters-oi
                self.opt_iters(max_iter=opt_iters,optsteps=opt_steps,rtype=rtype)
        else:
            print "Exiting early"
        print "Finished GSM!"  

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
        print " adding node: %i between %i %i" %(n2,n1,n3)
        # calcDiff using DLC class between node1.xyz and node2.xyz

        return DLC.add_node(self.nodes[n1],self.nodes[n3],self.nnodes,self.nn)

    def set_active(self,nR,nP):
        #print(" Here is active:",self.active)
        if nR!=nP and self.growth_direction==0:
            print(" setting active nodes to %i and %i"%(nR,nP))
        elif self.growth_direction==1:
            print(" setting active node to %i "%nR)
        elif self.growth_direction==2:
            print(" setting active node to %i "%nP)
        else:
            print(" setting active node to %i "%nR)

        for i in range(self.nnodes):
            if self.nodes[i] !=0:
                self.optimizer[i].options['OPTTHRESH'] = self.options['CONV_TOL']*2.
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction==1:
            self.active[nP]=False
        if self.growth_direction==2:
            self.active[nR]=False
        #print(" Here is new active:",self.active)

    def tangent(self,n1,n2):
        #print" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2)
        return DLC.tangent_1(self.nodes[n2],self.nodes[n1])

    def check_if_grown(self):
        isDone=False
        if self.nn==self.nnodes:
            isDone=True
            if self.growth_direction==1:
                self.nodes[-1].update_ics()
                # copy previous node PES and calculate E
                lot1 = self.nodes[-2].PES.lot.copy(
                        self.nodes[-2].PES.lot,
                        self.nnodes-1)
                if self.nodes[-2].PES.__class__.__name__=="Avg_PES":
                    self.nodes[-1].PES = Avg_PES(self.nodes[-2].PES.PES1,self.nodes[2].PES.PES2,lot1)
                else:
                    self.nodes[-1].PES = PES(self.nodes[-2].PES.options.copy().set_values({
                        "lot": lot1,
                        }))
                self.nodes[-1].energy = self.nodes[-1].PES.get_energy(self.nodes[-1].geom)
                if self.nodes[-1].PES.__class__.__name__=="Avg_PES":
                    print "final dE = ",self.nodes[-1].PES.dE

        return isDone

    def check_add_node(self):
        success=True 
        if self.nodes[self.nR-1].gradrms < self.gaddmax and self.growth_direction!=2:
            #self.active[self.nR-1] = False
            if self.nodes[self.nR] == 0:
                self.interpolateR()
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction!=1:
            #self.active[self.nnodes-self.nP] = False
            if self.nodes[-self.nP-1] == 0:
                self.interpolateP()
        return success

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
        if self.nR == 0: nlist[2*ncurrent] += 1
        if self.nP == 0: nlist[2*ncurrent+1] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes -self.nP
        nlist[2*ncurrent+1] = self.nR-1
        #TODO is this actually used?
        if self.nR == 0: nlist[2*ncurrent+1] += 1
        if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        return ncurrent,nlist

    def check_opt(self,totalgrad,fp,rtype):
        isDone=False
        #if rtype==self.stage: 
        # previously checked if rtype equals and 'stage' -- a previuos definition of climb/find were equal
        if True:
            if self.nodes[self.TSnode].gradrms<self.options['CONV_TOL'] and self.dE_iter<0.1: #TODO should check totalgrad
                isDone=True
                self.tscontinue=False
            if totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*self.options['CONV_TOL']: #TODO extra crit here
                isDone=True
                self.tscontinue=False
        return isDone

    def set_V0(self):
        self.nodes[0].gradrms = 0.
        self.nodes[0].energy = self.nodes[0].V0 = self.nodes[0].PES.get_energy(self.nodes[0].geom)
        if self.growth_direction!=1:
            self.nodes[-1].energy = self.nodes[-1].PES.get_energy(self.nodes[-1].geom)
            self.nodes[-1].gradrms = 0.
            print " Energy of the end points are %4.3f, %4.3f" %(self.nodes[0].energy,self.nodes[-1].energy)
            print " relative E %4.3f, %4.3f" %(0.0,self.nodes[-1].energy-self.nodes[0].energy)
        else:
            print " Energy of end points are %4.3f " % self.nodes[0].energy
            self.nodes[-1].energy = self.nodes[0].energy
            self.nodes[-1].gradrms = 0.

    def print_msg(self):
        msg="""
        __        __   _                            _        
        \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
         \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
          \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
           \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
                                        ____ ____  __  __ 
                           _ __  _   _ / ___/ ___||  \/  |
                          | '_ \| | | | |  _\___ \| |\/| |
                          | |_) | |_| | |_| |___) | |  | |
                          | .__/ \__, |\____|____/|_|  |_|
                          |_|    |___/                    
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |# 
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#
               """

        print msg


if __name__=='__main__':
    from qchem import QChem
    from pes import PES
    #from dlc import DLC
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

    geom1 = manage_xyz.read_xyz(filepath1)
    geom2 = manage_xyz.read_xyz(filepath2)

    lot1=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional=functional,nproc=nproc,geom=geom1)
    lot2 = QChem(lot1.options.copy().set_values({'geom':geom2}))

    pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    pes2 = PES(pes1.options.copy().set_values({'lot':lot2}))

    M1 = Molecule.from_options(fnm=filepath1,PES=pes1)
    M2 = Molecule.from_options(fnm=filepath2,PES=pes2)
   
    DLC1 = DelocalizedInternalCoordinates.from_options(molecule=M1)
    DLC2 = DelocalizedInternalCoordinates.from_options(molecule=M2)
    DLC_Union = DelocalizedInternalCoordinates.union(DLC1,DLC2)

    optimizer=eigenvector_follow.from_options(print_level=1)  #default parameters fine here/opt_type will get set by GSM

    gsm = GSM.from_options(reactant=M1,product=M2,nnodes=9,optimizer=optimizer,DLC=DLC_Union)
    exit()


    gsm.restart_string()
    gsm.climb=gsm.find=True
    print gsm.nnodes
    gsm.TSnode=4
    gsm.get_tangents_1()
    gsm.get_eigenv_finite(4)

    #gsm.tscontinue=False
    gsm.go_gsm(rtype=2,opt_steps=3)

