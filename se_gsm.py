import numpy as np
import options
import os
from base_gsm import *
from dlc import *
from pes import *
import pybel as pb


class SE_GSM(Base_Method):

    @staticmethod
    def from_options(**kwargs):
        return SE_GSM(SE_GSM.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        super(SE_GSM,self).__init__(options)
        self.nn=1
        self.isomer_init()

    def isomer_init(self):
        for i in self.driving_coords:
            if "ADD" or "BREAK" in i:
                bond=(i[1],i[2])
                if not self.icoords[0].bond_exists(bond):
                    print "adding bond ",bond
                    self.icoords[0].mol.OBMol.AddBond(bond[0],bond[1],1)
                    self.icoords[0].BObj.bonds.append(bond)
            if "ANGLE" in i:
                angle=(i[1],i[2],i[3])
                if not self.icoords[0].angle_exists(angle):
                    self.icoords[0].AObj.angles.append(bond)
                    print "adding angle ",angle
            if "TORSION" in i:
                torsion=(i[1],i[2],i[3],i[4])
                if not self.icoords[0].torsion_exists(torsion):
                    print "adding torsion ",torsion
                    self.icoords[0].TObj.torsions.append(torsion)

        self.icoords[0].madeBonds=True
        self.icoords[0].setup()

    def go_gsm(self,max_iters,max_steps):
        if self.isRestarted==False:
            self.icoords[0].gradrms = 0.
            self.icoords[0].energy = self.icoords[0].PES.get_energy(self.icoords[0].geom)
            print " Initial energy is %1.4f" % self.icoords[0].energy
            self.interpolate(1) 
            self.icoords[1].energy = self.icoords[1].PES.get_energy(self.icoords[1].geom)
            self.growth_iters(iters=max_iters,maxopt=max_steps)
            if self.tscontinue==True:
                if self.pastts==1: #normal over the hill
                    #self.add_node(self.nR-1,self.nR)
                    self.interpolateR(1)
                    self.add_last_node(2)
                elif self.pastts==2 or self.pastts==3: #when cgrad is positive
                    self.add_last_node(1)
                    if self.icoords[self.nR-1].gradrms>5.*self.CONV_TOL:
                        self.add_last_node(1)
                elif self.pastts==3: #product detected by bonding
                    self.add_last_node(1)

            self.nnodes=self.nR
            print " Number of nodes is ",self.nnodes
            print " Warning last node still not optimized fully"
            self.write_xyz_files(iters=1,base='grown_string',nconstraints=1)
            print " SSM growth phase over"

            print " beginning opt phase"
            print "Setting all interior nodes to active"
            for n in range(1,self.nnodes-1):
                self.active[n]=True
                self.icoords[n].OPTTHRESH=self.CONV_TOL

        print " initial ic_reparam"
        self.ic_reparam()
        if self.tscontinue==True:
            self.opt_iters(max_iter=max_iters,optsteps=3) #opt steps fixed at 3
        else:
            print "Exiting early"

        print "Finished GSM!"  


    def add_node(self,n1,n2,n3=None):
        print "adding node: %i from node %i" %(n2,n1)
        return DLC.add_node_SE(self.icoords[n1],self.driving_coords)

    def add_last_node(self,rtype):
        assert rtype==1 or rtype==2, "rtype must be 1 or 2"
        samegeom=False
        if rtype==1:
            print "copying last node, opting"
            self.icoords[self.nR] = DLC.copy_node(self.icoords[self.nR-1],self.nR)
        elif rtype==2:
            print "already created node, opting"
        noptsteps=15
        print " Optimizing node %i" % self.nR
        self.icoords[self.nR].OPTTHRESH = self.CONV_TOL
        self.optimize(n=self.nR,nsteps=noptsteps)
        self.active[self.nR]=True
        if (self.icoords[self.nR].coords == self.icoords[self.nR-1].coords).all():
            print "Opt did not produce new geometry"
        else:
            self.nR+=1
        return


    def check_add_node(self):
        if self.icoords[self.nR-1].gradrms < self.gaddmax:
            self.active[self.nR-1] = False
            if self.icoords[self.nR] == 0:
                self.interpolateR()

    def interpolate(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.interpolateR()

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        pass

    def set_active(self,nR,nP=None):
        #print(" Here is active:",self.active)
        print(" setting active node to %i "%nR)

        for i in range(self.nnodes):
            if self.icoords[i] !=0:
                self.active[i] = False
                self.icoords[i].OPTTHRESH = self.CONV_TOL*2.
        self.icoords[nR].OPTTHRESH = self.ADD_NODE_TOL
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

    def tangent(self,n1,n2):
        if n2 ==self.nR-1:
            print" getting tangent from node ",n2
            return DLC.tangent_SE(self.icoords[n2],self.driving_coords)
        elif self.icoords[n2]!=0 and self.icoords[n1]!=0: 
            print" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2)
            return DLC.tangent_1(self.icoords[n2],self.icoords[n1])
        else:
            raise ValueError("can't make tan")

    def check_if_grown(self):
        self.pastts = self.past_ts()
        isDone=False
        #TODO break planes
        if self.pastts and self.nn>3: #TODO extra criterion here
            print "pastts is ",self.pastts
            isDone=True
        fp = self.find_peaks(1)
        if fp==-1 and self.energies[self.nR-1]>200.:
            print "growth_iters over: all uphill and high energy"
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True
        if fp==-2:
            print "growth_iters over: all uphill and flattening out"
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True
        return isDone

    def check_opt(self,totalgrad,fp):
        isDone=False
        added=False
        if self.nmax==self.nnodes-2 and (self.stage==2 or totalgrad<0.2) and fp==1:
            if self.icoords[self.nR-1].gradrms>self.CONV_TOL:

                print "TS node is second to last node, adding one more node"
                self.add_last_node(1)
                self.nnodes=self.nR
                self.active[self.nnodes-1]=False #GSM makes self.active[self.nnodes-1]=True as well
                self.active[self.nnodes-2]=True #GSM makes self.active[self.nnodes-1]=True as well
                added=True
                print "done adding node"
                print "nnodes = ",self.nnodes
                self.get_tangents_1()
            return isDone

        # => check string profile <= #
        if fp==-1: #total string is uphill
            print "fp == -1, check V_profile"

            if self.tstype==2:
                print "check for total dissociation"
                #check break
                #TODO
                isDone=True
            if self.tstype!=2:
                print "flatland? set new start node to endpoint"
                self.tscontinue=0
                isDone=True

        if fp==--2:
            print "termination due to dissociation"
            self.tscontinue=False
            self.endearly=True #bools
            isDone=True

        # check for intermediates
        if self.stage==1 and fp>0:
            fp=self.find_peaks(3)
            if fp>1:
                rxnocc,wint = self.check_for_reaction()
            if fp >1 and rxnocc==True and wint<self.nnodes-1:
                print "Need to trim string"
                isDone=True

        # => Convergence Criteria
        if (((self.stage==1 and self.tstype==1) or self.stage==2) and
                self.icoords[self.TSnode].gradrms< self.CONV_TOL):
            self.tscontinue=False
            isDone=True
        if (((self.stage==1 and self.tstype==1) or self.stage==2) and totalgrad<0.1 and
                self.icoords[self.TSnode].gradrms<2.5*self.CONV_TOL and self.emaxp+0.02> self.emax and 
                self.emaxp-0.02< self.emax):
            self.tscontinue=False
            isDone=True

        return isDone

