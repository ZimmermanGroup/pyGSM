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

        self.icoords[0].setup()

    def go_gsm(self,max_iters,max_steps):
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
        for n in range(1,self.nnodes):
            self.active[n]==True

        print " initial ic_reparam"
        self.ic_reparam()

        if self.tscontinue==True:
            self.opt_iters(max_iter=max_iters,optsteps=max_steps)
        else:
            print "Exiting early"

        print "Finished GSM!"  


    def add_node(self,n1,n2,n3=None):
        print "adding node: %i from node %i" %(n2,n1)
        return DLC.add_node_SE(self.icoords[n1],self.driving_coords)

    def add_last_node(self,rtype):
        samegeom=False
        if rtype==1:
            print "copying last node, opting"
            self.icoords[self.nR] = DLC.copy_node(self.icoords[self.nR-1],self.nR)
        elif rtype==2:
            print "already created node, opting"
        noptsteps=15
        self.optimize(n=self.nR,nsteps=noptsteps)
        if (self.icoords[self.nR].coords == self.icoords[self.nR-1].coords).all():
            samegeom=True

        if samegeom:
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
                self.active[i] = False;
                self.icoords[i].OPTTHRESH = self.CONV_TOL*2.;
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
            #print" getting tangent from node ",n2
            return DLC.tangent_SE(self.icoords[n2],self.driving_coords)
        elif self.icoords[n2]!=0 and self.icoords[n1]!=0: 
            #print" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2)
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


if __name__ == '__main__':
#    from icoord import *
    ORCA=False
    QCHEM=False
    PYTC=True
    nproc=8

    if QCHEM:
        from qchem import *
    if ORCA:
        from orca import *
    if PYTC:
        from pytc import *
    import manage_xyz

    if False:
        filepath="tests/fluoroethene.xyz"
        nocc=11
        nactive=2
    if False:
        filepath="tests/ethylene.xyz"
        nocc=7
        nactive=2

    if False:
        filepath="tests/SiH4.xyz"
        nocc=8
        nactive=2

    if True:
        filepath="tests/butadiene_ethene.xyz"
        nocc=21
        nactive=4

    mol=pb.readfile("xyz",filepath).next()
    if QCHEM:
        lot=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
    if ORCA:
        lot=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
    if PYTC:
        lot=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31g')
        lot.cas_from_file(filepath)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

    print "\n IC1 \n"
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=0)

    if True:
        print "\n Starting GSM \n"
        #gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=9,nconstraints=1,CONV_TOL=0.001,driving_coords=[("TORSION",2,1,4,6,90.)])
        gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,CONV_TOL=0.001,driving_coords=[("ADD",6,4),("ADD",5,1)],ADD_NODE_TOL=0.05,tstype=2)
        gsm.go_gsm(max_iters=30,max_steps=3)


