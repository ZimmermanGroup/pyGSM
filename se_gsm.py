import numpy as np
import options
import os
from base_gsm import *
from dlc import *
from pes import *
import pybel as pb
import sys


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

    def add_node(self,n1,n2,n3=None):
        print "adding node: %i from node %i" %(n2,n1)
        return DLC.add_node_SE(self.icoords[n1],self.driving_coords)

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
        print(" Here is active:",self.active)
        print(" setting active node to %i "%nR)

        for i in range(self.nnodes):
            if self.icoords[i] !=0:
                self.active[i] = False;
                self.icoords[i].OPTTHRESH = self.CONV_TOL*2.;
        self.icoords[nR].OPTTHRESH = self.ADD_NODE_TOL
        self.active[nR] = True
        print(" Here is new active:",self.active)

    def tangent(self,n1,n2):
        if n2 ==self.nR-1:
            return DLC.tangent_SE(self.icoords[n1],self.icoords[n2])
        else:
            return DLC.tangent_1(self.icoords[n1],self.icoords[n2])


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

    if True:
        filepath="tests/fluoroethene.xyz"
        filepath2="tests/stretched_fluoroethene.xyz"
        nocc=11
        nactive=2

    if False:
        filepath2="tests/SiH2H2.xyz"
        filepath="tests/SiH4.xyz"
        nocc=8
        nactive=2

    mol=pb.readfile("xyz",filepath).next()
    mol2=pb.readfile("xyz",filepath2).next()
    if QCHEM:
        lot=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
        lot2=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
    
    if ORCA:
        lot=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
        lot2=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
    if PYTC:
        lot=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot.cas_from_file(filepath)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

    print "\n IC1 \n"
    ic1=DLC.from_options(mol=mol,PES=pes)

    if True:
        print "\n Starting GSM \n"
        gsm=SE_GSM.from_options(ICoord1=ic1,nnodes=9,nconstraints=1,CONV_TOL=0.001,driving_coords=[("BREAK",1,2),("TORSION",5,2,1,3,90.)])
        print "about to interpolate"
        gsm.interpolate(4)
        gsm.write_node_xyz()

    if False:
        print DLC.tangent_1(gsm.icoords[0],gsm.icoords[-1])
    
    if False:
        gsm.get_tangents_1(n0=0)

    if False:
        gsm.ic_reparam_g()


    if False:
        gsm.grow_string(50)
        #gsm.growth_iters(iters=50,maxopt=3,nconstraints=1)
        #gsm.opt_iters()
        if ORCA:
            os.system('rm temporcarun/*')

