import numpy as np
import options
import os
from se_gsm import *
from dlc import *
from penalty_pes import *
import pybel as pb
import sys


class SE_Cross(SE_GSM):

    @staticmethod
    def from_options(**kwargs):
        return SE_Cross(SE_Cross.default_options().set_values(kwargs))

    def go_gsm(self,iters):
        self.icoords[0].gradrms=0.
        self.icoords[0].energy = self.icoords[0].PES.get_energy(self.icoords[0].geom)
        print 'initial energy is {:1.4f}'.format(self.icoords[0].energy)
        sys.stdout.flush()
        self.interpolate(1)
        self.icoords[1].energy = self.icoords[1].PES.get_energy(self.icoords[1].geom)
        self.growth_iters(iters=iters,maxopt=3,nconstraints=self.nconstraints)

        print 'SE_Cross growth phase over'
        print 'Warning last node still not fully optimized'

        if self.check_if_grown():
            self.icoords[self.nR] = DLC.copy_node_X(self.icoords[self.nR-2],self.nR)
            self.nR += 1
        self.optimize(n=self.nR-1,nsteps=50)
            
        self.write_xyz_files(iters=1,base="grown_string",nconstraints=self.nconstraints)

    def add_node(self,n1,n2,n3=None):
        print "adding node: %i from node %i"%(n2,n1)
        return DLC.add_node_SE_X(self.icoords[n1],self.driving_coords)
    
#    def opt_steps(self,maxopt,nconstraints):
#        for i in range(1):
#            for n in range(self.nnodes):
#                if self.icoords[n] != 0 and self.active[n]==True:
#                    print "optimizing node %i" % n
#                    self.icoords[n].opt_constraint(self.ictan[n])
#                    self.icoords[n].smag = self.optimize(n,maxopt,nconstraints)

    def check_if_grown(self):
        isDone = False
        epsilon = 1.5
        pes1dE = self.icoords[self.nR-1].PES.dE
        pes2dE = self.icoords[self.nR-2].PES.dE
        if abs(pes1dE) < epsilon:
            if abs(self.icoords[self.nR-1].bdist) <= 0.5 * abs(self.icoords[1].bdist) and abs(pes1dE) > abs(pes2dE):
                isDone = True
        return isDone

    def check_add_node(self):
        if self.icoords[self.nR-1].gradrms < self.gaddmax:
            self.active[self.nR-1] = False
            if self.nR == self.nnodes:
                print " Ran out of nodes, exiting GSM"
                raise ValueError
            if self.icoords[self.nR] == 0:
                self.interpolateR()


if __name__ == '__main__':
    ORCA=False
    QCHEM=True
    PYTC=False
    nproc=8
    
    ethylene=False
    sih4=False
    fluoroethene=False
    butadiene_ethene=False
    FeCO5=False
    FeO_H2=False
    NiL2Br2=False
    NiL2Br2_tetr=True

    if QCHEM: from qchem import *
    elif ORCA: from orca import *
    elif PYTC: from pytc import *

    states = [(1,0),(3,0)]
    basis = '6-31G(d)'
    charge=0

    if fluoroethene:
        filepath = 'tests/fluoroethene.xyz'
        nocc=11
        nactive=2
        driving_coords = [('BREAK',1,2,0.2)]
    elif ethylene:
        filepath = 'tests/ethylene.xyz'
        nocc=7
        nactive=2
        driving_coords = [('BREAK',1,2,0.2)]
    elif sih4:
        filepath = 'tests/SiH4.xyz'
        nocc=8
        nactive=2
        driving_coords = [('ADD',3,4,0.2),('BREAK',1,3,0.2),("BREAK",1,4,0.2)]
    elif butadiene_ethene:
        filepath = 'tests/butadiene_ethene.xyz'
        nocc=21
        nactive=4
    elif FeCO5:
        filepath = 'tests/FeCO5.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('BREAK',1,6,0.2)]
    elif FeO_H2:
        filepath = 'tests/FeO_H2.xyz'
        states = [(4,0),(6,0)]
        driving_coords = [('ADD',1,3,0.2),('ADD',2,4,0.2)]
        charge=1
    elif NiL2Br2:
        filepath = 'tests/NiL2Br2_sqpl.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('TORSION',18,12,13,23,10.)]
    elif NiL2Br2_tetr:
        filepath = 'tests/NiL2Br2_tetr.xyz'
        states = [(1,0),(3,0)]
        driving_coords = [('TORSION',16,14,1,13,10.)]

    mol = pb.readfile('xyz',filepath).next()
    if QCHEM:
        lot = QChem.from_options(states=states,charge=charge,basis=basis,functional='B3LYP',nproc=nproc)
    elif ORCA:
        lot = Orca.from_options(states=states,charge=charge,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
    elif PYTC:
        lot = PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31g')
        lot.cas_from_file(filepath)

    pes1 = PES.from_options(lot=lot,ad_idx=0,multiplicity=states[0][0])
    pes2 = PES.from_options(lot=lot,ad_idx=0,multiplicity=states[1][0])
    pes = Penalty_PES(pes1,pes2)
    print ' IC1 '
    ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1,resetopt=False)

    if True:
        print ' Starting GSM '
        gsm = SE_Cross.from_options(ICoord1=ic1,nnodes=20,nconstraints=1,CONV_TOL=0.001,driving_coords=driving_coords,ADD_NODE_TOL=0.05)
        gsm.go_gsm(20)


