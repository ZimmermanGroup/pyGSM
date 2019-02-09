import numpy as np
import openbabel as ob
import pybel as pb
import options
import elements 
import os
from units import *
import itertools
import manage_xyz
from _icoord import ICoords
from dlc import DLC
#from _hbmat import HBmat
from _obutils import Utils

np.set_printoptions(precision=4)
#np.set_printoptions(suppress=True)
np.set_printoptions(threshold=np.inf)


class Hybrid_DLC(DLC): # write new mixins _Hyb_ICoords for hybrid water,_Hyb_Bmat,
    """
    Hybrid DLC for systems containing a large amount of atoms, the coordinates are partitioned 
    into a QM-region which is simulated with ICs, and a MM-region which is modeled with Cartesians. 
    """
    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Hybrid_DLC(Hybrid_DLC.default_options().set_values(kwargs))

    def get_nxyzics(self):
        '''
        This function gets the number of xyz ics,
        a list of booleans that describes whether the atom is xyz ic or not,
        and finally it saves the xyzics_coords
        '''
        startidx=0
        self.nxyzatoms=0
        for i,res in enumerate(ob.OBResidueIter(self.mol.OBMol)):
            if res.GetName() not in self.IC_region:
                self.nxyzatoms+=res.GetNumAtoms()
                for j,a in enumerate(ob.OBResidueAtomIter(res)):
                    self.xyzatom_bool[startidx+j]=True
            startidx+= res.GetNumAtoms()
        self.xyzics_coords = np.zeros((self.nxyzatoms,3))
        startidx=0
        resid=0
        for res in ob.OBResidueIter(self.mol.OBMol):
            if res.GetName() not in self.IC_region:
                for j,a in enumerate(ob.OBResidueAtomIter(res)):
                    self.xyzics_coords[startidx+j,0] = a.GetX()
                    self.xyzics_coords[startidx+j,1] = a.GetY()
                    self.xyzics_coords[startidx+j,2] = a.GetZ()
                resid+=1
                startidx= resid*res.GetNumAtoms()

    def set_nicd(self):
        self.nicd=3*(self.natoms-self.nxyzatoms)-6+3*self.nxyzatoms

    def bmatp_create(self):
        only_ics= self.BObj.nbonds + self.AObj.nangles + self.TObj.ntor
        bmatp=super(DLC,self).bmatp_create()
        bmatp[only_ics:,(self.natoms-self.nxyzatoms)*3:] = np.eye(self.nxyzatoms*3)
        #print bmatp
        #print bmatp[:only_ics,:(self.natoms-self.nxyzatoms)*3]
        return bmatp

    def q_create(self):
        q=super(DLC,self).q_create()

        tmpcoords = np.copy(self.coords)
        xyzic_atoms=np.zeros((self.nxyzatoms,3))
        count=0
        for i,a in enumerate(tmpcoords):
            if self.xyzatom_bool[i]==True:
                xyzic_atoms[count]=a
                count+=1
        xyzic_atoms=xyzic_atoms.flatten()
            
        for i in range(self.nicd):
            q[i]+=np.dot(self.Ut[i,self.num_ics_p:],xyzic_atoms)
        return q


if __name__ =='__main__':
    #filepath="r001.ttt_meh_oh_ff-solvated-dftb3-dyn1_000.pdb"
    filepath="solvated.pdb"
    from qchem import *
    from pes import *
    lot1=QChem.from_options(states=[(1,0)],basis='6-31gs')
    pes = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    mol1=pb.readfile("pdb",filepath).next()
    #ic1=Hybrid_DLC.from_options(mol=mol1,PES=pes,IC_region=['TTT'])
    ic1=Hybrid_DLC.from_options(mol=mol1,PES=pes,IC_region=['LIG'])

