import numpy as np
import openbabel as ob
import pybel as pb
from units import *
import elements 

""" 
This file contains ICoord utilities that center around the usage of openbabel
"""

Elements = elements.ElementData()
class Utils:

    def set_xyz(self,coords):
        self.coords = np.copy(coords)
        for i,xyz in enumerate(coords):
            self.mol.OBMol.GetAtom(i+1).SetVector(xyz[0],xyz[1],xyz[2])

    @staticmethod
    def make_mol_from_coords(coords,atomic_symbols):
        mol = ob.OBMol()
        for s,xyz in zip(atomic_symbols,coords):
            i = mol.NewAtom()
            a = Elements.from_symbol(s).atomic_num
            i.SetAtomicNum(a)
            i.SetVector(xyz[0],xyz[1],xyz[2])
        return pb.Molecule(mol)

    def print_xyz(self):
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            print(" %1.4f %1.4f %1.4f" %(a.GetX(), a.GetY(), a.GetZ()) )

    def distance(self,i,j):
        """ for some reason openbabel has this one based """
        a1=self.mol.OBMol.GetAtom(i)
        a2=self.mol.OBMol.GetAtom(j)
        return a1.GetDistance(a2)

    def getIndex(self,i):
        """ be careful here I think it's 0 based"""
        return self.mol.OBMol.GetAtom(i).GetIndex()

    def getCoords(self,i):
        a= self.mol.OBMol.GetAtom(i+1)
        return [a.GetX(),a.GetY(),a.GetZ()]

    def getAllCoords(self,i):
        for i in range(self.natoms):
            getCoords(i)

    def getAtomicNums(self):
        atomic_nums = [ self.getAtomicNum(i+1) for i in range(self.natoms) ]
        return atomic_nums

    def getAtomicNum(self,i):
        a = self.mol.OBMol.GetAtom(i)
        return a.GetAtomicNum()

    def get_angle(self,i,j,k):
        a=self.mol.OBMol.GetAtom(i)
        b=self.mol.OBMol.GetAtom(j)
        c=self.mol.OBMol.GetAtom(k)
        return self.mol.OBMol.GetAngle(a,b,c) #b is the vertex #in degrees

    def get_torsion(self,i,j,k,l):
        a=self.mol.OBMol.GetAtom(i)
        b=self.mol.OBMol.GetAtom(j)
        c=self.mol.OBMol.GetAtom(k)
        d=self.mol.OBMol.GetAtom(l)
        tval=self.mol.OBMol.GetTorsion(a,b,c,d)*np.pi/180.
        #if tval >3.14159:
        if tval>=np.pi:
            tval-=2.*np.pi
        #if tval <-3.14159:
        if tval<=-np.pi:
            tval+=2.*np.pi
        return tval*180./np.pi

    def isTM(self,i):
        anum= self.getIndex(i)
        if anum>20:
            if anum<31:
                return True
            elif anum >38 and anum < 49:
                return True
            elif anum >71 and anum <81:
                return True

    def get_element_VDW(self,a):
        return Elements.from_atomic_number(a).vdw_radius

    def close_bond(self,bond):
        A = 0.2
        d = self.distance(bond[0],bond[1])
        #dr = (vdw_radii.radii[self.getAtomicNum(bond[0])] + vdw_radii.radii[self.getAtomicNum(bond[1])] )/2
        a=self.getAtomicNum(bond[0])
        b=self.getAtomicNum(bond[1])
        dr = (Elements.from_atomic_number(a).vdw_radius + Elements.from_atomic_number(b).vdw_radius )/2.
        val = np.exp(-A*(d-dr))
        if val>1: val=1
        return val

    def coord_num(self):
        coordn=[]
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            count=0
            for nbr in ob.OBAtomAtomIter(a):
                count+=1
            coordn.append(count)
        #print coordn
        return coordn

    def make_frags(self):
        """ Currently only works for two fragments """

        print(" making frags")
        nfrags=0
        frags=[]
        frag1=[]
        frag2=[]
        for n,a in enumerate(ob.OBMolAtomIter(self.mol.OBMol)):
            found=False
            if n==0:
                frag1.append((0,n+1))
            else:
                found=False
                for nbr in ob.OBAtomAtomIter(a):
                    if (0,nbr.GetIndex()+1) in frag1:
                        found=True
                if found==True:
                    frag1.append((0,a.GetIndex()+1))
                if found==False:
                    frag2.append((1,a.GetIndex()+1))

        if not frag2:
            nfrags=1
        else:
            nfrags=2
        frags=frag1+frag2
        for i in frags:
            print(" atom[%i]: %i " % (i[1],i[0]))
        print(" nfrags: %i" % (nfrags))
        return nfrags,frags

    def draw(self):
        #self.mol.draw(filename="path.png")
        self.mol.write("svg", "can.svg", overwrite=True)

