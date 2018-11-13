import numpy as np
import openbabel as ob
import pybel as pb
from units import *
from collections import Counter

'''
This file contains a ICoord class which is a Mixin for deloc_ic class
and it contains three slot classes which help with management of IC data
'''

class ICoords:

    def make_bonds(self):
        bonds=[]
        bondd=[]
        nbonds=0
        for bond in ob.OBMolBondIter(self.mol.OBMol):
            nbonds+=1
            a=bond.GetBeginAtomIdx()
            b=bond.GetEndAtomIdx()
            if a>b:
                bonds.append((a,b))
            else:
                bonds.append((b,a))

        bonds = sorted(bonds)
        for bond in bonds:
            bondd.append(self.distance(bond[0],bond[1]))
        print " number of bonds is %i" %nbonds
        print " printing bonds"
        for bond,bod in zip(bonds,bondd):
            print "%s: %1.2f" %(bond, bod)

        return Bond_obj(bonds,nbonds,bondd)

    def make_angles(self):
        nangles=0
        angles=[]
        anglev=[]
        # openbabels iterator doesn't work because not updating properly
        for i,bond1 in enumerate(self.BObj.bonds):
            for bond2 in self.BObj.bonds[:i]:
                found=False
                if bond1[0] == bond2[0]:
                    angle = (bond1[1],bond1[0],bond2[1])
                    found=True
                elif bond1[0] == bond2[1]:
                    angle = (bond1[1],bond1[0],bond2[0])
                    found=True
                elif bond1[1] == bond2[0]:
                    angle = (bond1[0],bond1[1],bond2[1])
                    found=True
                elif bond1[1] == bond2[1]:
                    angle = (bond1[0],bond1[1],bond2[0])
                    found=True
                if found==True:
                    angv = self.get_angle(angle[0],angle[1],angle[2])
                    if angv>30.:
                        anglev.append(angv)
                        angles.append(angle)
                        nangles +=1

        print " number of angles is %i" %nangles
        print " printing angles"
        for angle,angv in zip(angles,anglev):
            print "%s %1.2f" %(angle,angv)
        return Ang_obj(angles,nangles,anglev)

    def make_torsions(self):
        ntor=0
        torsions=[]
        torv=[]
        # doesn't work because not updating properly
        for i,angle1 in enumerate(self.AObj.angles):
            for angle2 in self.AObj.angles[:i]:
                found = False
                a1=angle1[0]
                b1=angle1[1]
                c1=angle1[2]
                a2=angle2[0]
                b2=angle2[1]
                c2=angle2[2]
                if b1==c2 and b2==c1:
                    torsion = (a1,b1,b2,a2)
                    found = True
                elif b1==a2 and b2==c1:
                    torsion = (a1,b1,b2,c2)
                    found = True
                elif b1==c2 and b2==a1:
                    torsion = (c1,b1,b2,a2)
                    found = True
                elif b1==a2 and b2==a1:
                    torsion = (c1,b1,b2,c2)
                    found = True
                if found==True and (torsion[0] != torsion[2]) and torsion[0] != torsion[3] : 
                    ntor+=1
                    torsions.append(torsion)
                    t = self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3])
                    torv.append(t)

        print " number of torsions is %i" %ntor
        print " printing torsions"
        for n,torsion in enumerate(torsions):
            print "%s: %1.2f" %(torsion, torv[n])
        return Tor_obj(torsions,ntor,torv)

    @staticmethod
    def tangent_1(ICoord1,ICoord2):
        ictan = []

        for bond1,bond2 in zip(ICoord1.BObj.bondd,ICoord2.BObj.bondd):
            ictan.append(bond1 - bond2)
        for angle1,angle2 in zip(ICoord1.AObj.anglev,ICoord2.AObj.anglev):
            ictan.append((angle1-angle2)*np.pi/180.)
        for torsion1,torsion2 in zip(ICoord1.TObj.torv,ICoord2.TObj.torv):
            temptorsion = (torsion1-torsion2)*np.pi/180.0
            if temptorsion > np.pi:
                ictan.append(-1*((2*np.pi) - temptorsion))
            elif temptorsion < -np.pi:
                ictan.append((2*np.pi)+temptorsion)
            else:
                ictan.append(temptorsion)
        #print " printing ictan"
        #for i in range(ICoord1.BObj.nbonds):
        #    print "%1.2f " %ictan[i],
        #print 
        #for i in range(ICoord1.BObj.nbonds,ICoord1.AObj.nangles+ICoord1.BObj.nbonds):
        #    print "%1.2f " %ictan[i],
        #print 
        #for i in range(ICoord1.BObj.nbonds+ICoord1.AObj.nangles,ICoord1.AObj.nangles+ICoord1.BObj.nbonds+ICoord1.TObj.ntor):
        #    print "%1.2f " %ictan[i],
        #print "\n"
        return np.asarray(ictan).reshape((ICoord1.num_ics,1))

    def tangent_SE(ICoord1,driving_coordinates):
        ictan = []
        bdist = 0.
        nadds = driving_coordinates.count("ADD")
        nbreaks = driving_coordinates.count("BREAK")
        nangles = driving_coordinates.count("ANGLE")
        ntorsions = driving_coordinates.count("TORSION")
        ictan = np.zeros((ICoord1.num_ics,1),dtype=float)
        breakdq = 0.3

        for i in driving_coordinates:
            if "ADD" in i:
                bond = (i[1],i[2])
                wbond = ICoord1.BObj.bond_num(bond)
                d0 = (ICoord1.get_element_VDW(bond[0]) + ICoord1.get_element_VDW(bond[1]))/2.8
                if ICoord1.distance(bond[0],bond[1])>d0:
                    ictan[wbond] = -1*(d0-ICoord1.distance(bond[0],bond[1]))
                if nbreaks>0:
                    ictan[wbond] *= 2
                if ICoord1.print_level>0:
                    print "bond %s d0: %4.3f diff: %4.3f " % (i[1],d0,ictan[wbond])
            if "BREAK" in i:
                bond = (i[1],i[2])
                wbond = ICoord1.BObj.bond_num(bond)
                d0 = (ICoord1.get_element_VDW(bond[0]) + ICoord1.get_element_VDW(bond[1]))*2.
                if ICoord1.distance(bond[0],bond[1])<d0:
                    ictan[wbond] = -1*(d0-ICoord1.distance(bond[0],bond[1])) # NOTE THAT THIS IS NOT HOW THE ORIGINAL GSM WORKED 
                if ICoord1.print_level>0:
                    print "bond %s d0: %4.3f diff: %4.3f " % (i[1],d0,ictan[wbond])
            if "ANGLE" in i:
                angle=(i[1],i[2],i[3])
                ang_idx = ICoord1.AObj.angle_num(angle)
                anglet = i[4]
                ang_diff = (anglet -ICoord1.AObj.anglev[ang_idx]) *np.pi/180.
                print(" angle: %s is index %i " %(angle,ang_idx))
                print(" anglev: %4.3f align to %4.3f diff(rad): %4.3f" %(ICoord1.AObj.anglev[ang_idx],anglet,ang_diff))
                ictan[ICoord1.BObj.nbonds+ang_idx] = -ang_diff
            if "TORSION" in i:
                torsion=(i[1],i[2],i[3],i[4])
                tor_idx = ICoord1.TObj.torsion_num(torsion)
                tort = i[5]
                tor_diff = (tort -ICoord1.TObj.torv[tor_idx]) 
                if tor_diff>180.:
                    tor_diff-=360.
                elif tor_diff<-180.:
                    tor_diff+=360.
                if tor_diff*np.pi/180.>0.1 or tor_diff*np.pi/180.<0.1:
                    ictan[ICoord1.BObj.nbonds+ICoord1.AObj.nangles+tor_idx] = -tor_diff*np.pi/180.
                print(" torsion: %s is index %i "%(i[1],tor_idx))
                print(" torv: %4.3f align to %4.3f diff(rad): %4.3f" %(ICoord1.TObj.torv[tor_idx],tort,tor_diff))

        bdist = np.linalg.norm(ictan)

        return ictan,bdist


######################  IC objects #####################################
class Bond_obj(object):
    __slots__ = ["nbonds","bonds","bondd"]
    def __init__(self,bonds,nbonds,bondd):
        self.bonds=bonds
        self.nbonds=nbonds
        self.bondd=bondd

    def update(self,mol):
        self.bondd=[]
        self.nbonds = len(self.bonds)
        for bond in self.bonds:
            self.bondd.append(self.distance(mol,bond[0],bond[1]))

    def bond_num(self,bond):
        for b in [bond,tuple(reversed(bond))]:
            try:
                return self.bonds.index(b)
            except ValueError:
                pass
        raise ValueError('The bond %s does not exist' % b)

    def distance(self,mol,i,j):
        """ for some reason openbabel has this one based """
        a1=mol.OBMol.GetAtom(i)
        a2=mol.OBMol.GetAtom(j)
        return a1.GetDistance(a2)

class Ang_obj(object):
    __slots__ = ["nangles","angles","anglev"]
    def __init__(self,angles,nangles,anglev):
        self.angles=angles
        self.nangles=nangles
        self.anglev=anglev

    def update(self,mol):
        self.anglev=[]
        self.nangles = len(self.angles)
        for angle in self.angles:
            self.anglev.append(self.get_angle(mol,angle[0],angle[1],angle[2]))

    def get_angle(self,mol,i,j,k):
        a=mol.OBMol.GetAtom(i)
        b=mol.OBMol.GetAtom(j)
        c=mol.OBMol.GetAtom(k)
        return mol.OBMol.GetAngle(a,b,c) #b is the vertex #in degrees

    def angle_num(self,angle):
        for a in angle,tuple(reversed(angle)):
            try:
                return self.angles.index(a)
            except ValueError:
                pass
        raise ValueError('The angle does not exist')


class Tor_obj(object):
    __slots__ = ["ntor","torsions","torv"]
    def __init__(self,torsions,ntor,torv):
        self.torsions=torsions
        self.ntor=ntor
        self.torv=torv

    def update(self,mol):
        self.torv=[]
        self.ntor = len(self.torsions)
        for torsion in self.torsions:
            self.torv.append(self.get_torsion(mol,torsion[0],torsion[1],torsion[2],torsion[3]))

    def get_torsion(self,mol,i,j,k,l):
        a=mol.OBMol.GetAtom(i)
        b=mol.OBMol.GetAtom(j)
        c=mol.OBMol.GetAtom(k)
        d=mol.OBMol.GetAtom(l)
        tval=mol.OBMol.GetTorsion(a,b,c,d)*np.pi/180.
        if tval>=np.pi:
            tval-=2.*np.pi
        if tval<=-np.pi:
            tval+=2.*np.pi
        return tval*180./np.pi

    def torsion_num(self,torsion):
        for t in torsion,tuple(reversed(torsion)):
            try:
                return self.torsions.index(t)
            except ValueError:
                pass
        raise ValueError('The torsion does not exist')



    """
    def make_imptor(self):
        self.imptor=[]
        self.nimptor=0
        self.imptorv=[]
        count=0
        for i in self.AObj.angles:
            #print i
            try:
                for j in self.AObj.angles[0:count]:
                    found=False
                    a1=i[0]
                    m1=i[1]
                    c1=i[2]
                    a2=j[0]
                    m2=j[1]
                    c2=j[2]
                    #print(" angle: %i %i %i angle2: %i %i %i" % (a1,m1,c1,a2,m2,c2))
                    if m1==m2:
                        if a1==a2:
                            found=True
                            d=self.mol.OBMol.GetAtom(c2+1)
                        elif a1==c2:
                            found=True
                            d=self.mol.OBMol.GetAtom(a2+1)
                        elif c1==c2:
                            found=True
                            d=self.mol.OBMol.GetAtom(a2+1)
                        elif c1==a2:
                            found=True
                            d=self.mol.OBMol.GetAtom(c2+1)
                    if found==True:
                        a=self.mol.OBMol.GetAtom(c1+1)
                        b=self.mol.OBMol.GetAtom(a1+1)
                        c=self.mol.OBMol.GetAtom(m1+1)
                        imptorvt=self.mol.OBMol.GetTorsion(a,b,c,d)
                        #print imptorvt
                        if abs(imptorvt)>12.0 and abs(imptorvt-180.)>12.0:
                            found=False
                        else:
                            self.imptorv.append(imptorvt)
                            self.imptor.append((a.GetIndex(),b.GetIndex(),c.GetIndex(),d.GetIndex()))
                            self.nimptor+=1
            except Exception as e: print(e)
            count+=1
            return

    def make_nonbond(self):
        self.nonbond=[]
        for i in range(self.natoms):
            for j in range(i):
                found=False
                for k in range(self.BObj.nbonds):
                    if found==True:
                        break
                    if (self.BObj.bonds[k][0]==i and self.BObj.bonds[k][1]==j) or (self.BObj.bonds[k][0]==j and self.BObj.bonds[k][1]==i):
                        found=True
                for k in range(self.AObj.nangles):
                    if found==True:
                        break
                    if self.AObj.angles[k][0]==i:
                        if self.AObj.angles[k][1]==j:
                            found=True
                        elif self.AObj.angles[k][2]==j:
                            found=True
                    elif self.AObj.angles[k][1]==i:
                        if self.AObj.angles[k][0]==j:
                            found=True
                        elif self.AObj.angles[k][2]==j:
                            found=True
                    elif self.AObj.angles[k][2]==i:
                        if self.AObj.angles[k][0]==j:
                            found=True
                        elif self.AObj.angles[k][1]==j:
                            found=True
                if found==False:
                   self.nonbond.append(self.distance(i,j))
        #print self.nonbond

    """
