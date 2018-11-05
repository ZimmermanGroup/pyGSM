import options
import numpy as np
import os
import pybel as pb
import icoord as ico
from copy import deepcopy

global DQMAG_SSM_SCALE
DQMAG_SSM_SCALE=1.5
global DQMAG_SSM_MAX
DQMAG_SSM_MAX=0.8
global DQMAG_SSM_MIN
DQMAG_SSM_MIN=0.2

class BaseGSM(object):
    
    @staticmethod
    def default_options():
        if hasattr(BaseGSM, '_default_options'): return BaseGSM._default_options.copy()

        opt = options.Options() 
        
        opt.add_option(
            key='ICoord1',
            required=True,
            allowed_types=[ico.ICoord],
            doc='')

        opt.add_option(
            key='ICoord2',
            required=False,
            allowed_types=[ico.ICoord],
            doc='')

        opt.add_option(
            key='nnodes',
            required=False,
            value=0,
            allowed_types=[int],
            doc='number of string nodes')
        
        opt.add_option(
            key='isSSM',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify SSM or DSM')

        opt.add_option(
            key='isomers',
            required=False,
            value=[],
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify atoms\
                 indexed at 1')

        opt.add_option(
            key='isMAP_SE',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify isMAP_SE')

        opt.add_option(
            key='nconstraints',
            required=False,
            value=0,
            allowed_types=[int])

        BaseGSM._default_options = opt
        return BaseGSM._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return BaseGSM(BaseGSM.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes

        #TODO What is optCG Ask Paul
        self.optCG = False
        self.isTSnode =False
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        self.icoords[-1] = self.options['ICoord2']
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.isSSM = self.options['isSSM']
        self.isMAP_SE = self.options['isMAP_SE']
        self.active = [False] * self.nnodes
        self.active[0] = False
        self.active[-1] = False
        self.isomers = self.options['isomers']
        self.isomer_init()
        self.nconstraints = self.options['nconstraints']

    @staticmethod
    def get_ssm_dqmag(bdist):
        dqmag = 0.
        minmax = DQMAG_SSM_MAX - DQMAG_SSM_MIN
        a = bdist/DQMAG_SSM_SCALE
        if a > 1.:
            a = 1.
        dqmag = DQMAG_SSM_MIN + minmax * a
        
        if dqmag < DQMAG_SSM_MIN:
            dqmag = DQMAG_SSM_MIN
        print " dqmag: {:3} from bdist: {:3} \n".format(dqmag,bdist)
        return dqmag

    @staticmethod
    def tangent_1(ICoord1,ICoord2):
        ictan = []
        print "starting tangent 1"
        for bond1,bond2 in zip(ICoord1.bondd,ICoord2.bondd):
            ictan.append(bond1 - bond2)
        for angle1,angle2 in zip(ICoord1.anglev,ICoord2.anglev):
            ictan.append((angle1-angle2)*np.pi/180.)
        for torsion1,torsion2 in zip(ICoord1.torv,ICoord2.torv):
            temptorsion = (torsion1-torsion2)*np.pi/180.0
            if temptorsion > np.pi:
                ictan.append(-1*((2*np.pi) - temptorsion))
            elif temptorsion < -np.pi:
                ictan.append((2*np.pi)+temptorsion)
            else:
                ictan.append(temptorsion)
        print 'ending tangent 1'
        return ictan
            
    @staticmethod
    def tangent_1a(ICoord1,ICoord2):
        ictan = []
        print "starting tangent 1a"
        for bond1,bondd1 in zip(ICoord1.bonds,ICoord1.bondd):
            for bond2,bondd2 in zip(ICoord2.bonds,ICoord2.bondd):
                if (bond1==bond2) or (bond1[::-1]==bond2):
                    ictan.append(bondd1 - bondd2)
                    break
        for angle1,anglev1 in zip(ICoord1.angles,ICoord1.anglev):
            for angle2,anglev2 in zip(ICoord2.angles,ICoord2.anglev):
                if (angle1==angle2) or (angle1[::-1]==angle2):
                    ictan.append((anglev1 - anglev2)*np.pi/180.)
                    break
        for torsion1,torsionv1 in zip(ICoord1.torsions,ICoord1.torv):
            for torsion2,torsionv2 in zip(ICoord2.torsions,ICoord2.torv):
                if (torsion1==torsion2) or (torsion1[::-1]==torsion2):
                    temptorsion = (torsionv1-torsionv2)*np.pi/180.
                    if temptorsion > np.pi:
                        ictan.append(-1*((2*np.pi) - temptorsion))
                    elif temptorsion < -np.pi:
                        ictan.append((2*np.pi)+temptorsion)
                    else:
                        ictan.append(temptorsion)
                    break
        return ictan



    def tangent_1b(self,ICoord1):
        print '\n'
        nbonds = ICoord1.nbonds
        nangles = ICoord1.nangles
        ntor = ICoord1.ntor
        size_ic = nbonds + nangles + ntor
        len_d = ICoord1.nicd

        bdist = 0.
        ictan = [0.] * size_ic

        if hasattr(self,'add'):
            for a1,a2 in self.add:
                add1 = (a1-1,a2-1)
                add2 = add1[::-1]
                add1exists = ICoord1.bond_exists(add1)
                add2exists = ICoord1.bond_exists(add2) 
                if (not add1exists) and (not add2exists):
                    raise Exception(" WARNING: Bond %d %d not found! \n"%(a1,a3))
                if add1exists:
                    wbond = ICoord1.bonds.index(add1)
                if add2exists:
                    wbond = ICoord1.bonds.index(add1)

                a = ICoord1.getAtomicNum(a1-1)
                b = ICoord1.getAtomicNum(a2-1)
                d0 = (ICoord1.Elements.from_atomic_number(a).vdw_radius + ICoord1.Elements.from_atomic_number(b).vdw_radius)/2.8 
                if ICoord1.distance(a1-1,a2-1) > d0:
                    ictan[wbond] = -1 * (d0 - ICoord1.distance(a1-1,a2-1))
                else:
                    ictan[wbond] = 0.0
                if hasattr(self,'brk'):
                    ictan[wbond] = ictan[wbond] * 2
                print "bond %d %d d0: %4.4f diff: %4.4f \n"%(a1,a2,d0,ictan[wbond])
                bdist += ictan[wbond]**2

        if hasattr(self,'brk'):
            breakdq = 0.3
            for b1,b2 in self.brk:
                brk1 = (b1-1,b2-1)
                brk2 = brk1[::-1]
                brk1exists = ICoord1.bond_exists(brk1)
                brk2exists = ICoord1.bond_exists(brk2)
                if (not brk1exists) and (not brk2exists):
                    raise Exception(" Warning: Bond %d %d not found! \n"%(b1,b2))
                if brk1exists:
                    wbond = ICoord1.bonds.index(brk1)
                if brk2exists:
                    wbond = ICoord1.bonds.index(brk2)

                a = ICoord1.getAtomicNum(b1-1)
                b = ICoord1.getAtomicNum(b2-1)
                d0 = (ICoord1.Elements.from_atomic_number(a).vdw_radius + ICoord1.Elements.from_atomic_number(b).vdw_radius)
                if self.isMAP_SE:
                    d0 = d0/1.5
                if ICoord1.distance(b1-1,b2-1) < d0:
                    ictan[wbond] = -1 * (d0 - ICoord1.distance(b1-1,b2-1))
                print " bond %d %d d0: %4.3f dist: %4.3f diff: %4.3f \n"%(b1,b2,d0,ICoord1.distance(b1-1,b2-1),ictan[wbond])
                

        if hasattr(self,'angles'):
            count = 0
            for b1,b2,b3 in self.angles:
                if (b1,b2,b3) in ICoord1.angles:
                    an1 = ICoord1.angles.index((b1,b2,b3))
                elif (b3,b2,b1) in ICoord1.angles:
                    an1 = ICoord1.angles.index((b3,b2,b1))
                print " tangent angle: % % % is % \n"%(b1,b2,b3,an1)
                print " anglev: %4.3f anglet: %4.3f diff(rad): %4.3f \n"%(ICoord1.anglev[an1],self.anglet[count],(self.anglet[count] - ICoord1.anglev[an1])*np.pi/180.0)
                ictan[nbonds+an1] = -(self.anglet[count] - ICoord1.anglev[an1]) * np.pi/180.
                count += 1
        if hasattr(self,'tors'):
            count = 0
            for btor in self.tors:
                an1 = 0
                if btor in ICoord1.torsions:
                    an1 = ICoord1.torsions.index(btor)
                elif btor[::-1] in ICoord1.torsions:
                    an1 = ICoord1.torsions.index(btor[::-1])
                tordiff = self.tort[count] - ICoord1.torv[an1]
                print " tordiff= %1.2f\n"%tordiff
                torfix = 0.
                if tordiff > 180.0:
                    torfix = -360.0
                elif tordiff < -180.:
                    torfix = 360
                if (((tordiff+torfix)*np.pi/180.) > 0.1 ) or (((tordiff+torfix)*np.pi/180.) < 0.1):
                    ictan[nbonds+nangles+an1] = - (tordiff + torfix) * np.pi/180.
                else:
                    ictan[nbonds+nangles+an1] = 0.
                print " tangent tor: {} {} {} {} is #{} \n".format(btor[0],btor[1],btor[2],btor[3],an1)
                print " torv: %4.3f tort: %4.3f diff(rad): %4.3f \n"%(ICoord1.torv[an1],self.tort[count],(tordiff+torfix)*np.pi/180.)
                count += 1
        norm = np.linalg.norm(ictan)
        ictan = ictan/norm 
        bdist = norm
        return bdist,ictan



    def isomer_init(self):
        if len(self.isomers)==0:
            return False
        print "reading isomers"
        nadd = 0
        nbrk = 0
        nangle = 0
        ntors = 0
        nbond = 0

        maxab = 10
        for tup in self.isomers:
            if 'bond' in tup[0].lower():
                if nbond == 0:
                    self.bond = []
                self.bond.append((tup[1],tup[2]))
                print " bond for coordinate system: {} {} \n".format(tup[1],tup[2])
                nbond += 1
                if nbond > maxab:
                    break
            elif 'add' in tup[0].lower():
                if nadd == 0:
                    self.add = []
                self.add.append((tup[1],tup[2]))
                print " adding bond: {} {} \n".format(tup[1],tup[2])
                nadd += 1
                if nadd > maxab:
                    break
            elif 'break' in tup[0].lower():
                if nbrk == 0:
                    self.brk = []
                self.brk.append((tup[1],tup[2]))
                print " breaking bond: {} {} \n".format(tup[1],tup[2])
                nbrk += 1
                if nbrk > maxab:
                    break
            elif 'angle' in tup[0].lower():
                if nangle ==0:
                    self.angles = []
                    self.anglet = []
                self.angles.append((tup[1],tup[2],tup[3]))
                self.anglet.append(tup[4])
                print " angle: %d %d %d align to %4.3f \n"%(tup[1],tup[2],tup[3],tup[4])
                nangle += 1
                if nangle > maxab:
                    break
            elif 'torsion' in tup[0].lower():
                if ntors ==0:
                    self.tors = []
                    self.tort = []
                self.tors.append((tup[1],tup[2],tup[3],tup[4]))
                self.tort.append(tup[5])
                print " tor: %d %d %d %d align to %4.3f \n"%(tup[1],tup[2],tup[3],tup[4],tup[5])
                ntors += 1
                if ntors > maxab:
                    break
        if nadd > 0 or nbrk > 0 or nangle > 0 or ntors > 0:
            return True
        else:
            return False



    def add_node(self,n1,n2,n3):
        
        if self.isSSM:
            raise Exception("Cannot use add_node with SSM. Use add_node_SSM")

        print "Adding Node {} between Nodes {} and {}".format(n2,n1,n3)
        if n1 == n2 or n1 == n3 or n2 == n3:
            print "Cannot add node {} between {} and {}".format(n2,n1,n3)
            raise ValueError("n1==n2, or n1==n3, or n2==n3")

        BDISTMIN = 0.01
        bdist = 0.

        iR = n1
        iP = n3
        iN = n2


        self.icoords[iR] = ico.ICoord.union_ic(self.icoords[iR],self.icoords[iP])
        self.icoords[iP] = ico.ICoord.union_ic(self.icoords[iP],self.icoords[iR])
        self.icoords[iR].update_ics()
        self.icoords[iP].update_ics()

        print " iR,iP: %d %d iN: %d "%(iR,iP,iN)
        mol = pb.Molecule(pb.ob.OBMol(self.icoords[iR].mol.OBMol))
        mol2 = pb.Molecule(pb.ob.OBMol(self.icoords[iP].mol.OBMol))
        lot = deepcopy(self.icoords[iR].lot)
        lot2 = deepcopy(self.icoords[iP].lot)
        newic = ico.ICoord.from_options(mol=mol,lot=lot)
        intic = ico.ICoord.from_options(mol=mol2,lot=lot2)

        newic = ico.ICoord.union_ic(newic,intic)
        intic = ico.ICoord.union_ic(intic,newic)

        newic.update_ics()
        intic.update_ics()

        dq0 = [0.] * newic.nicd
        dq0 = np.asarray(dq0).reshape(newic.nicd,1)
        ictan = ico.ICoord.tangent_1(newic,intic)

        dqmag = 0.0
        
        newic.bmatp_create()
        newic.bmatp_to_U()
        newic.opt_constraint(ictan)

        ictan0 = ictan[:]
        
        dqmag += np.dot(ictan0,newic.Ut[-1])

        print " dqmag: %1.3f"%dqmag

        newic.bmat_create()

        if self.nnodes-self.nn != 1:
            dq0[newic.nicd-1] = -dqmag/float(self.nnodes-self.nn)
        else:
            dq0[newic.nicd-1] = -dqmag/2.0;
        
        print " dq0[constraint]: %1.3f \n"%dq0[newic.nicd-1]
        
        newic.ic_to_xyz(dq0)

        newic.update_ics()

        self.icoords[iN] = newic
        
        #TODO com_rotate_move(n1,n3,n2,1.0)
        
        self.icoords[iN].bmatp_create()
        self.icoords[iN].bmatp_to_U()
        self.icoords[iN].bmat_create()

        self.icoords[iN].make_Hint()
        self.icoords[iN].newHess = 5
        
        self.active[iN] = True;

        success = True
        
        self.nn += 1
        
       # self.icoords[n1].mol.write('xyz','noden1.xyz')
       # self.icoords[n2].mol.write("xyz","noden2.xyz")
       # self.icoords[n3].mol.write("xyz","noden3.xyz")
       # print "ictan: ",ictan0
        print "\niR xyz:"
        self.icoords[iR].print_xyz()

        print "\niN xyz:"
        self.icoords[iN].print_xyz()

        print "\niP xyz:"
        self.icoords[iP].print_xyz()
        
        self.icoords[iR].mol.write('xyz','temp{:02}.xyz'.format(iR),overwrite=True)
        self.icoords[iN].mol.write('xyz','temp{:02}.xyz'.format(iN),overwrite=True)
        self.icoords[iP].mol.write('xyz','temp{:02}.xyz'.format(iP),overwrite=True)

    def add_node_SSM(self,n1,n2):
        if not self.isSSM:
            return
        print "Adding node to {}".format(n1)
        BDISTMIN = 0.01
        bdist = 0.

        print "\n n2 \n"
        mol = pb.Molecule(pb.ob.OBMol(self.icoords[n1].mol.OBMol)) 
        self.icoords[n2] = ico.ICoord.from_options(mol=mol,lot=self.icoords[n1].lot) 
        self.icoords[n2].update_ics()

        dq0 = [0.] * self.icoords[n2].nicd
        
        bdist,ictan = self.tangent_1b(self.icoords[n2])

        
#        self.icoords[n2].bdist = bdist
#
#        print " bdist: {:3} \n".format(bdist)
#        if bdist < BDISTMIN:
#            print " bdist > BDISTMIN"
#            return
#        if bdist > self.icoords[n1].bdist:
#            print " bdist not getting smaller"
#            return        
        
        dqmag = 0.0
        self.icoords[n2].bmatp_create()
        self.icoords[n2].bmatp_to_U()
        self.icoords[n2].opt_constraint(ictan)
        ictan0 = ictan[:]
        print 'ictan0:\n',ictan0        
 
        dqmag = BaseGSM.get_ssm_dqmag(bdist)
        
        print " dqmag: %1.3f"%dqmag

        self.icoords[n2].bmat_create()
        dq0[-1] = -dqmag

        print " dq0[-1]: %1.3f"%dq0[-1]
        

        self.icoords[n2].ic_to_xyz(dq0)
        self.icoords[n2].update_ics()

        self.icoords[n2].bmatp_create()
        self.icoords[n2].bmatp_to_U()
        self.icoords[n2].bmat_create()

        for i in range(self.icoords[n2].nbonds + self.icoords[n2].nangles + self.icoords[n2].ntor):
            self.icoords[n2].Hintp[i] = self.icoords[n1].Hintp[i]
        self.icoords[n2].newHess = 2

        self.active[n2] = 1        
        self.nn += 1

        print "printing xyz"
        self.icoords[n1].print_xyz()
        print 'n2'
        self.icoords[n2].print_xyz()
        
#        if (bdist >= BDISTMIN) and (self.icoords[n2].bdist < self.icoords[n1].bdist):
#            self.nn += 1
#            return True
#        else:
#            return False

