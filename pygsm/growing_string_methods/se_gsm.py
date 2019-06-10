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
        self.isomer_init()

        print(" Done initializing isomer")
        self.nodes[0].form_Primitive_Hessian()
        print(" Primitive Internal Coordinates")
        print(self.nodes[0].primitive_internal_coordinates[0:50])
        print(" number of primitives is", self.nodes[0].num_primitives)

        # stash bdist for node 0
        ictan,self.nodes[0].bdist = self.tangent(0,None)
        self.nodes[0].update_coordinate_basis(constraints=ictan)
        self.set_V0()

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy
        #TODO should be actual gradient
        self.nodes[0].gradrms = 0.

    def isomer_init(self):
        #TODO ANGLE, TORSION or OOP between fragments will not work if using TRIC with BLOCK LA
        changed_top = False
        for i in self.driving_coords:
            if "ADD" in i or "BREAK" in i:
                bond = Distance(i[1]-1,i[2]-1)
                self.nodes[0].coord_obj.Prims.add(bond,verbose=True)
                changed_top =True
            if "ANGLE" in i:
                angle = Angle(i[1]-1,i[2]-1,i[3]-1)
                self.nodes[0].coord_obj.Prims.add(angle,verbose=True)
            if "TORSION" in i:
                torsion = Dihedral(i[1]-1,i[2]-1,i[3]-1,i[4]-1)
                self.nodes[0].coord_obj.Prims.add(torsion,verbose=True)
            if "OOP" in i:
                oop = OutOfPlane(i[1]-1,i[2]-1,i[3]-1,i[4]-1)
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
            self.interpolate(1) 
            self.growth_iters(iters=max_iters,maxopt=opt_steps)
            if self.tscontinue==True:
                if self.pastts==1: #normal over the hill
                    self.interpolateR(1)
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
            self.TSnode = np.argmax(self.energies)
            self.emax = self.energies[self.TSnode]

            print(" Number of nodes is ",self.nnodes)
            print(" Warning last node still not optimized fully")
            self.write_xyz_files(iters=1,base='grown_string',nconstraints=1)
            print(" SSM growth phase over")
            self.done_growing=True

            print(" beginning opt phase")
            print("Setting all interior nodes to active")
            for n in range(1,self.nnodes-1):
                self.active[n]=True
                self.optimizer[n].conv_grms=self.options['CONV_TOL']

        print(" initial ic_reparam")
        self.ic_reparam()
        if self.tscontinue==True:
            self.opt_iters(max_iter=max_iters,optsteps=3,rtype=rtype) #opt steps fixed at 3
        else:
            print("Exiting early")

        print("Finished GSM!")  


    def add_node(self,n1,n2,n3=None):
        # n3 is not used!
        BDISTMIN=0.05
        print(" adding node: %i from node %i" %(n2,n1))
        ictan,bdist =  self.tangent(n1,None)

        if bdist<BDISTMIN:
            print("bdist too small %.3f" % bdist)
            return 0
        new_node = Molecule.copy_from_options(self.nodes[n1],new_node_id=n2)
        Vecs = new_node.update_coordinate_basis(constraints=ictan)
        constraint = new_node.constraints
        sign=-1.

        dqmag_scale=1.5
        minmax = self.DQMAG_MAX - self.DQMAG_MIN
        a = bdist/dqmag_scale
        if a>1.:
            a=1.
        dqmag = sign*(self.DQMAG_MIN+minmax*a)
        print(" dqmag: %4.3f from bdist: %4.3f" %(dqmag,bdist))

        dq0 = dqmag*constraint
        print(" dq0[constraint]: %1.3f" % dqmag)

        new_node.update_xyz(dq0)
        new_node.bdist = bdist

        return new_node

    def add_last_node(self,rtype):
        assert rtype==1 or rtype==2, "rtype must be 1 or 2"
        samegeom=False
        noptsteps=100
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
                        )
            self.active[self.nR]=True
            if (self.nodes[self.nR].xyz == self.nodes[self.nR-1].xyz).all():
                print(" Opt did not produce new geometry")
            else:
                self.nR+=1
        elif rtype==2:
            print(" already created node, opting")
            self.optimizer[self.nR-1].optimize(
                        molecule=self.nodes[self.nR-1],
                        refE=self.nodes[0].V0,
                        opt_steps=noptsteps,
                        )
        return

    def check_add_node(self):
        success=True
        #if self.nodes[self.nR-1].gradrms < self.gaddmax:
        if self.nodes[self.nR-1].gradrms < self.options['ADD_NODE_TOL']:
            if self.nR == self.nnodes:
                print(" Ran out of nodes, exiting GSM")
                raise ValueError
            self.active[self.nR-1] = False
            if self.nodes[self.nR] == None:
                success=self.interpolateR()
        return success

    def interpolate(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.interpolateR()

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        pass

    def set_active(self,nR,nP=None):
        #print(" Here is active:",self.active)
        print((" setting active node to %i "%nR))

        for i in range(self.nnodes):
            if self.nodes[i] != None:
                self.active[i] = False
                self.optimizer[i].conv_grms = self.options['CONV_TOL']
                print(" CONV_TOL of node %d is %.4f" % (i,self.optimizer[i].conv_grms))
        #self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[nR].conv_grms = self.options['CONV_TOL']
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
        if n2 ==None or n2==n1:
            print(" getting tangent from node ",n1)
            nadds = self.driving_coords.count("ADD")
            nbreaks = self.driving_coords.count("BREAK")
            nangles = self.driving_coords.count("ANGLE")
            ntorsions = self.driving_coords.count("TORSION")
            ictan = np.zeros((self.nodes[n1].num_primitives,1),dtype=float)
            breakdq = 0.3
            bdist=0.0
            atoms = self.nodes[n1].atoms
            xyz = self.nodes[n1].xyz

            for i in self.driving_coords:
                if "ADD" in i:
                    index = [i[1]-1, i[2]-1]
                    bond = Distance(index[0],index[1])
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(bond)
                    if len(i)==3:
                        #TODO why not just use the covalent radii?
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2.8
                    elif len(i)==4:
                        d0=i[3]
                    current_d =  bond.value(xyz)
                    ictan[prim_idx] = -1*(d0-current_d)
                    #if nbreaks>0:
                    #    ictan[prim_idx] *= 2
                    # => calc bdist <=
                    if current_d>d0:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if self.print_level>0:
                        print(" bond %s target (less than): %4.3f current d: %4.3f diff: %4.3f " % ((i[1],i[2]),d0,current_d,ictan[prim_idx]))

                if "BREAK" in i:
                    index = [i[1]-1, i[2]-1]
                    bond = Distance(index[0],index[1])
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(bond)
                    if len(i)==3:
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2.8
                    elif len(i)==4:
                        d0=i[3]

                    current_d =  bond.value(xyz)
                    ictan[prim_idx] = -1*(d0-current_d) 

                    # => calc bdist <=
                    if current_d<d0:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])

                    if self.print_level>0:
                        print(" bond %s target (greater than): %4.3f, current d: %4.3f diff: %4.3f " % ((i[1],i[2]),d0,current_d,ictan[prim_idx]))
                if "ANGLE" in i:

                    index = [i[1]-1, i[2]-1,i[3]-1]
                    angle = Angle(index[0],index[1],index[3])
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(angle)
                    anglet = i[4]
                    ang_value = angle.value(xyz)
                    ang_diff = anglet*np.pi/180. - ang_value
                    #print(" angle: %s is index %i " %(angle,ang_idx))
                    if self.print_level>0:
                        print((" anglev: %4.3f align to %4.3f diff(rad): %4.3f" %(ang_value,anglet,ang_diff)))
                    ictan[prim_idx] = -ang_diff
                    #TODO need to come up with an adist
                    #if abs(ang_diff)>0.1:
                    #    bdist+=ictan[ICoord1.BObj.nbonds+ang_idx]*ictan[ICoord1.BObj.nbonds+ang_idx]
                if "TORSION" in i:
                    #torsion=(i[1],i[2],i[3],i[4])
                    index = [i[1]-1, i[2]-1,i[3]-1,i[4]-1]
                    torsion = Dihedral(index[0],index[1],index[2],index[3])
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(torsion)
                    tort = i[5]
                    torv = torsion.value(xyz)
                    tor_diff = tort - torv*180./np.pi
                    if tor_diff>180.:
                        tor_diff-=360.
                    elif tor_diff<-180.:
                        tor_diff+=360.
                    ictan[prim_idx] = -tor_diff*np.pi/180.

                    if tor_diff*np.pi/180.>0.1 or tor_diff*np.pi/180.<0.1:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if self.print_level>0:
                        print((" current torv: %4.3f align to %4.3f diff(deg): %4.3f" %(torv*180./np.pi,tort,tor_diff)))

                trans = ['TranslationX', 'TranslationY', 'TranslationZ']
                if any(elem in trans for elem in i):
                    fragid = i[1]
                    destination = i[2]
                    indices = self.nodes[n1].get_frag_atomic_index(fragid)
                    atoms=range(indices[0]-1,indices[1])
                    #print('indices of frag %i is %s' % (fragid,indices))
                    T_class = getattr(sys.modules[__name__], i[0])
                    translation = T_class(atoms,w=np.ones(len(atoms))/len(atoms))
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(translation)
                    trans_curr = translation.value(xyz)
                    trans_diff = destination-trans_curr
                    ictan[prim_idx] = -trans_diff
                    bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if self.print_level>0:
                        print((" current trans: %4.3f align to %4.3f diff: %4.3f" %(trans_curr,destination,trans_diff)))

                #TODO
                rots = ['RotationA','RotationB','RotationC']
                if any(elem in rots for elem in i):
                    fragid = i[1]
                    rot_angle = i[2]
                    indices = self.nodes[n1].get_frag_atomic_index(fragid)
                    atoms=range(indices[0]-1,indices[1])
                    R_class = getattr(sys.modules[__name__], i[0])
                    coords = self.nodes[n1].xyz
                    sel = coords.reshape(-1,3)[atoms,:]
                    sel -= np.mean(sel,axis=0)
                    rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                    rotation = R_class(atoms,coords,self.nodes[n1].coord_obj.Prims.Rotators,w=rg)
                    prim_idx = self.nodes[n1].coord_obj.Prims.dof_index(rotation)
                    rot_curr = rotation.value(xyz)
                    rot_diff = rot_angle-rot_curr
                    #print('rot_diff before periodic %.3f' % rot_diff)
                    #if rot_diff > 2*np.pi:
                    #    rot_diff -= 2*np.pi
                    #elif rot_diff< -2*np.pi:
                    #    rot_diff += 2*np.pi

                    ictan[prim_idx] = -rot_diff
                    bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if self.print_level>0:
                        print((" current rot: %4.3f align to %4.3f diff: %4.3f" %(rot_curr,rot_angle,rot_diff)))

            bdist = np.sqrt(bdist)
            if np.all(ictan==0.0):
                raise RuntimeError(" All elements are zero")
            return ictan,bdist
        else:
            print(" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2))
            assert self.nodes[n2]!=None,'node n2 is None'
            Q1 = self.nodes[n1].primitive_internal_values 
            Q2 = self.nodes[n2].primitive_internal_values 
            PMDiff = Q2-Q1
            #for i in range(len(PMDiff)):
            for k,prim in zip(list(range(len(PMDiff))),self.nodes[n1].primitive_internal_coordinates):
                if prim.isPeriodic:
                    Plus2Pi = PMDiff[k] + 2*np.pi
                    Minus2Pi = PMDiff[k] - 2*np.pi
                    if np.abs(PMDiff[k]) > np.abs(Plus2Pi):
                        PMDiff[k] = Plus2Pi
                    if np.abs(PMDiff[k]) > np.abs(Minus2Pi):
                        PMDiff[k] = Minus2Pi
            return np.reshape(PMDiff,(-1,1)),None


    def check_if_grown(self):
        self.pastts = self.past_ts()
        isDone=False
        #TODO break planes
        condition1 = (abs(self.nodes[self.nR-1].bdist) <=(1-self.BDIST_RATIO)*abs(self.nodes[0].bdist))
        print(" bdist %.3f" % self.nodes[self.nR-1].bdist)

        if self.pastts and self.nn>3 and condition1: #TODO extra criterion here
            print("pastts is ",self.pastts)
            isDone=True
        fp = self.find_peaks(1)
        if fp==-1 and self.energies[self.nR-1]>200.:
            print("growth_iters over: all uphill and high energy")
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True
        if fp==-2:
            print("growth_iters over: all uphill and flattening out")
            self.end_early=2
            self.tscontinue=False
            self.nnodes=self.nR
            isDone=True
        return isDone

    def check_opt(self,totalgrad,fp,rtype):
        isDone=False
        added=False
        #if self.TSnode==self.nnodes-2 and (self.stage==2 or totalgrad<0.2) and fp==1:
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
            fp=self.find_peaks(3)
            if fp>1:
                rxnocc,wint = self.check_for_reaction()
            if fp >1 and rxnocc==True and wint<self.nnodes-1:
                print("Need to trim string")
                self.tscontinue=False
                isDone=True
                return isDone

        # => Convergence Criteria
        #if (((self.stage==1 and rtype==1) or self.stage==2) and
        if (((self.climb and rtype==1) or self.find) and
                self.nodes[self.TSnode].gradrms< self.options['CONV_TOL']):
            self.tscontinue=False
            isDone=True
            return isDone
        #if (((self.stage==1 and rtype==1) or self.stage==2) and totalgrad<0.1 and
        if (((self.climb and rtype==1) or self.find) and totalgrad<0.1 and
                self.nodes[self.TSnode].gradrms<2.5*self.options['CONV_TOL'] and self.emaxp+0.02> self.emax and 
                self.emaxp-0.02< self.emax):
            self.tscontinue=False
            isDone=True
            return isDone

    def restart_string(self,xyzbase='restart'):
        super(SE_Cross,self).restart_string(xyzbase)
        self.done_growing=False
        self.nnodes=20
        self.nR -=1 
        # stash bdist for node 0
        _,self.nodes[0].bdist = self.tangent(0,None)

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
