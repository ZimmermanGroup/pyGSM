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

        print(" Forming Union of primitive coordinates")
        self.nodes[0].coord_obj = self.nodes[0].coord_obj.union(self.nodes[-1].coord_obj)
        self.nodes[0].form_Primitive_Hessian()
        print(" Done forming union")
        self.nodes[-1].coord_obj = self.nodes[0].coord_obj.copy(self.nodes[-1].xyz)
        self.nodes[-1].form_Primitive_Hessian()

        # this tests if the primitives are the same
        assert self.nodes[0].coord_obj == self.nodes[-1].coord_obj, "They should be the same."

        print(" Primitive Internal Coordinates")
        print(self.nodes[0].primitive_internal_coordinates)
        print(" number of primitives is", self.nodes[0].num_primitives)

    def restart_string(self,xyzbase='restart'):
        self.growth_direction=0
        xyzfile=xyzbase+".xyz"
        with open(xyzfile) as f:
            nlines = sum(1 for _ in f)
        #print "number of lines is ", nlines
        with open(xyzfile) as f:
            natoms = int(f.readlines()[2])

        #print "number of atoms is ",natoms
        nstructs = (nlines-6)/ (natoms+5) #this is for three blocks after GEOCON
        
        #print "number of structures in restart file is %i" % nstructs
        coords=[]
        grmss = []
        atomic_symbols=[]
        dE = []
        with open(xyzfile) as f:
            f.readline()
            f.readline() #header lines
            # get coords
            for struct in range(nstructs):
                tmpcoords=np.zeros((natoms,3))
                f.readline() #natoms
                f.readline() #space
                for a in range(natoms):
                    line=f.readline()
                    tmp = line.split()
                    tmpcoords[a,:] = [float(i) for i in tmp[1:]]
                    if struct==0:
                        atomic_symbols.append(tmp[0])
                coords.append(tmpcoords)
            # Get energies
            f.readline() # line
            f.readline() #energy
            for struct in range(nstructs):
                self.energies[struct] = float(f.readline())
            # Get grms
            f.readline() # max-force
            for struct in range(nstructs):
                grmss.append(float(f.readline()))
            # Get dE
            f.readline()
            for struct in range(nstructs):
                dE.append(float(f.readline()))

        # create newic object
        self.newic  = Molecule.copy_from_options(self.nodes[0])

        # initial energy
        self.nodes[0].V0 = self.nodes[0].energy 
        self.nodes[0].gradrms=grmss[0]
        self.nodes[0].PES.dE = dE[0]
        self.nodes[-1].gradrms=grmss[-1]
        self.nodes[-1].PES.dE = dE[-1]
        self.emax = float(max(self.energies[1:-1]))
        self.TSnode = np.argmax(self.energies)
        print(" initial energy is %3.4f" % self.nodes[0].energy)

        for struct in range(1,nstructs-1):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[0],coords[struct],struct)
            self.nodes[struct].gradrms=grmss[struct]
            self.nodes[struct].PES.dE = dE[struct]
            self.nodes[struct].newHess=5

        self.nnodes=self.nR=nstructs
        self.isRestarted=True
        self.done_growing=True
        self.nodes[self.TSnode].isTSnode=True
        print(" setting all interior nodes to active")
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.optimizer[n].options['OPTTHRESH']=self.options['CONV_TOL']*2
            self.optimizer[n].options['DMAX'] = 0.05
        print(" V_profile: ", end=' ')
        for n in range(self.nnodes):
            print(" {:7.3f}".format(float(self.energies[n])), end=' ')
        print()
        print(" grms_profile: ", end=' ')
        for n in range(self.nnodes):
            print(" {:7.3f}".format(float(self.nodes[n].gradrms)), end=' ')
        print()
        print(" dE_profile: ", end=' ')
        for n in range(self.nnodes):
            print(" {:7.3f}".format(float(self.nodes[n].difference_energy)), end=' ')
        print()

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
            print(" initial ic_reparam")
            self.get_tangents_1()
            self.ic_reparam(ic_reparam_steps=25)
            self.write_xyz_files(iters=1,base='initial_ic_reparam',nconstraints=1)
        else:
            oi=0
            self.get_tangents_1()
        for i in range(self.nnodes):
            if self.nodes[i] !=None:
                self.optimizer[i].options['OPTTHRESH'] = self.options['CONV_TOL']*2

        if self.tscontinue==True:
            if max_iters-oi>0:
                opt_iters=max_iters-oi
                self.opt_iters(max_iter=opt_iters,optsteps=opt_steps,rtype=rtype)
        else:
            print("Exiting early")
        print("Finished GSM!")  

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
        print(" adding node: %i between %i %i from %i" %(n2,n1,n3,n1))
        ictan =  self.tangent(n3,n1)
        Vecs = self.nodes[n1].update_coordinate_basis(constraints=ictan)

        dq0 = np.zeros((Vecs.shape[1],1))
        dqmag = np.dot(Vecs[:,0],ictan)
        print(" dqmag: %1.3f"%dqmag)

        if self.nnodes-self.nn > 1:
            dq0[0] = -dqmag/float(self.nnodes-self.nn)
        else:
            dq0[0] = -dqmag/2.0;
        print(" dq0[constraint]: %1.3f" % dq0[0])
        old_xyz = self.nodes[n1].xyz.copy()
        new_xyz = self.nodes[n1].coord_obj.newCartesian(old_xyz,dq0)
        new_node = Molecule.copy_from_options(self.nodes[n1],new_xyz,n2)
        return new_node

    def set_active(self,nR,nP):
        #print(" Here is active:",self.active)
        if nR!=nP and self.growth_direction==0:
            print((" setting active nodes to %i and %i"%(nR,nP)))
        elif self.growth_direction==1:
            print((" setting active node to %i "%nR))
        elif self.growth_direction==2:
            print((" setting active node to %i "%nP))
        else:
            print((" setting active node to %i "%nR))

        for i in range(self.nnodes):
            if self.nodes[i] != None:
                self.optimizer[i].options['OPTTHRESH'] = self.options['CONV_TOL']*2.
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction==1:
            self.active[nP]=False
        if self.growth_direction==2:
            self.active[nR]=False
        #print(" Here is new active:",self.active)

    def check_if_grown(self):
        isDone=False
        if self.nn==self.nnodes:
            isDone=True
            if self.growth_direction==1:
                print("need to add last node")
                raise NotImplementedError
                #TODO

        return isDone

    def check_add_node(self):
        success=True 
        if self.nodes[self.nR-1].gradrms < self.gaddmax and self.growth_direction!=2:
            if self.nodes[self.nR] == None:
                self.interpolateR()
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction!=1:
            if self.nodes[-self.nP-1] == None:
                self.interpolateP()
        return success

    def tangent(self,n1,n2):
        print(" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2))
        # this could have been done easier but it is nicer to do it this way
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
        self.nodes[0].V0 = self.nodes[0].energy

        #TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction!=1:
            self.nodes[-1].gradrms = 0.
            print(" Energy of the end points are %4.3f, %4.3f" %(self.nodes[0].energy,self.nodes[-1].energy))
            print(" relative E %4.3f, %4.3f" %(0.0,self.nodes[-1].energy-self.nodes[0].energy))
        else:
+            print(" Energy of end points are %4.3f " % self.nodes[0].energy)
        else:
            print(" Energy of end points are %4.3f " % self.nodes[0].energy)
            self.nodes[-1].energy = self.nodes[0].energy
            self.nodes[-1].gradrms = 0.


if __name__=='__main__':
    from qchem import QChem
    from pes import PES
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
    #filepath1='reactant.xyz'
    #filepath2='product.xyz'

    lot1=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional=functional,nproc=nproc,fnm=filepath1)
    lot2 = QChem(lot1.options.copy().set_values({'fnm':filepath2}))

    pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    pes2 = PES(pes1.options.copy().set_values({'lot':lot2}))

    M1 = Molecule.from_options(fnm=filepath1,PES=pes1,coordinate_type="DLC")
    M2 = Molecule.from_options(fnm=filepath2,PES=pes2,coordinate_type="DLC")

    optimizer=eigenvector_follow.from_options(print_level=1)  #default parameters fine here/opt_type will get set by GSM

    gsm = GSM.from_options(reactant=M1,product=M2,nnodes=9,optimizer=optimizer,print_level=1)
    gsm.go_gsm(rtype=2,opt_steps=3)

