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

    def go_gsm(self,max_iters=50,opt_steps=3,rtype=0):
        """rtype=0 MECI search
           rtype=1 MESX search
        """
        assert rtype in [0,1], "rtype not defined"
        print "*********************************************************************"
        if rtype==0:
            print "Doing MECI search"
        else:
            print "Doing MESX search"
        print "*********************************************************************"

        self.icoords[0].gradrms=0.
        self.icoords[0].energy = self.icoords[0].V0 = self.icoords[0].PES.get_energy(self.icoords[0].geom)
        print ' Initial energy is {:1.4f}'.format(self.icoords[0].energy)
        sys.stdout.flush()

        # stash bdist for node 0
        _,self.icoords[0].bdist = DLC.tangent_SE(self.icoords[0],self.driving_coords,quiet=True)
        print " Initial bdist is %1.3f" %self.icoords[0].bdist

        # interpolate first node
        self.interpolate(1)

        # grow string
        self.growth_iters(iters=max_iters,maxopt=opt_steps,nconstraints=1)
        print ' SE_Cross growth phase over'
        print ' Warning last node still not fully optimized'

        if rtype==0:
            # doing extra constrained penalty optimization for MECI
            self.icoords[self.nR-1].OPTTHRESH=0.01
            oiters=50
            ictan = DLC.tangent_1(self.icoords[self.nR-1],self.icoords[self.nR-2])
            self.icoords[self.nR-1].PES.sigma=3.5
            self.optimize(n=self.nR-1,opt_type=1,nsteps=oiters,ictan=ictan)
        else:
            # unconstrained penalty optimization
            self.icoords[self.nR-1].OPTTHRESH=self.CONV_TOL
            oiters=100
            nconstraints=0
            ictan=None
            self.optimize(n=self.nR-1,opt_type=0,nsteps=oiters)
            
        if rtype==0:
            self.write_xyz_files(iters=1,base="after_penalty",nconstraints=1)
            self.icoords[self.nR] = DLC.copy_node_X(self.icoords[self.nR-1],new_node_id=self.nR,rtype=5)
            self.icoords[self.nR].OPTTHRESH=self.CONV_TOL
            self.optimize(n=self.nR,nsteps=oiters*2,opt_type=5) #MECI opt
            self.write_xyz_files(iters=1,base="grown_string",nconstraints=1)
        else:
            self.write_xyz_files(iters=1,base="grown_string",nconstraints=1)

    def opt_string(self,max_iters=50,optsteps=3,rtype=0):
        self.nnodes=self.nR
        print "getting energies"
        for ico in self.icoords[0:self.nR]:
            if ico!=0:
                lot = ico.PES.lot.copy(ico.PES.lot,ico.PES.lot.node_id)
                pes = PES(ico.PES.PES2.options.copy().set_values({
                    "lot":lot,
                    }))
                ico.PES = pes
                ico.energy = ico.PES.get_energy(ico.geom)
        self.icoords[0].V0 = self.icoords[0].energy 
        print "initial energy is %4.3f" % self.icoords[0].V0

        self.store_energies()
        print " V_profile: ",
        for n in range(self.nnodes):
            print " {:7.3f}".format(float(self.energies[n])),
        print
        print "Setting all interior nodes to active"
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.icoords[n].OPTTHRESH=self.CONV_TOL
        self.ic_reparam(ic_reparam_steps=25)
        self.write_xyz_files(iters=1,base='initial_ic_reparam',nconstraints=1)
        self.opt_iters(max_iter=max_iters,optsteps=optsteps,rtype=rtype)
        self.icoords[self.TSnode].mol.write('xyz','TS.xyz',overwrite=True)

    def add_node(self,n1,n2,n3=None):
        print " adding node: %i from node %i"%(n2,n1)
        return DLC.add_node_SE_X(self.icoords[n1],self.driving_coords,dqmag_max=self.DQMAG_MAX,dqmag_min=self.DQMAG_MIN)
    
    def converged(self,n,opt_type):

        if opt_type==0:
            tmp1 = np.copy(self.icoords[n].PES.grad1)
            tmp2 = np.copy(self.icoords[n].PES.grad2)
            print 'norm1: {:1.4f} norm2: {:1.4f}'.format(np.linalg.norm(tmp1),np.linalg.norm(tmp2)),
            print 'ratio: {:1.4f}'.format(np.linalg.norm(tmp1)/np.linalg.norm(tmp2))
            tmp1 = tmp1/np.linalg.norm(tmp1)
            tmp2 = tmp2/np.linalg.norm(tmp2)
            print 'normalized gradient dot product:',float(np.dot(tmp1.T,tmp2))
            sys.stdout.flush()
            if self.icoords[n].gradrms<self.CONV_TOL and 1.-abs(float(np.dot(tmp1.T,tmp2))) <= 0.02 and abs(self.icoords[n].PES.dE) <= 1.25:
                return True
            else:
                return False
        elif opt_type==1: #constrained growth
            if self.icoords[n].gradrms<self.icoords[n].OPTTHRESH:
                return True
            else:
                return False
        elif opt_type==5:
            if self.icoords[n].gradrms<self.CONV_TOL and abs(self.icoords[n].PES.dE) <= 1.0:
                return True
            else:
                return False

    def check_if_grown(self):
        isDone = False
        epsilon = 1.5
        pes1dE = self.icoords[self.nR-1].PES.dE
        pes2dE = self.icoords[self.nR-2].PES.dE
        condition1 = (abs(self.icoords[self.nR-1].bdist) <=(1-self.BDIST_RATIO)*abs(self.icoords[0].bdist) and (abs(pes1dE) > abs(pes2dE)))
        if condition1:
            isDone = True
        return isDone

