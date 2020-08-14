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
from .se_gsm import SE_GSM
from potential_energy_surfaces import Avg_PES


class SE_Cross(SE_GSM):

    def go_gsm(self,max_iters=50,opt_steps=3,rtype=0):
        """rtype=0 MECI search
           rtype=1 MESX search
        """
        assert rtype in [0,1], "rtype not defined"
        if rtype==0:
            nifty.printcool("Doing SE-MECI search")
        else:
            nifty.printcool("Doing SE-MESX search")

        self.nodes[0].gradrms=0.
        self.nodes[0].V0 = self.nodes[0].energy
        print(' Initial energy is {:1.4f}'.format(self.nodes[0].energy))
        sys.stdout.flush()

        # stash bdist for node 0
        _,self.nodes[0].bdist = Base_Method.tangent(self.nodes[0],None,driving_coords=self.driving_coords)
        print(" Initial bdist is %1.3f" %self.nodes[0].bdist)

        # interpolate first node
        self.add_GSM_nodeR()

        # grow string
        self.growth_iters(iters=max_iters,maxopt=opt_steps,nconstraints=1)
        print(' SE_Cross growth phase over')
        print(' Warning last node still not fully optimized')


        if True:
            # doing extra constrained penalty optimization for MECI
            print(" extra constrained optimization for the nnR-1 = %d" % (self.nR-1))
            self.optimizer[self.nR-1].conv_grms=0.01
            ictan,_ = Base_Method.tangent(self.nodes[self.nR-1],self.nodes[self.nR-2])
            self.nodes[self.nR-1].PES.sigma=3.5

            self.optimizer[self.nR-1].optimize(
                    molecule=self.nodes[self.nR-1],
                    refE=self.nodes[0].V0,
                    opt_type='ICTAN',
                    opt_steps=50,
                    ictan=ictan,
                    )

        self.optimizer[self.nR].opt_cross=True
        if rtype==0:
            # MECI optimization
            self.write_xyz_files(iters=1,base="after_penalty",nconstraints=1)
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR-1],new_node_id=self.nR)
            avg_pes = Avg_PES.create_pes_from(self.nodes[self.nR].PES)
            self.nodes[self.nR].PES = avg_pes
            self.optimizer[self.nR].conv_grms=self.options['CONV_TOL']
            self.optimizer[self.nR].optimize(
                    molecule=self.nodes[self.nR],
                    refE=self.nodes[0].V0,
                    opt_type='MECI',
                    opt_steps=100,
                    )
            self.write_xyz_files(iters=1,base="grown_string",nconstraints=1)
        else:
            # unconstrained penalty optimization
            #TODO make unctonstrained "CROSSING" which checks for dE convergence
            self.nodes[self.nR] = Molecule.copy_from_options(self.nodes[self.nR-1],new_node_id=self.nR)
            self.nodes[self.nR].PES.sigma = 10.0
            print(" sigma for node %d is %.3f" %(self.nR,self.nodes[self.nR].PES.sigma))
            self.optimizer[self.nR].opt_cross=True
            self.optimizer[self.nR].conv_grms=self.options['CONV_TOL']
            self.optimizer[self.nR].conv_gmax = self.options['CONV_TOL']
            self.optimizer[self.nR].conv_Ediff = 0.1
            self.optimizer[self.nR].optimize(
                    molecule=self.nodes[self.nR],
                    refE=self.nodes[0].V0,
                    opt_type='UNCONSTRAINED',
                    opt_steps=200,
                    )
            self.write_xyz_files(iters=1,base="grown_string",nconstraints=1)

        
        tmp = []
        for n in range(self.nnodes):
            tmp.append(self.energies[n])
        self.energies = np.asarray(tmp)
    
    def converged(self,n,opt_type):
        if opt_type=="UNCSONTRAINED":
            tmp1 = np.copy(self.nodes[n].PES.grad1)
            tmp2 = np.copy(self.nodes[n].PES.grad2)
            print('norm1: {:1.4f} norm2: {:1.4f}'.format(np.linalg.norm(tmp1),np.linalg.norm(tmp2)))
            print('ratio: {:1.4f}'.format(np.linalg.norm(tmp1)/np.linalg.norm(tmp2)))
            tmp1 = tmp1/np.linalg.norm(tmp1)
            tmp2 = tmp2/np.linalg.norm(tmp2)
            print('normalized gradient dot product:',float(np.dot(tmp1.T,tmp2)))
            sys.stdout.flush()
            if self.nodes[n].gradrms<self.options['CONV_TOL'] and 1.-abs(float(np.dot(tmp1.T,tmp2))) <= 0.02 and abs(self.nodes[n].PES.dE) <= 1.25:
                return True
            else:
                return False
        elif opt_type=="ICTAN": #constrained growth
            if self.nodes[n].gradrms<self.optimizer[n].conv_grms:
                return True
            else:
                return False
        elif opt_type=="MECI":
            if self.nodes[n].gradrms<self.options['CONV_TOL'] and abs(self.nodes[n].PES.dE) <= 1.0:
                return True
            else:
                return False

    def check_if_grown(self):
        isDone = False
        epsilon = 1.5
        pes1dE = self.nodes[self.nR-1].PES.dE
        pes2dE = self.nodes[self.nR-2].PES.dE
        condition1 = (abs(self.nodes[self.nR-1].bdist) <=(1-self.BDIST_RATIO)*abs(self.nodes[0].bdist) and (abs(pes1dE) > abs(pes2dE)))
        condition2= ((self.nodes[self.nR-1].bdist+0.1>self.nodes[self.nR-2].bdist) and (1-self.BDIST_RATIO)*abs(self.nodes[0].bdist))
        if condition1:
            print(" Condition 1 satisfied")
            print(" bdist current %1.3f" % abs(self.nodes[self.nR-1].bdist))
            print(" bdist target %1.3f" % (abs(self.nodes[0].bdist)*(1-self.BDIST_RATIO)))
            print(" Growth-phase over")
            isDone=True
        #elif condition2:
        #    print(" Condition 2 satisfied")
        #    print(" Growth-phase over")
        #    isDone = True
        return isDone

    def restart_string(self,xyzfile='restart.xyz'):
        super(SE_Cross,self).restart_string(xyzfile)
        self.done_growing=False
        self.nnodes=20
        self.nR -=1 
        # stash bdist for node 0
        _,self.nodes[0].bdist = Base_Method.tangent(self.nodes[0],None,driving_coords=self.driving_coords)

