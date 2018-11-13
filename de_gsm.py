import numpy as np
import options
import os
from base_gsm import *
from dlc import *
from pes import *
import pybel as pb
import sys

class GSM(Base_Method):

    def interpolate(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
        sign = -1
        for i in range(newnodes):
            print "Adding node",i
            sign *= -1
            if sign == 1:
                self.interpolateR()
            else:
                self.interpolateP()

    def __init__(
            self,
            options,
            ):

        super(GSM,self).__init__(options)

        tmp = self.options['ICoord2']
        self.icoords[0] = DLC.union_ic(self.icoords[0],tmp)
        print "after union"
        lot1 = tmp.PES.lot.copy(
                tmp.PES.lot, 
                self.nnodes-1)
        PES1 = PES(tmp.PES.options.copy().set_values({
            "lot": lot1,
            }))
        self.icoords[-1] = DLC(self.icoords[0].options.copy().set_values(dict(
            mol= tmp.mol,
            PES=PES1,
            )))

    def add_node(self,n1,n2,n3):
        print "adding node: %i between %i %i" %(n2,n1,n3)
        return DLC.add_node(self.icoords[n1],self.icoords[n3],self.nnodes,self.nn)

    def set_active(self,nR,nP):
        print(" Here is active:",self.active)
        if nR!=nP:
            print(" setting active nodes to %i and %i"%(nR,nP))
        else:
            print(" setting active node to %i "%nR)

        for i in range(self.nnodes):
            if self.icoords[i] !=0:
                self.active[i] = False;
                self.icoords[i].OPTTHRESH = self.CONV_TOL*2.;
        self.active[nR] = True
        self.active[nP] = True
        print(" Here is new active:",self.active)

    def tangent(self,n1,n2):
        return DLC.tangent_1(self.icoords[n1],self.icoords[n2])

    def grow_string(self,maxiters=20):
        print 'Starting Growth Phase'
        self.write_node_xyz("nodes_xyz_file0.xyz")
        iters = 1
        while True:
            print "beginning iteration:",iters
            sys.stdout.flush()
            do_growth = False
            for act in self.active:
                if act:
                    do_growth = True
                    break
            if do_growth:
                self.growth_iters(nconstraints=1,current=iters)
                sys.stdout.flush()
            else:
                print 'All nodes added. String done growing'
                break
            iters += 1
            if iters > maxiters:
                raise ValueError("reached max number of growth iterations")
        self.write_node_xyz()

    def opt_steps(self,maxopt,nconstraints):
        for i in range(maxopt):
            for n in range(self.nnodes):
                if self.icoords[n] != 0 and self.active[n]==True:
                    print "optimizing node %i" % n
                    self.icoords[n].opt_constraint(self.ictan[n])
                    print self.icoords[n].coords
                    self.icoords[n].smag = self.optimize(n,3,nconstraints)

    def check_add_node(self):
        if self.icoords[self.nR-1].gradrms < self.gaddmax:
            self.active[self.nR-1] = False
            if self.icoords[self.nR] == 0:
                self.interpolateR()
        if self.icoords[self.nnodes-self.nP].gradrms < self.gaddmax:
            self.active[self.nnodes-self.nP] = False
            if self.icoords[-self.nP-1] == 0:
                self.interpolateP()

    def get_tangents_1e(self,n0=0):
        size_ic = self.icoords[0].num_ics
        nbonds = self.icoords[0].BObj.nbonds
        nangles = self.icoords[0].AObj.nangles
        ntor = self.icoords[0].TObj.ntor
        ictan = np.zeros((self.nnodes,size_ic))
        ictan0 = np.copy(ictan[0])
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes,self.nnodes))
        
        print " V_profile: ",
        for n in range(self.nnodes):
            print " {:1.1}".format(self.energies[n]),
        print
        
        TSnode = self.nmax
        for n in range(n0+1,self.nnodes-1):
            do3 = False
            if not self.find:
                if self.energies[n+1] > self.energies[n] and self.energies[n] > self.energies[n-1]:
                    n_sm = n
                    n_lg = n+1
                elif self.energies[n-1] > self.energies[n] and self.energies[n] > self.energies[n+1]:
                    n_sm = n-1
                    n_lg = n
                else:
                    do3 = True
                    newic = n
                    intic = n+1
                    int2ic = n-1
            else:
                if n < TSnode:
                    n_sm = n
                    n_lg = n+1
                elif n> TSnode:
                    n_sm = n-1
                    n_lg = n
                else:
                    do3 = True
                    newic = n
                    intic = n+1
                    int2ic = n-1
            if not do3:
                ictan[n] = np.transpose(tangent_1(self.icoords[n_lg],self.icoords[n_sm]))
            else:
                f1 = 0.
                dE1 = abs(self.energies[n+1]-self.energies[n])
                dE2 = abs(self.energies[n] - self.energies[n-1])
                dEmax = max(dE1,dE2)
                dEmin = min(dE1,dE2)
                if self.energies[n+1]>self.energies[n-1]:
                    pass
        
        self.dqmaga = dqmaga
        self.ictan = ictan

    def start_string(self):
        print "\n"
        self.interpolate(2) 
        self.nn=2
        self.nR=1
        self.nP=1

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        """size_ic = self.icoords[0].num_ics; len_d = self.icoords[0].nicd"""
        num_ics = self.icoords[0].num_ics
        
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)

        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(self.nnodes)
        edist = np.zeros(self.nnodes)
        
        TSnode = -1 # What is this?
        emax = -1000 # And this?

        for i in range(ic_reparam_steps):
            self.get_tangents_1g()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP:self.nnodes])
            print " totaldqmag (without inner): {:1.2}\n".format(totaldqmag)
            print " printing spacings dqmaga: "
            for n in range(self.nnodes):
                print " {:1.2}".format(self.dqmaga[n])
            print ''
            
            if i == 0:
                for n in range(n0+1,self.nR):
                    rpart[n] += 1.0/(self.nn-2)
                for n in range(self.nnodes-self.nP,self.nnodes-1):
                    rpart[n] += 1.0/(self.nn-2)
#                rpart[0] = 0.0
#                rpart[-1] = 0.0
                print " rpart: "
                for n in range(1,self.nnodes):
                    print " {:1.2}".format(rpart[n]),
                print
            nR0 = self.nR
            nP0 = self.nP

            if False:
                if self.nnodes-self.nn > 2:
                    nR0 -= 1
                    nP0 -= 0
            
            deltadq = 0.0
            for n in range(n0+1,nR0):
                deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes-nP0,self.nnodes-1):
                deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n]) > MAXRE:
                    rpmove[n] = float(np.sign(rpmove[n])*MAXRE)

            for n in range(n0+1,self.nnodes-1):
                print " disp[{}]: {:1.2f}".format(n,rpmove[n]),
            print

            disprms = float(np.linalg.norm(rpmove[n0+1:self.nnodes-1]))
            lastdispr = disprms
            print " disprms: {:1.3}\n".format(disprms)

            if disprms < 1e-2:
                break

                            #TODO check how range is defined in gstring, uses n0...
            for n in range(n0+1,self.nnodes-1):
                if isinstance(self.icoords[n],DLC):
                    if rpmove[n] > 0:
                        if len(self.ictan[n]) != 0:# and self.active[n]: #may need to change active requirement TODO
                            #print "May need to make copy_CI"
                            #This does something to ictan0
                            self.icoords[n].update_ics()
                            self.icoords[n].bmatp = self.icoords[n].bmatp_create()
                            self.icoords[n].bmatp_to_U()
                            self.icoords[n].opt_constraint(self.ictan[n])
                            self.icoords[n].bmat_create()
                            dq0 = np.zeros((self.icoords[n].nicd,1))
                            dq0[self.icoords[n].nicd-nconstraints] = rpmove[n]
                            print " dq0[constraint]: {:1.3}".format(dq0[self.icoords[n].nicd-nconstraints])
                            self.icoords[n].ic_to_xyz(dq0)
                            self.icoords[n].update_ics()
                        else:
                            pass
                else:
                    pass
        print " spacings (end ic_reparam, steps: {}):".format(ic_reparam_steps),
        for n in range(self.nnodes):
            print " {:1.2}".format(self.dqmaga[n]),
        print "  disprms: {:1.3}".format(disprms)
        #Failed = check_array(self.nnodes,self.dqmaga)
        #If failed, do exit 1

    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))


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

    if False:
        filepath="tests/fluoroethene.xyz"
        filepath2="tests/stretched_fluoroethene.xyz"
        nocc=11
        nactive=2

    if True:
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
        lot2=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot2.casci_from_file_from_template(filepath,filepath2,nocc,nocc)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot2,ad_idx=0,multiplicity=1)

    print "\n IC1 \n"
    ic1=DLC.from_options(mol=mol,PES=pes)
    print "\n IC2 \n"
    ic2=DLC.from_options(mol=mol2,PES=pes2)

    if True:
        print "\n Starting GSM \n"
        gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9,nconstraints=1,CONV_TOL=0.001)
        gsm.icoords[0].gradrms = 0.
        gsm.icoords[-1].gradrms = 0.
        gsm.icoords[0].energy = gsm.icoords[0].PES.get_energy(gsm.icoords[0].geom)
        gsm.icoords[-1].energy = gsm.icoords[-1].PES.get_energy(gsm.icoords[-1].geom)
        print 'gsm.icoords[0] E:',gsm.icoords[0].energy
        print 'gsm.icoords[-1]E:',gsm.icoords[-1].energy
        gsm.interpolate(2)
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

