import numpy as np
import options
import os
from base_gsm import *
from icoord import *
import pybel as pb
import sys
from pes import *

class GSM(BaseGSM):

    def starting_string(self):
        #dq
        #add_node()
        #add_node()
        return

    def de_gsm(self):
        #grow string
        #for i in range(max_iters)
        #grow_node if okay
        #optimize string
        #find TS
        return

    def interpolateR(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
            return
        for i in range(newnodes):
            tempR = ICoord.union_ic(self.icoords[self.nR-1],self.icoords[-self.nP])
            tempP = ICoord.union_ic(self.icoords[-self.nP],self.icoords[self.nR-1])
            self.icoords[self.nR] = ICoord.add_node(tempR,tempP)
            self.active[self.nR] = True
#            ictan = ICoord.tangent_1(self.icoords[self.nR],self.icoords[-self.nP])
#            self.icoords[self.nR].opt_constraint(ictan)
            self.nn+=1
            self.nR+=1

    def interpolateP(self,newnodes=1):
        if self.nn+newnodes > self.nnodes:
            print("Adding too many nodes, cannot interpolate")
            return
        for i in range(newnodes):
            tempP = ICoord.union_ic(self.icoords[-self.nP],self.icoords[self.nR-1])
            tempR = ICoord.union_ic(self.icoords[self.nR-1],self.icoords[-self.nP])
            self.icoords[-self.nP-1] = ICoord.add_node(tempP,tempR)
#            ictan = ICoord.tangent_1(self.icoords[-self.nP-1],self.icoords[self.nR-1])
#            self.icoords[-self.nP-1].opt_constraint(ictan)
            self.active[-self.nP-1] = True
            self.nn+=1
            self.nP+=1

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

    def interpolate2(self,newnodes=1):
        sign = -1
        for i in range(newnodes):
            sign *= -1
            if sign == 1:
                self.add_node(self.nR-1,self.nR,-self.nP)
                self.nR += 1
            elif sign == -1:
                self.add_node(-self.nP,-self.nP-1,self.nR-1)
                self.nP += 1            

    def write_xyz_files(self,iters=0):
        xyzfile = os.getcwd()+'/'+'opt_nodes_{:03}.xyz'.format(iters)
        stringxyz = pb.Outputfile('xyz',xyzfile,overwrite=True)
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat('xyz')
        opt_nodes = []
        for ico,act in zip(self.icoords,self.active):
            if act:
                mol = obconversion.WriteString(ico.mol.OBMol)
                opt_nodes.append(mol)
            elif ico != 0:
                mol = obconversion.WriteString(ico.mol.OBMol)
                opt_nodes.append(mol)

        for mol in opt_nodes:
            stringxyz.write(pb.readstring('xyz',mol))

        with open(xyzfile,'r+') as f:
            content = f.read()
            f.seek(0,0)
            f.write("[Molden Format]\n[Geometries] (XYZ)\n"+content)
#            print "writing geometries to",xyzfile
        with open(xyzfile, 'a') as f:
            f.write("[GEOCONV]\n")
            f.write('energy\n')
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.V0+ico.energy))
                elif ico != 0:
                    f.write('{}\n'.format(ico.PES.get_energy(ico.geom)))
#            print "writing energies to",xyzfile
            f.write("max-force\n")
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.gradrms))
                #elif ico != 0:
                #    f.write('{}\n'.format(ico.gradrms))
            f.write('max-step \n')
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.smag))
                elif ico != 0:
                    f.write('{}\n'.format(0.))
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.dqmag))
                elif ico != 0:
                    f.write('{}\n'.format(0.))
        f.close()

    def growth_iters(self,maxrounds=1,maxopt=3,nconstraints=0,iters=0):
        for n in range(maxrounds):
            self.opt_step(maxopt,nconstraints)
        self.write_xyz_files(iters)

    def opt_step(self,maxopt,nconstraints):
        for i in range(maxopt):
            for n in range(self.nnodes):
                if self.icoords[n] != 0:
                    if self.icoords[n].gradrms < 5e-3:
                        self.active[n] = False
                        try:
                            if self.icoords[n+1] == 0:
                                self.interpolateR()
                                self.active[n+1] = True
                            elif self.icoords[n-1] == 0:
                                self.interpolateP()
                                self.active[n-1] = True
                        except IndexError:
                            pass
                if self.active[n] == True:
                    self.icoords[n].smag = self.icoords[n].optimize(1,nconstraints)

    def write_node_xyz(self,xyzfile = "nodes_xyz_file.xyz"):
        xyzfile = os.getcwd()+"/"+xyzfile
        nodesXYZ = pb.Outputfile("xyz",xyzfile,overwrite=True)
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat('xyz')
        opt_mols = []
        for ico in self.icoords:
            if ico != 0:
                opt_mols.append(obconversion.WriteString(ico.mol.OBMol))
        for mol in opt_mols:
            nodesXYZ.write(pb.readstring("xyz",mol))

    def get_tangents_1g(self):
        num_ics = self.icoords[0].num_ics
        ictan0 = [0]*self.nnodes
        ictan = [[]]*self.nnodes
        dqmaga = [0.] * self.nnodes
        dqa = [[]]*self.nnodes
        for n in range(len(self.icoords)-1):
            if self.icoords[n] != 0:
                if self.icoords[n+1] != 0:
                    self.icoords[n] = ICoord.union_ic(self.icoords[n],self.icoords[n+1])
                    self.icoords[n+1] = ICoord.union_ic(self.icoords[n+1],self.icoords[n])
                    
                    #TODO tangent_1b() for SSSM
                    ictan[n] = ICoord.tangent_1(self.icoords[n+1],self.icoords[n])            
                elif self.icoords[n+1] == 0:
                    self.icoords[n] = ICoord.union_ic(self.icoords[n],self.icoords[-self.nP])
                    self.icoords[-self.nP] = ICoord.union_ic(self.icoords[-self.nP],self.icoords[n])
                    ictan[n] = ICoord.tangent_1(self.icoords[n],self.icoords[-self.nP])
                
                ictan0 = np.copy(ictan[n])
    
                self.icoords[n].bmatp_create()
                self.icoords[n].bmatp_to_U()
                self.icoords[n].opt_constraint(ictan[n])
                self.icoords[n].bmat_create()
                if self.icoords[n+1] != 0:
                    dqmaga[n] = np.dot(ictan0.T,self.icoords[n+1].Ut[-1,:])
                elif self.icoords[n+1] == 0:
                    dqmaga[n] = np.dot(ictan0.T,self.icoords[-self.nP].Ut[-1,:])
                dqmaga[n] = float(np.sqrt(dqmaga[n]))
                self.icoords[n].dqmag = dqmaga[n]
            else:
                pass        

        self.dqmaga = dqmaga
        self.ictan = ictan
            

    def get_tangents_1g_2(self):
        size_ic = self.icoords[0].num_ics
        ictan0 = [0]*self.nnodes
        ictan = [[]]*self.nnodes
        nlist = [0]*(2*self.nnodes)
        ncurrent = 0
        dqmaga = [0.]*self.nnodes
        dqa = [[],]*self.nnodes

        for n in range(self.nR):
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
        if self.nR == 0: 
            nlist[2*ncurrent+1] += 1
        if self.nP == 0:
            nlist[2*ncurrent] -= 1
        ncurrent += 1

        for n in range(ncurrent):
            self.icoords[nlist[2*n+1]] = ICoord.union_ic(self.icoords[nlist[2*n+1]],self.icoords[nlist[2*n+0]])
            self.icoords[nlist[2*n+0]] = ICoord.union_ic(self.icoords[nlist[2*n+0]],self.icoords[nlist[2*n+1]])

            if self.isSSM:
                pass #do tangent_1b()
            else:
                ictan[nlist[2*n]] = ICoord.tangent_1(self.icoords[nlist[2*n+1]],self.icoords[nlist[2*n+0]])

            ictan0 = np.copy(ictan[nlist[2*n]])

            if True:
                self.icoords[nlist[2*n+1]].bmatp_create()
                self.icoords[nlist[2*n+1]].bmatp_to_U()
                self.icoords[nlist[2*n+1]].opt_constraint(ictan[nlist[2*n]])
            self.icoords[nlist[2*n+1]].bmat_create()
        
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,self.icoords[nlist[2*n+1]].Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(dqmaga[nlist[2*n]]))

        self.dqmaga = dqmaga
        self.ictan = ictan

    def ic_reparam_g(self,ic_reparam_steps=4):
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
            totaldqmag = np.sum(self.dqmaga)
            print " totaldqmag (without inner): {:1.2}\n".format(totaldqmag)
            print " printing spacings dqmaga: "
            for n in range(self.nnodes):
                print " {:1.2}".format(self.dqmaga[n])
            
            if i == 0:
                rpart += 1.0/(self.nn-2)
                rpart[0] = 0.0
                rpart[-1] = 0.0
                print " rpart: "
                for n in range(1,self.nnodes):
                    print " {:1.2}".format(rpart[n])
            nR0 = self.nR
            nP0 = self.nP
            if self.nnodes-self.nn > 2:
                nR0 -= 1
                nP0 -= 0
            
            deltadq = 0.0
            for n in range(1,nR0):
                deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes-nP0,self.nnodes-1):
                deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(1,self.nnodes-1):
                if abs(rpmove[n]) > MAXRE:
                    repmove[n] = float(np.sign(rpmove[n])*MAXRE)

            disprms = float(np.linalg.norm(rpmove))
            lastdispr = disprms
            print " disprms: {:1.3}\n".format(disprms)

            if disprms < 1e-2:
                break

            for n in range(1,self.nnodes-1):
                if isinstance(self.icoords[n],ICoord):
                    if rpmove[n] > 0:
                        if len(self.ictan[n]) != 0:
                            #print "May need to make copy_CI"
                            #This does something to ictan0
                            self.icoords[n].update_ics()
                            self.icoords[n].bmatp_create()
                            self.icoords[n].bmatp_to_U()
                            
                            self.icoords[n].opt_constraint(self.ictan[n])
                            self.icoords[n].bmat_create()
                            dq0 = np.zeros((self.icoords[n].nicd,1))
                            dq0[-1] = rpmove[n]
                            print " dq0[constraint]: {:1.3}".format(dq0[-1])
                            self.icoords[n].ic_to_xyz(dq0)
                            self.icoords[n].update_ics()
                        else:
                            pass
                else:
                    pass
        print " spacings (end ic_reparam, steps: {}):".format(ic_reparam_steps)
        for n in range(self.nnodes):
            print " {:1.2}".format(self.dqmaga[n])
            print "  disprms: {:1.3}".format(disprms)
        #Failed = check_array(self.nnodes,self.dqmaga)
        #If failed, do exit 1



    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))


if __name__ == '__main__':
#    from icoord import *
    from qchem import *
    import manage_xyz

    if True:
        filepath="tests/fluoroethene.xyz"
        filepath2="tests/stretched_fluoroethene.xyz"

    if True:
        filepath2="tests/SiH2H2.xyz"
        filepath="tests/SiH4.xyz"

    mol=pb.readfile("xyz",filepath).next()
    mol2=pb.readfile("xyz",filepath2).next()
#    geom = manage_xyz.read_xyz(filepath,scale=1)
#    geom2 = manage_xyz.read_xyz(filepath2,scale=1)
    lot=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=8)
    lot2=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=8)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot2,ad_idx=0,multiplicity=1)

    print "\n IC1 \n\n"
    ic1=ICoord.from_options(mol=mol,PES=pes)
    print "\n IC2 \n\n"
    ic2=ICoord.from_options(mol=mol2,PES=pes2)

    if True:
        print "\n Starting GSM \n\n"
        gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9,nconstraints=1)

        gsm.interpolate(2) 
        gsm.ic_reparam_g()
        #gsm.interpolate2(7)
        gsm.write_node_xyz("nodes_xyz_file0.xyz")

    if False:
        gsm.ic_reparam_g()
    
    if False:
        gsm.get_tangents_1g()
        print "DQMAGA:",gsm.dqmaga
        print "Printing Tangents:"
        for tan in gsm.ictan:
            print np.transpose(tan)

    if True:
        iters = 0
        while True:
            print "beginning iteration:",iters
            sys.stdout.flush()
            if iters >= 20:
                break
            gsm.ic_reparam_g()
            do_growth = False
            for act in gsm.active:
                if act:
                    do_growth = True
                    break
            if do_growth:
                gsm.growth_iters(nconstraints=1,iters=iters)
                sys.stdout.flush()
            else:
                break
            iters += 1
        #gsm.growth_iters(nconstraints=1)
        gsm.write_node_xyz()



