import numpy as np
import options
import os
from base_gsm import *
from icoord import *
import pybel as pb

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
            #self.icoords[self.nR-1] = tempR
            #self.icoords[-self.nP] = tempP
            #self.icoords[self.nR] = ICoord.add_node(self.icoords[self.nR-1],self.icoords[-self.nP])
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
            #self.icoords[-self.nP] = tempP
            #self.icoords[self.nR-1] = tempR
            #self.icoords[-self.nP-1] = ICoord.add_node(self.icoords[-self.nP],self.icoords[self.nR-1])
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

    def writenodes(self):
        count = 0
        for ico in self.icoords:
            if ico != 0:
                ico.mol.write('xyz','temp{:02}.xyz'.format(count),overwrite=True)
            count+=1

    def growth_iters(self,maxrounds=1,maxopt=1,nconstraints=0):
        for n in range(maxrounds):
            self.opt_step(maxopt,nconstraints)
        xyzfile = os.getcwd()+'/'+'opt_nodes.xyz'
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
                    f.write('{}\n'.format(ico.lot.getEnergy()))
#            print "writing energies to",xyzfile
            f.write("max-force\n")
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.gradrms))
                elif ico != 0:
                    f.write('{}\n'.format(ico.lot.getGrad()))
            f.write('max-step \n')
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.smag))
                elif ico != 0:
                    f.write('{}\n'.format(0.))
        f.close()

    def opt_step(self,maxopt,nconstraints):
        for i in range(maxopt):
            for n in range(self.nnodes):
                if self.icoords[n] != 0:
                    if self.icoords[n].gradrms < 5e-3:
                        self.active[n] = False
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
        size_ic = self.icoords[0].nbonds+self.icoords[0].nangles+self.icoords[0].ntor
        ictan0 = [0]*self.nnodes
        ictan = [[],]*self.nnodes
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
            newic = self.icoords[nlist[2*n+1]]
            intic = self.icoords[nlist[2*n+0]]

            if self.isSSM:
                pass #do tangent_1b()
            else:
                ictan[nlist[2*n]] = ICoord.tangent_1(newic,intic)

            ictan0 = np.copy(ictan[nlist[2*n]])

            if True:
                newic.bmatp_create()
                newic.bmatp_to_U()
                newic.opt_constraint(ictan[nlist[2*n]])
            newic.bmat_create()
        
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,newic.Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(dqmaga[nlist[2*n]]))

        self.dqmaga = dqmaga
        self.ictan = ictan

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
    geom = manage_xyz.read_xyz(filepath,scale=1)
    geom2 = manage_xyz.read_xyz(filepath2,scale=1)
    lot=QChem.from_options(E_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP',nproc=2)
    lot2=QChem.from_options(E_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP',nproc=2)

    print "\n IC1 \n\n"
    ic1=ICoord.from_options(mol=mol,lot=lot)
    print "\n IC2 \n\n"
    ic2=ICoord.from_options(mol=mol2,lot=lot2)

    if True:
        print "\n Starting GSM \n\n"
        gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9,nconstraints=1)

        gsm.interpolate(7) 
        #gsm.interpolate2(7)
        gsm.write_node_xyz("nodes_xyz_file0.xyz")

    if False:
        gsm.get_tangents_1g()
        print "DQMAGA:",gsm.dqmaga
        print "Printing Tangents:"
        for tan in gsm.ictan:
            print np.transpose(tan)

    if False:
        gsm.growth_iters(maxrounds=3,nconstraints=1)
        gsm.write_node_xyz()

   


