import numpy as np
import options
import os
from base_gsm import *
from icoord import *

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
            self.icoords[self.nR] = ICoord.add_node(self.icoords[self.nR-1],self.icoords[-self.nP])
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
            self.icoords[-self.nP-1] = ICoord.add_node(self.icoords[-self.nP],self.icoords[self.nR-1])
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

    def writenodes(self):
        count = 0
        for ico in self.icoords:
            if ico != 0:
                ico.mol.write('xyz','temp{:02}.xyz'.format(count),overwrite=True)
            count+=1

    def growth_iters(self,maxrounds=1,maxopt=1,nconstraints=0):
        for n in range(maxrounds):
            self.opt_step(maxopt,nconstraints)

    def opt_step(self,maxopt,nconstraints):
        for i in range(maxopt):
            for n in range(self.nnodes):
                if self.icoords[n] != 0:
                    if self.icoords[n].gradrms < 5e-3:
                        self.active[n] = False
                if self.active[n] == True:
                    self.icoords[n].optimize(1,nconstraints)

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
    

    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))


if __name__ == '__main__':
#    from icoord import *
    from qchem import *
    import manage_xyz
    filepath="tests/fluoroethene.xyz"
    filepath2="tests/stretched_fluoroethene.xyz"

    # LOT object
#    nocc=23
#    nactive=2
#    lot=PyTC.from_options(calc_states=[(0,0)],filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
    #lot.cas_from_geom()


    mol=pb.readfile("xyz",filepath).next()
    mol2=pb.readfile("xyz",filepath2).next()
    geom = manage_xyz.read_xyz(filepath,scale=1)
    geom2 = manage_xyz.read_xyz(filepath2,scale=1)
    lot=QChem.from_options(E_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP')
    lot2=QChem.from_options(E_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP')

    print "\n IC1 \n\n"
    ic1=ICoord.from_options(mol=mol,lot=lot)
    print "\n IC2 \n\n"
    ic2=ICoord.from_options(mol=mol2,lot=lot2)

    print "\n Starting GSM \n\n"
    gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=10,nconstraints=1)

    gsm.interpolate() 
#    gsm.write_node_xyz("nodes_xyz_file0.xyz")
#    gsm.growth_iters(maxrounds=1,nconstraints=1)
#    gsm.write_node_xyz()

#    gsm.interpolate()
#    gsm.write_node_xyz()
#    gsm.icoords[1].optimize(20,1)
#    gsm.icoords[1].ic_create()
#    gsm.icoords[1].bmatp_create()
#    gsm.icoords[1].bmatp_to_U()
#    gsm.icoords[1].make_Hint()
#    for ico in gsm.icoords:
#        if ico != 0:
#            ico.optimize(20,1)
   


