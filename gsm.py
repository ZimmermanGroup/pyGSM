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

    def interpolate(self,newnodes=1):
        count = 0
        while count < newnodes:
            self.icoords[self.nn-1] = ICoord.add_node(self.icoords[self.nn-2],self.icoords[-1])
            self.nn+=1
            count+=1

    def writenodes(self):
        count = 0
        for ico in self.icoords:
            if ico != 0:
                ico.mol.write('xyz','temp{:02}.xyz'.format(count),overwrite=True)
            count+=1

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
    gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9)
#    gsm.icoords[1]=ICoord.add_node(gsm.icoords[0],gsm.icoords[8])
    gsm.interpolate(7) 
    gsm.writenodes()
    gsm.icoords[1].optimize(20,1)
    gsm.icoords[1].ic_create()
    gsm.icoords[1].bmatp_create()
    gsm.icoords[1].bmatp_to_U()
    gsm.icoords[1].make_Hint()
#    for ico in gsm.icoords:
#        if ico != 0:
#            ico.optimize(20,1)
   


