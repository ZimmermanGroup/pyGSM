import numpy as np
import options
import os
from base_gsm import *

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

if __name__ == '__main__':
    from icoords import *
    from qchem import *
    import manage_xyz
    filepath="tests/fluoroethene.xyz"

    # LOT object
#    nocc=23
#    nactive=2
#    lot=PyTC.from_options(calc_states=[(0,0)],filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
    #lot.cas_from_geom()


    mol=pb.readfile("xyz",filepath).next()
    geom = manage_xyz.read_xyz(filepath,scale=1)
    lot=QChem.from_options(calc_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP')


    ic1=ICoord.from_options(mol=mol,lot=lot)
#    ic2=ICoord(ic1.options.copy().set_values(dict(filepath=filepath2)))

    gsm=GSM.from_options(ICoord1=ic1,nnodes=9)


