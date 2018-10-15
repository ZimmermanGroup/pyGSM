import numpy as np
import options
import os

class GSM(object):

    @staticmethod
    def default_options():
        if hasattr(GSM, '_default_options'): return GSM._default_options.copy()

        opt = options.Options() 

        GSM._default_options = opt
        return GSM._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes

        #TODO What is optCG Ask Paul
        self.optCG = False
        self.isTSnode =False


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

    
    filepath="tests/fluoroethene.xyz"

    # LOT object
    nocc=23
    nactive=2
    lot=PyTC.from_options(calc_states=[(0,0)],filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
    lot.cas_from_geom()

    mol=pb.readfile("xyz",filepath).next()
    ic1=ICoord.from_options(mol=mol,lot=lot)
    ic2=ICoord(ic1.options.copy().set_values(dict(filepath=filepath2)))

    gsm = GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9)


