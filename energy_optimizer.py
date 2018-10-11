import numpy as np
import openbabel as ob
import pybel as pb
import options
import ico as ic
#import grad

class EOpt(object):

    @staticmethod
    def default_options():
        if hasattr(EOpt, '_default_options'): return EOpt._default_options.copy()
        opt = options.Options() 
        opt.add_option(
            key='nsteps',
            value=1,
            required=False,
            allowed_types=[int],
            doc='Number of steps the optimizer will take')

        opt.add_option(
                key='penalty',
                value=1.0,
                required=False,
                allowed_types=[float],
                doc='Penalty parameter for penalty optimizer')

        opt.add_option(
                key='ICoord',
                required=False,
                allowed_types=[ic.ICoord],
                doc='Internal coord class object ')

        opt.add_option(
                key='filepath',
                value='initial0000.xyz',
                required=False,
                allowed_types=[str],
                doc='Path to xyz file')

        opt.add_option(
                key='PES',
                required=True,
                doc='Potential energy surface calculator')

        EOpt._default_options = opt
        return EOpt._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return EOpt(EOpt.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes
        self.nsteps = self.options['nsteps']
        self.ICoord = self.options['ICoord']
        self.filepath = self.options['filepath']
        self.PES = self.options['PES']

    def optimize(self):
        #Hintp_to_Hint()
        energy=0.



if __name__ == '__main__':

    from obutils import *
    from lot import *
    
    filepath="fluoroethene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    ic1=ic.ICoord.from_options(mol=mol)
    ic1.ic_create()
    ic1.bmatp_create()
    ic1.bmatp_to_U()
    ic1.bmat_create()
    
    nocc=23
    nactive=2
    calculator=LOT.from_options(wstate=0,wspin=0,filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
    calculator.cas_from_geom()
    print(calculator.getEnergy())

    #opt.optimize()

