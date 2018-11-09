import options
import manage_xyz
import numpy as np
from units import *
import elements 
ELEMENT_TABLE = elements.ElementData()

class PES(object):
    """ PES object """

    @staticmethod
    def default_options():

        if hasattr(PES, '_default_options'): return PES._default_options.copy()
        opt = options.Options() 

        opt.add_option(
                key='lot',
                value=None,
                required=True,
                doc='Level of theory object')

        opt.add_option(
                key='ad_idx',
                value=0,
                required=True,
                doc='adiabatic index')

        opt.add_option(
                key='multiplicity',
                value=1,
                required=True,
                doc='multiplicity')

        PES._default_options = opt
        return PES._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PES(PES.default_options().set_values(kwargs))

    def __init__(self,
            options,
            ):
        """ Constructor """
        self.options = options

        self.lot = self.options['lot']
        self.ad_idx = self.options['ad_idx']
        self.multiplicity = self.options['multiplicity']
        #self.do_coupling = False
        #self.dE = None

    def get_energy(self,geom):
        #if self.checked_input == False:
        #    self.check_input(geom)
        return self.lot.get_energy(geom,self.multiplicity,self.ad_idx)

    def get_gradient(self,geom):
        tmp =self.lot.get_gradient(geom,self.multiplicity,self.ad_idx)
        return np.reshape(tmp,(3*len(tmp),1))

    def check_input(self,geom):
        atoms = manage_xyz.get_atoms(self.geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.checked_input =True

if __name__ == '__main__':

    from qchem import *
    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    lot=QChem.from_options(states=[(1,0),(3,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    print pes.get_energy(geom)
    print pes.get_gradient(geom)

