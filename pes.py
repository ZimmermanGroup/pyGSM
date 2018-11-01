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
                required=False,
                doc='adiabatic index')

        PES._default_options = opt
        return PES._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """
        self.options = options

        self.lot = self.options['lot']
        self.ad_idx = self.options['ad_idx']

    def get_energy(self,geom):
        if self.checked_input == False:
            self.check_input(geom)
        self.lot.get_energy(coords,self.ad_idx,self.multiplicity)

    def check_input(self,geom):
        atoms = manage_xyz.get_atoms(self.geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.checked_input =True

if __name__ == '__main__':


