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
        self.dE = 1000.
        print ' PES object parameters:'
        print ' Multiplicity:',self.multiplicity,'ad_idx:',self.ad_idx

    def get_energy(self,geom):
        #if self.checked_input == False:
        #    self.check_input(geom)
        return self.lot.get_energy(geom,self.multiplicity,self.ad_idx)

    def initial_energy(self,reffile,filepath,ref_nocc,nocc,
            fomo_temp=0.3,
            flip_occ_to_act=None,
            flip_vir_to_act=None,
            ):
        # need to make onlly for pytc
        if self.lot.from_template:
            self.lot.casci_from_file_from_template(reffile,filepath,ref_nocc,nocc,fomo_temp,flip_occ_to_act,flip_vir_to_act) 
        else:
           self.lot.casci_from_file(filepath)
        geom=manage_xyz.read_xyz(filepath,scale=1)   
        return self.get_energy(geom)

    def get_gradient(self,geom):
        tmp =self.lot.get_gradient(geom,self.multiplicity,self.ad_idx)
        return np.reshape(tmp,(3*len(tmp),1))

    def check_input(self,geom):
        atoms = manage_xyz.get_atoms(self.geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.checked_input =True

if __name__ == '__main__':

    if QCHEM:
        from qchem import *
    elif PYTC:
        from pytc import *

    filepath="tests/fluoroethene.xyz"
    nocc=11
    nactive=2
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    if QCHEM:
        lot=QChem.from_options(states=[(1,0),(3,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    elif PYTC:
        lot=PyTC.from_options(states=[(1,0),(3,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot.casci_from_file_from_template(x,x,nocc,nocc) # hack to get it to work,need casci1

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    print pes.get_energy(geom)
    print pes.get_gradient(geom)

