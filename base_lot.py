import options
import manage_xyz
import numpy as np
from units import *
import elements 
ELEMENT_TABLE = elements.ElementData()


class Lot(object):
    """ Lot object for level of theory calculators """

    @staticmethod
    def default_options():
        """ Lot default options. """

        if hasattr(Lot, '_default_options'): return Lot._default_options.copy()
        opt = options.Options() 


        opt.add_option(
            key='states',
            allowed_types=[list],
            doc='list of states 0-indexed')

        opt.add_option(
                key='functional',
                required=False,
                allowed_types=[str],
                doc='density functional')

        opt.add_option(
                key='nocc',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of occupied orbitals (for CAS)')

        opt.add_option(
                key='nactive',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of active orbitals (for CAS)')

        opt.add_option(
                key='basis',
                value=0,
                required=False,
                allowed_types=[str],
                doc='Basis set')

        opt.add_option(
                key='charge',
                value=0,
                required=False,
                allowed_types=[int],
                doc='charge of molecule')

        opt.add_option(
                key='nproc',
                required=False,
                value=1,
                doc="number of processors",
                )

        opt.add_option(
                key='do_coupling',
                required=False,
                value=False,
                doc='derivative coupling'
                )

        opt.add_option(
                key="node_id",
                required=False,
                value=0,
                doc='unique id used for storing orbs,etc'
                )

        Lot._default_options = opt
        return Lot._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """

        self.options = options
        # Cache some useful atributes
        self.states =self.options['states']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']
        self.functional=self.options['functional']
        self.nproc=self.options['nproc']
        self.charge = self.options['charge']
        self.do_coupling=self.options['do_coupling']
        self.node_id=self.options['node_id']
        self.hasRanForCurrentCoords =False
        self.has_nelectrons =False

    def check_multiplicity(self,multiplicity):
        if multiplicity > self.n_electrons + 1:
            raise ValueError("Spin multiplicity too high.")
        if (self.n_electrons + multiplicity + 1) % 2:
            raise ValueError("Inconsistent charge/multiplicity.")
            
    def get_nelec(self,geom,multiplicity):
        atoms = manage_xyz.get_atoms(geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.n_electrons = sum(atomic_num) - self.charge
        if self.n_electrons < 0:
            raise ValueError("Molecule has fewer than 0 electrons!!!")
        self.check_multiplicity(multiplicity)
        return 

    def get_energy(self,geom,mulitplicity,state):
        raise NotImplementedError()

    def get_gradient(self,geom,multiplicity,state):
        raise NotImplementedError()

    def get_coupling(self,geom,multiplicity,state1,state2):
        raise NotImplementedError()

    def finite_difference(self):
        print("Not yet implemented")
        return 0

    def runall(self,geom):
        self.E=[]
        self.grada = []
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets) 
        if len_singlets is not 0:
            self.run(geom,1)
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets) 
        if len_triplets is not 0:
            self.run(geom,3)
        doublets=self.search_tuple(self.states,2)
        len_doublets=len(doublets) 
        if len_doublets is not 0:
            self.run(geom,2)
        quartets=self.search_tuple(self.states,4)
        len_quartets=len(quartets) 
        if len_quartets is not 0:
            self.run(geom,4)
        pentets=self.search_tuple(self.states,5)
        len_pentets=len(pentets) 
        if len_pentets is not 0:
            self.run(geom,5)
        hextets=self.search_tuple(self.states,6)
        len_hextets=len(hextets) 
        if len_hextets is not 0:
            self.run(geom,6)
        self.hasRanForCurrentCoords=True

    def search_tuple(self,tups, elem):
        return filter(lambda tup: elem==tup[0], tups)

