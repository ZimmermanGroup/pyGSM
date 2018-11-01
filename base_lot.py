import options
import manage_xyz
import numpy as np
from units import *
import elements 


class Base(object):
    """ Base object for level of theory calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Base, '_default_options'): return Base._default_options.copy()
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
                )


        Base._default_options = opt
        return Base._default_options.copy()

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
        self.hasRanForCurrentCoords =False
        self.got_electrons =False

        print self.states

    def check_multiplicity(self):
        if multiplicity > self.n_electrons + 1:
            raise ValueError("Spin multiplicity too high.")
        if (self.n_electrons + multiplicity + 1) % 2:
            raise ValueError("Inconsistent charge/multiplicity.")
            
    def get_nelec(self,geom):
        atoms = manage_xyz.get_atoms(geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.got_electrons =True
        self.n_electrons = sum(atomic_num) - self.charge
        if self.n_electrons < 0:
            raise ValueError("Molecule has fewer than 0 electrons!!!")
        self.check_multiplicity()

        return 



    def get_energy(self,geom,mulitplicity,charge,state):
        raise NotImplementedError()

    def get_gradient(self,geom,multiplicity,charge,state):
        raise NotImplementedError()

    def finite_difference(self):
        print("Not yet implemented")
        return 0

    def search_tuple(self,tups, elem):
        return filter(lambda tup: elem==tup[0], tups)

