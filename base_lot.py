import options
import manage_xyz
import numpy as np
from units import *
import elements 
import os
ELEMENT_TABLE = elements.ElementData()

#TODO take out all job-specific data -- encourage external files since those are most customizable
class Lot(object):
    """ Lot object for level of theory calculators """

    @staticmethod
    def default_options():
        """ Lot default options. """

        if hasattr(Lot, '_default_options'): return Lot._default_options.copy()
        opt = options.Options() 

        opt.add_option(
                key='fnm',
                value=None,
                required=False,
                allowed_types=[str],
                doc='File name to create the LOT object from. Only used if geom is none.'
                )

        opt.add_option(
                key='geom',
                value=None,
                required=False,
                doc='geometry object required to get the atom names and initial coords'
                )

        opt.add_option(
            key='states',
            value=[(1,0)],
            required=False,
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

        opt.add_option(
                key="lot_inp_file",
                required=False,
                value=None,
                doc='file name storing LOT input section. Used for custom basis sets,\
                     custom convergence criteria, etc. Will override nproc, basis and\
                     functional. Do not specify charge or spin in this file. Charge \
                     and spin should be specified in charge and states options.\
                     for QChem, include $molecule line. For ORCA, do not include *xyz\
                     line.'
                     )

        opt.add_option(
                key='job_data',
                value={},
                allowed_types=[dict],
                doc='extra key-word arguments to define level of theory object. e.g.\
                     TeraChem Cloud requires a TeraChem client and options dictionary.'
                )

        Lot._default_options = opt
        return Lot._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """
        self.options = options

        self.geom=self.options['geom']
        if self.geom is not None:
            print(" initializing LOT from geom")
        elif self.options['fnm'] is not None:
                print(" initializing LOT from file")
                if not os.path.exists(self.options['fnm']):
                    logger.error('Tried to create LOT object from a file that does not exist: %s\n' % self.options['fnm'])
                    raise IOError
                self.geom = manage_xyz.read_xyz(self.options['fnm'],scale=1.)
                self.atoms = manage_xyz.get_atoms(self.geom)
        else:
            raise RuntimeError("Need to initialize LOT object")

        # Cache some useful atributes
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.states =self.options['states']

        #TODO remove some of these options 
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
        self.lot_inp_file = self.options['lot_inp_file']

        #package  specific implementation
        self.options['job_data']['tcc_options'] = self.options['job_data'].get('tcc_options',{})
        self.options['job_data']['TC'] = self.options['job_data'].get('TC',None)
        self.options['job_data']['orbfile'] = self.options['job_data'].get('orbfile','')
        self.options['job_data']['psiw'] = self.options['job_data'].get('psiw',None)
        self.options['job_data']['simulation'] = self.options['job_data'].get('simulation',None)

    @classmethod
    def from_options(cls,**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    @classmethod
    def copy(cls,lot,**kwargs):
        return cls(lot.options.copy().set_values(kwargs))

    def check_multiplicity(self,multiplicity):
        if multiplicity > self.n_electrons + 1:
            raise ValueError("Spin multiplicity too high.")
        if (self.n_electrons + multiplicity + 1) % 2:
            print(self.n_electrons)
            print(multiplicity)
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
        septets=self.search_tuple(self.states,7)
        len_septets=len(septets) 
        if len_septets is not 0:
            self.run(geom,7)
        self.hasRanForCurrentCoords=True

    def search_PES_tuple(self,tups, multiplicity,state):
        '''returns tuple in list of tuples that matches multiplicity and state'''
        return [tup for tup in tups if multiplicity==tup[0] and state==tup[1]]

    def search_tuple(self,tups,multiplicity):
        return [tup for tup in tups if multiplicity==tup[0]]

