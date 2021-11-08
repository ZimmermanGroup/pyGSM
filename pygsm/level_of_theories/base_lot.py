# standard library imports
from collections import namedtuple
import os

# third party
import numpy as np

# local application imports
from utilities import manage_xyz, options, elements, nifty, units
try:
    from .file_options import File_Options
except:
    from file_options import File_Options

ELEMENT_TABLE = elements.ElementData()

# TODO take out all job-specific data -- encourage external files since those are most customizable
# TODO fix tuple searches
# TODO Make energies,grada dictionaries


def copy_file(path1, path2):
    cmd = 'cp -r ' + path1 + ' ' + path2
    print(" copying scr files\n {}".format(cmd))
    os.system(cmd)
    os.system('wait')


class LoTError(Exception):
    pass


class Lot(object):
    """ Lot object for level of theory calculators """

    @staticmethod
    def default_options():
        """ Lot default options. """

        if hasattr(Lot, '_default_options'):
            return Lot._default_options.copy()
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
            value=[(1, 0)],
            required=False,
            doc='list of states 0-indexed')

        opt.add_option(
            key='gradient_states',
            value=None,
            required=False,
            doc='list of states to calculate gradients for, will assume same as states if not given'
        )

        opt.add_option(
            key='coupling_states',
            value=None,
            required=False,
            doc='states to calculate derivative coupling. Currently only one coupling can be calculated per level of theory object.'
        )

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
            allowed_types=[int],
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
            allowed_types=[int],
            doc='node id used for storing orbs,etc'
        )

        opt.add_option(
            key="ID",
            required=False,
            value=0,
            allowed_types=[int],
            doc=' id used for storing orbs,etc for string'
        )

        opt.add_option(
            key="calc_grad",
            required=False,
            value=True,
            allowed_types=[bool],
            doc=' calculate gradient or not'
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

        opt.add_option(
            key='file_options',
            value=None,
            allowed_types=[File_Options],
            doc='A specialized dictionary containing lot specific options from file\
                        including checks on dependencies and clashes. Not all packages\
                        require'
        )

        opt.add_option(
            key='xTB_Hamiltonian',
            value=None,
            required=False,
            allowed_types=[str],
            doc='xTB hamiltonian'
        )

        opt.add_option(
            key='xTB_accuracy',
            value=None,
            required=False,
            allowed_types=[float],
            doc='xTB accuracy'
        )

        opt.add_option(
            key='xTB_electronic_temperature',
            value=None,
            required=False,
            allowed_types=[float],
            doc='xTB electronic_temperature'
        )

        opt.add_option(
            key='solvent',
            value=None,
            required=False,
            allowed_types=[str],
            doc='xTB solvent'
        )

        Lot._default_options = opt
        return Lot._default_options.copy()

    def __init__(self,
                 options,
                 ):
        """ Constructor """
        self.options = options
        # properties
        self.Energy = namedtuple('Energy', 'value unit')
        self.Gradient = namedtuple('Gradient', 'value unit')
        self.Coupling = namedtuple('Coupling', 'value unit')
        self._Energies = {}
        self._Gradients = {}
        self._Couplings = {}

        # count number of states
        singlets = self.search_tuple(self.states, 1)
        doublets = self.search_tuple(self.states, 2)
        triplets = self.search_tuple(self.states, 3)
        quartets = self.search_tuple(self.states, 4)
        quintets = self.search_tuple(self.states, 5)

        # TODO do this for all states, since it catches if states are put in lazy e.g [(1,1)]
        if singlets:
            len_singlets = max(singlets, key=lambda x: x[1])[1]+1
        else:
            len_singlets = 0
        len_doublets = len(doublets)
        len_triplets = len(triplets)
        len_quartets = len(quartets)
        len_quintets = len(quintets)

        # DO this before fixing states if put in lazy
        if self.options['gradient_states'] is None and self.calc_grad:
            print(" Assuming gradient states are ", self.states)
            self.options['gradient_states'] = self.options['states']

        if len(self.states) < len_singlets+len_doublets+len_triplets+len_quartets+len_quintets:
            print('fixing states to be proper length')
            tmp = []
            # TODO put in rest of fixed states
            for i in range(len_singlets):
                tmp.append((1, i))
            for i in range(len_triplets):
                tmp.append((3, i))
            self.states = tmp
            print(' New states ', self.states)

        self.geom = self.options['geom']
        if self.geom is not None:
            print(" initializing LOT from geom")
        elif self.options['fnm'] is not None:
            print(" initializing LOT from file")
            if not os.path.exists(self.options['fnm']):
                # logger.error('Tried to create LOT object from a file that does not exist: %s\n' % self.options['fnm'])
                raise IOError
            self.geom = manage_xyz.read_xyz(self.options['fnm'], scale=1.)
        else:
            raise RuntimeError("Need to initialize LOT object")

        # Cache some useful atributes - other useful attributes are properties
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.atoms = manage_xyz.get_atoms(self.geom)
        self.ID = self.options['ID']
        self.nproc = self.options['nproc']
        self.charge = self.options['charge']
        self.node_id = self.options['node_id']
        self.lot_inp_file = self.options['lot_inp_file']
        self.xTB_Hamiltonian = self.options['xTB_Hamiltonian']
        self.xTB_accuracy = self.options['xTB_accuracy']
        self.xTB_electronic_temperature = self.options['xTB_electronic_temperature']
        self.solvent = self.options['solvent']

        # Bools for running
        self.hasRanForCurrentCoords = False
        self.has_nelectrons = False

        # Read file options if they exist and not already set
        if self.file_options is None:
            self.file_options = File_Options(self.lot_inp_file)

        # package  specific implementation
        # TODO MOVE to specific package !!!
        # tc cloud
        self.options['job_data']['orbfile'] = self.options['job_data'].get('orbfile', '')
        # pytc? TODO
        self.options['job_data']['lot'] = self.options['job_data'].get('lot', None)

        print(" making folder scratch/{:03}/{}".format(self.ID, self.node_id))
        os.system('mkdir -p scratch/{:03}/{}'.format(self.ID, self.node_id))

    @classmethod
    def from_options(cls, **kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    @property
    def Energies(self):
        '''
        A list of tuples with multiplicity, state and energy
        '''
        # assert type(self._E) is dict,"E must be dictionary"
        return self._Energies

    @Energies.setter
    def Energies(self, value):
        # assert type(value) is dict,"E must be dictionary"
        self._Energies = value

    @property
    def Gradients(self):
        '''
        A list of tuples with multiplicity, state and energy 
        '''

        # assert type(self._) is dict,"grada must be dictionary"
        return self._Gradients

    @Gradients.setter
    def Gradients(self, value):
        assert type(value) is dict, "grada must be dictionary"
        self._Gradients = value

    @property
    def Couplings(self):
        '''
        '''
        return self._Couplings

    @Couplings.setter
    def Couplings(self, value):
        self._Couplings = value

    @property
    def file_options(self):
        return self.options['file_options']

    @file_options.setter
    def file_options(self, value):
        assert type(value) == File_Options, "incorrect type for file options"
        self.options['file_options'] = value

    @property
    def do_coupling(self):
        return self.options['do_coupling']

    @do_coupling.setter
    def do_coupling(self, value):
        assert type(value) == bool, "incorrect type for do_coupling"
        self.options['do_coupling'] = value

    @property
    def coupling_states(self):
        return self.options['coupling_states']

    @coupling_states.setter
    def coupling_states(self, value):
        assert type(value) == tuple, "incorrect type for coupling,currently only support a tuple"
        self.options['coupling_states'] = value

    @property
    def gradient_states(self):
        return self.options['gradient_states']

    @gradient_states.setter
    def gradient_states(self, value):
        assert type(value) == list, "incorrect type for gradient"
        self.options['gradient_states'] = value

    @property
    def states(self):
        return self.options['states']

    @states.setter
    def states(self, value):
        assert type(value) == list, "incorrect type for gradient"
        self.options['states'] = value

    @property
    def calc_grad(self):
        return self.options['calc_grad']

    @calc_grad.setter
    def calc_grad(self, value):
        assert type(value) == bool, "incorrect type for calc_grad"
        self.options['calc_grad'] = value

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        return cls(lot.options.copy().set_values(options))

    def check_multiplicity(self, multiplicity):
        if multiplicity > self.n_electrons + 1:
            raise ValueError("Spin multiplicity too high.")
            print(self.n_electrons)
            print(multiplicity)
            raise ValueError("Inconsistent charge/multiplicity.")

    def get_nelec(self, geom, multiplicity):
        atoms = manage_xyz.get_atoms(geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.n_electrons = sum(atomic_num) - self.charge
        if self.n_electrons < 0:
            raise ValueError("Molecule has fewer than 0 electrons!!!")
        self.check_multiplicity(multiplicity)
        return

    def get_energy(self, coords, multiplicity, state, runtype=None):
        if self.hasRanForCurrentCoords is False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom, runtype)

        Energy = self.Energies[(multiplicity, state)]
        if Energy.unit == "Hartree":
            return Energy.value*units.KCAL_MOL_PER_AU
        elif Energy.unit == 'kcal/mol':
            return Energy.value
        elif Energy.unit is None:
            return Energy.value

    def get_gradient(self, coords, multiplicity, state, frozen_atoms=None):
        if self.hasRanForCurrentCoords is False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom)
        Gradient = self.Gradients[(multiplicity, state)]
        if Gradient.value is not None:
            if frozen_atoms is not None:
                for a in frozen_atoms:
                    Gradient.value[a, :] = 0.
            if Gradient.unit == "Hartree/Bohr":
                return Gradient.value * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            elif Gradient.unit == "kcal/mol/Angstrom":
                return Gradient.value * units.KCAL_MOL_TO_AU  # kcalmol/A*Ha/kcalmol=Ha/ang
            else:
                raise NotImplementedError
        else:
            return None

    def get_coupling(self, coords, multiplicity, state1, state2, frozen_atoms=None):
        if self.hasRanForCurrentCoords is False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom, self.currentCoords)
            self.runall(geom)
        Coupling = self.Couplings[(state1, state2)]

        if Coupling.value is not None:
            if frozen_atoms is not None:
                for a in [3*i for i in frozen_atoms]:
                    Coupling.value[a:a+3, 0] = 0.
            if Coupling.unit == "Hartree/Bohr":
                return Coupling.value * units.ANGSTROM_TO_AU  # Ha/bohr*bohr/ang=Ha/ang
            else:
                raise NotImplementedError
        else:
            return None
        # return np.reshape(self.coup,(3*len(self.geom),1))*units.ANGSTROM_TO_AU

    def write_E_to_file(self):
        with open('scratch/{:03}/E_{}.txt'.format(self.ID, self.node_id), 'w') as f:
            for key, Energy in self.Energies.items():
                f.write('{} {} {:9.7f} Hartree\n'.format(key[0], key[1], Energy.value))

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        raise NotImplementedError

    def runall(self, geom, runtype=None):
        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}
        for state in self.states:
            mult, ad_idx = state
            if state in self.gradient_states or runtype == "gradient":
                self.run(geom, mult, ad_idx)
            elif state in self.coupling_states:
                self.run(geom, mult, ad_idx, 'coupling')
            else:
                self.run(geom, mult, ad_idx, 'energy')

    #    self.E=[]
    #    self.grada = []
    #    singlets=self.search_tuple(self.states,1)
    #    len_singlets=len(singlets)
    #    if len_singlets is not 0:
    #        self.run(geom,1)
    #    triplets=self.search_tuple(self.states,3)
    #    len_triplets=len(triplets)
    #    if len_triplets is not 0:
    #        self.run(geom,3)
    #    doublets=self.search_tuple(self.states,2)
    #    len_doublets=len(doublets)
    #    if len_doublets is not 0:
    #        self.run(geom,2)
    #    quartets=self.search_tuple(self.states,4)
    #    len_quartets=len(quartets)
    #    if len_quartets is not 0:
    #        self.run(geom,4)
    #    pentets=self.search_tuple(self.states,5)
    #    len_pentets=len(pentets)
    #    if len_pentets is not 0:
    #        self.run(geom,5)
    #    hextets=self.search_tuple(self.states,6)
    #    len_hextets=len(hextets)
    #    if len_hextets is not 0:
    #        self.run(geom,6)
    #    septets=self.search_tuple(self.states,7)
    #    len_septets=len(septets)
    #    if len_septets is not 0:
    #        self.run(geom,7)
    #    self.hasRanForCurrentCoords=True

    def search_PES_tuple(self, tups, multiplicity, state):
        '''returns tuple in list of tuples that matches multiplicity and state'''
        return [tup for tup in tups if multiplicity == tup[0] and state == tup[1]]

    def search_tuple(self, tups, multiplicity):
        return [tup for tup in tups if multiplicity == tup[0]]
