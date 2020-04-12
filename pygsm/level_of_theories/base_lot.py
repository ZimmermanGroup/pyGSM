# standard library imports
import os

# third party 
import numpy as np

# local application imports
from utilities import manage_xyz,options,elements,nifty
from .file_options import File_Options

ELEMENT_TABLE = elements.ElementData()

#TODO take out all job-specific data -- encourage external files since those are most customizable
#TODO fix tuple searches


class Lot(File_Options):
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
                key='gradient_states',
                value=[],
                required=False,
                doc='list of states to calculate gradients for, will assume same as states if not given'
                )

        opt.add_option(
                key='coupling_states',
                value=[],
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

        Lot._default_options = opt
        return Lot._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """

        self.options = options

        # count number of states
        singlets=self.search_tuple(self.states,1)
        doublets=self.search_tuple(self.states,2)
        triplets=self.search_tuple(self.states,3)
        quartets=self.search_tuple(self.states,4)
        quintets=self.search_tuple(self.states,5)

        #TODO do this for all states, since it catches if states are put in lazy
        if singlets:
            len_singlets= max(singlets,key=lambda x: x[1])[1]+1
        else:
            len_singlets=0
        len_doublets=len(doublets)
        len_triplets=len(triplets)
        len_quartets=len(quartets)
        len_quintets=len(quintets)

        # DO this before fixing states
        # THIS is wrong if you set gradient states to [] to purposefully prevent gradients
        # will none work? use calc_grad?
        if not self.options['gradient_states'] and self.calc_grad:
            print(" Assuming gradient states are ",self.states)
            self.options['gradient_states']=self.options['states']

        if len(self.states)<len_singlets+len_doublets+len_triplets+len_quartets+len_quintets:
            print('fixing states to be proper length')
            tmp = []
            # TODO put in rest of fixed states 
            for i in range(len_singlets):
                tmp.append((1,i))
            for i in range(len_triplets):
                tmp.append((3,i))
            self.states = tmp
            print(' New states ',self.states)

        self.geom=self.options['geom']
        if self.geom is not None:
            print(" initializing LOT from geom")
        elif self.options['fnm'] is not None:
                print(" initializing LOT from file")
                if not os.path.exists(self.options['fnm']):
                    logger.error('Tried to create LOT object from a file that does not exist: %s\n' % self.options['fnm'])
                    raise IOError
                self.geom = manage_xyz.read_xyz(self.options['fnm'],scale=1.)
        else:
            raise RuntimeError("Need to initialize LOT object")

        # Cache some useful atributes
        self.currentCoords = manage_xyz.xyz_to_np(self.geom)
        self.atoms = manage_xyz.get_atoms(self.geom)
        self.ID = self.options['ID']

        #TODO remove some of these options  make others properties
        #self.nocc=self.options['nocc']
        #self.nactive=self.options['nactive']
        self.nproc=self.options['nproc']
        self.charge = self.options['charge']
        self.node_id=self.options['node_id']
        self.hasRanForCurrentCoords =False
        self.has_nelectrons =False
        self.lot_inp_file = self.options['lot_inp_file']

        # Parse input file saving options to self, only do this for classes which need it
        # E.g. Q-Chem does not currently need to save the file arguments because it can simply concatenate 
        # lot_inp_file and geometry
        if self.__class__.__name__ =='pDynamo':
            super(Lot,self).__init__(self.lot_inp_file)


        #package  specific implementation
        # tc cloud
        self.options['job_data']['orbfile'] = self.options['job_data'].get('orbfile','')
        # pytc? TODO
        self.options['job_data']['lot'] = self.options['job_data'].get('lot',None)
        # openmm
        self.options['job_data']['simulation'] = self.options['job_data'].get('simulation',None)
        #pDynamo
        self.options['job_data']['system'] = self.options['job_data'].get('simulation',None)

    @classmethod
    def from_options(cls,**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    @property
    def do_coupling(self):
        return self.options['do_coupling']

    @do_coupling.setter
    def do_coupling(self,value):
        assert type(value)==bool, "incorrect type for do_coupling"
        self.options['do_coupling']=value

    @property
    def coupling_states(self):
        return self.options['coupling_states']

    @coupling_states.setter
    def coupling_states(self,value):
        assert type(value)==list or type(value)==tuple, "incorrect type for coupling"
        self.options['coupling_states']=value

    @property
    def gradient_states(self):
        return self.options['gradient_states']

    @gradient_states.setter
    def gradient_states(self,value):
        assert type(value)==list, "incorrect type for gradient"
        self.options['gradient_states']=value

    @property
    def states(self):
        return self.options['states']

    @states.setter
    def states(self,value):
        assert type(value)==list, "incorrect type for gradient"
        self.options['states']=value

    @property
    def calc_grad(self):
        return self.options['calc_grad']

    @calc_grad.setter
    def calc_grad(self,value):
        assert type(value)==bool, "incorrect type for calc_grad"
        self.options['calc_grad']=value


    @classmethod
    def copy(cls,lot,options,copy_wavefunction=True):
        return cls(lot.options.copy().set_values(options))

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

