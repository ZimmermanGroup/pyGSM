"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""

# standard library imports
import sys
import os
from os import path
from time import time

# third party
import logging
import numpy as np
from collections import Counter
#import openbabel as ob
#import pybel as pb

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from potential_energy_surfaces import PES
from potential_energy_surfaces import Avg_PES
from potential_energy_surfaces import Penalty_PES
from coordinate_systems import DelocalizedInternalCoordinates
from coordinate_systems import CartesianCoordinates

logger = logging.getLogger(__name__)
ELEMENT_TABLE = elements.ElementData()

# TOC:
# constructors
# methods
# properties

"""Specify a molecule by its atom composition, coordinates, charge
and spin multiplicities.
"""
class Molecule(object):
    
    @staticmethod
    def default_options():
        if hasattr(Molecule, '_default_options'): return Molecule._default_options.copy()
        opt = options.Options() 

        opt.add_option(
                key='fnm',
                value=None,
                required=False,
                allowed_types=[str],
                doc='File name to create the Molecule object from. Only used if geom is none.'
                    )
        opt.add_option(
                key='ftype',
                value=None,
                required=False,
                allowed_types=[str],
                doc='filetype (optional) will attempt to read filetype if not given'
                )

        opt.add_option(
                key='coordinate_type',
                required=False,
                value='Cartesian',
                allowed_values=['Cartesian','DLC','HDLC','TRIC'],
                doc='The type of coordinate system to build'
                )

        opt.add_option(
                key='coord_obj',
                required=False,
                value=None,
                allowed_types=[DelocalizedInternalCoordinates,CartesianCoordinates],
                doc='A coordinate object.'
                )


        opt.add_option(
                key='geom',
                required=False,
                allowed_types=[list],
                doc='geometry including atomic symbols'
                )

        opt.add_option(
                key='xyz',
                required=False,
                allowed_types=[np.ndarray],
                doc='The Cartesian coordinates in Angstrom'
                )

        opt.add_option(
                key='PES',
                required=True,
                allowed_types=[PES,Avg_PES,Penalty_PES],
                doc='potential energy surface object to evaulate energies, gradients, etc. Pes is defined by charge, state, multiplicity,etc. '
                       
                )

        opt.add_option(
                key='Primitive_Hessian',
                value=None,
                required=False,
                doc='Primitive hessian save file for doing optimization.'
                )

        opt.add_option(
                key='Hessian',
                value=None,
                required=False,
                doc='Hessian save file in the basis of coordinate_type.'
                )

        opt.add_option(
                key='Form_Hessian',
                value=True,
                doc='Form the Hessian in the current basis -- takes time for large molecules.'
                )

        opt.add_option(
                key="top_settings",
                value={},
                doc='some extra kwargs for forming coordinate object.'
                )

        opt.add_option(
                key='comment',
                required=False,
                value='',
                doc='A string that is saved on the molecule, used for descriptive purposes'
                )

        opt.add_option(
                key='node_id',
                required=False,
                value=0,
                doc='used to specify level of theory node identification',
                )


        Molecule._default_options = opt
        return Molecule._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Molecule(Molecule.default_options().set_values(kwargs))

    def __repr__(self):
        lines = [" molecule object"]
        lines.append(self.Data.__str__())
        return'\n'.join(lines)

    @staticmethod
    def copy_from_options(MoleculeA,xyz=None,fnm=None,new_node_id=1):
        """Create a copy of MoleculeA"""
        #lot = MoleculeA.PES.lot.copy(MoleculeA.PES.lot,node_id=new_node_id)
        #PES = MoleculeA.PES.create_pes_from(PES=MoleculeA.PES,options={'node_id': new_node_id})

        if xyz is not None:
            new_geom = manage_xyz.np_to_xyz(MoleculeA.geometry,xyz)
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy().set_values({"xyz":xyz}))
        elif fnm is not None:
            new_geom = manage_xyz.read_xyz(fnm)
            xyz = manage_xyz.xyz_to_np(new_geom)
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy().set_values({"xyz":xyz}))
        else:
            new_geom = MoleculeA.geometry
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy())

        return Molecule(MoleculeA.Data.copy().set_values({
            'coord_obj':coord_obj,
            'geom':new_geom,
            'node_id':new_node_id,
            }))


    def __init__(self,
            options,
            **kwargs
            ):

        self.Data=options

        # => Read in the coordinates <= #
        # important first try to read in geom

        t0 = time()
        if self.Data['geom'] is not None:
            print(" getting cartesian coordinates from geom")
            atoms=manage_xyz.get_atoms(self.Data['geom'])
            xyz=manage_xyz.xyz_to_np(self.Data['geom'])
            #mol = nifty.make_mol_from_coords(xyz,atoms)
        elif self.Data['fnm'] is not None:
            print(" reading cartesian coordinates from file")
            if self.Data['ftype'] is None:
                self.Data['ftype'] = os.path.splitext(self.Data['fnm'])[1][1:]
            if not os.path.exists(self.Data['fnm']):
                logger.error('Tried to create Molecule object from a file that does not exist: %s\n' % self.Data['fnm'])
                raise IOError
            #mol=next(pb.readfile(self.Data['ftype'],self.Data['fnm']))
            #xyz = nifty.getAllCoords(mol)
            #atoms =  nifty.getAtomicSymbols(mol)
            xyz = manage_xyz.read_xyz(self.Data['fnm'])
            atoms = manage_xyz.get_atoms(xyz)

        else:
            raise RuntimeError

        t1 = time()
        print(" Time to get coords= %.3f" % (t1 - t0))
        #resid=[]
        #for a in ob.OBMolAtomIter(mol.OBMol):
        #    res = a.GetResidue()
        #    resid.append(res.GetName())
        #self.resid = resid

        # Perform all the sanity checks and cache some useful attributes

        #TODO make PES property
        self.PES = type(self.Data['PES']).create_pes_from(self.Data['PES'],{'node_id':self.Data['node_id']})
        if not hasattr(atoms, "__getitem__"):
            raise TypeError("atoms must be a sequence of atomic symbols")

        for a in atoms:
            if not isinstance(a, str):
                raise TypeError("atom symbols must be strings")

        if type(xyz) is not np.ndarray:
            raise TypeError("xyz must be a numpy ndarray")
        if xyz.shape != (len(atoms), 3):
            raise ValueError("xyz must have shape natoms x 3")
        self.Data['xyz'] = xyz.copy()

        # create a dictionary from atoms
        # atoms contain info you need to know about the atoms
        self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]

        tx = time()
        print(" Time to create PES,elements %.3f" % (tx-t1))
        t1=tx

        if not isinstance(self.Data['comment'], str):
            raise TypeError("comment for a Molecule must be a string")

        # Create the coordinate system
        if self.Data['coord_obj'] is not None:
            print(" getting coord_object from options")
            self.coord_obj = self.Data['coord_obj']
        elif self.Data['coordinate_type'] == "Cartesian":
            self.coord_obj = CartesianCoordinates.from_options(xyz=self.xyz,atoms=self.atoms)
        elif self.Data['coordinate_type'] == "DLC":
            print(" building coordinate object")
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,connect=True,extra_kwargs =self.Data['top_settings'])
        elif self.Data['coordinate_type'] == "HDLC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,addcart=True,extra_kwargs =self.Data['top_settings']) 
        elif self.Data['coordinate_type'] == "TRIC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,addtr=True,extra_kwargs =self.Data['top_settings']) 
        self.Data['coord_obj']=self.coord_obj

        t2 = time() 
        print(" Time  to build coordinate system= %.3f" % (t2-t1))

        #TODO
        self.gradrms = 100.
        self.TSnode=False
        self.bdist =0.

        #TODO BUGGGGGGG make property so it copies over properly...
        self.newHess = 10
        ###
        if self.Data['Primitive_Hessian'] is None and type(self.coord_obj) is not CartesianCoordinates:
            self.form_Primitive_Hessian()
        t3 = time()
        print(" Time to build Prim Hessian %.3f" % (t3-t2))

        if self.Data['Hessian'] is None and self.Data['Form_Hessian']:
            if self.Data['Primitive_Hessian'] is not None:
                print(" forming Hessian in basis")
                self.form_Hessian_in_basis()
            else:
                self.form_Hessian()

        #logger.info("Molecule %s constructed.", repr(self))
        #logger.debug("Molecule %s constructed.", repr(self))
        print("molecule constructed")

    def __add__(self,other):
        """ add method for molecule objects. Concatenates"""
        raise NotImplementedError

    def reorder(self, new_order):
        """Reorder atoms in the molecule"""
        #TODO doesn't work probably CRA 3/2019
        for field in ["atoms", "xyz"]:
            self.__dict__[field] = self.__dict__[field][list(new_order)]
        self.atoms = [self.atoms[i] for i in new_order]

    def reorder_according_to(self,other):
        """

        Reorder atoms according to some other Molecule object.  This
        happens when we run a program like pdb2gmx or pdbxyz and it
        scrambles our atom ordering, forcing us to reorder the atoms
        for all frames in the current Molecule object.

        Directions:
        (1) Load up the scrambled file as a new Molecule object.
        (2) Call this function: Original_Molecule.reorder_according_to(scrambled)
        (3) Save Original_Molecule to a new file name.

        """
        raise NotImplementedError

    def center(self, center_mass = False):
        """ Move geometric center to the origin. """
        if center_mass:
            com = self.center_of_mass
            self.xyz -= com
        else:
            self.xyz -= self.xyz.mean(0)

    @property
    def atomic_mass(self):
        return np.array([units.AMU_TO_AU * ele.mass_amu for ele in self.atoms])

    @property
    def mass_amu(self):
        return np.array([ele.mass_amu for ele in self.atoms])

    @property
    def atomic_num(self):
        return [ele.atomic_num for ele in self.atoms]

    @property
    def total_mass_au(self):
        """Returns the total mass of the molecule"""
        return np.sum(self.atomic_mass)

    @property
    def total_mass_amu(self):
        """Returns the total mass of the molecule"""
        return np.sum(self.mass_amu)

    @property
    def natoms(self):
        """The number of atoms in the molecule"""
        return len(self.atoms)

    def atom_data(self):
        uniques = list(set(M.atoms))
        for a in uniques:
            nifty.printcool_dictionary(a._asdict())

    @property
    def center_of_mass(self):
        M = self.total_mass_au
        return np.sum([self.xyz[i,:]*self.atomic_mass[i]/M for i in range(self.natoms)],axis=0)

    @property
    def mass_weighted_cartesians(self):
        return np.asarray( [self.xyz[i,:]*self.atomic_mass[i]/M for i in range(self.natoms)] )

    @property
    def radius_of_gyration(self):
        com = self.center_of_mass
        M = self.total_mass_au
        rgs = []
        xyz1 = self.xyz.copy()
        xyz1 -= com
        return np.sum( self.atomic_mass[i]*np.dot(x,x) for i,x in enumerate(xyz1))/M

    @property
    def geometry(self):
        symbols =[a.symbol for a in self.atoms]
        return manage_xyz.combine_atom_xyz(symbols,self.xyz)

    @property
    def energy(self):
        return self.PES.get_energy(self.xyz)
        #return 0.

    @property
    def gradient(self):
        gradx = self.PES.get_gradient(self.xyz) 
        return self.coord_obj.calcGrad(self.xyz,gradx)  #CartesianCoordinate just returns gradx

    @property
    def derivative_coupling(self):
        dvecx = self.PES.get_coupling(self.xyz) 
        return self.coord_obj.calcGrad(self.xyz,dvecx)

    @property
    def difference_gradient(self):
        dgradx = self.PES.get_dgrad(self.xyz) 
        return self.coord_obj.calcGrad(self.xyz,dgradx)
    
    @property
    def difference_energy(self):
        self.energy
        return self.PES.dE

    @property
    def exactHessian(self):
        raise NotImplementedError

    @property
    def Primitive_Hessian(self):
        return self.Data['Primitive_Hessian']
    
    @Primitive_Hessian.setter
    def Primitive_Hessian(self,value):
        self.Data['Primitive_Hessian'] = value

    def form_Primitive_Hessian(self):
        print(" making primitive Hessian")
        self.Data['Primitive_Hessian'] = self.coord_obj.Prims.guess_hessian(self.xyz)
        self.newHess = 10
    
    def update_Primitive_Hessian(self,change=None):
        print(" updating prim hess")
        if change is not None:
            self.Primitive_Hessian += change
        return  self.Primitive_Hessian

    @property
    def Hessian(self):
        return self.Data['Hessian']

    @Hessian.setter
    def Hessian(self,value):
        self.Data['Hessian'] = value

    def form_Hessian(self):
        self.Data['Hessian'] = self.coord_obj.guess_hessian(self.xyz)
        self.newHess = 5

    def update_Hessian(self,change=None):
        #print " in update Hessian"
        if change is not None:
            self.Hessian += change
        return self.Hessian

    def form_Hessian_in_basis(self):
        #print " forming Hessian in current basis"
        self.Hessian = block_matrix.dot( block_matrix.dot(block_matrix.transpose(self.coord_basis),self.Primitive_Hessian),self.coord_basis)

        #print(" Hessian")
        #print(self.Hessian)
        return self.Hessian

    @property
    def xyz(self):
        return self.Data['xyz']

    @xyz.setter
    def xyz(self,newxyz=None):
        if newxyz is not None:
            self.Data['xyz']=newxyz

    def update_xyz(self,dq=None,verbose=True):
        #print " updating xyz"
        if dq is not None:
            self.xyz = self.coord_obj.newCartesian(self.xyz,dq,verbose)
        return self.xyz

    @property
    def finiteDifferenceHessian(self):
        return self.PES.get_finite_difference_hessian(self.xyz)
   
    @property
    def primitive_internal_coordinates(self):
        return self.coord_obj.Prims.Internals

    @property
    def num_primitives(self):
        return len(self.primitive_internal_coordinates)

    @property
    def num_bonds(self):
        count=0
        for ic in self.coord_obj.Prims.Internals:
            if type(ic) == "Distance":
                count+=1
        return count

    @property
    def primitive_internal_values(self):
        ans =self.coord_obj.Prims.calculate(self.xyz)
        return np.asarray(ans)

    @property
    def coord_basis(self):
        return self.coord_obj.Vecs

    def update_coordinate_basis(self,constraints=None):
        if self.coord_obj.__class__.__name__=='CartesianCoordinates':
            return
        #if constraints is not None:
            #assert constraints.shape[0] == self.coord_basis.shape[0], '{} does not equal {} dimensions'.format(constraints.shape[0],self.coord_basis.shape[0])

        print(" updating coord basis")
        self.coord_obj.clearCache()
        self.coord_obj.build_dlc(self.xyz,constraints)
        return self.coord_basis

    @property
    def constraints(self):
        return self.coord_obj.Vecs.cnorms

    @property
    def coordinates(self):
        return np.reshape(self.coord_obj.calculate(self.xyz),(-1,1))

    @property
    def num_coordinates(self):
        return len(self.coordinates)

    @property
    def frag_atomic_indices(self):
        return self.coord_obj.Prims.frag_atomic_indices

    def get_frag_atomic_index(self,fragid):
        return self.frag_atomic_indices[fragid]


if __name__=='__main__':
    from level_of_theories import Molpro
    filepath='../../data/ethylene.xyz'
    molpro = Molpro.from_options(states=[(1,0)],fnm=filepath,lot_inp_file='../../data/ethylene_molpro.com')

    pes = PES.from_options(lot=molpro,ad_idx=0,multiplicity=1)

    
    reactant = Molecule.from_options(fnm=filepath,PES=pes,coordinate_type="TRIC",Form_Hessian=False)

    print(reactant.coord_basis)
