"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""
import copy
import logging
import numpy as np
#import fileio
import elements
import units as un
from collections import Counter
import openbabel as ob
import pybel as pb
import os
import manage_xyz
import options
from pes import PES
from avg_pes import Avg_PES
from penalty_pes import Penalty_PES
from dlc_new import DelocalizedInternalCoordinates
from cartesian import CartesianCoordinates

logger = logging.getLogger(__name__)
ELEMENT_TABLE = elements.ElementData()


# Utils to get things started
def getAllCoords(mol):
    natoms = mol.OBMol.NumAtoms()
    tmpcoords = np.zeros((natoms,3))
    for i in range(natoms):
        a= mol.OBMol.GetAtom(i+1)
        tmpcoords[i,:] = [a.GetX(),a.GetY(),a.GetZ()]
    return tmpcoords

def getAtomicNum(mol,i):
    a = mol.OBMol.GetAtom(i)
    return a.GetAtomicNum()

def getAtomicSymbols(mol):
    natoms = mol.OBMol.NumAtoms()
    atomic_nums = [ getAtomicNum(mol,i+1) for i in range(natoms) ]
    atomic_symbols = [ ELEMENT_TABLE.from_atomic_number(i).symbol for i in atomic_nums ] 
    return atomic_symbols

def make_mol_from_coords(coords,atomic_symbols):
    mol = ob.OBMol()
    for s,xyz in zip(atomic_symbols,coords):
        i = mol.NewAtom()
        a = ELEMENT_TABLE.from_symbol(s).atomic_num
        i.SetAtomicNum(a)
        i.SetVector(xyz[0],xyz[1],xyz[2])
    return pb.Molecule(mol)


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
                key='comment',
                required=False,
                value='',
                doc='A string that is saved on the molecule, used for descriptive purposes'
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

    @classmethod
    def copy_from_options(cls,MoleculeA,xyz=None,new_node_id=1):
        """Create a copy of MoleculeA"""
        lot = MoleculeA.PES.lot.copy(new_node_id)
        PES = type(MoleculeA.PES).create_pes_from(MoleculeA.PES,lot)

        if xyz is not None:
            new_geom = manage_xyz.np_to_xyz(MoleculeA.geometry,xyz)
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy().set_values({"xyz":xyz}))
        else:
            new_geom = MoleculeA.geometry
            coord_obj = type(MoleculeA.coord_obj)(MoleculeA.coord_obj.options.copy())

        new_mol = cls(MoleculeA.Data.copy().set_values({"PES":PES,'coord_obj':coord_obj,'geom':new_geom}))

        return new_mol

    def __init__(self,
            options,
            **kwargs
            ):

        self.Data=options

        # => Read in the coordinates <= #
        # important first try to read in geom
        if self.Data['geom'] is not None:
            print " getting cartesian coordinates from geom"
            atoms=manage_xyz.get_atoms(self.Data['geom'])
            xyz=manage_xyz.xyz_to_np(self.Data['geom'])
            mol = make_mol_from_coords(xyz,atoms)
        elif self.Data['fnm'] is not None:
            print " reading cartesian coordinates from file"
            if self.Data['ftype'] is None:
                self.Data['ftype'] = os.path.splitext(self.Data['fnm'])[1][1:]
            if not os.path.exists(self.Data['fnm']):
                logger.error('Tried to create Molecule object from a file that does not exist: %s\n' % self.Data['fnm'])
                raise IOError
            mol=pb.readfile(self.Data['ftype'],self.Data['fnm']).next()
            xyz = getAllCoords(mol)
            atoms =  getAtomicSymbols(mol)
        else:
            raise RuntimeError

        resid=[]
        for a in ob.OBMolAtomIter(mol.OBMol):
            res = a.GetResidue()
            resid.append(res.GetName())
        self.resid = resid

        # Perform all the sanity checks and cache some useful attributes
        self.PES = self.Data['PES']
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

        if not isinstance(self.Data['comment'], basestring):
            raise TypeError("comment for a Molecule must be a string")

        # Create the coordinate system
        if self.Data['coord_obj'] is not None:
            print " getting coord_object from options"
            self.coord_obj = self.Data['coord_obj']
        elif self.Data['coordinate_type'] == "Cartesian":
            self.coord_obj = CartesianCoordinates.from_options()
        elif self.Data['coordinate_type'] == "DLC":
            print " building coordinate object"
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,connect=True)
        elif self.Data['coordinate_type'] == "HDLC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,addcart=True) 
        elif self.Data['coordinate_type'] == "TRIC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,addtr=True) 
        self.Data['coord_obj']=self.coord_obj

        logger.info("Molecule %s constructed.", repr(self))
        logger.debug("Molecule %s constructed.", repr(self))

        #TODO
        self.gradrms = 0.
        self.TSnode=False

        self.newHess = 0
        if self.Data['Hessian'] is None:
            self.form_Hessian()

        if self.Data['Primitive_Hessian'] is None and type(self.coord_obj) is not CartesianCoordinates:
            self.form_Primitive_Hessian()


    def __add__(self,other):
        """ add method for molecule objects. Concatenates"""
        raise NotImplementedError

    def add_internal(self,dof):
        self.coord_obj.Prims.add(dof)
        M.coord_obj.build_dlc(self.xyz)

        #might need to reset Hessian if not None
        if self.Data['Hessian'] is None:
            self.Data['Hessian'] = self.coord_obj.guess_hessian(self.xyz)
        if self.Data['Primitive_Hessian'] is None and type(self.coord_obj) is not CartesianCoordinates:
            self.Data['Primitive_Hessian'] = self.coord_obj.Prims.guess_hessian(self.xyz)

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
        return np.array([un.AMU_TO_AU * ele.mass_amu for ele in self.atoms])

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

    @property
    def center_of_mass(self):
        M = self.total_mass_au
        return np.sum([self.xyz[i,:]*self.atomic_mass[i]/M for i in range(self.natoms)],axis=0)

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
        #return self.PES.get_energy(self.xyz)
        return 0.

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
        dgradx = self.PES.get_coupling(self.xyz) 
        return self.coord_obj.calcGrad(self.xyz,dgradx)
    
    @property
    def difference_energy(self):
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
        print " making primitive Hessian"
        self.Data['Primitive_Hessian'] = self.coord_obj.Prims.guess_hessian(self.xyz)
        self.newHess = 5
    
    def update_Primitive_Hessian(self,change=None):
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
        self.Hessian = np.linalg.multi_dot([self.coord_basis.T,self.Primitive_Hessian,self.coord_basis])
        return self.Hessian

    @property
    def xyz(self):
        return self.Data['xyz']

    @xyz.setter
    def xyz(self,newxyz=None, dq=None):
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
        self.coord_obj.build_dlc(self.xyz,constraints)
        return self.coord_basis

    @property
    def coordinates(self):
        return np.reshape(self.coord_obj.calculate(self.xyz),(-1,1))

    @property
    def num_coordinates(self):
        return len(self.coordinates)

if __name__ =='__main__':
    from molpro import Molpro

    #m = Molecule('s1minima_with_h2o.pdb',fragments=True)
    
    nocc=11
    nactive=2
    filepath='examples/tests/fluoroethene.xyz'
    geom=manage_xyz.read_xyz(filepath,scale=1.)

    lot=Molpro.from_options(states=[(1,0),(1,1)],charge=0,nocc=nocc,nactive=nactive,basis='6-31G',do_coupling=True,nproc=4,fnm=filepath)
    pes1 = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot,ad_idx=1,multiplicity=1)
    pes = Avg_PES(pes1,pes2,lot=lot)

    M = Molecule.from_options(fnm=filepath,PES=pes,coordinate_type="DLC")
    print "done constructing"
    #print M
    #print M.primitive_internal_coordinates
    #coords = M.coordinates
    #print coords

    #print M.Primitive_Hessian
    #print M.Hessian
    #print M.coord_basis
    #print M.update_coordinate_basis()

    #print M.update_Primitive_Hessian(change=np.ones((M.Primitive_Hessian.shape),dtype=float))
    Hess =M.Hessian
    print Hess
    print M.update_Hessian(np.eye(Hess.shape[0]))

    #dq = np.zeros_like(M.coordinates)
    #dq[0] = 0.5
    #print M.update_xyz(dq)

    #print M.primitive_internal_values

    #hess= M.form_Hessian_in_basis()
    #print hess
    #print M.coordinate_basis


    exit()

    #print m.atomic_num
    #print m.center_of_mass
    #print m.radius_of_gyration
    M.build_bonds()
    print(M.bonds)
    M.build_topology()
    print M.geometry
    M.Data['Hessian'] = 10
    print M.Data['Hessian']

    M2 = M.copy(node_id=2)
    print "done copying"
    print M2.geometry
    print M2.Data['Hessian']
    #print M.energy
    #print M.gradient

