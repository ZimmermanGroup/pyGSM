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
                doc='File name to create the Molecule object from.  If provided,\
                    the file will be parsed and used to fill in the fields such as \
                    elem (elements), xyz (coordinates) and so on.  If ftype is not \
                    provided, will automatically try to determine file type from file \
                    extension.  If not provided, will create an empty object. ',
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
                doc=''
                )

        opt.add_option(
                key='PES',
                required=False,
                allowed_types=[PES,Avg_PES,Penalty_PES],
                doc='potential energy surface object to evaulate energies, gradients, etc.\
                        pes is defined by charge, state, multiplicity,etc. '
                )

        opt.add_option(
                key='Hessian',
                value=None,
                required=False,
                doc='Hessian save file for doing optimization -- no particular format'
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

    def copy(self, xyz=None,new_node_id=1):
        """Create a copy of this molecule"""

        lot = self.PES.lot.copy(new_node_id)
        PES = type(self.PES).create_pes_from(self.PES,lot)

        #CRA 3/2019 I don't know what this is
        #for key, value in self.__dict__.iteritems():
        #    mol.__dict__[key] = copy.copy(value)

        if xyz is not None:
            new_geom = manage_xyz.np_to_xyz(self.geometry,xyz)
            coord_obj = type(self.coord_obj)(self.coord_obj.options.copy().set_values({"xyz":xyz}))
        else:
            new_geom = self.geometry
            coord_obj = type(self.coord_obj)(self.coord_obj.options.copy())

        new_mol = type(self)(self.Data.copy().set_values({"PES":PES,'coord_obj':coord_obj,'geom':new_geom}))

        return new_mol

    def __init__(self,
            options,
            **kwargs
            ):

        self.Data=options

        # => Read in the coordinates <= #
        if self.Data['geom'] is not None:
            print "getting cartesian coordinates from geom"
            atoms=manage_xyz.get_atoms(self.Data['geom'])
            xyz=manage_xyz.xyz_to_np(self.Data['geom'])
            mol = make_mol_from_coords(xyz,atoms)
        elif self.Data['fnm'] is not None:
            print "reading cartesian coordinates from file"
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
        self.xyz = xyz.copy()

        # create a dictionary from atoms
        # atoms contain info you need to know about the atoms
        self.atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]

        if not isinstance(self.Data['comment'], basestring):
            raise TypeError("comment for a Molecule must be a string")

        # Create the coordinate system
        if self.Data['coord_obj'] is not None:
            print "getting coord_object from options"
            self.coord_obj = self.Data['coord_obj']
        elif self.Data['coordinate_type'] == "Cartesian":
            self.coord_obj = Cartesian.from_options()
        elif self.Data['coordinate_type'] == "DLC":
            print "building coordinate object"
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,connect=True)
        elif self.Data['coordinate_type'] == "HDLC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=sel.xyz,atoms=self.atoms) 
        elif self.Data['coordinate_type'] == "TRIC":
            self.coord_obj = DelocalizedInternalCoordinates.from_options(xyz=self.xyz,atoms=self.atoms,addtr=True) 
        self.Data['coord_obj']=self.coord_obj

        logger.info("Molecule %s constructed.", repr(self))
        logger.debug("Molecule %s constructed.", repr(self))


    def __add__(self,other):
        """ add method for molecule objects. Concatenates"""
        raise NotImplementedError

    def reorder(self, new_order):
        """Reorder atoms in the molecule"""
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
        return self.PES.get_energy(self.xyz)

    @property
    def gradient(self):
        gradx = self.PES.get_gradient(self.xyz) 
        return self.coord_obj.calcGrad(self.xyz,gradx)

    @property
    def derivative_coupling(self):
        return self.PES.get_coupling(self.xyz)

    @property
    def difference_gradient(self):
        return self.PES.get_dgrad(self.xyz)

    @property
    def Hessian(self):
        return 0

    @property
    def primitive_internal_coordinates(self):
        return self.coord_obj.Prims.Internals

    @property
    def primitive_internal_coordinate_values(self):
        #ans = [prim.value(self.xyz) for prim in  self.coord_obj.Prims.Internals ]
        ans =self.coord_obj.Prims.calculate(self.xyz)
        return np.asarray(ans)

    def get_coordinate_basis(self,constraints=None):
        #print "old vecs"
        #print self.coord_obj.Vecs
        cVec = self.coord_obj.form_cVecs_from_prim_Vecs(constraints)
        self.coord_obj.build_dlc(self.xyz,cVec)
        #print "new vecs"
        #print self.coord_obj.Vecs
        return self.coord_obj.Vecs
    coordinate_basis=property(get_coordinate_basis)

    @property
    def coordinates(self):
        return self.coord_obj.calculate(self.xyz)

if __name__ =='__main__':
    from molpro import Molpro

    #m = Molecule('s1minima_with_h2o.pdb',fragments=True)
    
    nocc=11
    nactive=2
    filepath='examples/tests/fluoroethene.xyz'
    geom=manage_xyz.read_xyz(filepath,scale=1.)

    lot=Molpro.from_options(states=[(1,0),(1,1)],charge=0,nocc=nocc,nactive=nactive,basis='6-31G',do_coupling=True,nproc=4,geom=geom)
    pes1 = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot,ad_idx=1,multiplicity=1)
    pes = Avg_PES(pes1,pes2,lot=lot)

    M = Molecule.from_options(fnm=filepath,PES=pes,coordinate_type="DLC")
    print M.primitive_internal_coordinates
    print M.primitive_internal_coordinate_values
    coords = M.coordinates
    print coords

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

