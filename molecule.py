"""Class structures of important chemical concepts
This class is the combination of Martinez group and Lee Ping's molecule class.
"""
import copy
import logging
import numpy as np
#import fileio
import elements
import units as un
import itertools
from collections import Counter
import openbabel as ob
import pybel as pb
import os
from pkg_resources import parse_version
import manage_xyz
import options
from pes import PES
from avg_pes import Avg_PES
from penalty_pes import Penalty_PES

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


def AtomContact(xyz, pairs, box=None, displace=False):
    """
    Compute distances between pairs of atoms.

    Parameters
    ----------
    xyz : np.ndarray
        Nx3 array of atom positions
    pairs : list
        List of 2-tuples of atom indices
    box : np.ndarray, optional
        An array of three numbers (xyz box vectors).

    Returns
    -------
    np.ndarray
        A Npairs-length array of minimum image convention distances
    np.ndarray (optional)
        if displace=True, return a Npairsx3 array of displacement vectors
    """
    # Obtain atom selections for atom pairs
    parray = np.array(pairs)
    sel1 = parray[:,0]
    sel2 = parray[:,1]
    xyzpbc = xyz.copy()
    # Minimum image convention: Place all atoms in the box
    # [-xbox/2, +xbox/2); [-ybox/2, +ybox/2); [-zbox/2, +zbox/2)
    if box is not None:
        xbox = box[0]
        ybox = box[1]
        zbox = box[2]
        while any(xyzpbc[:,0] < -0.5*xbox):
            xyzpbc[:,0] += (xyzpbc[:,0] < -0.5*xbox)*xbox
        while any(xyzpbc[:,1] < -0.5*ybox):
            xyzpbc[:,1] += (xyzpbc[:,1] < -0.5*ybox)*ybox
        while any(xyzpbc[:,2] < -0.5*zbox):
            xyzpbc[:,2] += (xyzpbc[:,2] < -0.5*zbox)*zbox
        while any(xyzpbc[:,0] >= 0.5*xbox):
            xyzpbc[:,0] -= (xyzpbc[:,0] >= 0.5*xbox)*xbox
        while any(xyzpbc[:,1] >= 0.5*ybox):
            xyzpbc[:,1] -= (xyzpbc[:,1] >= 0.5*ybox)*ybox
        while any(xyzpbc[:,2] >= 0.5*zbox):
            xyzpbc[:,2] -= (xyzpbc[:,2] >= 0.5*zbox)*zbox
    # Obtain atom selections for the pairs to be computed
    # These are typically longer than N but shorter than N^2.
    xyzsel1 = xyzpbc[sel1]
    xyzsel2 = xyzpbc[sel2]
    # Calculate xyz displacement
    dxyz = xyzsel2-xyzsel1
    # Apply minimum image convention to displacements
    if box is not None:
        dxyz[:,0] += (dxyz[:,0] < -0.5*xbox)*xbox
        dxyz[:,1] += (dxyz[:,1] < -0.5*ybox)*ybox
        dxyz[:,2] += (dxyz[:,2] < -0.5*zbox)*zbox
        dxyz[:,0] -= (dxyz[:,0] >= 0.5*xbox)*xbox
        dxyz[:,1] -= (dxyz[:,1] >= 0.5*ybox)*ybox
        dxyz[:,2] -= (dxyz[:,2] >= 0.5*zbox)*zbox
    dr2 = np.sum(dxyz**2,axis=1)
    dr = np.sqrt(dr2)
    if displace:
        return dr, dxyz
    else:
        return dr

#===========================#
#|   Connectivity graph    |#
#|  Good for doing simple  |#
#|     topology tricks     |#
#===========================#
try:
    import networkx as nx
    class MyG(nx.Graph):
        def __init__(self):
            super(MyG,self).__init__()
            self.Alive = True
        def __eq__(self, other):
            # This defines whether two MyG objects are "equal" to one another.
            if not self.Alive:
                return False
            if not other.Alive:
                return False
            return nx.is_isomorphic(self,other,node_match=nodematch)
        def __hash__(self):
            """ The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash. """
            return 1
        def L(self):
            """ Return a list of the sorted atom numbers in this graph. """
            return sorted(list(self.nodes()))
        def AStr(self):
            """ Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' . """
            return ','.join(['%i' % i for i in self.L()])
        def e(self):
            """ Return an array of the elements.  For instance ['H' 'C' 'C' 'H']. """
            elems = nx.get_node_attributes(self,'e')
            return [elems[i] for i in self.L()]
        def ef(self):
            """ Create an Empirical Formula """
            Formula = list(self.e())
            return ''.join([('%s%i' % (k, Formula.count(k)) if Formula.count(k) > 1 else '%s' % k) for k in sorted(set(Formula))])
        def x(self):
            """ Get a list of the coordinates. """
            coors = nx.get_node_attributes(self,'x')
            return np.array([coors[i] for i in self.L()])
except ImportError:
    logger.warning("NetworkX cannot be imported (topology tools won't work).  Most functionality should still work though.")


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
                key='atoms',
                required=False,
                allowed_types=[np.ndarray],
                doc='list of atomic symbols'
                )

        opt.add_option(
                key='xyz',
                required=False,
                allowed_types=[np.ndarray],
                doc='cartesian coordinates not including the atomic symbols'
                )

        opt.add_option(
                key='geom',
                doc='geometry including atomic symbols')

        opt.add_option(
                key='PES',
                required=True,
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
                key='resid',
                value=None,
                required=False,
                doc='list of residue ids'
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

    def __init__(self,
            options,
            **kwargs
            ):

        self.Data=options

        if self.Data['fnm'] is not None:
            if self.Data['ftype'] is None:
                ## Try to determine from the file name using the extension.
                self.Data['ftype'] = os.path.splitext(self.Data['fnm'])[1][1:]
            if not os.path.exists(self.Data['fnm']):
                logger.error('Tried to create Molecule object from a file that does not exist: %s\n' % self.Data['fnm'])
                raise IOError
            # read the molecule and save info
            mol=pb.readfile(self.Data['ftype'],self.Data['fnm']).next()
            xyz = getAllCoords(mol)
            atoms =  getAtomicSymbols(mol)
        elif self.Data['geom'] is not None:
            atoms=manage_xyz.get_atoms(self.Data['geom'])
            xyz=manage_xyz.xyz_to_np(self.Data['geom'])
            mol = make_mol_from_coords(xyz,atoms)

        resid=[]
        for a in ob.OBMolAtomIter(mol.OBMol):
            res = a.GetResidue()
            resid.append(res.GetName())
        self.Data['resid'] = resid

        # Perform all the sanity checks and cache some useful attributes
        self.PES = self.Data['PES']
        if not hasattr(atoms, "__getitem__"):
            raise TypeError("atoms must be a sequence of atomic symbols")

        for a in atoms:
            if not isinstance(a, str):
                raise TypeError("atom symbols must be strings")
        self.Data['atoms'] = np.array(atoms, dtype=np.dtype('S2'))

        if type(xyz) is not np.ndarray:
            raise TypeError("xyz must be a numpy ndarray")
        if xyz.shape != (self.Data['atoms'].shape[0], 3):
            raise ValueError("xyz must have shape natoms x 3")
        self.xyz = xyz.copy()

        self.elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]

        if not isinstance(self.Data['comment'], basestring):
            raise TypeError("comment for a Molecule must be a string")

        # build the topology for the molecule
        self.built_bonds = False
        ## Topology settings
        self.top_settings = {'toppbc' : kwargs.get('toppbc', False),
                             'topframe' : kwargs.get('topframe', 0),
                             'Fac' : kwargs.get('Fac', 1.2),
                             'read_bonds' : False,
                             'fragment' : kwargs.get('fragment', False),
                             'radii' : kwargs.get('radii', {})}


        logger.info("Molecule %s constructed.", repr(self))
        logger.debug("Molecule %s constructed.", repr(self))

    def copy(self, xyz=None,node_id=1):
        """Create a copy of this molecule"""

        lot = self.PES.lot.copy(node_id)
        PES = type(self.PES).create_pes_from(self.PES,lot)
        mol = type(self)(self.Data.copy().set_values({"PES":PES}))

        #CRA 3/2019 I don't know what this is
        #for key, value in self.__dict__.iteritems():
        #    mol.__dict__[key] = copy.copy(value)

        if xyz is not None:
            xyz = np.array(xyz, dtype=float)
            mol.xyz = xyz.reshape((-1, 3))
        else:
            mol.xyz = self.xyz.copy()

        return mol

    def __add__(self,other):
        """ add method for molecule objects. Concatenates"""
        raise NotImplementedError

    def reorder(self, new_order):
        """Reorder atoms in the molecule"""
        for field in ["atoms", "xyz"]:
            self.__dict__[field] = self.__dict__[field][list(new_order)]
        self.elements = [self.elements[i] for i in new_order]

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

    def build_bonds(self):
        """ Build the bond connectivity graph. """
        sn = self.top_settings['topframe']
        toppbc = self.top_settings['toppbc']
        Fac = self.top_settings['Fac']
        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.
        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones
        R = np.array([self.top_settings['radii'].get(i.symbol, i.covalent_radius) for i in self.elements])
        # Create a list of 2-tuples corresponding to combinations of atomic indices using a grid algorithm.
        mins = np.min(self.xyz,axis=0)
        maxs = np.max(self.xyz,axis=0)
        # Grid size in Angstrom.  This number is optimized for speed in a 15,000 atom system (united atom pentadecane).
        gsz = 6.0
        if hasattr(self, 'boxes'):
            xmin = 0.0
            ymin = 0.0
            zmin = 0.0
            xmax = self.boxes[sn].a
            ymax = self.boxes[sn].b
            zmax = self.boxes[sn].c
            if any([i != 90.0 for i in [self.boxes[sn].alpha, self.boxes[sn].beta, self.boxes[sn].gamma]]):
                logger.warning("Warning: Topology building will not work with broken molecules in nonorthogonal cells.")
                toppbc = False
        else:
            xmin = mins[0]
            ymin = mins[1]
            zmin = mins[2]
            xmax = maxs[0]
            ymax = maxs[1]
            zmax = maxs[2]
            toppbc = False

        xext = xmax-xmin
        yext = ymax-ymin
        zext = zmax-zmin

        if toppbc:
            gszx = xext/int(xext/gsz)
            gszy = yext/int(yext/gsz)
            gszz = zext/int(zext/gsz)
        else:
            gszx = gsz
            gszy = gsz
            gszz = gsz

        # Run algorithm to determine bonds.
        # Decide if we want to use the grid algorithm.
        use_grid = toppbc or (np.min([xext, yext, zext]) > 2.0*gsz)
        if use_grid:
            # Inside the grid algorithm.
            # 1) Determine the left edges of the grid cells.
            # Note that we leave out the rightmost grid cell,
            # because this may cause spurious partitionings.
            xgrd = np.arange(xmin, xmax-gszx, gszx)
            ygrd = np.arange(ymin, ymax-gszy, gszy)
            zgrd = np.arange(zmin, zmax-gszz, gszz)
            # 2) Grid cells are denoted by a three-index tuple.
            gidx = list(itertools.product(range(len(xgrd)), range(len(ygrd)), range(len(zgrd))))
            # 3) Build a dictionary which maps a grid cell to itself plus its neighboring grid cells.
            # Two grid cells are defined to be neighbors if the differences between their x, y, z indices are at most 1.
            gngh = OrderedDict()
            amax = np.array(gidx[-1])
            amin = np.array(gidx[0])
            n27 = np.array(list(itertools.product([-1,0,1],repeat=3)))
            for i in gidx:
                gngh[i] = []
                ai = np.array(i)
                for j in n27:
                    nj = ai+j
                    for k in range(3):
                        mod = amax[k]-amin[k]+1
                        if nj[k] < amin[k]:
                            nj[k] += mod
                        elif nj[k] > amax[k]:
                            nj[k] -= mod
                    gngh[i].append(tuple(nj))
            # 4) Loop over the atoms and assign each to a grid cell.
            # Note: I think this step becomes the bottleneck if we choose very small grid sizes.
            gasn = OrderedDict([(i, []) for i in gidx])
            for i in range(self.na):
                xidx = -1
                yidx = -1
                zidx = -1
                for j in xgrd:
                    xi = self.xyzs[sn][i][0]
                    while xi < xmin: xi += xext
                    while xi > xmax: xi -= xext
                    if xi < j: break
                    xidx += 1
                for j in ygrd:
                    yi = self.xyzs[sn][i][1]
                    while yi < ymin: yi += yext
                    while yi > ymax: yi -= yext
                    if yi < j: break
                    yidx += 1
                for j in zgrd:
                    zi = self.xyzs[sn][i][2]
                    while zi < zmin: zi += zext
                    while zi > zmax: zi -= zext
                    if zi < j: break
                    zidx += 1
                gasn[(xidx,yidx,zidx)].append(i)

            # 5) Create list of 2-tuples corresponding to combinations of atomic indices.
            # This is done by looping over pairs of neighboring grid cells and getting Cartesian products of atom indices inside.
            # It may be possible to get a 2x speedup by eliminating forward-reverse pairs (e.g. (5, 4) and (4, 5) and duplicates (5,5).)
            AtomIterator = []
            for i in gasn:
                for j in gngh[i]:
                    apairs = cartesian_product2([gasn[i], gasn[j]])
                    if len(apairs) > 0: AtomIterator.append(apairs[apairs[:,0]>apairs[:,1]])
            AtomIterator = np.ascontiguousarray(np.vstack(AtomIterator))
        else:
            # Create a list of 2-tuples corresponding to combinations of atomic indices.
            # This is much faster than using itertools.combinations.
            AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[range(i+1,self.natoms) for i in range(self.natoms)]),dtype=np.int32))).T)

        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        BT0 = R[AtomIterator[:,0]]
        BT1 = R[AtomIterator[:,1]]
        BondThresh = (BT0+BT1) * Fac
        BondThresh = (BondThresh > mindist) * BondThresh + (BondThresh < mindist) * mindist
        if hasattr(self, 'boxes') and toppbc:
            dxij = AtomContact(self.xyz, AtomIterator, box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c]))
        else:
            dxij = AtomContact(self.xyz, AtomIterator)

        # Update topology settings with what we learned
        self.top_settings['toppbc'] = toppbc

        # Create a list of atoms that each atom is bonded to.
        atom_bonds = [[] for i in range(self.natoms)]
        bond_bool = dxij < BondThresh
        for i, a in enumerate(bond_bool):
            if not a: continue
            (ii, jj) = AtomIterator[i]
            if ii == jj: continue
            atom_bonds[ii].append(jj)
            atom_bonds[jj].append(ii)
        bondlist = []

        for i, bi in enumerate(atom_bonds):
            for j in bi:
                if i == j: continue
                # Do not add a bond between resids if fragment is set to True.
                if self.top_settings['fragment'] and 'resid' in self.Data.keys() and self.resid[i] != self.resid[j]:
                    continue
                elif i < j:
                    bondlist.append((i, j))
                else:
                    bondlist.append((j, i))
        bondlist = sorted(list(set(bondlist)))
        self.bonds = sorted(list(set(bondlist)))
        self.built_bonds = True

    def build_topology(self, force_bonds=True, **kwargs):
        """
        Create self.topology and self.molecules; these are graph
        representations of the individual molecules (fragments)
        contained in the Molecule object.

        Parameters
        ----------
        force_bonds : bool
            Build the bonds from interatomic distances.  If the user
            calls build_topology from outside, assume this is the
            default behavior.  If creating a Molecule object using
            __init__, do not force the building of bonds by default
            (only build bonds if not read from file.)
        topframe : int, optional
            Provide the frame number used for reading the bonds.  If
            not provided, this will be taken from the top_settings
            field.  If provided, this will take priority and write
            the value into top_settings.
        """
        sn = kwargs.get('topframe', self.top_settings['topframe'])
        self.top_settings['topframe'] = sn
        if self.natoms > 100000:
            logger.warning("Warning: Large number of atoms (%i), topology building may take a long time" % self.na)
        # Build bonds from connectivity graph if not read from file.
        if (not self.top_settings['read_bonds']) or force_bonds:
            self.build_bonds()
        # Create a NetworkX graph object to hold the bonds.
        G = MyG()
        for i, a in enumerate(self.Data['atoms']):
            G.add_node(i)
            if parse_version(nx.__version__) >= parse_version('2.0'):
                nx.set_node_attributes(G,{i:a}, name='e')
                nx.set_node_attributes(G,{i:self.xyz[i]}, name='x')
            else:
                nx.set_node_attributes(G,'e',{i:a})
                nx.set_node_attributes(G,'x',{i:self.xyz[i]})
        for (i, j) in self.bonds:
            G.add_edge(i, j)
        # The Topology is simply the NetworkX graph object.
        self.topology = G
        # LPW: Molecule.molecules is a funny misnomer... it should be fragments or substructures or something
        self.molecules = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in self.molecules: g.__class__ = MyG
        # Deprecated in networkx 2.2
        # self.molecules = list(nx.connected_component_subgraphs(G))

        # show topology
        #import matplotlib.pyplot as plt
        #plt.plot()
        #nx.draw(m.topology,with_labels=True,font_weight='bold')
        #plt.show()
        #plt.svefig('tmp.png')

    def distance_matrix(self, pbc=True):
        """ Obtain distance matrix between all pairs of atoms. """
        AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[range(i+1,self.natoms) for i in range(self.natoms)]),dtype=np.int32))).T)
        drij = []
        if hasattr(self, 'boxes') and pbc:
            drij.append(AtomContact(self.xyz,AtomIterator,box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c])))
        else:
            drij.append(AtomContact(self.xyz,AtomIterator))
        return AtomIterator, drij

    def distance_displacement(self):
        """ Obtain distance matrix and displacement vectors between all pairs of atoms. """
        AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[range(i+1,self.natoms) for i in range(self.natoms)]),dtype=np.int32))).T)
        drij = []
        dxij = []
        if hasattr(self, 'boxes') and pbc:
            drij_i, dxij_i = AtomContact(self.xyz,AtomIterator,box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c]),displace=True)
        else:
            drij_i, dxij_i = AtomContact(self.xyz,AtomIterator,box=None,displace=True)
        drij.append(drij_i)
        dxij.append(dxij_i)
        return AtomIterator, drij, dxij

    def find_angles(self):

        """ Return a list of 3-tuples corresponding to all of the
        angles in the system.  Verified for lysine and tryptophan
        dipeptide when comparing to TINKER's analyze program. """

        if not hasattr(self, 'topology'):
            logger.error("Need to have built a topology to find angles\n")
            raise RuntimeError

        angidx = []
        # Iterate over separate molecules
        for mol in self.molecules:
            # Iterate over atoms in the molecule
            for a2 in list(mol.nodes()):
                # Find all bonded neighbors to this atom
                friends = sorted(list(nx.neighbors(mol, a2)))
                if len(friends) < 2: continue
                # Double loop over bonded neighbors
                for i, a1 in enumerate(friends):
                    for a3 in friends[i+1:]:
                        # Add bonded atoms in the correct order
                        angidx.append((a1, a2, a3))
        return angidx

    def find_dihedrals(self):

        """ Return a list of 4-tuples corresponding to all of the
        dihedral angles in the system.  Verified for alanine and
        tryptophan dipeptide when comparing to TINKER's analyze
        program. """

        if not hasattr(self, 'topology'):
            logger.error("Need to have built a topology to find dihedrals\n")
            raise RuntimeError

        dihidx = []
        # Iterate over separate molecules
        for mol in self.molecules:
            # Iterate over bonds in the molecule
            for edge in list(mol.edges()):
                # Determine correct ordering of atoms (middle atoms are ordered by convention)
                a2 = edge[0] if edge[0] < edge[1] else edge[1]
                a3 = edge[1] if edge[0] < edge[1] else edge[0]
                for a1 in sorted(list(nx.neighbors(mol, a2))):
                    if a1 != a3:
                        for a4 in sorted(list(nx.neighbors(mol, a3))):
                            if a4 != a2 and len({a1, a2, a3, a4}) == 4:
                                dihidx.append((a1, a2, a3, a4))
        return dihidx

    def measure_distances(self, i, j):
        distances = []
        for s in range(self.ns):
            x1 = self.xyzs[s][i]
            x2 = self.xyzs[s][j]
            distance = np.linalg.norm(x1-x2)
            distances.append(distance)
        return distances

    def measure_angles(self, i, j, k):
        angles = []
        for s in range(self.ns):
            x1 = self.xyzs[s][i]
            x2 = self.xyzs[s][j]
            x3 = self.xyzs[s][k]
            v1 = x1-x2
            v2 = x3-x2
            n = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
            angle = np.arccos(n)
            angles.append(angle * 180/ np.pi)
        return angles

    def measure_dihedrals(self, i, j, k, l):
        """ Return a series of dihedral angles, given four atom indices numbered from zero. """
        phis = []
        if 'bonds' in self.Data:
            if any(p not in self.bonds for p in [(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))]):
                logger.warning([(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))])
                logger.warning("Measuring dihedral angle for four atoms that aren't bonded.  Hope you know what you're doing!")
        else:
            logger.warning("This molecule object doesn't have bonds defined, sanity-checking is off.")
        for s in range(self.ns):
            x4 = self.xyzs[s][l]
            x3 = self.xyzs[s][k]
            x2 = self.xyzs[s][j]
            x1 = self.xyzs[s][i]
            v1 = x2-x1
            v2 = x3-x2
            v3 = x4-x3
            t1 = np.linalg.norm(v2)*np.dot(v1,np.cross(v2,v3))
            t2 = np.dot(np.cross(v1,v2),np.cross(v2,v3))
            phi = np.arctan2(t1,t2)
            phis.append(phi * 180 / np.pi)
            #phimod = phi*180/pi % 360
            #phis.append(phimod)
            #print phimod
        return phis

    @property
    def atomic_mass(self):
        return np.array([un.AMU_TO_AU * ele.mass_amu for ele in self.elements])

    @property
    def mass_amu(self):
        return np.array([ele.mass_amu for ele in self.elements])

    @property
    def atomic_num(self):
        return [ele.atomic_num for ele in self.elements]

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
        return len(self.Data['atoms'])

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
        return manage_xyz.combine_atom_xyz(self.Data['atoms'],self.xyz)

    @property
    def energy(self):
        return self.PES.get_energy(self.xyz)

    @property
    def gradient(self):
        return self.PES.get_gradient(self.xyz)

    @property
    def derivative_coupling(self):
        return self.PES.get_coupling(self.xyz)

    @property
    def difference_gradient(self):
        return self.PES.get_dgrad(self.xyz)

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
    M = Molecule.from_options(geom=geom,PES=pes)

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

