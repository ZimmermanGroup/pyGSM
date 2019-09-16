from __future__ import print_function

# standard library imports
import time
import sys
from os import path

# i don't know what this is doing
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))

import numpy as np
import itertools
from pkg_resources import parse_version
from collections import OrderedDict, defaultdict

try:
    import networkx as nx
except ImportError:
    nifty.logger.warning("NetworkX cannot be imported (topology tools won't work).  Most functionality should still work though.")

from utilities import *

#===========================#
#|   Connectivity graph    |#
#|  Good for doing simple  |#
#|     topology tricks     |#
#===========================#
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

    #print("sel1, sel2")
    #print(sel1)
    #print(sel2)
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


class Topology():
    def __init__(self,**kwargs):

        return

    @staticmethod
    def build_topology( xyz, atoms, add_bond=None,hybrid_indices=None, force_bonds=True, bondlistfile=None, **kwargs):
        """
        Create topology and fragments; these are graph
        representations of the individual molecule fragments
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

        natoms = len(atoms)

        # can do an assert for xyz here CRA TODO
        if natoms > 100000:
            nifty.logger.warning("Warning: Large number of atoms (%i), topology building may take a long time" % natoms)

        # Get hybrid indices
        hybrid_indices=hybrid_indices
        hybrid_idx_start_stop = []
        prim_idx_start_stop = []
        if hybrid_indices ==None:
            primitive_indices = range(len(atoms))
        else:
            # specify Hybrid TRIC we need to specify which atoms to build topology for
            primitive_indices = []
            for i in range(len(atoms)):
                if i not in hybrid_indices:
                    primitive_indices.append(i)
            #print("non-cartesian indices")
            #print(primitive_indices)

            # get the hybrid start and stop indices
            new=True
            for i in range(natoms+1):
                if i in hybrid_indices:
                    if new==True:
                        start=i
                        new=False
                else:
                    if new==False:
                        end=i-1
                        new=True
                        hybrid_idx_start_stop.append((start,end))

        if force_bonds:
            nifty.printcool(" building bonds")
            bonds = Topology.build_bonds(xyz,atoms,primitive_indices)
            assert bondlistfile is None
        elif bondlistfile:
            bonds = read_bonds_from_file(bondlistfile)

        if add_bond:
            print(" adding extra bonds")
            for bond in add_bond:
                bonds.append(bond)
            #print(bonds)

        # Create a NetworkX graph object to hold the bonds.
        G = MyG()

        for i in primitive_indices:
            element = atoms[i]
            a = element.name
            G.add_node(i)
            if parse_version(nx.__version__) >= parse_version('2.0'):
                nx.set_node_attributes(G,{i:a}, name='e')
                nx.set_node_attributes(G,{i:xyz[i]}, name='x')
            else:
                nx.set_node_attributes(G,'e',{i:a})
                nx.set_node_attributes(G,'x',{i:xyz[i]})
        for (i, j) in bonds:
            G.add_edge(i, j)

        # The Topology is simply the NetworkX graph object.
        topology = G
        fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in fragments: g.__class__ = MyG


        #print("nodes of Graph")
        #print(topology.L())

        # Deprecated in networkx 2.2
        # fragments = list(nx.connected_component_subgraphs(G))

        # show topology
        #import matplotlib as mpl
        #mpl.use('Agg')
        #import matplotlib.pyplot as plt
        #plt.plot()
        #nx.draw(topology,with_labels=True,font_weight='bold')
        #plt.show()
        #plt.savefig('tmp.png')

        return G

    @staticmethod
    def rebuild_topology_from_prim_bonds(xyz):

        raise NotImplementedError
        bonds = []
        for p in Internals:
            if type(p)==Distance:
                bonds.append(p)

        # Create a NetworkX graph object to hold the bonds.
        G = MyG()
        for i, a_dict in enumerate(atoms):
            a = a_dict.name
            G.add_node(i)
            if parse_version(nx.__version__) >= parse_version('2.0'):
                nx.set_node_attributes(G,{i:a}, name='e')
                nx.set_node_attributes(G,{i:xyz[i]}, name='x')
            else:
                nx.set_node_attributes(G,'e',{i:a})
                nx.set_node_attributes(G,'x',{i:xyz[i]})
        for bond in bonds:
            atoms = bond.atoms
            G.add_edge(atoms[0],atoms[1])
        
        # The Topology is simply the NetworkX graph object.
        topology = G
        fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in fragments: g.__class__ = MyG

    @staticmethod
    def build_bonds(xyz,atoms,primitive_indices,**kwargs):
        """ Build the bond connectivity graph. """

        print(" In build bonds")
        top_settings = {
                            'toppbc' : kwargs.get('toppbc', False),
                            'topframe' : kwargs.get('topframe', 0),
                            'Fac' : kwargs.get('Fac', 1.2),
                            'radii' : kwargs.get('radii', {}),
                             }

        # leftover from LPW code
        sn = top_settings['topframe']
        toppbc = top_settings['toppbc']
        Fac = top_settings['Fac']
        natoms = len(xyz)

        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.
        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones
        #R = np.array([top_settings['radii'].get(i.symbol, i.covalent_radius) for i in atoms])
        R = np.array([atom.covalent_radius for atom in atoms ])
        # Create a list of 2-tuples corresponding to combinations of atomic indices using a grid algorithm.
        mins = np.min(xyz,axis=0)
        maxs = np.max(xyz,axis=0)
        # Grid size in Angstrom.  This number is optimized for speed in a 15,000 atom system (united atom pentadecane).
        gsz = 6.0
        #if hasattr( 'boxes'):
        if top_settings['toppbc']:
            raise NotImplementedError
            xmin = 0.0
            ymin = 0.0
            zmin = 0.0
            xmax = boxes[sn].a
            ymax = boxes[sn].b
            zmax = boxes[sn].c
            if any([i != 90.0 for i in [boxes[sn].alpha, boxes[sn].beta, boxes[sn].gamma]]):
                nifty.logger.warning("Warning: Topology building will not work with broken molecules in nonorthogonal cells.")
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
            gidx = list(itertools.product(list(range(len(xgrd))), list(range(len(ygrd))), list(range(len(zgrd)))))
            # 3) Build a dictionary which maps a grid cell to itplus its neighboring grid cells.
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

            # TODO 9/2019 build-bonds only for non-cartesian indices
            gasn = OrderedDict([(i, []) for i in gidx])
            #for i in range(natoms):
            for i in primitive_indices:
                xidx = -1
                yidx = -1
                zidx = -1
                for j in xgrd:
                    xi = xyz[i][0]
                    while xi < xmin: xi += xext
                    while xi > xmax: xi -= xext
                    if xi < j: break
                    xidx += 1
                for j in ygrd:
                    yi = xyz[i][1]
                    while yi < ymin: yi += yext
                    while yi > ymax: yi -= yext
                    if yi < j: break
                    yidx += 1
                for j in zgrd:
                    zi = xyz[i][2]
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
                    apairs = nifty.cartesian_product2([gasn[i], gasn[j]])
                    if len(apairs) > 0: AtomIterator.append(apairs[apairs[:,0]>apairs[:,1]])
            AtomIterator = np.ascontiguousarray(np.vstack(AtomIterator))
        else:
            # Create a list of 2-tuples corresponding to combinations of atomic indices.
            # This is much faster than using itertools.combinations.
            # TODO 9/2019 build-bonds only for non-cartesian indices
            num_atoms_nc = len(primitive_indices)

            # the original version
            # AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(natoms-i-1) for i in range(natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[list(range(i+1,natoms)) for i in range(natoms)]),dtype=np.int32))).T)


            #first_o =np.fromiter(itertools.chain(*[[i]*(natoms-i-1) for i in range(natoms)]),dtype=np.int32)


            #print("prim indices")
            # need the primitive start and stop indices
            prim_idx_start_stop = []
            new=True
            for i in range(natoms+1):
                if i in primitive_indices:
                    if new==True:
                        start=i
                        new=False
                else:
                    if new==False:
                        end=i-1
                        new=True
                        prim_idx_start_stop.append((start,end))

            #print(prim_idx_start_stop)
            first_list =[]
            for tup in prim_idx_start_stop:
                for i in range(tup[0],tup[1]):
                    first_list.append([i]*(tup[1]-i))
                    #first_list.append([i]*(tup[1]-i-1) for i in range(tup[0],tup[1]))
            #print(first_list)

            second_list =[]
            for tup in prim_idx_start_stop:
                for i in range(tup[0],tup[1]):
                    second_list.append(list(range(i+1,tup[1]+1)))
            #print(second_list)

            #first = np.fromiter(itertools.chain(*[[i]*(natoms-i-1) for i in primitive_indices]),dtype=np.int32)
            first = np.fromiter(itertools.chain(*first_list),dtype=np.int32)
            second = np.fromiter(itertools.chain(*second_list),dtype=np.int32)

            #print("HERE")
            #print(first_o.T)
            #print("\n")
            #print(first.T)
            ##print(second.T)

            AtomIterator = np.ascontiguousarray(
                    np.vstack((
                            np.fromiter(itertools.chain(*first_list),dtype=np.int32),
                            np.fromiter(itertools.chain(*second_list),dtype=np.int32)
                            )).T
                    )
                    #np.fromiter(itertools.chain(*[[i]*(natoms-i-1) for i in primitive_indices]),dtype=np.int32), 


        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        BT0 = R[AtomIterator[:,0]]
        BT1 = R[AtomIterator[:,1]]
        BondThresh = (BT0+BT1) * Fac
        BondThresh = (BondThresh > mindist) * BondThresh + (BondThresh < mindist) * mindist
        #if hasattr( 'boxes') and toppbc:
        if toppbc:
            raise NotImplementedError
            dxij = AtomContact(xyz, AtomIterator, box=np.array([boxes[sn].a, boxes[sn].b, boxes[sn].c]))
        else:
            dxij = AtomContact(xyz, AtomIterator)

        # Update topology settings with what we learned
        top_settings['toppbc'] = toppbc

        # Create a list of atoms that each atom is bonded to.
        atom_bonds = [[] for i in range(natoms)]
        #atom_bonds = [[] for i in primitive_indices]

        bond_bool = dxij < BondThresh
        for i, a in enumerate(bond_bool):
            if not a: continue
            (ii, jj) = AtomIterator[i]
            if ii == jj: continue
            atom_bonds[ii].append(jj)
            atom_bonds[jj].append(ii)
        bondlist = []

        #print("atom_bonds")
        #print(atom_bonds)

        for i, bi in enumerate(atom_bonds):
            for j in bi:
                if i == j: continue
                # Do not add a bond between resids if fragment is set to True.
                #if top_settings['fragment'] and 'resid' in Data.keys() and resid[i] != resid[j]:
                #    continue
                elif i < j:
                    bondlist.append((i, j))
                else:
                    bondlist.append((j, i))
        bondlist = sorted(list(set(bondlist)))
        bonds = sorted(list(set(bondlist)))
        built_bonds = True

        #print('bond list')
        #print(bondlist)

        return bonds

    def read_bonds_from_file(filename):
        print("reading bonds")
        bondlist = np.loadtxt(filename)
        
        bonds=[]
        for b in bondlist:
            i = int(b[0])
            j = int(b[1])
            if i>j:
                bonds.append((i,j))
            else:
                bonds.append((j,i))
        
        sorted_bonds = sorted(list(set(bonds)))
        built_bonds = True
        print(sorted_bonds[:10])

        return sorted_bonds

    def distance_matrix(self,xyz, pbc=True):
        """ Obtain distance matrix between all pairs of atoms. """
        AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[list(range(i+1,self.natoms)) for i in range(self.natoms)]),dtype=np.int32))).T)
        drij = []
        if hasattr(self, 'boxes') and pbc:
            drij.append(AtomContact(xyz,AtomIterator,box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c])))
        else:
            drij.append(AtomContact(xyz,AtomIterator))
        return AtomIterator, drij

    def distance_displacement(xyz,self):
        """ Obtain distance matrix and displacement vectors between all pairs of atoms. """
        AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[list(range(i+1,self.natoms)) for i in range(self.natoms)]),dtype=np.int32))).T)
        drij = []
        dxij = []
        if hasattr(self, 'boxes') and pbc:
            drij_i, dxij_i = AtomContact(xyz,AtomIterator,box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c]),displace=True)
        else:
            drij_i, dxij_i = AtomContact(xyz,AtomIterator,box=None,displace=True)
        drij.append(drij_i)
        dxij.append(dxij_i)
        return AtomIterator, drij, dxij

    # these aren't used 
    def find_angles(self):

        """ Return a list of 3-tuples corresponding to all of the
        angles in the system.  Verified for lysine and tryptophan
        dipeptide when comparing to TINKER's analyze program. """

        if not hasattr(self, 'topology'):
            logger.error("Need to have built a topology to find angles\n")
            raise RuntimeError

        angidx = []
        # Iterate over separate molecules
        for mol in self.fragments:
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

    # these aren't used 
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
        for mol in self.fragments:
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

if __name__ =='__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    #filepath='../../data/butadiene_ethene.xyz'
    #filepath='crystal.xyz'
    filepath1='multi1.xyz'
    filepath2='multi2.xyz'

    geom1 = manage_xyz.read_xyz(filepath1)
    geom2 = manage_xyz.read_xyz(filepath2)
    atom_symbols  = manage_xyz.get_atoms(geom1)
    xyz1 = manage_xyz.xyz_to_np(geom1)
    xyz2 = manage_xyz.xyz_to_np(geom2)

    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    #print(atoms)

    #hybrid_indices = list(range(0,10)) + list(range(21,26))
    hybrid_indices =  list(range(16,26))


    G1 = Topology.build_topology(xyz1,atoms,hybrid_indices=hybrid_indices)
    G2 = Topology.build_topology(xyz2,atoms,hybrid_indices=hybrid_indices)

    for bond in G2.edges():
        if bond in G1.edges:
            pass
        elif (bond[1],bond[0]) in G1.edges():
            pass
        else:
            print(" Adding bond {} to top1".format(bond))
            if bond[0]>bond[1]:
                G1.add_edge(bond[0],bond[1])
            else:
                G1.add_edge(bond[1],bond[0])


    #print(" G")
    #print(G.L())

    #fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    #for g in fragments: g.__class__ = MyG
   
    #print(" fragments")
    #for frag in fragments:
    #    print(frag.L())
    ##print(len(mytop.fragments))
    ##print(mytop.fragments)

    ## need the primitive start and stop indices
    #prim_idx_start_stop=[]
    #new=True
    #for frag in fragments:
    #    nodes=frag.L()
    #    prim_idx_start_stop.append((nodes[0],nodes[-1]))
    #print("prim start stop")
    #print(prim_idx_start_stop)

    #prim_idx =[]
    #for info in prim_idx_start_stop:
    #    prim_idx += list(range(info[0],info[1]+1))
    #print('prim_idx')
    #print(prim_idx)

    #new_hybrid_indices=list(range(len(atoms)))
    #for elem in prim_idx:
    #    new_hybrid_indices.remove(elem)
    #print('hybr')
    #print(new_hybrid_indices)

    #hybrid_idx_start_stop=[]
    ## get the hybrid start and stop indices
    #new=True
    #for i in range(len(atoms)+1):
    #    if i in new_hybrid_indices:
    #        print(i)
    #        if new==True:
    #            start=i
    #            new=False
    #    else:
    #        if new==False:
    #            end=i-1
    #            new=True
    #            hybrid_idx_start_stop.append((start,end))
    #print(hybrid_idx_start_stop)
