#!/usr/bin/env python

import time
from collections import OrderedDict, defaultdict
import numpy as np
from numpy.linalg import multi_dot
import elements
from nifty import click, commadash, logger,cartesian_product2
from _math_utils import *
import options
from slots import *
import itertools
import networkx as nx
from pkg_resources import parse_version


ELEMENT_TABLE = elements.ElementData()

CacheWarning = False

class InternalCoordinates(object):

    @staticmethod
    def default_options():
        ''' InternalCoordinates default options.'''

        if hasattr(InternalCoordinates, '_default_options'): return InternalCoordinates._default_options.copy()
        opt = options.Options() 

        opt.add_option(
                key="xyz",
                required=True,
                doc='cartesian coordinates in angstrom'
                )

        opt.add_option(
                key='atoms',
                required=True,
                #allowed_types=[],
                doc='atom element named tuples/dictionary must be of type list[elements].'
                )

        opt.add_option(
                key="frozen_atoms",
                value=None,
                required=False,
                doc='Atoms to be left unoptimized/unmoved',
                )

        opt.add_option(
                key='connect',
                value=False,
                allowed_types=[bool],
                doc="Connect the fragments/residues together with a minimum spanning bond,\
                    use for DLC, Don't use for TRIC, or HDLC.",
                )

        opt.add_option(
                key='addcart',
                value=False,
                allowed_types=[bool],
                doc="Add cartesian coordinates\
                    use to form HDLC ,Don't use for TRIC, DLC.",
                )

        opt.add_option(
                key='addtr',
                value=False,
                allowed_types=[bool],
                doc="Add translation and rotation coordinates\
                    use for TRIC.",
                )

        opt.add_option(
                key='constraints',
                value=None,
                allowed_types=[list],
                doc='A list of Distance,Angle,Torsion constraints (see slots.py),\
                    This is only useful if doing a constrained geometry optimization\
                    since GSM will handle the constraint automatically.'
                )
        opt.add_option(
                key='cVals',
                value=None,
                allowed_types=[list],
                doc='List of Distance,Angle,Torsion constraints values'
                )

        opt.add_option(
                key='extra_kwargs',
                value={},
                doc='Extra keyword arguments -- THis is leftover from LPW code but \
                        maybe useful in the future'
                        )

        opt.add_option(
                key='primitives',
                value=None,
                doc='This is a Primitive internal coordinates object -- can be used instead \
                        of creating new primitive object'
                )

        opt.add_option(
                key='print_level',
                value=1,
                required=False,
                allowed_types=[int],
                doc='0-- no printing, 1-- printing')

        InternalCoordinates._default_options = opt
        return InternalCoordinates._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return InternalCoordinates(InternalCoordinates.default_options().set_values(kwargs))

    def __init__(self,
            options
            ):

        self.options = options
        self.stored_wilsonB = OrderedDict()

    def addConstraint(self, cPrim, cVal):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def haveConstraints(self):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def augmentGH(self, xyz, G, H):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def calcGradProj(self, xyz, gradx):
        raise NotImplementedError("Constraints not supported with Cartesian coordinates")

    def clearCache(self):
        self.stored_wilsonB = OrderedDict()

    def wilsonB(self, xyz):
        """
        Given Cartesian coordinates xyz, return the Wilson B-matrix
        given by dq_i/dx_j where x is flattened (i.e. x1, y1, z1, x2, y2, z2)
        """
        global CacheWarning
        t0 = time.time()
        xhash = hash(xyz.tostring())
        ht = time.time() - t0
        if xhash in self.stored_wilsonB:
            ans = self.stored_wilsonB[xhash]
            return ans
        WilsonB = []
        Der = self.derivatives(xyz)
        for i in range(Der.shape[0]):
            WilsonB.append(Der[i].flatten())
        self.stored_wilsonB[xhash] = np.array(WilsonB)
        if len(self.stored_wilsonB) > 1000 and not CacheWarning:
            logger.warning("\x1b[91mWarning: more than 100 B-matrices stored, memory leaks likely\x1b[0m")
            CacheWarning = True
        ans = np.array(WilsonB)
        return ans

    def GMatrix(self, xyz):
        """
        Given Cartesian coordinates xyz, return the G-matrix
        given by G = BuBt where u is an arbitrary matrix (default to identity)
        """
        #t0 = time.time()
        Bmat = self.wilsonB(xyz)
        #t1 = time.time()
        BuBt = np.dot(Bmat,Bmat.T)
        #t2 = time.time()
        #t10 = t1-t0
        #t21 = t2-t1
        #print("time to form B-matrix %.3f" % t10)
        #print("time to mat-mult B %.3f" % t21)
        return BuBt

    def GInverse_SVD(self, xyz):
        xyz = xyz.reshape(-1,3)
        # Perform singular value decomposition
        click()
        loops = 0
        while True:
            try:
                G = self.GMatrix(xyz)
                time_G = click()
                U, S, VT = np.linalg.svd(G)
                time_svd = click()
            except np.linalg.LinAlgError:
                logger.warning("\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m")
                xyz = xyz + 1e-2*np.random.random(xyz.shape)
                loops += 1
                if loops == 10:
                    raise RuntimeError('SVD failed too many times')
                continue
            break
        # print "Build G: %.3f SVD: %.3f" % (time_G, time_svd),
        V = VT.T
        UT = U.T
        Sinv = np.zeros_like(S)
        LargeVals = 0
        for ival, value in enumerate(S):
            # print "%.5e % .5e" % (ival,value)
            if np.abs(value) > 1e-6:
                LargeVals += 1
                Sinv[ival] = 1/value
        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        Sinv = np.diag(Sinv)
        Inv = multi_dot([V, Sinv, UT])
        return Inv

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1,3)
        click()
        G = self.GMatrix(xyz)
        time_G = click()
        Gi = np.linalg.inv(G)
        time_inv = click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return Gi

    def checkFiniteDifference(self, xyz):
        xyz = xyz.reshape(-1,3)
        Analytical = self.derivatives(xyz)
        FiniteDifference = np.zeros_like(Analytical)
        h = 1e-5
        for i in range(xyz.shape[0]):
            for j in range(3):
                x1 = xyz.copy()
                x2 = xyz.copy()
                x1[i,j] += h
                x2[i,j] -= h
                PMDiff = self.calcDiff(x1,x2)
                FiniteDifference[:,i,j] = PMDiff/(2*h)
        for i in range(Analytical.shape[0]):
            logger.info("IC %i/%i : %s" % (i, Analytical.shape[0], self.Internals[i]))
            lines = [""]
            maxerr = 0.0
            for j in range(Analytical.shape[1]):
                lines.append("Atom %i" % (j+1))
                for k in range(Analytical.shape[2]):
                    error = Analytical[i,j,k] - FiniteDifference[i,j,k]
                    if np.abs(error) > 1e-5:
                        color = "\x1b[91m"
                    else:
                        color = "\x1b[92m"
                    lines.append("%s % .5e % .5e %s% .5e\x1b[0m" % ("xyz"[k], Analytical[i,j,k], FiniteDifference[i,j,k], color, Analytical[i,j,k] - FiniteDifference[i,j,k]))
                    if maxerr < np.abs(error):
                        maxerr = np.abs(error)
            if maxerr > 1e-5:
                logger.info('\n'.join(lines))
            else:
                logger.info("Max Error = %.5e" % maxerr)
        logger.info("Finite-difference Finished")

    def calcGrad(self, xyz, gradx):
        #q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx)
        Gq = multi_dot([Ginv, Bmat, gradx])
        return Gq

    def readCache(self, xyz, dQ):
        if not hasattr(self, 'stored_xyz'):
            return None
        #xyz = xyz.flatten()
        #dQ = dQ.flatten()
        if np.linalg.norm(self.stored_xyz - xyz) < 1e-10:
            if np.linalg.norm(self.stored_dQ - dQ) < 1e-10:
                return self.stored_newxyz
        return None

    def writeCache(self, xyz, dQ, newxyz):
        #xyz = xyz.flatten()
        #dQ = dQ.flatten()
        #newxyz = newxyz.flatten()
        self.stored_xyz = xyz.copy()
        self.stored_dQ = dQ.copy()
        self.stored_newxyz = newxyz.copy()

    def newCartesian(self, xyz, dQ, verbose=True):
        cached = self.readCache(xyz, dQ)
        if cached is not None:
            #print "Returning cached result"
            return cached
        xyz1 = xyz.copy()
        dQ1 = dQ.flatten()
        # Iterate until convergence:
        microiter = 0
        ndqs = []
        rmsds = []
        self.bork = False
        # Damping factor
        damp = 1.0
        # Function to exit from loop
        def finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1):
            if ndqt > 1e-1:
                if verbose: logger.info(" Failed to obtain coordinates after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
                self.bork = True
                self.writeCache(xyz, dQ, xyz_iter1)
                return xyzsave.reshape((-1,3))
            elif ndqt > 1e-3:
                if verbose: logger.info(" Approximate coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
            else:
                if verbose: logger.info(" Cartesian coordinates obtained after %i microiterations (rmsd = %.3e |dQ| = %.3e)\n" % (microiter, rmsdt, ndqt))
            self.writeCache(xyz, dQ, xyzsave)
            return xyzsave.reshape((-1,3))
        fail_counter = 0
        while True:
            microiter += 1
            Bmat = self.wilsonB(xyz1)

            #CRA 3/2019
            Ginv = self.GInverse(xyz1)
            #Ginv = np.linalg.inv(np.dot(Bmat,Bmat.T))
            # Get new Cartesian coordinates
            dxyz = damp*multi_dot([Bmat.T,Ginv,dQ1])
            xyz2 = xyz1 + dxyz.reshape((-1,3))
            if microiter == 1:
                xyzsave = xyz2.copy()
                xyz_iter1 = xyz2.copy()
            # Calculate the actual change in internal coordinates
            dQ_actual = self.calcDiff(xyz2, xyz1)
            rmsd = np.sqrt(np.mean((np.array(xyz2-xyz1).flatten())**2))
            ndq = np.linalg.norm(dQ1-dQ_actual)
            if len(ndqs) > 0:
                if ndq > ndqt:
                    if verbose: logger.info(" Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Bad)\n" % (microiter, ndq, ndqt, rmsd, damp))
                    damp /= 2
                    fail_counter += 1
                    # xyz2 = xyz1.copy()
                else:
                    if verbose: logger.info(" Iter: %i Err-dQ (Best) = %.5e (%.5e) RMSD: %.5e Damp: %.5e (Good)\n" % (microiter, ndq, ndqt, rmsd, damp))
                    fail_counter = 0
                    damp = min(damp*1.2, 1.0)
                    rmsdt = rmsd
                    ndqt = ndq
                    xyzsave = xyz2.copy()
            else:
                if verbose: logger.info(" Iter: %i Err-dQ = %.5e RMSD: %.5e Damp: %.5e\n" % (microiter, ndq, rmsd, damp))
                rmsdt = rmsd
                ndqt = ndq
            ndqs.append(ndq)
            rmsds.append(rmsd)
            # Check convergence / fail criteria
            if rmsd < 1e-6 or ndq < 1e-6:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if fail_counter >= 5:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            if microiter == 50:
                return finish(microiter, rmsdt, ndqt, xyzsave, xyz_iter1)
            # Figure out the further change needed
            dQ1 = dQ1 - dQ_actual
            xyz1 = xyz2.copy()

    def rebuild_topology_from_prim_bonds(self,xyz):
        bonds = []
        for p in self.Internals:
            if type(p)==Distance:
                bonds.append(p)

        # Create a NetworkX graph object to hold the bonds.
        G = MyG()
        for i, a_dict in enumerate(self.atoms):
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
        self.topology = G
        self.fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in self.fragments: g.__class__ = MyG

    def build_topology(self, xyz, force_bonds=True, **kwargs):
        """
        Create self.topology and self.fragments; these are graph
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
        # can do an assert for xyz here CRA TODO
        if self.natoms > 100000:
            logger.warning("Warning: Large number of atoms (%i), topology building may take a long time" % self.natoms)

        bonds = self.build_bonds(xyz)

        # Create a NetworkX graph object to hold the bonds.
        G = MyG()
        for i, a_dict in enumerate(self.atoms):
            a = a_dict.name
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
        self.topology = G
        self.fragments = [G.subgraph(c).copy() for c in nx.connected_components(G)]
        for g in self.fragments: g.__class__ = MyG
        # Deprecated in networkx 2.2
        # self.fragments = list(nx.connected_component_subgraphs(G))

        # show topology
        #import matplotlib.pyplot as plt
        #plt.plot()
        #nx.draw(m.topology,with_labels=True,font_weight='bold')
        #plt.show()
        #plt.svefig('tmp.png')

        return G

    def build_bonds(self,xyz):
        """ Build the bond connectivity graph. """

        # leftover from LPW code
        sn = self.top_settings['topframe']
        toppbc = self.top_settings['toppbc']
        Fac = self.top_settings['Fac']

        mindist = 1.0 # Any two atoms that are closer than this distance are bonded.
        # Create an atom-wise list of covalent radii.
        # Molecule object can have its own set of radii that overrides the global ones
        R = np.array([self.top_settings['radii'].get(i.symbol, i.covalent_radius) for i in self.atoms])
        # Create a list of 2-tuples corresponding to combinations of atomic indices using a grid algorithm.
        mins = np.min(xyz,axis=0)
        maxs = np.max(xyz,axis=0)
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
            gidx = list(itertools.product(list(range(len(xgrd))), list(range(len(ygrd))), list(range(len(zgrd)))))
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
            for i in range(self.natoms):
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
                    apairs = cartesian_product2([gasn[i], gasn[j]])
                    if len(apairs) > 0: AtomIterator.append(apairs[apairs[:,0]>apairs[:,1]])
            AtomIterator = np.ascontiguousarray(np.vstack(AtomIterator))
        else:
            # Create a list of 2-tuples corresponding to combinations of atomic indices.
            # This is much faster than using itertools.combinations.
            AtomIterator = np.ascontiguousarray(np.vstack((np.fromiter(itertools.chain(*[[i]*(self.natoms-i-1) for i in range(self.natoms)]),dtype=np.int32), np.fromiter(itertools.chain(*[list(range(i+1,self.natoms)) for i in range(self.natoms)]),dtype=np.int32))).T)

        # Create a list of thresholds for determining whether a certain interatomic distance is considered to be a bond.
        BT0 = R[AtomIterator[:,0]]
        BT1 = R[AtomIterator[:,1]]
        BondThresh = (BT0+BT1) * Fac
        BondThresh = (BondThresh > mindist) * BondThresh + (BondThresh < mindist) * mindist
        if hasattr(self, 'boxes') and toppbc:
            dxij = AtomContact(xyz, AtomIterator, box=np.array([self.boxes[sn].a, self.boxes[sn].b, self.boxes[sn].c]))
        else:
            dxij = AtomContact(xyz, AtomIterator)

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
                #if self.top_settings['fragment'] and 'resid' in self.Data.keys() and self.resid[i] != self.resid[j]:
                #    continue
                elif i < j:
                    bondlist.append((i, j))
                else:
                    bondlist.append((j, i))
        bondlist = sorted(list(set(bondlist)))
        bonds = sorted(list(set(bondlist)))
        self.built_bonds = True

        return bonds

            
   # # CRA  3/2019 these should be utils -- not part of the class
   # def measure_distances(self, i, j):
   #     distances = []
   #     for s in range(self.ns):
   #         x1 = self.xyzs[s][i]
   #         x2 = self.xyzs[s][j]
   #         distance = np.linalg.norm(x1-x2)
   #         distances.append(distance)
   #     return distances

   # def measure_angles(self, i, j, k):
   #     angles = []
   #     for s in range(self.ns):
   #         x1 = self.xyzs[s][i]
   #         x2 = self.xyzs[s][j]
   #         x3 = self.xyzs[s][k]
   #         v1 = x1-x2
   #         v2 = x3-x2
   #         n = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))
   #         angle = np.arccos(n)
   #         angles.append(angle * 180/ np.pi)
   #     return angles

   # def measure_dihedrals(self, i, j, k, l):
   #     """ Return a series of dihedral angles, given four atom indices numbered from zero. """
   #     phis = []
   #     if 'bonds' in self.Data:
   #         if any(p not in self.bonds for p in [(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))]):
   #             logger.warning([(min(i,j),max(i,j)),(min(j,k),max(j,k)),(min(k,l),max(k,l))])
   #             logger.warning("Measuring dihedral angle for four atoms that aren't bonded.  Hope you know what you're doing!")
   #     else:
   #         logger.warning("This molecule object doesn't have bonds defined, sanity-checking is off.")
   #     for s in range(self.ns):
   #         x4 = self.xyzs[s][l]
   #         x3 = self.xyzs[s][k]
   #         x2 = self.xyzs[s][j]
   #         x1 = self.xyzs[s][i]
   #         v1 = x2-x1
   #         v2 = x3-x2
   #         v3 = x4-x3
   #         t1 = np.linalg.norm(v2)*np.dot(v1,np.cross(v2,v3))
   #         t2 = np.dot(np.cross(v1,v2),np.cross(v2,v3))
   #         phi = np.arctan2(t1,t2)
   #         phis.append(phi * 180 / np.pi)
   #         #phimod = phi*180/pi % 360
   #         #phis.append(phimod)
   #         #print phimod
   #     return phis


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
