from __future__ import print_function

# standard library imports
import time
import sys
from os import path

# i don't know what this is doing
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))

# third party
from copy import deepcopy
import numpy as np
import networkx as nx
np.set_printoptions(precision=4,suppress=True)
import itertools
from collections import OrderedDict, defaultdict

# local application imports

try:
    from .internal_coordinates import InternalCoordinates
    from .topology import Topology,MyG
    from .slots import *
except:
    from internal_coordinates import InternalCoordinates
    from topology import Topology,MyG
    from slots import *

from utilities import *

CacheWarning = False

class PrimitiveInternalCoordinates(InternalCoordinates):

    def __init__(self,
            options
            ):
        
        super(PrimitiveInternalCoordinates, self).__init__(options)

        # Cache some useful attributes
        self.options = options
        self.atoms = options['atoms']
        #extra_kwargs=options['extra_kwargs']

        # initialize 
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.Rotators = OrderedDict()
        self.natoms = len(self.atoms)
        self.built_bonds = False

        ## Topology settings  -- CRA 3/2019 leftovers from Lee-Ping's code
        # but maybe useful in the future
        #self.top_settings = {
        #                    #'build_topology' : extra_kwargs.get('build_topology',True),
        #                    'make_primitives' : extra_kwargs.get('make_primitives',True),
        #                     }
        #bondfile = extra_kwargs.get('bondfile',None)

        xyz = options['xyz']
        self.topology = self.options['topology']
        #make_prims = self.top_settings['make_primitives']

        # setup
        if self.options['form_primitives']:
            if self.topology is None:
                print(" Warning it's better to build the topology before calling PrimitiveInternals\n Only the most basic option is enabled here \n You get better control of the topology by controlling extra bonds, angles etc.")
                self.topology = Topology.build_topology(xyz,self.atoms)
                print(" done making topology")
    
            self.fragments = [self.topology.subgraph(c).copy() for c in nx.connected_components(self.topology)]
            for g in self.fragments: g.__class__ = MyG

            self.get_hybrid_indices(xyz)
            nifty.click()
            self.newMakePrimitives(xyz)
            print(" done making primitives")
            time_build = nifty.click()
            print(" make prim %.3f" % time_build)

        # Reorder primitives for checking with cc's code in TC.
        # Note that reorderPrimitives() _must_ be updated with each new InternalCoordinate class written.
        #self.reorderPrimitives()
        #time_reorder = nifty.click()
        #print("done reordering %.3f" % time_reorder)


        #self.makeConstraints(xyz, constraints, cvals)


    @classmethod
    def copy(cls,Prims):
        newPrims = cls(Prims.options.copy().set_values({'form_primitives':False})) 
        newPrims.hybrid_idx_start_stop = Prims.hybrid_idx_start_stop
        newPrims.topology = deepcopy(Prims.topology)
        newPrims.Internals = deepcopy(Prims.Internals)
        newPrims.block_info = deepcopy(Prims.block_info)
        newPrims.atoms = newPrims.options['atoms']
        newPrims.fragments = [Prims.topology.subgraph(c).copy() for c in nx.connected_components(Prims.topology)]
        for g in newPrims.fragments: g.__class__ = MyG

        return newPrims

    def makePrimitives(self, xyz):

        self.Internals =[]
        connect=self.options['connect']
        addcart=self.options['addcart']
        addtr=self.options['addtr']

        # LPW also uses resid from molecule . . . 
        frags = [m.nodes() for m in self.fragments]
        
        # coordinates in Angstrom
        coords = xyz.flatten()

        # Build a list of noncovalent distances
        noncov = []
        # Connect all non-bonded fragments together
        if connect:
            # Make a distance matrix mapping atom pairs to interatomic distances
            AtomIterator, dxij = self.distance_matrix(xyz,pbc=False)
            D = {}
            for i, j in zip(AtomIterator, dxij[0]):
                assert i[0] < i[1]
                D[tuple(i)] = j
            dgraph = nx.Graph()
            for i in range(self.natoms):
                dgraph.add_node(i)
            for k, v in list(D.items()):
                dgraph.add_edge(k[0], k[1], weight=v)
            mst = sorted(list(nx.minimum_spanning_edges(dgraph, data=False)))
            for edge in mst:
                if edge not in list(self.topology.edges()):
                    print("Adding %s from minimum spanning tree" % str(edge))
                    self.topology.add_edge(edge[0], edge[1])
                    noncov.append(edge)
        else:
            if addcart:
                for i in range(self.natoms):
                    self.add(CartesianX(i, w=1.0))
                    self.add(CartesianY(i, w=1.0))
                    self.add(CartesianZ(i, w=1.0))
            elif addtr:
                for i in frags:
                    if len(i) >= 2:
                        self.add(TranslationX(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationY(i, w=np.ones(len(i))/len(i)))
                        self.add(TranslationZ(i, w=np.ones(len(i))/len(i)))
                        sel = coords.reshape(-1,3)[i,:] 
                        sel -= np.mean(sel, axis=0)
                        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                        self.add(RotationA(i, coords, self.Rotators, w=rg))
                        self.add(RotationB(i, coords, self.Rotators, w=rg))
                        self.add(RotationC(i, coords, self.Rotators, w=rg))
                    else:
                        for j in i:
                            self.add(CartesianX(j, w=1.0))
                            self.add(CartesianY(j, w=1.0))
                            self.add(CartesianZ(j, w=1.0))
            else:
                if len(frags)>1:
                    raise RuntimeError("need someway to define the intermolecular interaction")

        # # Build a list of noncovalent distances
        # Add an internal coordinate for all interatomic distances
        for (a, b) in self.topology.edges():
            self.add(Distance(a, b))

        # Add an internal coordinate for all angles
        # This number works best for the iron complex
        LinThre = 0.95
        AngDict = defaultdict(list)
        for b in self.topology.nodes():
            for a in self.topology.neighbors(b):
                for c in self.topology.neighbors(b):
                    if a < c:
                        # if (a, c) in self.topology.edges() or (c, a) in self.topology.edges(): continue
                        Ang = Angle(a, b, c)
                        nnc = (min(a, b), max(a, b)) in noncov
                        nnc += (min(b, c), max(b, c)) in noncov
                        # if nnc >= 2: continue
                        # logger.info("LPW: cosine of angle", a, b, c, "is", np.abs(np.cos(Ang.value(coords))))
                        if np.abs(np.cos(Ang.value(coords))) < LinThre:
                            self.add(Angle(a, b, c))
                            AngDict[b].append(Ang)
                        elif connect or not addcart:
                            # logger.info("Adding linear angle")
                            # Add linear angle IC's
                            # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in selfs (e.g. HCCCN)
                            # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                            # Bringing back old code to use "translations" for the latter case, but should be investigated
                            # more deeply in the future.
                            if nnc == 0:
                                self.add(LinearAngle(a, b, c, 0))
                                self.add(LinearAngle(a, b, c, 1))
                            else:
                                # Unit vector connecting atoms a and c
                                nac = xyz[c] - xyz[a]
                                nac /= np.linalg.norm(nac)
                                # Dot products of this vector with the Cartesian axes
                                dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                # Functions for adding Cartesian coordinate
                                # carts = [CartesianX, CartesianY, CartesianZ]
                                #print("warning, adding translation, did you mean this?")
                                trans = [TranslationX, TranslationY, TranslationZ]
                                w = np.array([-1.0, 2.0, -1.0])
                                # Add two of the most perpendicular Cartesian coordinates
                                for i in np.argsort(dots)[:2]:
                                    self.add(trans[i]([a, b, c], w=w))
                            
        for b in self.topology.nodes():
            for a in self.topology.neighbors(b):
                for c in self.topology.neighbors(b):
                    for d in self.topology.neighbors(b):
                        if a < c < d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(b, d), max(b, d)) in noncov
                            # if nnc >= 1: continue
                            for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                Ang1 = Angle(b,i,j)
                                Ang2 = Angle(i,j,k)
                                if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                                if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                                if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > LinThre:
                                    self.delete(Angle(i, b, j))
                                    self.add(OutOfPlane(b, i, j, k))
                                    break
                                
        # Find groups of atoms that are in straight lines
        atom_lines = [list(i) for i in self.topology.edges()]
        while True:
            # For a line of two atoms (one bond):
            # AB-AC
            # AX-AY
            # i.e. AB is the first one, AC is the second one
            # AX is the second-to-last one, AY is the last one
            # AB-AC-...-AX-AY
            # AB-(AC, AX)-AY
            atom_lines0 = deepcopy(atom_lines)
            for aline in atom_lines:
                # Imagine a line of atoms going like ab-ac-ax-ay.
                # Our job is to extend the line until there are no more
                ab = aline[0]
                ay = aline[-1]
                for aa in self.topology.neighbors(ab):
                    if aa not in aline:
                        # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                        # Add AA to the front of the list
                        if all([np.abs(np.cos(Angle(aa, ab, ac).value(coords))) > LinThre for ac in aline[1:] if ac != ab]):
                            aline.insert(0, aa)
                for az in self.topology.neighbors(ay):
                    if az not in aline:
                        if all([np.abs(np.cos(Angle(ax, ay, az).value(coords))) > LinThre for ax in aline[:-1] if ax != ay]):
                            aline.append(az)
            if atom_lines == atom_lines0: break
        atom_lines_uniq = []
        for i in atom_lines:    # 
            if tuple(i) not in set(atom_lines_uniq):
                atom_lines_uniq.append(tuple(i))
        lthree = [l for l in atom_lines_uniq if len(l) > 2]
        # TODO: Perhaps should reduce the times this is printed out in reaction paths
        # if len(lthree) > 0:
        #     print "Lines of three or more atoms:", ', '.join(['-'.join(["%i" % (i+1) for i in l]) for l in lthree])

        # Normal dihedral code
        for aline in atom_lines_uniq:
            # Go over ALL pairs of atoms in a line
            for (b, c) in itertools.combinations(aline, 2):
                if b > c: (b, c) = (c, b)
                # Go over all neighbors of b
                for a in self.topology.neighbors(b):
                    # Go over all neighbors of c
                    for d in self.topology.neighbors(c):
                        # Make sure the end-atoms are not in the line and not the same as each other
                        if a not in aline and d not in aline and a != d:
                            nnc = (min(a, b), max(a, b)) in noncov
                            nnc += (min(b, c), max(b, c)) in noncov
                            nnc += (min(c, d), max(c, d)) in noncov
                            # print aline, a, b, c, d
                            Ang1 = Angle(a,b,c)
                            Ang2 = Angle(b,c,d)
                            # Eliminate dihedrals containing angles that are almost linear
                            # (should be eliminated already)
                            if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                            if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                            self.add(Dihedral(a, b, c, d))

    # overwritting parent internal coordinate wilsonB with a block matrix representation
    def wilsonB(self,xyz):
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
        xyz = xyz.reshape(-1,3)
        Blist = []
        #sa=0
        #for nprim,na in zip(self.nprims_frag,self.natoms_frag):

        #sp=0
        for info in self.block_info:
            WilsonB = []
            sa = info[0]
            ea = info[1]
            sp = info[2]
            ep = info[3]
            #nprim = info[2]
            #ep=sp+nprim
            Der = np.array( [ p.derivative(xyz[sa:ea,:],start_idx=sa) for p in self.Internals[sp:ep] ])
            for i in range(Der.shape[0]):
                WilsonB.append(Der[i].flatten())
            Blist.append(np.asarray(WilsonB))

            #if info[-1]=="P":
            #    #TODO 9/2019 this is not efficient because it forms the derivative for the entire system
            #    # each derivative is (N,3) dimensional so that is why we only take the sa:ea atoms
            #    Der = np.array([ p.derivative(xyz)[sa:ea,:] for p in self.Internals[sp:ep] ])
            #    for i in range(Der.shape[0]):
            #        WilsonB.append(Der[i].flatten())
            #    Blist.append(np.asarray(WilsonB))
            #else:
            #    # derivative of hybrid region is the identity
            #    # should only append diagonal elements -- means more blocks but minimizes energy and N^2 multiplication
            #    #Blist.append(np.eye(3*leng))
            #    #leng = 3*(ea - sa)
            #    #for elem in range(leng):
            #    #    Blist.append(np.asarray([[1]],dtype=int))
            #    #Blist.append(np.asarray([[1]],dtype=int))
            #    Blist.append(np.eye(3,dtype=int))
            ##sp=ep


        ans = block_matrix(Blist)
        #print(block_matrix.full_matrix(ans))
        #print("total B shape ",ans.shape)
        #print(" num blocks ",ans.num_blocks)
        #for block in ans.matlist:
        #    print(block)
        #    print(block.shape)

        self.stored_wilsonB[xhash] = ans
        if len(self.stored_wilsonB) > 1000 and not CacheWarning:
            nifty.logger.warning("\x1b[91mWarning: more than 100 B-matrices stored, memory leaks likely\x1b[0m")
            CacheWarning = True
        return ans
    
    def GMatrix(self,xyz):
        #if len(self.nprims_frag)==1:
        #    return block_matrix(super(PrimitiveInternalCoordinates,self).GMatrix(xyz))
        t0 = time.time()
        Bmat = self.wilsonB(xyz)
        t1 = time.time()
        #print(" done getting Bmat {}".format(t1-t0))

        #block_list=[]
        #for B,info in zip(Bmat.matlist,self.block_info):
        #    if info[3]=="H":
        #        block_list.append(B)
        #    else:
        #        block_list.append(np.dot(B,B.T))
        #return block_matrix(block_list)

        return block_matrix.dot(Bmat,block_matrix.transpose(Bmat))


    def GInverse_SVD(self, xyz):
        xyz = xyz.reshape(-1,3)
        # Perform singular value decomposition
        nifty.click()
        loops = 0
        while True:
            try:
                G = self.GMatrix(xyz)
                time_G = nifty.click()
                start=0
                tmpUvecs=[]
                tmpVvecs=[]
                tmpSvecs=[]
                for Gmat in G.matlist:
                    U, s, VT = np.linalg.svd(Gmat)
                    tmpVvecs.append(VT.T)
                    tmpUvecs.append(U.T)
                    tmpSvecs.append(np.diag(s))
                V = block_matrix(tmpVvecs)
                UT = block_matrix(tmpUvecs)
                S = block_matrix(tmpSvecs)
                time_svd = nifty.click()
            except np.linalg.LinAlgError:
                logger.warning("\x1b[1;91m SVD fails, perturbing coordinates and trying again\x1b[0m")
                xyz = xyz + 1e-2*np.random.random(xyz.shape)
                loops += 1
                if loops == 10:
                    raise RuntimeError('SVD failed too many times')
                continue
            break
        print("Build G: %.3f SVD: %.3f" % (time_G, time_svd))

        LargeVals = 0
        
        tmpSinv = []
        for smat in S.matlist:
            sinv = np.zeros_like(smat)
            for ival,value in enumerate(np.diagonal(smat)):
                if np.abs(value) >1e-6:
                    LargeVals += 1
                    sinv[ival,ival] = 1./value
            tmpSinv.append(sinv)
        Sinv = block_matrix(tmpSinv)

        # print "%i atoms; %i/%i singular values are > 1e-6" % (xyz.shape[0], LargeVals, len(S))
        #Inv = multi_dot([V, Sinv, UT])
        tmpInv = []
        for v,sinv,ut in zip(V.matlist,Sinv.matlist,UT.matlist):
            tmpInv.append(np.dot(v,np.dot(sinv,ut)))
        
        return block_matrix(tmpInv)

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1,3)
        nifty.click()
        G = self.GMatrix(xyz)
        time_G = nifty.click()

        matlist=[]
        for Gmat in G.matlist:
            matlist.append(np.linalg.inv(Gmat))
        
        Gt = block_matrix(matlist)
        time_inv = nifty.click()
        #print("G-time: %.3f Inv-time: %.3f" % (time_G, time_inv))

        return Gt

    def calcGrad(self, xyz, gradx):
        #q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx)
        #Gq = multi_dot([Ginv, Bmat, gradx])
        #return Gq
        return block_matrix.dot( Ginv,block_matrix.dot(Bmat,gradx) )

    def makeConstraints(self, xyz, constraints, cvals=None):
        # Add the list of constraints. 
        xyz = xyz.flatten()
        if cvals is None and constraints is not None:
            cvals=[]
            # If coordinates are provided instead of a constraint value, 
            # then calculate the constraint value from the positions.
            # If both are provided, then the coordinates are ignored.
            for c in constraints:
                cvals.append(c.value(xyz))
            if len(constraints) != len(cvals):
                raise RuntimeError("List of constraints should be same length as constraint values")
            for cons, cval in zip(constraints, cvals):
                self.addConstraint(cons, cval, xyz)

    def __repr__(self):
        lines = ["Internal coordinate system (atoms numbered from 1):"]
        typedict = OrderedDict()
        for Internal in self.Internals:
            lines.append(Internal.__repr__())
            if str(type(Internal)) not in typedict:
                typedict[str(type(Internal))] = 1
            else:
                typedict[str(type(Internal))] += 1
        if len(lines) > 200:
            # Print only summary if too many
            lines = []
        for k, v in list(typedict.items()):
            lines.append("%s : %i" % (k, v))
        return '\n'.join(lines)

    def __eq__(self, other):
        answer = True
        for i in self.Internals:
            if i not in other.Internals:
                print("this prim is in p1 but not p2 ",i)
                answer = False
        for i in other.Internals:
            if i not in self.Internals:
                print("this prim is in p2 but not p1",i)
                answer = False
        return answer

    def __ne__(self, other):
        return not self.__eq__(other)

    def update(self, other):
        Changed = False
        for i in self.Internals:
            if i not in other.Internals:
                if hasattr(i, 'inactive'):
                    i.inactive += 1
                else:
                    i.inactive = 0
                if i.inactive == 1:
                    logger.info("Deleting:", i)
                    self.Internals.remove(i)
                    Changed = True
            else:
                i.inactive = 0
        for i in other.Internals:
            if i not in self.Internals:
                logger.info("Adding:  ", i)
                self.Internals.append(i)
                Changed = True
        return Changed

    def join(self, other,bonds_only=False):
        Changed = False
        for i in other.Internals:
            if i not in self.Internals:
                if bonds_only and type(i)!="Distance":
                    pass
                else:
                    #logger.info("Adding:  ", i)
                    print(("Adding ",i))
                    self.Internals.append(i)
                    Changed = True
        return Changed

    def repr_diff(self, other):
        alines = ["-- Added: --"]
        for i in other.Internals:
            if i not in self.Internals:
                alines.append(i.__repr__())
        dlines = ["-- Deleted: --"]
        for i in self.Internals:
            if i not in other.Internals:
                dlines.append(i.__repr__())
        output = []
        if len(alines) > 1:
            output += alines
        if len(dlines) > 1:
            output += dlines
        return '\n'.join(output)

    def resetRotations(self, xyz):
        for Internal in self.Internals:
            if type(Internal) is LinearAngle:
                Internal.reset(xyz)
        for rot in list(self.Rotators.values()):
            rot.reset(xyz)

    def largeRots(self):
        for Internal in self.Internals:
            if type(Internal) is LinearAngle:
                if Internal.stored_dot2 > 0.75:
                    # Linear angle is almost parallel to reference axis
                    return True
            if type(Internal) in [RotationA, RotationB, RotationC]:
                if Internal in self.cPrims:
                    continue
                if Internal.Rotator.stored_norm > 0.9*np.pi:
                    # Molecule has rotated by almost pi
                    return True
                if Internal.Rotator.stored_dot2 > 0.9:
                    # Linear molecule is almost parallel to reference axis
                    return True
        return False

    def calculate(self, xyz):
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.value(xyz))
        return np.array(answer)

    def calculateDegrees(self, xyz):
        answer = []
        for Internal in self.Internals:
            value = Internal.value(xyz)
            if Internal.isAngular:
                value *= 180/np.pi
            answer.append(value)
        return np.array(answer)

    def getRotatorNorms(self):
        rots = []
        for Internal in self.Internals:
            if type(Internal) in [RotationA]:
                rots.append(Internal.Rotator.stored_norm)
        return rots

    def getRotatorDots(self):
        dots = []
        for Internal in self.Internals:
            if type(Internal) in [RotationA]:
                dots.append(Internal.Rotator.stored_dot2)
        return dots

    def printRotations(self, xyz):
        rotNorms = self.getRotatorNorms()
        if len(rotNorms) > 0:
            logger.info("Rotator Norms: ", " ".join(["% .4f" % i for i in rotNorms]))
        rotDots = self.getRotatorDots()
        if len(rotDots) > 0 and np.max(rotDots) > 1e-5:
            logger.info("Rotator Dots : ", " ".join(["% .4f" % i for i in rotDots]))
        linAngs = [ic.value(xyz) for ic in self.Internals if type(ic) is LinearAngle]
        if len(linAngs) > 0:
            logger.info("Linear Angles: ", " ".join(["% .4f" % i for i in linAngs]))

    def derivatives(self, xyz):
        self.calculate(xyz)
        answer = [ p.derivative(xyz) for p in self.Internals]
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        return np.array(answer)

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        Q1 = self.calculate(coord1)
        Q2 = self.calculate(coord2)
        PMDiff = (Q1-Q2)
        for k in range(len(PMDiff)):
            # TODO periodic boundary conditions
            #if self.Internals[k].isPeriodicBoundary:
            #    PlusL = PMdiff[k] + self.boundary
            #    MinsL = PMdiff[k] - self.boundary
            if self.Internals[k].isPeriodic:
                Plus2Pi = PMDiff[k] + 2*np.pi
                Minus2Pi = PMDiff[k] - 2*np.pi
                if np.abs(PMDiff[k]) > np.abs(Plus2Pi):
                    PMDiff[k] = Plus2Pi
                if np.abs(PMDiff[k]) > np.abs(Minus2Pi):
                    PMDiff[k] = Minus2Pi
        return PMDiff

    def GInverse(self, xyz):
        #9/2019 CRA what is the difference in performace/stability for SVD vs regular inverse?

        return self.GInverse_EIG(xyz)
        #return self.GInverse_SVD(xyz)

    def add(self, dof,verbose=False):
        if dof not in self.Internals:
            if verbose:
                print((" adding ",dof))
            self.Internals.append(dof)
            return True
        else:
            return False
    
    def dof_index(self,dof):
        return self.Internals.index(dof)

    def delete(self, dof):
        found=False
        for ii in range(len(self.Internals))[::-1]:
            if dof == self.Internals[ii]:
                del self.Internals[ii]
                found=True
        return found

    def addConstraint(self, cPrim, cVal=None, xyz=None):
        if cVal is None and xyz is None:
            raise RuntimeError('Please provide either cval or xyz')
        if cVal is None:
            # If coordinates are provided instead of a constraint value, 
            # then calculate the constraint value from the positions.
            # If both are provided, then the coordinates are ignored.
            cVal = cPrim.value(xyz)
            print(cVal)
        if cPrim in self.cPrims:
            iPrim = self.cPrims.index(cPrim)
            if np.abs(cVal - self.cVals[iPrim]) > 1e-6:
                logger.info("Updating constraint value to %.4e" % cVal)
            self.cVals[iPrim] = cVal
        else:
            if cPrim not in self.Internals:
                self.Internals.append(cPrim)
            self.cPrims.append(cPrim)
            self.cVals.append(cVal)

    def reorderPrimitives(self):
        # Reorder primitives to be in line with cc's code
        newPrims = []
        for cPrim in self.cPrims:
            newPrims.append(cPrim)

        for typ in [Distance, Angle, LinearAngle, MultiAngle, OutOfPlane, Dihedral, MultiDihedral, CartesianX, CartesianY, CartesianZ, TranslationX, TranslationY, TranslationZ, RotationA, RotationB, RotationC]:
            for p in self.Internals:
                if type(p) is typ and p not in self.cPrims:
                    newPrims.append(p)
        if len(newPrims) != len(self.Internals):
            raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")
        self.Internals = newPrims

        if not self.options['connect']:
            self.reorderPrimsByFrag()
        else:
            # all atoms are considered one "fragment"
            self.block_info = [(1,self.natoms,len(newPrims),'P')]
        

    def newMakePrimitives(self,xyz):
        self.Internals = []
        self.block_info=[]
        # coordinates in Angstrom
        coords = xyz.flatten()
        connect=self.options['connect']
        addcart=self.options['addcart']
        addtr=self.options['addtr']

        print(" Creating block info")
        tmp_block_info=[]
        # get primitive blocks
        for frag in self.fragments:
            nodes = frag.L()
            tmp_block_info.append((nodes[0],nodes[-1]+1,frag,'reg'))
            #TODO can assert blocks are contiguous here
        print(" number of primitive blocks is ",len(self.fragments))
        
        # get hybrid blocks
        for tup in self.hybrid_idx_start_stop:
            # Add primitive Cartesians for each atom in hybrid block
            sa = tup[0]
            ea = tup[1]
            leng = ea-sa
            for atom in range(sa,ea+1):
                tmp_block_info.append((atom,atom+1,None,'hyb'))

        # sort the blocks
        tmp_block_info.sort(key=lambda tup: tup[0])
        #print("block info")
        #print(tmp_block_info)
        print(" Done creating block info,\n Now Making Primitives by block")

        sp=0
        for info in tmp_block_info:
            nprims=0
            if info[-1]=='reg':
                frag = info[2]
                noncov = []
                # Connect all non-bonded fragments together
                if connect:
                    # Make a distance matrix mapping atom pairs to interatomic distances
                    AtomIterator, dxij = self.distance_matrix(xyz,pbc=False)
                    D = {}
                    for i, j in zip(AtomIterator, dxij[0]):
                        assert i[0] < i[1]
                        D[tuple(i)] = j
                    dgraph = nx.Graph()
                    for i in range(self.natoms):
                        dgraph.add_node(i)
                    for k, v in list(D.items()):
                        dgraph.add_edge(k[0], k[1], weight=v)
                    mst = sorted(list(nx.minimum_spanning_edges(dgraph, data=False)))
                    for edge in mst:
                        if edge not in list(self.topology.edges()):
                            print("Adding %s from minimum spanning tree" % str(edge))
                            self.topology.add_edge(edge[0], edge[1])
                            noncov.append(edge)
           
                else:
                    #Add Cart or TR
                    if addcart:
                        for i in range(info[0],info[1]):
                            self.add(CartesianX(i, w=1.0))
                            self.add(CartesianY(i, w=1.0))
                            self.add(CartesianZ(i, w=1.0))
                            nprims+=3
                    elif addtr:
                        nodes=frag.nodes()
                        if len(nodes) >= 2:
                            self.add(TranslationX(nodes, w=np.ones(len(nodes))/len(nodes)))
                            self.add(TranslationY(nodes, w=np.ones(len(nodes))/len(nodes)))
                            self.add(TranslationZ(nodes, w=np.ones(len(nodes))/len(nodes)))
                            sel = xyz.reshape(-1,3)[nodes,:] 
                            sel -= np.mean(sel, axis=0)
                            rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                            self.add(RotationA(nodes, coords, self.Rotators, w=rg))
                            self.add(RotationB(nodes, coords, self.Rotators, w=rg))
                            self.add(RotationC(nodes, coords, self.Rotators, w=rg))
                            nprims+=6
                        else:
                            for j in nodes:
                                self.add(CartesianX(j, w=1.0))
                                self.add(CartesianY(j, w=1.0))
                                self.add(CartesianZ(j, w=1.0))
                                nprims+=3
                # # Build a list of noncovalent distances
                # Add an internal coordinate for all interatomic distances
                for (a, b) in frag.edges():
                    #if a in list(range(info[0],info[1])):
                    if self.add(Distance(a, b)):
                        nprims+=1

                # Add an internal coordinate for all angles
                # This number works best for the iron complex
                LinThre = 0.95
                AngDict = defaultdict(list)
                for b in frag.nodes():
                    for a in frag.neighbors(b):
                        for c in frag.neighbors(b):
                            if a < c:
                                # if (a, c) in self.topology.edges() or (c, a) in self.topology.edges(): continue
                                Ang = Angle(a, b, c)
                                nnc = (min(a, b), max(a, b)) in noncov
                                nnc += (min(b, c), max(b, c)) in noncov
                                # if nnc >= 2: continue
                                # logger.info("LPW: cosine of angle", a, b, c, "is", np.abs(np.cos(Ang.value(coords))))
                                if np.abs(np.cos(Ang.value(coords))) < LinThre:
                                    if self.add(Angle(a, b, c)):
                                        nprims+=1
                                    AngDict[b].append(Ang)
                                elif connect or not addcart:
                                    # logger.info("Adding linear angle")
                                    # Add linear angle IC's
                                    # LPW 2019-02-16: Linear angle ICs work well for "very" linear angles in selfs (e.g. HCCCN)
                                    # but do not work well for "almost" linear angles in noncovalent systems (e.g. H2O6).
                                    # Bringing back old code to use "translations" for the latter case, but should be investigated
                                    # more deeply in the future.
                                    if nnc == 0:
                                        if self.add(LinearAngle(a, b, c, 0)):
                                            nprims+=1
                                        if self.add(LinearAngle(a, b, c, 1)):
                                            nprims+=1
                                    else:
                                        # Unit vector connecting atoms a and c
                                        nac = xyz[c] - xyz[a]
                                        nac /= np.linalg.norm(nac)
                                        # Dot products of this vector with the Cartesian axes
                                        dots = [np.abs(np.dot(ei, nac)) for ei in np.eye(3)]
                                        # Functions for adding Cartesian coordinate
                                        # carts = [CartesianX, CartesianY, CartesianZ]
                                        #print("warning, adding translation, did you mean this?")
                                        trans = [TranslationX, TranslationY, TranslationZ]
                                        w = np.array([-1.0, 2.0, -1.0])
                                        # Add two of the most perpendicular Cartesian coordinates
                                        for i in np.argsort(dots)[:2]:
                                            if self.add(trans[i]([a, b, c], w=w)):
                                                nprims+=1
                                    
                for b in frag.nodes():
                    for a in frag.neighbors(b):
                        for c in frag.neighbors(b):
                            for d in frag.neighbors(b):
                                if a < c < d:
                                    nnc = (min(a, b), max(a, b)) in noncov
                                    nnc += (min(b, c), max(b, c)) in noncov
                                    nnc += (min(b, d), max(b, d)) in noncov
                                    # if nnc >= 1: continue
                                    for i, j, k in sorted(list(itertools.permutations([a, c, d], 3))):
                                        Ang1 = Angle(b,i,j)
                                        Ang2 = Angle(i,j,k)
                                        if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                                        if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                                        if np.abs(np.dot(Ang1.normal_vector(coords), Ang2.normal_vector(coords))) > LinThre:
                                            if self.delete(Angle(i, b, j)):
                                                nprims-=1
                                            if self.add(OutOfPlane(b, i, j, k)):
                                                nprims+=1
                                            break
                                        
                # Find groups of atoms that are in straight lines
                atom_lines = [list(i) for i in frag.edges()]
                while True:
                    # For a line of two atoms (one bond):
                    # AB-AC
                    # AX-AY
                    # i.e. AB is the first one, AC is the second one
                    # AX is the second-to-last one, AY is the last one
                    # AB-AC-...-AX-AY
                    # AB-(AC, AX)-AY
                    atom_lines0 = deepcopy(atom_lines)
                    for aline in atom_lines:
                        # Imagine a line of atoms going like ab-ac-ax-ay.
                        # Our job is to extend the line until there are no more
                        ab = aline[0]
                        ay = aline[-1]
                        for aa in frag.neighbors(ab):
                            if aa not in aline:
                                # If the angle that AA makes with AB and ALL other atoms AC in the line are linear:
                                # Add AA to the front of the list
                                if all([np.abs(np.cos(Angle(aa, ab, ac).value(coords))) > LinThre for ac in aline[1:] if ac != ab]):
                                    aline.insert(0, aa)
                        for az in frag.neighbors(ay):
                            if az not in aline:
                                if all([np.abs(np.cos(Angle(ax, ay, az).value(coords))) > LinThre for ax in aline[:-1] if ax != ay]):
                                    aline.append(az)
                    if atom_lines == atom_lines0: break
                atom_lines_uniq = []
                for i in atom_lines:    # 
                    if tuple(i) not in set(atom_lines_uniq):
                        atom_lines_uniq.append(tuple(i))
                lthree = [l for l in atom_lines_uniq if len(l) > 2]
                # TODO: Perhaps should reduce the times this is printed out in reaction paths
                # if len(lthree) > 0:
                #     print "Lines of three or more atoms:", ', '.join(['-'.join(["%i" % (i+1) for i in l]) for l in lthree])

                # Normal dihedral code
                for aline in atom_lines_uniq:
                    # Go over ALL pairs of atoms in a line
                    for (b, c) in itertools.combinations(aline, 2):
                        if b > c: (b, c) = (c, b)
                        # Go over all neighbors of b
                        for a in frag.neighbors(b):
                            # Go over all neighbors of c
                            for d in frag.neighbors(c):
                                # Make sure the end-atoms are not in the line and not the same as each other
                                if a not in aline and d not in aline and a != d:
                                    nnc = (min(a, b), max(a, b)) in noncov
                                    nnc += (min(b, c), max(b, c)) in noncov
                                    nnc += (min(c, d), max(c, d)) in noncov
                                    # print aline, a, b, c, d
                                    Ang1 = Angle(a,b,c)
                                    Ang2 = Angle(b,c,d)
                                    # Eliminate dihedrals containing angles that are almost linear
                                    # (should be eliminated already)
                                    if np.abs(np.cos(Ang1.value(coords))) > LinThre: continue
                                    if np.abs(np.cos(Ang2.value(coords))) > LinThre: continue
                                    if self.add(Dihedral(a, b, c, d)):
                                        nprims+=1

            else:   # THIS ELSE CORRESPONS TO FRAGMENTS BUILT WITH THE HYBRID REGION
                #self.add(CartesianX(info[0], w=1.0))
                #self.add(CartesianY(info[0], w=1.0))
                #self.add(CartesianZ(info[0], w=1.0))
                nprims=3

            ep = sp+nprims
            self.block_info.append((info[0],info[1],sp,ep))
            sp = ep


        #print(self.Internals)
        for info1,info2 in zip(tmp_block_info,self.block_info):
            if info1[-1]=='hyb':
                #for i in range(info2[2],info2[3]):
                i=info2[2]
                j=i+1
                k=i+2
                #print(" Inserting Cart at elements {} {} {}".format(i,j,k))
                self.Internals.insert(i,CartesianX(info1[0], w=1.0))
                self.Internals.insert(j,CartesianY(info1[0], w=1.0))
                self.Internals.insert(k,CartesianZ(info1[0], w=1.0))

        print(" Done making primitives")
        print(" Made a total of {} primitives".format(len(self.Internals)))
        #print(self.Internals)
        #print(" block info")
        #print(self.block_info)
        print(" num blocks ",len(self.block_info))


        #if len(newPrims) != len(self.Internals):
        #    #print(np.setdiff1d(self.Internals,newPrims))
        #    raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")

        self.clearCache()
        return

    def insert_block_primitives(self,prims,reform_topology):
        '''
        The SE-GSM needs to add primitives, we have to do this carefully because of the blocks
        '''

        return


    def reorderPrimsByFrag(self):
        '''
        Warning this assumes that the fragments aren't intermixed. you shouldn't do that!!!!
        '''

        # these are the subgraphs
        #frags = [m for m in self.fragments]
        newPrims = []

        # Orders the primitives by fragment, also takes into accoutn hybrid fragments (those that don't contain primitives)
        # if it's 'P' then its primitive and the BMatrix uses the derivative
        # if it's 'H' then its hybrid and the BMatrix uses the diagonal 
        #TODO rename variables to reflect current understanding
        #TODO The 'P' and 'H' nomenclature is probably not necessary since the regions are 
        # distinguishable by the number of primitives they contain, 
        # gt 0 in the former and eq 0 in the latter

        #print(" Getting the block information")

        tmp_block_info=[]

        print(" Creating block info")
        # get primitive blocks
        for frag in self.fragments:
            nodes = frag.L()
            tmp_block_info.append((nodes[0],nodes[-1]+1,frag,'reg'))
            #TODO can assert blocks are contiguous here

        # get hybrid blocks
        for tup in self.hybrid_idx_start_stop:
            # Add primitive Cartesians for each atom in hybrid block
            sa = tup[0]
            ea = tup[1]
            leng = ea-sa
            for atom in range(sa,ea+1):
                tmp_block_info.append((atom,atom+1,None,'hyb'))

        # sort the blocks
        tmp_block_info.sort(key=lambda tup: tup[0])

        print(" Done creating block info,\n Now Ordering Primitives by block")

        # Order primitives by block
        # probably faster to just reform the primitives!!!!

        self.block_info=[]
        sp=0

        for info in tmp_block_info:
            nprims=0
            if info[-1]=='reg':
                # TODO OLD
                for p in self.Internals:
                    atoms = p.atoms
                    if all([atom in range(info[0],info[1]) for atom in atoms]):
                        newPrims.append(p)
                        nprims+=1
            else:
                newPrims.append(CartesianX(info[0], w=1.0))
                newPrims.append(CartesianY(info[0], w=1.0))
                newPrims.append(CartesianZ(info[0], w=1.0))
                nprims=3

            ep = sp+nprims
            self.block_info.append((info[0],info[1],sp,ep))
            sp = ep

        #print(" block info")
        #print(self.block_info)
        #print(" Done Ordering prims by block")
        #print("num blocks ",len(self.block_info))

        #if len(newPrims) != len(self.Internals):
        #    #print(np.setdiff1d(self.Internals,newPrims))
        #    raise RuntimeError("Not all internal coordinates have been accounted for. You may need to add something to reorderPrimitives()")
        self.Internals = newPrims
        
        print(self.Internals)
        self.clearCache()
        return

    def getConstraints_from(self, other):
        if other.haveConstraints():
            for cPrim, cVal in zip(other.cPrims, other.cVals):
                self.addConstraint(cPrim, cVal)

    def haveConstraints(self):
        return len(self.cPrims) > 0

    def getConstraintViolation(self, xyz):
        nc = len(self.cPrims)
        maxdiff = 0.0
        for ic, c in enumerate(self.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            current = c.value(xyz)/w
            reference = self.cVals[ic]/w
            diff = (current - reference)
            if c.isPeriodic:
                if np.abs(diff-2*np.pi) < np.abs(diff):
                    diff -= 2*np.pi
                if np.abs(diff+2*np.pi) < np.abs(diff):
                    diff += 2*np.pi
            if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance]:
                factor = 1.
            elif c.isAngular:
                factor = 180.0/np.pi
            if np.abs(diff*factor) > maxdiff:
                maxdiff = np.abs(diff*factor)
        return maxdiff
    
    def printConstraints(self, xyz, thre=1e-5):
        nc = len(self.cPrims)
        out_lines = []
        header = "Constraint                         Current      Target       Diff.\n"
        for ic, c in enumerate(self.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            current = c.value(xyz)/w
            reference = self.cVals[ic]/w
            diff = (current - reference)
            if c.isPeriodic:
                if np.abs(diff-2*np.pi) < np.abs(diff):
                    diff -= 2*np.pi
                if np.abs(diff+2*np.pi) < np.abs(diff):
                    diff += 2*np.pi
            if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance]:
                factor = 1.
            elif c.isAngular:
                factor = 180.0/np.pi
            #if np.abs(diff*factor) > thre:
            out_lines.append("%-30s  % 10.5f  % 10.5f  % 10.5f\n" % (str(c), current*factor, reference*factor, diff*factor))
        if len(out_lines) > 0:
            logger.info(header)
            logger.info('\n'.join(out_lines))
            # if type(c) in [RotationA, RotationB, RotationC]:
            #     print c, c.value(xyz)
            #     logArray(c.x0)

    def getConstraintTargetVals(self):
        nc = len(self.cPrims)
        cNames = []
        cVals = []
        for ic, c in enumerate(self.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            reference = self.cVals[ic]/w
            if type(c) in [TranslationX, TranslationY, TranslationZ, CartesianX, CartesianY, CartesianZ, Distance]:
                factor = 1.
            elif c.isAngular:
                factor = 180.0/np.pi
            cNames.append(str(c))
            cVals.append(reference*factor)
        return(cNames, cVals)

    def guess_hessian(self, coords):
        """
        Build a guess Hessian that roughly follows Schlegel's guidelines. 
        """
        xyzs = coords.reshape(-1,3)
        def covalent(a, b):
            r = np.linalg.norm(xyzs[a]-xyzs[b])
            rcov = self.atoms[a].covalent_radius + self.atoms[b].covalent_radius
            return r/rcov < 1.2
       
        Hdiag = []
        for ic in self.Internals:
            if type(ic) is Distance:
                r = np.linalg.norm(xyzs[ic.a]-xyzs[ic.b]) 
                elem1 = min(self.atoms[ic.a].atomic_num,self.atoms[ic.b].atomic_num)
                elem2 = max(self.atoms[ic.a].atomic_num,self.atoms[ic.b].atomic_num)
                #A = 1.734
                #if elem1 < 3:
                #    if elem2 < 3:
                #        B = -0.244
                #    elif elem2 < 11:
                #        B = 0.352
                #    else:
                #        B = 0.660
                #elif elem1 < 11:
                #    if elem2 < 11:
                #        B = 1.085
                #    else:
                #        B = 1.522
                #else:
                #    B = 2.068
                if covalent(ic.a, ic.b):
                    Hdiag.append(0.35)
                    #Hdiag.append(A/(r-B)**3)
                else:
                    Hdiag.append(0.1)
            elif type(ic) in [Angle, LinearAngle, MultiAngle]:
                if type(ic) in [Angle, LinearAngle]:
                    a = ic.a
                    c = ic.c
                else:
                    a = ic.a[-1]
                    c = ic.c[0]
                if min(self.atoms[a].atomic_num,
                        self.atoms[ic.b].atomic_num,
                        self.atoms[c].atomic_num) < 3:
                    A = 0.160
                else:
                    A = 0.250
                if covalent(a, ic.b) and covalent(ic.b, c):
                    Hdiag.append(A)
                else:
                    Hdiag.append(0.1)
            elif type(ic) in [Dihedral, MultiDihedral]:
                r = np.linalg.norm(xyzs[ic.b]-xyzs[ic.c])
                rcov = self.atoms[ic.b].covalent_radius + self.atoms[ic.c].covalent_radius
                # Hdiag.append(0.1)
                Hdiag.append(0.023)
            elif type(ic) is OutOfPlane:
                r1 = xyzs[ic.b]-xyzs[ic.a]
                r2 = xyzs[ic.c]-xyzs[ic.a]
                r3 = xyzs[ic.d]-xyzs[ic.a]
                d = 1 - np.abs(np.dot(r1,np.cross(r2,r3))/np.linalg.norm(r1)/np.linalg.norm(r2)/np.linalg.norm(r3))
                # Hdiag.append(0.1)
                if covalent(ic.a, ic.b) and covalent(ic.a, ic.c) and covalent(ic.a, ic.d):
                    Hdiag.append(0.045)
                else:
                    Hdiag.append(0.023)
            elif type(ic) in [CartesianX, CartesianY, CartesianZ]:
                Hdiag.append(0.05)
            elif type(ic) in [TranslationX, TranslationY, TranslationZ]:
                Hdiag.append(0.05)
            elif type(ic) in [RotationA, RotationB, RotationC]:
                Hdiag.append(0.05)
            else:
                raise RuntimeError('Failed to build guess Hessian matrix. Make sure all IC types are supported')
        return np.diag(Hdiag)


    #def apply_periodic_boundary(self,xyz,L):
    #    tot=0
    #    new_xyz = np.zeros_like(xyz)
    #    for num_prim in self.nprims_frag:
    #        index = num_prim + tot - 6  # the ICs are ordered [...Tx,Ty,Tz,Ra,Rb,Rc] per frag
    #        prims = self.Internals[index:index+3]
    #        atoms = prims[0].atoms   # all prims should have the same atoms so okay to use 0th
    #        translate = False
    #        # need to check all the atoms of the frag before deciding to translate
    #        for a in atoms:
    #            if any(abs(xyz[a,:]) > L):
    #                translate=True
    #            else:
    #                translate=False
    #                break
    #        # apply translation
    #        for a in atoms:
    #            if translate==True:
    #                for i,x in enumerate(xyz[a,:]):
    #                    if x<-L/2:
    #                        new_xyz[a,i]=x+L
    #                    elif x>=L/2:
    #                        new_xyz[a,i]=x-L
    #                    else:
    #                        new_xyz[a,i]=x
    #            else:
    #                new_xyz[a,:] = xyz[a,:]
    #        tot+=num_prim
    #    return new_xyz

    def second_derivatives(self, xyz):
        self.calculate(xyz)
        answer = []
        for Internal in self.Internals:
            answer.append(Internal.second_derivative(xyz))
        # This array has dimensions:
        # 1) Number of internal coordinates
        # 2) Number of atoms
        # 3) 3
        # 4) Number of atoms
        # 5) 3
        return np.array(answer)


    def get_hybrid_indices(self,xyz):
        '''
        Get the hybrid indices if they exist
        '''

        natoms = len(xyz)

        # need the primitive start and stop indices
        prim_idx_start_stop=[]
        new=True
        for frag in self.fragments:
            nodes=frag.L()
            prim_idx_start_stop.append((nodes[0],nodes[-1]))
        #print("prim start stop")
        #print(prim_idx_start_stop)

        prim_idx =[]
        for info in prim_idx_start_stop:
            prim_idx += list(range(info[0],info[1]+1))
        #print('prim indices')
        #print(prim_idx)

        new_hybrid_indices=list(range(natoms))
        for elem in prim_idx:
            new_hybrid_indices.remove(elem)
        #print('hybrid indices')
        #print(new_hybrid_indices)


        # get the hybrid start and stop indices
        self.hybrid_idx_start_stop=[]
        new=True
        for i in range(natoms+1):
            if i in new_hybrid_indices:
                if new==True:
                    start=i
                    new=False
            else:
                if new==False:
                    end=i-1
                    new=True
                    self.hybrid_idx_start_stop.append((start,end))
        #print(" hybrid start stop")
        #print(self.hybrid_idx_start_stop)


    def append_prim_to_block(self,prim):
        #for info in self.block_info:
        #print(self.block_info)
        total_blocks = len(self.block_info)

        count=0
        for info in self.block_info:
            if info[3]-info[2] != 3:  # this is a hybrid block skipping 
                if all([atom in range(info[0],info[1]) for atom in prim.atoms]):
                    break
            count+=1
        #print(" the prim lives in block {}".format(count))

        # the start and end of the primitives is stored in block info
        # the third element is the end index for that blocks prims
        elem = self.block_info[count][3]

        self.Internals.insert(elem,prim)
        #print(" prims after inserting at elem {}".format(elem))
        #print(self.Internals)

        new_block_info = []
        for i,info in enumerate(self.block_info):
            if i < count:
                # sa,ea,sp,ep --> therefore all sps before count are unaffected
                new_block_info.append((info[0],info[1],info[2],info[3]))
            elif i==count:
                new_block_info.append((info[0],info[1],info[2],info[3]+1))
            else:
                new_block_info.append((info[0],info[1],info[2]+1,info[3]+1))
        #print(new_block_info)
        self.block_info = new_block_info

        return

    def add_union_primitives(self,other):  

        # Can make this faster if only check primitive indices
        # Need the primitive internal coordinates -- not the Cartesian internal coordinates
        print(" Number of primitives before {}".format(len(self.Internals)))
        #print(' block info before')
        #print(self.block_info)

        prim_idx1 =[]
        for count,prim in enumerate(self.Internals):
            if type(prim) not in [CartesianX,CartesianY,CartesianZ]:
                prim_idx1.append(count)

        prim_idx2 =[]
        for count,prim in enumerate(other.Internals):
            if type(prim) not in [CartesianX,CartesianY,CartesianZ]:
                prim_idx2.append(count)

        tmp_internals1 = [self.Internals[i] for i in prim_idx1]
        tmp_internals2 = [other.Internals[i] for i in prim_idx1]

        #for i in other.Internals:
        #    if i not in self.Internals:
        for i in tmp_internals2:
            if i not in tmp_internals1:
                #print("this prim is in p2 but not p1",i)
                print("Adding prim {} that is in Other to Internals".format(i))
                self.append_prim_to_block(i)

        #print(self.Internals)
        #print(len(self.Internals))
        print(" Number of primitives after {}".format(len(self.Internals)))
        #print(' block info after')
        #print(self.block_info)


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

    test_prims=False
    if test_prims:
        # testing Cartesian
        prim = CartesianX(0,w=1.0)
        print(xyz[0,:])
        print(xyz[0,:].shape)
        der = prim.derivative(xyz[0,:])
        print(der)
        print(der.shape)

        # testing Translation
        print("testing translation")
        i = list(range(10,16))
        prim = TranslationX(i,w=np.ones(len(i))/len(i))
        print(xyz[10:16,:])
        print(xyz[10:16,:].shape)
        der = prim.derivative(xyz[10:16,:],start_idx=10)
        print(der)
        print(der.shape)


        print("testing rotation")
        Rotators = OrderedDict()
        i = list(range(10,16))
        sel = xyz.reshape(-1,3)[i,:] 
        sel -= np.mean(sel, axis=0)
        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
        rotation = RotationA(i, xyz, Rotators, w=rg)

        der1 = prim.derivative(xyz[10:16,:],start_idx=10)
        print(der1)
        print(der1.shape)
        der2 = prim.derivative(xyz)
        print(der2)
        print(der2.shape)


        print('testing distance')
        prim = Distance(10,11)
        print(prim)
        der1 = prim.derivative(xyz[10:16,:],start_idx=10)
        print(der1)
        print(der1.shape)
        der2 = prim.derivative(xyz)
        print(der2)
        print(der2.shape)


        print('testing angle')
        prim = Angle(10,11,14)
        print(prim)
        der1 = prim.derivative(xyz[10:16,:],start_idx=10)
        print(der1)
        print(der1.shape)
        der2 = prim.derivative(xyz)
        print(der2)
        print(der2.shape)


        print('testing dihedral')
        prim = Dihedral(12,10,11,14)
        print(prim)
        der1 = prim.derivative(xyz[10:16,:],start_idx=10)
        print(der1)
        print(der1.shape)
        der2 = prim.derivative(xyz)
        print(der2)
        print(der2.shape)


    hybrid_indices = list(range(0,5)) + list(range(21,26))
    #hybrid_indices = list(range(0,74)) + list(range(3348, 3358))
    #hybrid_indices = None
    #print(hybrid_indices)
    #with open('frozen.txt') as f:
    #    hybrid_indices = f.read().splitlines()
    #hybrid_indices = [int(x) for x in hybrid_indices]
    #print(hybrid_indices)

    print(" Making topology")
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


    print(" Making prim")
    p1 = PrimitiveInternalCoordinates.from_options(
            xyz=xyz1,
            atoms=atoms,
            addtr = True,
            topology=G1,
            #extra_kwargs = {  'hybrid_indices' : hybrid_indices},
            ) 

    p2 = PrimitiveInternalCoordinates.from_options(
            xyz=xyz2,
            atoms=atoms,
            addtr = True,
            topology=G1,
            #extra_kwargs = {  'hybrid_indices' : hybrid_indices},
            ) 

    #print("Does p1 equal p2? ", p1==p2)

    #print(" Adding Angle 7-6-11 to p1")
    #angle = Angle(6,5,10)
    #print(angle)
    #p1.append_prim_to_block(angle)

    #print(p.calculate(xyz))
    #print(len(p.Internals))

    p1.add_union_primitives(p2)
