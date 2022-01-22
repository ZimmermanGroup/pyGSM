from __future__ import print_function
# standard library imports
import sys
import os
from os import path

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from coordinate_systems import Distance, Angle, Dihedral, OutOfPlane
from utilities import nifty, options, block_matrix
from wrappers import Molecule
from utilities.manage_xyz import write_molden_geoms

# third party
import numpy as np
import multiprocessing as mp
from collections import Counter
from copy import copy
from itertools import chain


def worker(arg):
   obj, methname = arg[:2]
   return getattr(obj, methname)(*arg[2:])


#######################################################################################
#### This class contains the main constructor, object properties and staticmethods ####
#######################################################################################


# TODO interpolate is still sloppy. It shouldn't create a new molecule node itself
# but should create the xyz. GSM should create the new molecule based off that xyz.
# TODO nconstraints in ic_reparam and write_iters is irrelevant


class GSM(object):

    from utilities import units

    @staticmethod
    def default_options():
        if hasattr(GSM, '_default_options'):
            return GSM._default_options.copy()

        opt = options.Options()

        opt.add_option(
            key='reactant',
            required=True,
            # allowed_types=[Molecule,wrappers.Molecule],
            doc='Molecule object as the initial reactant structure')

        opt.add_option(
            key='product',
            required=False,
            # allowed_types=[Molecule,wrappers.Molecule],
            doc='Molecule object for the product structure (not required for single-ended methods.')

        opt.add_option(
            key='nnodes',
            required=False,
            value=1,
            allowed_types=[int],
            doc="number of string nodes"
        )

        opt.add_option(
            key='optimizer',
            required=True,
            doc='Optimzer object  to use e.g. eigenvector_follow, conjugate_gradient,etc. \
                        most of the default options are okay for here since GSM will change them anyway',
        )

        opt.add_option(
            key='driving_coords',
            required=False,
            value=[],
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify atoms\
                 indexed at 1')

        opt.add_option(
            key='CONV_TOL',
            value=0.0005,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_gmax',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_Ediff',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='CONV_dE',
            value=0.5,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold'
        )

        opt.add_option(
            key='ADD_NODE_TOL',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
            key="growth_direction",
            value=0,
            required=False,
            doc="how to grow string,0=Normal,1=from reactant"
        )

        opt.add_option(
            key="DQMAG_MAX",
            value=0.8,
            required=False,
            doc="max step along tangent direction for SSM"
        )
        opt.add_option(
            key="DQMAG_MIN",
            value=0.2,
            required=False,
            doc=""
        )

        opt.add_option(
            key='print_level',
            value=1,
            required=False
        )

        opt.add_option(
            key='xyz_writer',
            value=write_molden_geoms,
            required=False,
            doc='Function to be used to format and write XYZ files',
        )

        opt.add_option(
            key='mp_cores',
            value=1,
            doc='multiprocessing cores for parallel programming. Use this with caution.',
        )

        opt.add_option(
            key="BDIST_RATIO",
            value=0.5,
            required=False,
            doc="SE-Crossing uses this \
                        bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
                        to be considered grown.",
        )

        opt.add_option(
            key='ID',
            value=0,
            required=False,
            doc='A number for identification of Strings'
        )

        opt.add_option(
            key='interp_method',
            value='DLC',
                allowed_values=['Geodesic', 'DLC'],
            required=False,
            doc='Which reparameterization method to use',
        )

        opt.add_option(
            key='noise',
            value=100.0,
            allowed_types=[float],
            required=False,
            doc='Noise to check for intermediate',
        )

        GSM._default_options = opt
        return GSM._default_options.copy()

    @classmethod
    def from_options(cls, **kwargs):
        return cls(cls.default_options().set_values(kwargs))

    @classmethod
    def copy_from_options(cls, gsm_obj, reactant, product):
        new_gsm = cls.from_options(gsm_obj.options.copy().set_values({'reactant': reactant, 'product': product}))
        return new_gsm

    def __init__(
            self,
            options,
    ):
        """ Constructor """
        self.options = options

        os.system('mkdir -p scratch')

        # Cache attributes
        self.nnodes = self.options['nnodes']
        self.nodes = [None]*self.nnodes
        self.nodes[0] = self.options['reactant']
        self.nodes[-1] = self.options['product']
        self.driving_coords = self.options['driving_coords']
        self.growth_direction = self.options['growth_direction']
        self.isRestarted = False
        self.DQMAG_MAX = self.options['DQMAG_MAX']
        self.DQMAG_MIN = self.options['DQMAG_MIN']
        self.BDIST_RATIO = self.options['BDIST_RATIO']
        self.ID = self.options['ID']
        self.optimizer = []
        self.interp_method = self.options['interp_method']
        self.CONV_TOL = self.options['CONV_TOL']
        self.noise = self.options['noise']
        self.mp_cores = self.options['mp_cores']
        self.xyz_writer = self.options['xyz_writer']

        optimizer = options['optimizer']
        for count in range(self.nnodes):
            self.optimizer.append(optimizer.__class__(optimizer.options.copy()))
        self.print_level = options['print_level']

        # Set initial values
        self.current_nnodes = 2
        self.nR = 1
        self.nP = 1
        self.climb = False
        self.find = False
        self.ts_exsteps = 3  # multiplier for ts node
        self.n0 = 1  # something to do with added nodes? "first node along current block"
        self.end_early = False
        self.tscontinue = True  # whether to continue with TS opt or not
        self.found_ts = False
        self.rn3m6 = np.sqrt(3.*self.nodes[0].natoms-6.)
        self.gaddmax = self.options['ADD_NODE_TOL']  # self.options['ADD_NODE_TOL']/self.rn3m6;
        print(" gaddmax:", self.gaddmax)
        self.ictan = [None]*self.nnodes
        self.active = [False] * self.nnodes
        self.climber = False  # is this string a climber?
        self.finder = False   # is this string a finder?
        self.done_growing = False
        self.nclimb = 0
        self.nhessreset = 10  # are these used??? TODO
        self.hessrcount = 0   # are these used?!  TODO
        self.hess_counter = 0   # it is probably good to reset the hessian
        self.newclimbscale = 2.
        self.TS_E_0 = None
        self.dE_iter = 100.  # change in max TS node

        self.nopt_intermediate = 0     # might be a duplicate of endearly_counter
        self.flag_intermediate = False
        self.endearly_counter = 0  # Find the intermediate x time
        self.pot_min = []
        self.ran_out = False   # if it ran out of iterations

        self.newic = Molecule.copy_from_options(self.nodes[0])  # newic object is used for coordinate transformations

    @property
    def TSnode(self):
        '''
        The current node with maximum energy
        '''
        # Treat GSM with penalty a little different since penalty will increase energy based on energy
        # differences, which might not be great for Climbing Image
        if self.__class__.__name__ != "SE_Cross" and self.nodes[0].PES.__class__.__name__ == "Penalty_PES":
            energies = np.asarray([0.]*self.nnodes)
            for i, node in enumerate(self.nodes):
                if node is not None:
                    energies[i] = (node.PES.PES1.energy + node.PES.PES2.energy)/2.
            return int(np.argmax(energies))
        else:
            # make sure TS is not zero or last node
            return int(np.argmax(self.energies[1:self.nnodes-1])+1)

    @property
    def emax(self):
        return self.energies[self.TSnode]

    @property
    def npeaks(self):
        '''
        '''
        minnodes = []
        maxnodes = []
        energies = self.energies
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[self.nnodes-1] < energies[self.nnodes-2]:
            minnodes.append(self.nnodes-1)
        for n in range(self.n0, self.nnodes-1):
            if energies[n+1] > energies[n]:
                if energies[n] < energies[n-1]:
                    minnodes.append(n)
            if energies[n+1] < energies[n]:
                if energies[n] > energies[n-1]:
                    maxnodes.append(n)

        return len(maxnodes)

    @property
    def energies(self):
        '''
        Energies of string
        '''
        E = []
        for ico in self.nodes:
            if ico is not None:
                E.append(ico.energy - self.nodes[0].energy)
        return E

    @energies.setter
    def energies(self, list_of_E):
        '''
        setter for energies
        '''
        self.E = list_of_E

    @property
    def geometries(self):
        geoms = []
        for ico in self.nodes:
            if ico is not None:
                geoms.append(ico.geometry)
        return geoms

    @property
    def gradrmss(self):
        self._gradrmss = []
        for ico in self.nodes:
            if ico is not None:
                self._gradrmss.append(ico.gradrms)
        return self._gradrmss

    @property
    def dEs(self):
        self._dEs = []
        for ico in self.nodes:
            if ico is not None:
                self._dEs.append(ico.difference_energy)
        return self._dEs

    @property
    def ictan(self):
        return self._ictan

    @ictan.setter
    def ictan(self, value):
        self._ictan = value

    @property
    def dqmaga(self):
        return self._dqmaga

    @dqmaga.setter
    def dqmaga(self, value):
        self._dqmaga = value

    @staticmethod
    def add_xyz_along_tangent(
            xyz1,
            constraints,
            step,
            coord_obj,
    ):
        dq0 = step*constraints
        new_xyz = coord_obj.newCartesian(xyz1, dq0)

        return new_xyz

    @staticmethod
    def add_node(
            nodeR,
            nodeP,
            stepsize,
            node_id,
            **kwargs
    ):
        '''
        Add a node between  nodeR and nodeP or if nodeP is none use driving coordinate to add new node
        '''

        # get driving coord
        driving_coords = kwargs.get('driving_coords', None)
        DQMAG_MAX = kwargs.get('DQMAG_MAX', 0.8)
        DQMAG_MIN = kwargs.get('DQMAG_MIN', 0.2)

        if nodeP is None:

            if driving_coords is None:
                raise RuntimeError("You didn't supply a driving coordinate and product node is None!")

            BDISTMIN = 0.05
            ictan, bdist = GSM.get_tangent(nodeR, None, driving_coords=driving_coords)

            if bdist < BDISTMIN:
                print("bdist too small %.3f" % bdist)
                return None
            new_node = Molecule.copy_from_options(nodeR, new_node_id=node_id)
            new_node.update_coordinate_basis(constraints=ictan)
            constraint = new_node.constraints[:, 0]
            sign = -1.

            dqmag_scale = 1.5
            minmax = DQMAG_MAX - DQMAG_MIN
            a = bdist/dqmag_scale
            if a > 1.:
                a = 1.
            dqmag = sign*(DQMAG_MIN+minmax*a)
            if dqmag > DQMAG_MAX:
                dqmag = DQMAG_MAX
            print(" dqmag: %4.3f from bdist: %4.3f" % (dqmag, bdist))

            dq0 = dqmag*constraint
            print(" dq0[constraint]: %1.3f" % dqmag)

            new_node.update_xyz(dq0)
            new_node.bdist = bdist

        else:
            ictan, _ = GSM.get_tangent(nodeR, nodeP)
            nodeR.update_coordinate_basis(constraints=ictan)
            constraint = nodeR.constraints[:, 0]
            dqmag = np.linalg.norm(ictan)
            print(" dqmag: %1.3f" % dqmag)
            # sign=-1
            sign = 1.
            dqmag *= (sign*stepsize)
            print(" scaled dqmag: %1.3f" % dqmag)

            dq0 = dqmag*constraint
            old_xyz = nodeR.xyz.copy()
            new_xyz = nodeR.coord_obj.newCartesian(old_xyz, dq0)
            new_node = Molecule.copy_from_options(MoleculeA=nodeR, xyz=new_xyz, new_node_id=node_id)

        return new_node

    @staticmethod
    def interpolate_xyz(nodeR, nodeP, stepsize):
        '''
        Interpolate between two nodes
        '''
        ictan, _ = GSM.get_tangent(nodeR, nodeP)
        Vecs = nodeR.update_coordinate_basis(constraints=ictan)
        constraint = nodeR.constraints[:, 0]
        prim_constraint = block_matrix.dot(Vecs, constraint)
        dqmag = np.dot(prim_constraint.T, ictan)
        print(" dqmag: %1.3f" % dqmag)
        # sign=-1
        sign = 1.
        dqmag *= (sign*stepsize)
        print(" scaled dqmag: %1.3f" % dqmag)

        dq0 = dqmag*constraint
        old_xyz = nodeR.xyz.copy()
        new_xyz = nodeR.coord_obj.newCartesian(old_xyz, dq0)

        return new_xyz

    @staticmethod
    def interpolate(start_node, end_node, num_interp):
        '''
    
        '''
        nifty.printcool(" interpolate")

        num_nodes = num_interp + 2
        nodes = [None]*(num_nodes)
        nodes[0] = start_node
        nodes[-1] = end_node
        sign = 1
        nR = 1
        nP = 1
        nn = nR + nP

        for n in range(num_interp):
            if num_nodes - nn > 1:
                stepsize = 1./float(num_nodes - nn)
            else:
                stepsize = 0.5
            if sign == 1:
                iR = nR-1
                iP = num_nodes - nP
                iN = nR
                nodes[nR] = GSM.add_node(nodes[iR], nodes[iP], stepsize, iN)
                if nodes[nR] is None:
                    raise RuntimeError

                # print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                nR += 1
                nn += 1

            else:
                n1 = num_nodes - nP
                n2 = n1 - 1
                n3 = nR - 1
                nodes[n2] = GSM.add_node(nodes[n1], nodes[n3], stepsize, n2)
                if nodes[n2] is None:
                    raise RuntimeError
                # print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                nP += 1
                nn += 1
            sign *= -1

        return nodes

    @staticmethod
    def get_tangent_xyz(xyz1, xyz2, prim_coords):
        PMDiff = np.zeros(len(prim_coords))
        for k, prim in enumerate(prim_coords):
            if type(prim) is Distance:
                PMDiff[k] = 2.5 * prim.calcDiff(xyz2, xyz1)
            else:
                PMDiff[k] = prim.calcDiff(xyz2, xyz1)
        return np.reshape(PMDiff, (-1, 1))

    @staticmethod
    def get_tangent(node1, node2, print_level=1, **kwargs):
        '''
        Get internal coordinate tangent between two nodes, assumes they have unique IDs
        '''

        if node2 is not None and node1.node_id != node2.node_id:
            print(" getting tangent from between %i %i pointing towards %i" % (node2.node_id, node1.node_id, node2.node_id))
            assert node2 != None, 'node n2 is None'

            PMDiff = np.zeros(node2.num_primitives)
            for k, prim in enumerate(node2.primitive_internal_coordinates):
                if type(prim) is Distance:
                    PMDiff[k] = 2.5 * prim.calcDiff(node2.xyz, node1.xyz)
                else:
                    PMDiff[k] = prim.calcDiff(node2.xyz, node1.xyz)

            return np.reshape(PMDiff, (-1, 1)), None
        else:
            print(" getting tangent from node ", node1.node_id)

            driving_coords = kwargs.get('driving_coords', None)
            assert driving_coords is not None, " Driving coord is None!"

            c = Counter(elem[0] for elem in driving_coords)
            nadds = c['ADD']
            nbreaks = c['BREAK']
            nangles = c['nangles']
            ntorsions = c['ntorsions']

            ictan = np.zeros((node1.num_primitives, 1), dtype=float)
            # breakdq = 0.3
            bdist = 0.0
            atoms = node1.atoms
            xyz = node1.xyz.copy()

            for i in driving_coords:
                if "ADD" in i:

                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]

                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        # TODO why not just use the covalent radii?
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2.8
                    elif len(i) == 4:
                        d0 = i[3]
                    current_d = bond.value(xyz)

                    # TODO don't set tangent if value is too small
                    ictan[prim_idx] = -1*(d0-current_d)
                    # if nbreaks>0:
                    #    ictan[prim_idx] *= 2
                    # => calc bdist <=
                    if current_d > d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print(" bond %s target (less than): %4.3f current d: %4.3f diff: %4.3f " % ((i[1], i[2]), d0, current_d, ictan[prim_idx]))

                elif "BREAK" in i:
                    # order indices to avoid duplicate bonds
                    if i[1] < i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]
                    bond = Distance(index[0], index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Distance')
                    if len(i) == 3:
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)
                    elif len(i) == 4:
                        d0 = i[3]

                    current_d = bond.value(xyz)
                    ictan[prim_idx] = -1*(d0-current_d)

                    # => calc bdist <=
                    if current_d < d0:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])

                    if print_level > 0:
                        print(" bond %s target (greater than): %4.3f, current d: %4.3f diff: %4.3f " % ((i[1], i[2]), d0, current_d, ictan[prim_idx]))
                elif "ANGLE" in i:

                    if i[1] < i[3]:
                        index = [i[1]-1, i[2]-1, i[3]-1]
                    else:
                        index = [i[3]-1, i[2]-1, i[1]-1]
                    angle = Angle(index[0], index[1], index[2])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Angle')
                    anglet = i[4]
                    ang_value = angle.value(xyz)
                    ang_diff = anglet*np.pi/180. - ang_value
                    # print(" angle: %s is index %i " %(angle,ang_idx))
                    if print_level > 0:
                        print((" anglev: %4.3f align to %4.3f diff(rad): %4.3f" % (ang_value, anglet, ang_diff)))
                    ictan[prim_idx] = -ang_diff
                    # TODO need to come up with an adist
                    # if abs(ang_diff)>0.1:
                    #    bdist+=ictan[ICoord1.BObj.nbonds+ang_idx]*ictan[ICoord1.BObj.nbonds+ang_idx]
                elif "TORSION" in i:

                    if i[1] < i[4]:
                        index = [i[1]-1, i[2]-1, i[3]-1, i[4]-1]
                    else:
                        index = [i[4]-1, i[3]-1, i[2]-1, i[1]-1]
                    torsion = Dihedral(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'Dihedral')
                    tort = i[5]
                    torv = torsion.value(xyz)
                    tor_diff = tort - torv*180./np.pi
                    if tor_diff > 180.:
                        tor_diff -= 360.
                    elif tor_diff < -180.:
                        tor_diff += 360.
                    ictan[prim_idx] = -tor_diff*np.pi/180.

                    if tor_diff*np.pi/180. > 0.1 or tor_diff*np.pi/180. < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print((" current torv: %4.3f align to %4.3f diff(deg): %4.3f" % (torv*180./np.pi, tort, tor_diff)))

                elif "OOP" in i:
                    index = [i[1]-1, i[2]-1, i[3]-1, i[4]-1]
                    oop = OutOfPlane(index[0], index[1], index[2], index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(index, 'OutOfPlane')
                    oopt = i[5]
                    oopv = oop.value(xyz)
                    oop_diff = oopt - oopv*180./np.pi
                    if oop_diff > 180.:
                        oop_diff -= 360.
                    elif oop_diff < -180.:
                        oop_diff += 360.
                    ictan[prim_idx] = -oop_diff*np.pi/180.

                    if oop_diff*np.pi/180. > 0.1 or oop_diff*np.pi/180. < 0.1:
                        bdist += np.dot(ictan[prim_idx], ictan[prim_idx])
                    if print_level > 0:
                        print((" current oopv: %4.3f align to %4.3f diff(deg): %4.3f" % (oopv*180./np.pi, oopt, oop_diff)))

            bdist = np.sqrt(bdist)
            if np.all(ictan == 0.0):
                raise RuntimeError(" All elements are zero")
            return ictan, bdist

    @staticmethod
    def get_tangents(nodes, n0=0, print_level=0):
        '''
        Get the normalized internal coordinate tangents and magnitudes between all nodes
        '''
        nnodes = len(nodes)
        dqmaga = [0.]*nnodes
        ictan = [[]]*nnodes

        for n in range(n0+1, nnodes):
            # print "getting tangent between %i %i" % (n,n-1)
            assert nodes[n] is not None, "n is bad"
            assert nodes[n-1] is not None, "n-1 is bad"
            ictan[n] = GSM.get_tangent_xyz(nodes[n-1].xyz, nodes[n].xyz, nodes[0].primitive_internal_coordinates)

            dqmaga[n] = 0.
            # ictan0= np.copy(ictan[n])
            dqmaga[n] = np.linalg.norm(ictan[n])

            ictan[n] /= dqmaga[n]

            # NOTE:
            # vanilla GSM has a strange metric for distance
            # no longer following 7/1/2020
            # constraint = self.newic.constraints[:,0]
            # just a fancy way to get the normalized tangent vector
            # prim_constraint = block_matrix.dot(Vecs,constraint)
            # for prim in self.newic.primitive_internal_coordinates:
            #    if type(prim) is Distance:
            #        index = self.newic.coord_obj.Prims.dof_index(prim)
            #        prim_constraint[index] *= 2.5
            # dqmaga[n] = float(np.dot(prim_constraint.T,ictan0))
            # dqmaga[n] = float(np.sqrt(dqmaga[n]))
            if dqmaga[n] < 0.:
                raise RuntimeError

        # TEMPORORARY parallel idea
        # ictan = [0.]
        # ictan += [ Process(target=get_tangent,args=(n,)) for n in range(n0+1,self.nnodes)]
        # dqmaga = [ Process(target=get_dqmag,args=(n,ictan[n])) for n in range(n0+1,self.nnodes)]

        if print_level > 1:
            print('------------printing ictan[:]-------------')
            for n in range(n0+1, nnodes):
                print("ictan[%i]" % n)
                print(ictan[n].T)
        if print_level > 0:
            print('------------printing dqmaga---------------')
            for n in range(n0+1, nnodes):
                print(" {:5.4}".format(dqmaga[n]), end='')
                if (n) % 5 == 0:
                    print()
            print()
        return ictan, dqmaga

    @staticmethod
    def get_three_way_tangents(nodes, energies, find=True, n0=0, print_level=0):
        '''
        Calculates internal coordinate tangent with a three-way tangent at TS node
        '''
        nnodes = len(nodes)
        ictan = [[]]*nnodes
        dqmaga = [0.]*nnodes
        # TSnode = np.argmax(energies[1:nnodes-1])+1
        TSnode = np.argmax(energies)   # allow for the possibility of TS node to be endpoints?

        last_node_max = (TSnode == nnodes-1)
        first_node_max = (TSnode == 0)
        if first_node_max or last_node_max:
            print("*********** This will cause a range error in the following for loop *********")
            print("** Setting the middle of the string to be TS node to get proper directions **")
            TSnode = nnodes//2

        for n in range(n0, nnodes):
            do3 = False
            print('getting tan[{' + str(n) + '}]')
            if n < TSnode:
                # The order is very important here
                # the way it should be ;(
                intic_n = n+1
                newic_n = n

                # old way
                # intic_n = n
                # newic_n = n+1

            elif n > TSnode:
                # The order is very important here
                intic_n = n
                newic_n = n-1
            else:
                do3 = True
                newic_n = n
                intic_n = n+1
                int2ic_n = n-1

            if do3:
                if first_node_max or last_node_max:
                    t1, _ = GSM.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = GSM.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    print(" done 3 way tangent")
                    ictan0 = t1 + t2
                else:
                    f1 = 0.
                    dE1 = abs(energies[n+1]-energies[n])
                    dE2 = abs(energies[n] - energies[n-1])
                    dEmax = max(dE1, dE2)
                    dEmin = min(dE1, dE2)
                    if energies[n+1] > energies[n-1]:
                        f1 = dEmax/(dEmax+dEmin+0.00000001)
                    else:
                        f1 = 1 - dEmax/(dEmax+dEmin+0.00000001)

                    print(' 3 way tangent ({}): f1:{:3.2}'.format(n, f1))

                    t1, _ = GSM.get_tangent(nodes[intic_n], nodes[newic_n])
                    t2, _ = GSM.get_tangent(nodes[newic_n], nodes[int2ic_n])
                    print(" done 3 way tangent")
                    ictan0 = f1*t1 + (1.-f1)*t2
            else:
                ictan0, _ = GSM.get_tangent(nodes[newic_n], nodes[intic_n])

            ictan[n] = ictan0/np.linalg.norm(ictan0)
            dqmaga[n] = np.linalg.norm(ictan0)

        return ictan, dqmaga

    @staticmethod
    def ic_reparam(nodes, energies, climbing=False, ic_reparam_steps=8, print_level=1, NUM_CORE=1, MAXRE=0.25):
        '''
        Reparameterizes the string using Delocalizedin internal coordinatesusing three-way tangents at the TS node
        Only pushes nodes outwards during reparameterization because otherwise too many things change.
            Be careful, however, if the path is allup or alldown then this can cause
        Parameters
        ----------
        nodes : list of molecule objects
        energies : list of energies in kcal/mol
        ic_reparam_steps : int max number of reparameterization steps
        print_level : int verbosity
        '''
        nifty.printcool("reparametrizing string nodes")

        nnodes = len(nodes)
        rpart = np.zeros(nnodes)
        for n in range(1, nnodes):
            rpart[n] = 1./(nnodes-1)
        deltadqs = np.zeros(nnodes)
        TSnode = np.argmax(energies)
        disprms = 100
        if ((TSnode == nnodes-1) or (TSnode == 0)) and climbing:
            raise RuntimeError(" TS node shouldn't be the first or last node")

        ideal_progress_gained = np.zeros(nnodes)
        if climbing:
            for n in range(1, TSnode):
                ideal_progress_gained[n] = 1./(TSnode)
            for n in range(TSnode+1, nnodes):
                ideal_progress_gained[n] = 1./(nnodes-TSnode-1)
            ideal_progress_gained[TSnode] = 0.
        else:
            for n in range(1, nnodes):
                ideal_progress_gained[n] = 1./(nnodes-1)

        for i in range(ic_reparam_steps):

            ictan, dqmaga = GSM.get_tangents(nodes)
            totaldqmag = np.sum(dqmaga)

            if climbing:
                progress = np.zeros(nnodes)
                progress_gained = np.zeros(nnodes)
                h1dqmag = np.sum(dqmaga[:TSnode+1])
                h2dqmag = np.sum(dqmaga[TSnode+1:nnodes])
                if print_level > 0:
                    print(" h1dqmag, h2dqmag: %3.2f %3.2f" % (h1dqmag, h2dqmag))
                progress_gained[:TSnode] = dqmaga[:TSnode]/h1dqmag
                progress_gained[TSnode+1:] = dqmaga[TSnode+1:]/h2dqmag
                progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
                progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
            else:
                progress = np.cumsum(dqmaga)/totaldqmag
                progress_gained = dqmaga/totaldqmag

            if i == 0:
                orig_dqmaga = copy(dqmaga)
                orig_progress_gained = copy(progress_gained)

            if climbing:
                difference = np.zeros(nnodes)
                for n in range(TSnode):
                    difference[n] = ideal_progress_gained[n] - progress_gained[n]
                    deltadqs[n] = difference[n]*h1dqmag
                for n in range(TSnode+1, nnodes):
                    difference[n] = ideal_progress_gained[n] - progress_gained[n]
                    deltadqs[n] = difference[n]*h2dqmag
            else:
                difference = ideal_progress_gained - progress_gained
                deltadqs = difference*totaldqmag

            if print_level > 1:
                print(" ideal progress gained per step", end=' ')
                for n in range(nnodes):
                    print(" step [{}]: {:1.3f}".format(n, ideal_progress_gained[n]), end=' ')
                print()
                print(" path progress                 ", end=' ')
                for n in range(nnodes):
                    print(" step [{}]: {:1.3f}".format(n, progress_gained[n]), end=' ')
                print()
                print(" difference                    ", end=' ')
                for n in range(nnodes):
                    print(" step [{}]: {:1.3f}".format(n, difference[n]), end=' ')
                print()
                print(" deltadqs                      ", end=' ')
                for n in range(nnodes):
                    print(" step [{}]: {:1.3f}".format(n, deltadqs[n]), end=' ')
                print()

            # disprms = np.linalg.norm(deltadqs)/np.sqrt(nnodes-1)
            disprms = np.linalg.norm(deltadqs)/np.sqrt(nnodes-1)
            print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 0.02:
                break

            # Move nodes
            if climbing:
                deltadqs[TSnode-2] -= deltadqs[TSnode-1]
                deltadqs[nnodes-2] -= deltadqs[nnodes-1]
                for n in range(1, nnodes-1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n])*MAXRE
                for n in range(TSnode-1):
                    deltadqs[n+1] += deltadqs[n]
                for n in range(TSnode+1, nnodes-2):
                    deltadqs[n+1] += deltadqs[n]
                for n in range(nnodes):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n])*MAXRE

                if NUM_CORE > 1:

                    # 5/14/2021 TS node fucks this up?!
                    tans = [ictan[n] if deltadqs[n] < 0 else ictan[n+1] for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1))]  # + [ ictan[n] if deltadqs[n]<0 else ictan[n+1] for n in range(TSnode+1,nnodes-1)]
                    pool = mp.Pool(NUM_CORE)
                    Vecs = pool.map(worker, ((nodes[0].coord_obj, "build_dlc", node.xyz, tan) for node, tan in zip(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1], tans)))
                    pool.close()
                    pool.join()
                    for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1]):
                        node.coord_basis = Vecs[n]

                    # move the positions
                    dqs = [deltadqs[n]*nodes[n].constraints[:, 0] for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1))]
                    pool = mp.Pool(NUM_CORE)
                    newXyzs = pool.map(worker, ((node.coord_obj, "newCartesian", node.xyz, dq) for node, dq in zip(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1], dqs)))
                    pool.close()
                    pool.join()
                    for n, node in enumerate(nodes[1:TSnode] + nodes[TSnode+1:nnodes-1]):
                        node.xyz = newXyzs[n]
                else:
                    for n in chain(range(1, TSnode), range(TSnode+1, nnodes-1)):
                        if deltadqs[n] < 0:
                            # print(f" Moving node {n} along tan[{n}] this much {deltadqs[n]}")
                            print(" Moving node {} along tan[{}] this much {}".format(n, n, deltadqs[n]))
                            nodes[n].update_coordinate_basis(ictan[n])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n]*constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
                        elif deltadqs[n] > 0:
                            print(" Moving node {} along tan[{}] this much {}".format(n, n+1, deltadqs[n]))
                            nodes[n].update_coordinate_basis(ictan[n+1])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n]*constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
            else:
                # e.g 11-2 = 9, deltadq[9] -= deltadqs[10]
                deltadqs[nnodes-2] -= deltadqs[nnodes-1]
                for n in range(1, nnodes-1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n])*MAXRE
                for n in range(1, nnodes-2):
                    deltadqs[n+1] += deltadqs[n]
                for n in range(1, nnodes-1):
                    if abs(deltadqs[n]) > MAXRE:
                        deltadqs[n] = np.sign(deltadqs[n])*MAXRE

                if NUM_CORE > 1:
                    # Update the coordinate basis
                    tans = [ictan[n] if deltadqs[n] < 0 else ictan[n+1] for n in range(1, nnodes-1)]
                    pool = mp.Pool(NUM_CORE)
                    Vecs = pool.map(worker, ((nodes[0].coord_obj, "build_dlc", node.xyz, tan) for node, tan in zip(nodes[1:nnodes-1], tans)))
                    pool.close()
                    pool.join()
                    for n, node in enumerate(nodes[1:nnodes-1]):
                        node.coord_basis = Vecs[n]
                    # move the positions
                    dqs = [deltadqs[n]*nodes[n].constraints[:, 0] for n in range(1, nnodes-1)]
                    pool = mp.Pool(NUM_CORE)
                    newXyzs = pool.map(worker, ((node.coord_obj, "newCartesian", node.xyz, dq) for node, dq in zip(nodes[1:nnodes-1], dqs)))
                    pool.close()
                    pool.join()
                    for n, node in enumerate(nodes[1:nnodes-1]):
                        node.xyz = newXyzs[n]
                else:
                    for n in range(1, nnodes-1):
                        if deltadqs[n] < 0:
                            # print(f" Moving node {n} along tan[{n}] this much {deltadqs[n]}")
                            print(" Moving node {} along tan[{}] this much {}".format(n, n, deltadqs[n]))
                            nodes[n].update_coordinate_basis(ictan[n])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n]*constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))
                        elif deltadqs[n] > 0:
                            print(" Moving node {} along tan[{}] this much {}".format(n, n+1, deltadqs[n]))
                            nodes[n].update_coordinate_basis(ictan[n+1])
                            constraint = nodes[n].constraints[:, 0]
                            dq = deltadqs[n]*constraint
                            nodes[n].update_xyz(dq, verbose=(print_level > 1))

        if climbing:
            ictan, dqmaga = GSM.get_tangents(nodes)
            h1dqmag = np.sum(dqmaga[:TSnode+1])
            h2dqmag = np.sum(dqmaga[TSnode+1:nnodes])
            if print_level > 0:
                print(" h1dqmag, h2dqmag: %3.2f %3.2f" % (h1dqmag, h2dqmag))
            progress_gained[:TSnode] = dqmaga[:TSnode]/h1dqmag
            progress_gained[TSnode+1:] = dqmaga[TSnode+1:]/h2dqmag
            progress[:TSnode] = np.cumsum(progress_gained[:TSnode])
            progress[TSnode:] = np.cumsum(progress_gained[TSnode:])
        else:
            ictan, dqmaga = GSM.get_tangents(nodes)
            totaldqmag = np.sum(dqmaga)
            progress = np.cumsum(dqmaga)/totaldqmag
            progress_gained = dqmaga/totaldqmag
        print()
        if print_level > 0:
            print(" ideal progress gained per step", end=' ')
            for n in range(nnodes):
                print(" step [{}]: {:1.3f}".format(n, ideal_progress_gained[n]), end=' ')
            print()
            print(" original path progress        ", end=' ')
            for n in range(nnodes):
                print(" step [{}]: {:1.3f}".format(n, orig_progress_gained[n]), end=' ')
            print()
            print(" reparameterized path progress ", end=' ')
            for n in range(nnodes):
                print(" step [{}]: {:1.3f}".format(n, progress_gained[n]), end=' ')
            print()

        print(" spacings (begin ic_reparam, steps", end=' ')
        for n in range(nnodes):
            print(" {:1.2}".format(orig_dqmaga[n]), end=' ')
        print()
        print(" spacings (end ic_reparam, steps: {}/{}):".format(i+1, ic_reparam_steps), end=' ')
        for n in range(nnodes):
            print(" {:1.2}".format(dqmaga[n]), end=' ')
        print("\n  disprms: {:1.3}".format(disprms))

        return

    # TODO move to string utils or delete altogether
    #def get_current_rotation(self,frag,a1,a2):
    #    '''
    #    calculate current rotation for single-ended nodes
    #    '''
    #
    #    # Get the information on fragment to rotate
    #    sa,ea,sp,ep = self.nodes[0].coord_obj.Prims.prim_only_block_info[frag]
    #
    #    theta = 0.
    #    # Haven't added any nodes yet
    #    if self.nR==1:
    #        return theta

    #    for n in range(1,self.nR):
    #        xyz_frag = self.nodes[n].xyz[sa:ea].copy()
    #        axis = self.nodes[n].xyz[a2] - self.nodes[n].xyz[a1]
    #        axis /= np.linalg.norm(axis)
    #
    #        # only want the fragment of interest
    #        reference_xyz = self.nodes[n-1].xyz.copy()

    #        # Turn off
    #        ref_axis = reference_xyz[a2] - reference_xyz[a1]
    #        ref_axis /= np.linalg.norm(ref_axis)

    #        # ALIGN previous and current node to get rotation around axis of rotation
    #        #print(' Rotating reference axis to current axis')
    #        I = np.eye(3)
    #        v = np.cross(ref_axis,axis)
    #        if v.all()==0.:
    #            print('Rotation is identity')
    #            R=I
    #        else:
    #            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    #            c = np.dot(ref_axis,axis)
    #            s = np.linalg.norm(v)
    #            R = I + vx + np.dot(vx,vx) * (1. - c)/(s**2)
    #        new_ref_axis = np.dot(ref_axis,R.T)
    #        #print(' overlap of ref-axis and axis (should be 1.) %1.2f' % np.dot(new_ref_axis,axis))
    #        new_ref_xyz = np.dot(reference_xyz,R.T)

    #
    #        # Calculate dtheta
    #        ca = self.nodes[n].primitive_internal_coordinates[sp+3]
    #        cb = self.nodes[n].primitive_internal_coordinates[sp+4]
    #        cc = self.nodes[n].primitive_internal_coordinates[sp+5]
    #        dv12_a = ca.calcDiff(self.nodes[n].xyz,new_ref_xyz)
    #        dv12_b = cb.calcDiff(self.nodes[n].xyz,new_ref_xyz)
    #        dv12_c = cc.calcDiff(self.nodes[n].xyz,new_ref_xyz)
    #        dv12 = np.array([dv12_a,dv12_b,dv12_c])
    #        #print(dv12)
    #        dtheta = np.linalg.norm(dv12)  #?
    #
    #        dtheta = dtheta + np.pi % (2*np.pi) - np.pi
    #        theta += dtheta

    #    theta = theta/ca.w
    #    angle = theta * 180./np.pi
    #    print(angle)

    #    return theta

    @staticmethod
    def calc_optimization_metrics(nodes):
        '''
        '''

        nnodes = len(nodes)
        rn3m6 = np.sqrt(3*nodes[0].natoms-6)
        totalgrad = 0.0
        gradrms = 0.0
        sum_gradrms = 0.0
        for i, ico in enumerate(nodes[1:nnodes-1]):
            if ico != None:
                print(" node: {:02d} gradrms: {:.6f}".format(i, float(ico.gradrms)), end='')
                if i % 5 == 0:
                    print()
                totalgrad += ico.gradrms*rn3m6
                gradrms += ico.gradrms*ico.gradrms
                sum_gradrms += ico.gradrms
        print('')
        # TODO wrong for growth
        gradrms = np.sqrt(gradrms/(nnodes-2))
        return totalgrad, gradrms, sum_gradrms
