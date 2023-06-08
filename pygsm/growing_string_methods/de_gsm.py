from __future__ import print_function
# standard library imports
import sys
from os import path
from utilities import manage_xyz, nifty

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))


from pygsm.molecule import Molecule

try:
    from .main_gsm import MainGSM
except:
    from main_gsm import MainGSM


class DE_GSM(MainGSM):

    def __init__(
            self,
            options,
    ):

        super(DE_GSM, self).__init__(options)

        print(" Assuming primitives are union!")
        print(" number of primitives is", self.nodes[0].num_primitives)

    # TODO Change rtype to a more meaningful argument name
    def go_gsm(self, max_iters=50, opt_steps=3, rtype=2):
        """
        rtype=2 Find and Climb TS,
        1 Climb with no exact find,
        0 turning of climbing image and TS search
        """
        self.set_V0()

        if not self.isRestarted:
            if self.growth_direction == 0:
                self.add_GSM_nodes(2)
            elif self.growth_direction == 1:
                self.add_GSM_nodeR(1)
            elif self.growth_direction == 2:
                self.add_GSM_nodeP(1)

            # Grow String
            self.grow_string(max_iters=max_iters, max_opt_steps=opt_steps)
            nifty.printcool("Done Growing the String!!!")
            self.done_growing = True

            # nifty.printcool("initial ic_reparam")
            self.reparameterize()
            self.xyz_writer('grown_string_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)

        # Can check for intermediate at beginning but not doing that now.
        # else:
        #    if self.has_intermediate(self.noise):
        #        nifty.printcool(f" WARNING THIS REACTION HAS AN INTERMEDIATE within noise {self.noise}, opting out")
        #        try:
        #            self.optimize_string(max_iter=3,opt_steps=opt_steps,rtype=0)
        #        except Exception as error:
        #            print(" Done optimizing 3 times, checking if intermediate still exists")
        #            if self.has_intermediate(self.noise):
        #                self.tscontinue=False

        if self.tscontinue:
            try:
                self.optimize_string(max_iter=max_iters, opt_steps=opt_steps, rtype=rtype)
            except Exception as error:
                if str(error) == "Ran out of iterations":
                    print(error)
                    self.end_early = True
                else:
                    print(error)
                    self.end_early = True
        else:
            print("Exiting early")
            self.end_early = True

        filename = "opt_converged_{:03d}.xyz".format(self.ID)
        print(" Printing string to " + filename)
        self.xyz_writer(filename, self.geometries, self.energies, self.gradrmss, self.dEs)
        print("Finished GSM!")

        return

    def add_GSM_nodes(self, newnodes=1):
        if self.current_nnodes+newnodes > self.nnodes:
            print("Adding too many nodes, cannot add_GSM_node")
        sign = -1
        for i in range(newnodes):
            sign *= -1
            if sign == 1:
                self.add_GSM_nodeR()
            else:
                self.add_GSM_nodeP()

    def set_active(self, nR, nP):
        # print(" Here is active:",self.active)
        if nR != nP and self.growth_direction == 0:
            print((" setting active nodes to %i and %i" % (nR, nP)))
        elif self.growth_direction == 1:
            print((" setting active node to %i " % nR))
        elif self.growth_direction == 2:
            print((" setting active node to %i " % nP))
        else:
            print((" setting active node to %i " % nR))

        for i in range(self.nnodes):
            if self.nodes[i] is not None:
                self.optimizer[i].conv_grms = self.CONV_TOL*2.
        self.optimizer[nR].conv_grms = self.options['ADD_NODE_TOL']
        self.optimizer[nP].conv_grms = self.options['ADD_NODE_TOL']
        print(" conv_tol of node %d is %.4f" % (nR, self.optimizer[nR].conv_grms))
        print(" conv_tol of node %d is %.4f" % (nP, self.optimizer[nP].conv_grms))
        self.active[nR] = True
        self.active[nP] = True
        if self.growth_direction == 1:
            self.active[nP] = False
        if self.growth_direction == 2:
            self.active[nR] = False
        # print(" Here is new active:",self.active)

    def check_if_grown(self):
        '''
        Check if the string is grown
        Returns True if grown 
        '''

        return self.current_nnodes == self.nnodes

    def grow_nodes(self):
        '''
        Grow nodes
        '''

        if self.nodes[self.nR-1].gradrms < self.gaddmax and self.growth_direction != 2:
            if self.nodes[self.nR] is None:
                self.add_GSM_nodeR()
                print(" getting energy for node %d: %5.4f" % (self.nR-1, self.nodes[self.nR-1].energy - self.nodes[0].V0))
        if self.nodes[self.nnodes-self.nP].gradrms < self.gaddmax and self.growth_direction != 1:
            if self.nodes[-self.nP-1] is None:
                self.add_GSM_nodeP()
                print(" getting energy for node %d: %5.4f" % (self.nnodes-self.nP, self.nodes[-self.nP].energy - self.nodes[0].V0))
        return

    def make_tan_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent-2):
            if nlist[2*n] not in param_list:
                param_list.append(nlist[2*n])
        return param_list

    def make_move_list(self):
        ncurrent, nlist = self.make_difference_node_list()
        param_list = []
        for n in range(ncurrent):
            if nlist[2*n+1] not in param_list:
                param_list.append(nlist[2*n+1])
        return param_list

    def make_difference_node_list(self):
        '''
        Returns ncurrent and a list of indices that can be iterated over to produce
        tangents for the string pathway.
        '''
        # TODO: THis can probably be done more succinctly using a list of tuples
        ncurrent = 0
        nlist = [0]*(2*self.nnodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1

        for n in range(self.nnodes-self.nP+1, self.nnodes):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n-1
            ncurrent += 1

        nlist[2*ncurrent] = self.nR - 1
        nlist[2*ncurrent+1] = self.nnodes - self.nP

        if False:
            nlist[2*ncurrent+1] = self.nR - 2  # for isMAP_SE

        # TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent] += 1
        # if self.nP == 0: nlist[2*ncurrent+1] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes - self.nP
        nlist[2*ncurrent+1] = self.nR-1
        # #TODO is this actually used?
        # if self.nR == 0: nlist[2*ncurrent+1] += 1
        # if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        return ncurrent, nlist

    def set_V0(self):
        self.nodes[0].V0 = self.nodes[0].energy

        # TODO should be actual gradient
        self.nodes[0].gradrms = 0.
        if self.growth_direction != 1:
            self.nodes[-1].gradrms = 0.
            print(" Energy of the end points are %4.3f, %4.3f" % (self.nodes[0].energy, self.nodes[-1].energy))
            print(" relative E %4.3f, %4.3f" % (0.0, self.nodes[-1].energy-self.nodes[0].energy))
        else:
            print(" Energy of end points are %4.3f " % self.nodes[0].energy)
            # self.nodes[-1].energy = self.nodes[0].energy
            # self.nodes[-1].gradrms = 0.


if __name__ == '__main__':
    from level_of_theories.dummy_lot import Dummy
    from potential_energy_surfaces.pes import PES
    from coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates, Topology
    from optimizers import eigenvector_follow

    geoms = manage_xyz.read_molden_geoms('../growing_string_methods/opt_converged_000.xyz')
    lot = Dummy.from_options(geom=geoms[0])

    pes = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)
    atom_symbols = manage_xyz.get_atoms(geoms[0])

    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geoms[0])
    xyz2 = manage_xyz.xyz_to_np(geoms[-1])

    top1 = Topology.build_topology(
        xyz1,
        atoms,
    )

    # find union bonds
    xyz2 = manage_xyz.xyz_to_np(geoms[-1])
    top2 = Topology.build_topology(
        xyz2,
        atoms,
    )

    # Add bonds to top1 that are present in top2
    # It's not clear if we should form the topology so the bonds
    # are the same since this might affect the Primitives of the xyz1 (slightly)
    # Later we stil need to form the union of bonds, angles and torsions
    # However, I think this is important, the way its formulated, for identifiyin
    # the number of fragments and blocks, which is used in hybrid TRIC.
    for bond in top2.edges():
        if bond in top1.edges:
            pass
        elif (bond[1], bond[0]) in top1.edges():
            pass
        else:
            print(" Adding bond {} to top1".format(bond))
            if bond[0] > bond[1]:
                top1.add_edge(bond[0], bond[1])
            else:
                top1.add_edge(bond[1], bond[0])

    addtr = True
    connect = addcart = False
    p1 = PrimitiveInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top1,
    )

    p2 = PrimitiveInternalCoordinates.from_options(
        xyz=xyz2,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        topology=top1,  # Use the topology of 1 because we fixed it above
    )

    p1.add_union_primitives(p2)

    coord_obj1 = DelocalizedInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )

    optimizer = eigenvector_follow.from_options()

    reactant = Molecule.from_options(
        geom=geoms[0],
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=True,
    )
    product = Molecule.copy_from_options(
        reactant,
        xyz=xyz2,
        new_node_id=len(geoms)-1,
        copy_wavefunction=False,
    )

    gsm = DE_GSM.from_options(
        reactant=reactant,
        product=product,
        nnodes=len(geoms),
        optimizer=optimizer,
    )
    gsm.restart_from_geoms(geoms)
    gsm.find = True
    energies = [0.0, 0.31894656200893223, 1.0911851973214652, 2.435532565781614, 5.29310522499145, 20.137409660528647, -30.240701181493932, -39.4328096016543, -41.09534010407515, -44.007087726989994, -45.82765071728499]

    for e, m in zip(energies, gsm.nodes):
        m.PES.lot._Energies[(1, 0)] = lot.Energy(e, 'kcal/mol')
        m.PES.lot.hasRanForCurrentCoords = True

    #print(gsm.nodes[-1].PES.lot.get_energy(xyz2,1,0))
    print(gsm.nodes[-1].PES.lot.Energies)

    print(gsm.energies)

    print('reparameterizing')
    gsm.geodesic_reparam()

    manage_xyz.write_xyzs('rep.xyz', gsm.geometries)
