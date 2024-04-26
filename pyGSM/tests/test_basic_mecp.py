import pytest

from pyGSM.coordinate_systems.delocalized_coordinates import (
    DelocalizedInternalCoordinates,
)
from pyGSM.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pyGSM.coordinate_systems.topology import Topology
from pyGSM.level_of_theories.xtb_lot import xTB_lot
from pyGSM.molecule.molecule import Molecule
from pyGSM.optimizers import eigenvector_follow
from pyGSM.potential_energy_surfaces.penalty_pes import Penalty_PES
from pyGSM.potential_energy_surfaces.pes import PES
from pyGSM.utilities import elements, manage_xyz, nifty


def test_basic_penalty_opt():
    geom = manage_xyz.read_xyzs('pyGSM/data/diels_alder.xyz')[0]

    coupling_states = []

    lot1 = xTB_lot.from_options(
        ID=0,
        states=[(1, 0)],
        gradient_states=[0],
        coupling_states=coupling_states,
        geom=geom,
    )

    lot2 = xTB_lot.from_options(
        ID=1,
        states=[(1, 0)],
        gradient_states=[0],
        coupling_states=coupling_states,
        geom=geom,
    )

    pes1 = PES.from_options(
        lot=lot1,
        multiplicity=1,
        ad_idx=0,
    )

    pes2 = PES.from_options(
        lot=lot2,
        multiplicity=1,
        ad_idx=0,
    )

    pes = Penalty_PES(PES1=pes1, PES2=pes2, lot=lot1)

    nifty.printcool('Building the topology')
    atom_symbols = manage_xyz.get_atoms(geom)
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geom)
    top1 = Topology.build_topology(
        xyz1,
        atoms,
        bondlistfile=None,
    )

    nifty.printcool('Building Primitive Internal Coordinates')
    connect = False
    addtr = True
    addcart = False
    p1 = PrimitiveInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        connect=connect,
        addtr=addtr,
        addcart=addcart,
        topology=top1,
    )

    nifty.printcool('Building Delocalized Internal Coordinates')
    coord_obj1 = DelocalizedInternalCoordinates.from_options(
        xyz=xyz1,
        atoms=atoms,
        addtr=addtr,
        addcart=addcart,
        connect=connect,
        primitives=p1,
    )

    nifty.printcool('Building the molecule')

    Form_Hessian = True
    initial = Molecule.from_options(
        geom=geom,
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=Form_Hessian,
    )

    optimizer = eigenvector_follow.from_options(
        Linesearch='backtrack',  # a step size algorithm
        OPTTHRESH=0.0005,  # The gradrms threshold, this is generally easy to reach for large systems
        DMAX=0.01,  # The initial max step size, will be adjusted if optimizer is doing well. Max is 0.5
        conv_Ediff=0.1,  # convergence of difference energy
        conv_dE=0.1,  # convergence of energy difference between optimization steps
        conv_gmax=0.005,  # convergence of max gradient
        opt_cross=True,  # use difference energy criteria to determine if you are at crossing
    )

    print(' MECP optimization')
    geoms, energies = optimizer.optimize(
        molecule=initial,
        refE=initial.energy,
        opt_steps=150,  # The max number of optimization steps, use a small number until you have your final sigma
        verbose=True,
        opt_type='UNCONSTRAINED',
        xyzframerate=1,
    )

    print(f'{energies = }')
    assert energies[-1] == pytest.approx(-1.1495296961093118)
    print('Finished!')

    # print('Final energy is {:5.4f}'.format(initial.energy))
