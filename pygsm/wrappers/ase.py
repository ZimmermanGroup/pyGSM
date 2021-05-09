"""
Wrapper to the code using ASE

"""

import os

try:
    from ase import Atoms
    import ase.io
except ModuleNotFoundError:
    pass

from pygsm.coordinate_systems.delocalized_coordinates import DelocalizedInternalCoordinates
from pygsm.coordinate_systems.primitive_internals import PrimitiveInternalCoordinates
from pygsm.coordinate_systems.topology import Topology
from pygsm.growing_string_methods import DE_GSM
from pygsm.level_of_theories.ase import ASELoT
from pygsm.optimizers.eigenvector_follow import eigenvector_follow
from pygsm.optimizers.lbfgs import lbfgs
from pygsm.potential_energy_surfaces import PES
from pygsm.utilities import nifty
from pygsm.utilities.elements import ElementData
from pygsm.wrappers.molecule import Molecule
from .main import post_processing, cleanup_scratch


def minimal_wrapper_de_gsm(
        atoms_reactant: Atoms,
        atoms_product: Atoms,
        calculator,
        fixed_reactant=False,
        fixed_product=False,
):
    # LOT
    # 'EST_Package'
    # 'nproc': args.nproc,
    # 'states': None,

    # PES
    # pes_type = "PES"
    # 'PES_type': args.pes_type,
    # 'adiabatic_index': args.adiabatic_index,
    # 'multiplicity': args.multiplicity,
    # 'FORCE_FILE': args.FORCE_FILE,
    # 'RESTRAINT_FILE': args.RESTRAINT_FILE,
    # 'FORCE': None,
    # 'RESTRAINTS': None,

    # optimizer
    optimizer_method = "eigenvector_follow"  # OR "lbfgs"
    line_search = 'NoLineSearch'  # OR: 'backtrack'
    only_climb = True
    # 'opt_print_level': args.opt_print_level,
    step_size_cap = 0.1  # DMAX in the other wrapper

    # molecule
    coordinate_type = "TRIC"
    # 'hybrid_coord_idx_file': args.hybrid_coord_idx_file,
    # 'frozen_coord_idx_file': args.frozen_coord_idx_file,
    # 'prim_idx_file': args.prim_idx_file,

    # GSM
    # gsm_type = "DE_GSM"  # SE_GSM, SE_Cross
    num_nodes = 11  # 20 for SE-GSM
    # 'isomers_file': args.isomers,   # driving coordinates, this is a file
    add_node_tol = 0.1  # convergence for adding new nodes
    conv_tol = 0.0005  # Convergence tolerance for optimizing nodes
    conv_Ediff = 100.  # Energy difference convergence of optimization.
    # 'conv_dE': args.conv_dE,
    conv_gmax = 100.  # Max grad rms threshold
    # 'BDIST_RATIO': args.BDIST_RATIO,
    # 'DQMAG_MAX': args.DQMAG_MAX,
    # 'growth_direction': args.growth_direction,
    ID = 0
    # 'gsm_print_level': args.gsm_print_level,
    max_gsm_iterations = 100
    max_opt_steps = 3  # 20 for SE-GSM
    # 'use_multiprocessing': args.use_multiprocessing,
    # 'sigma': args.sigma,

    nifty.printcool('Parsed GSM')

    # LOT
    lot = ASELoT.from_options(calculator,
                              geom=[[x.symbol, *x.position] for x in atoms_reactant],
                              ID=ID)

    # PES
    pes_obj = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)

    # Build the topology
    nifty.printcool("Building the topologies")
    element_table = ElementData()
    elements = [element_table.from_symbol(sym) for sym in atoms_reactant.get_chemical_symbols()]

    topology_reactant = Topology.build_topology(
        xyz=atoms_reactant.get_positions(),
        atoms=elements
    )
    topology_product = Topology.build_topology(
        xyz=atoms_product.get_positions(),
        atoms=elements
    )

    # Union of bonds
    # debated if needed here or not
    for bond in topology_product.edges():
        if bond in topology_reactant.edges() or (bond[1], bond[0]) in topology_reactant.edges():
            continue
        print(" Adding bond {} to reactant topology".format(bond))
        if bond[0] > bond[1]:
            topology_reactant.add_edge(bond[0], bond[1])
        else:
            topology_reactant.add_edge(bond[1], bond[0])

    # primitive internal coordinates
    nifty.printcool("Building Primitive Internal Coordinates")

    prim_reactant = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        topology=topology_reactant,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    prim_product = PrimitiveInternalCoordinates.from_options(
        xyz=atoms_product.get_positions(),
        atoms=elements,
        topology=topology_product,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
    )

    # add product coords to reactant coords
    prim_reactant.add_union_primitives(prim_product)

    # Delocalised internal coordinates
    nifty.printcool("Building Delocalized Internal Coordinates")
    deloc_coords_reactant = DelocalizedInternalCoordinates.from_options(
        xyz=atoms_reactant.get_positions(),
        atoms=elements,
        connect=coordinate_type == "DLC",
        addtr=coordinate_type == "TRIC",
        addcart=coordinate_type == "HDLC",
        primitives=prim_reactant
    )

    # Molecules
    nifty.printcool("Building the reactant object with {}".format(coordinate_type))
    from_hessian = optimizer_method == "eigenvector_follow"

    molecule_reactant = Molecule.from_options(
        geom=[[x.symbol, *x.position] for x in atoms_reactant],
        PES=pes_obj,
        coord_obj=deloc_coords_reactant,
        Form_Hessian=from_hessian
    )

    molecule_product = Molecule.copy_from_options(
        molecule_reactant,
        xyz=atoms_product.get_positions(),
        new_node_id=num_nodes - 1,
        copy_wavefunction=False
    )

    # optimizer
    nifty.printcool("Building the Optimizer object")
    opt_options = dict(print_level=1,
                       Linesearch=line_search,
                       update_hess_in_bg=not (only_climb or optimizer_method == "lbfgs"),
                       conv_Ediff=conv_Ediff,
                       conv_gmax=conv_gmax,
                       DMAX=step_size_cap,
                       opt_climb=only_climb)
    if optimizer_method == "eigenvector_follow":
        optimizer_object = eigenvector_follow.from_options(**opt_options)
    elif optimizer_method == "lbfgs":
        optimizer_object = lbfgs.from_options(**opt_options)
    else:
        raise NotImplementedError

    # GSM
    nifty.printcool("Building the GSM object")
    gsm = DE_GSM.from_options(
        reactant=molecule_reactant,
        product=molecule_product,
        nnodes=num_nodes,
        CONV_TOL=conv_tol,
        CONV_gmax=conv_gmax,
        CONV_Ediff=conv_Ediff,
        ADD_NODE_TOL=add_node_tol,
        growth_direction=0,  # I am not sure how this works
        optimizer=optimizer_object,
        ID=ID,
        print_level=1,
        mp_cores=1,  # parallelism not tested yet with the ASE calculators
        interp_method="DLC",
    )

    # optimize reactant and product if needed
    if not fixed_reactant:
        nifty.printcool("REACTANT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", "0")
        optimizer_object.optimize(
            molecule=molecule_reactant,
            refE=molecule_reactant.energy,
            opt_steps=100,
            path=path
        )
    if not fixed_product:
        nifty.printcool("PRODUCT GEOMETRY NOT FIXED!!! OPTIMIZING")
        path = os.path.join(os.getcwd(), 'scratch', f"{ID:03}", str(num_nodes - 1))
        optimizer_object.optimize(
            molecule=molecule_product,
            refE=molecule_product.energy,
            opt_steps=100,
            path=path
        )

    # set 'rtype' as in main one (???)
    if only_climb:
        rtype = 1
    # elif no_climb:
    #     rtype = 0
    else:
        rtype = 2

    # do GSM
    nifty.printcool("Main GSM Calculation")
    gsm.go_gsm(max_gsm_iterations, max_opt_steps, rtype=rtype)

    # write the results into an extended xyz file
    string_ase, ts_ase = gsm_to_ase_atoms(gsm)
    ase.io.write(f"opt_converged_{gsm.ID:03d}_ase.xyz", string_ase)
    ase.io.write(f'TSnode_{gsm.ID}.xyz', string_ase)

    # post processing taken from the main wrapper, plots as well
    post_processing(gsm, have_TS=True)

    # cleanup
    cleanup_scratch(gsm.ID)


def gsm_to_ase_atoms(gsm: DE_GSM):
    # string
    frames = []
    for energy, geom in zip(gsm.energies, gsm.geometries):
        at = Atoms(symbols=[x[0] for x in geom], positions=[x[1:4] for x in geom])
        at.info["energy"] = energy
        frames.append(at)

    # TS
    ts_geom = gsm.nodes[gsm.TSnode].geometry
    ts_atoms = Atoms(symbols=[x[0] for x in ts_geom], positions=[x[1:4] for x in ts_geom])

    return frames, ts_atoms
