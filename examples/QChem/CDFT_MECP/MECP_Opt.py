# MECP Code from Khoi -> Kevin -> Josh & Taveechai
# MECP Optimization for
# State 1 = Ground State, State 2 = SET State

import argparse
import importlib
import os

import matplotlib as mpl

from pyGSM.coordinate_systems import (
    DelocalizedInternalCoordinates,
    PrimitiveInternalCoordinates,
    Topology,
)
from pyGSM.molecule import Molecule
from pyGSM.optimizers import eigenvector_follow
from pyGSM.potential_energy_surfaces import PES, Penalty_PES
from pyGSM.utilities import elements, manage_xyz, nifty

mpl.use('Agg')


# --Job Definitions--#
def main():
    try:
        os.environ['QCSCRATCH'] = os.environ['SLURM_LOCAL_SCRATCH']
    except:
        os.environ['QCSCRATCH'] = './'  # for debugging

    charge = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('-xyzfile', required=True, type=str)
    parser.add_argument('-coordinate_type', type=str, default='TRIC', required=False)
    parser.add_argument('-nproc', default=1, type=int, required=False)

    args = parser.parse_args()
    print('Using {} processors\n'.format(args.nproc))
    inpfileq = {
        'xyzfile': args.xyzfile,
        'EST_Package': 'QChem',
        'states': [0, 1],
        # PES
        'adiabatic_index': [0, 1],
        'multiplicity': [1, 1],
        'FORCE': None,
        'RESTRAINTS': None,
        # molecule
        'coordinate_type': args.coordinate_type,
    }

    nifty.printcool_dictionary(inpfileq, title='Using GSM Keys : Values')

    # LOT
    nifty.printcool('Build the QChem LOT object')
    est_package = importlib.import_module(
        'level_of_theories.' + inpfileq['EST_Package'].lower()
    )
    lot_class = getattr(est_package, inpfileq['EST_Package'])

    geoms = manage_xyz.read_xyzs(inpfileq['xyzfile'])

    inpfileq['states'] = [
        (int(m), int(s))
        for m, s in zip(inpfileq['multiplicity'], inpfileq['adiabatic_index'])
    ]
    do_coupling = True
    coupling_states = []

    multiplicity_1 = 5
    multiplicity_2 = 5

    lot1 = lot_class.from_options(
        ID=0,
        lot_inp_file='ground',
        states=[(multiplicity_1, 0)],
        gradient_states=[0],
        coupling_states=coupling_states,
        geom=geoms[0],
        nproc=args.nproc,
        charge=charge,
        do_coupling=do_coupling,
    )

    lot2 = lot_class.from_options(
        ID=1,
        lot_inp_file='set',
        states=[(multiplicity_2, 0)],
        gradient_states=[0],
        coupling_states=coupling_states,
        geom=geoms[0],
        nproc=args.nproc,
        charge=charge,
        do_coupling=do_coupling,
    )

    pes1 = PES.from_options(
        lot=lot1,
        multiplicity=multiplicity_1,  # JOSH
        ad_idx=0,
        FORCE=inpfileq['FORCE'],
        RESTRAINTS=inpfileq['RESTRAINTS'],
    )

    pes2 = PES.from_options(
        lot=lot2,
        multiplicity=multiplicity_2,  # JOSH
        ad_idx=0,
        FORCE=inpfileq['FORCE'],
        RESTRAINTS=inpfileq['RESTRAINTS'],
    )

    pes = Penalty_PES(PES1=pes1, PES2=pes2, lot=lot1, sigma=10.0, alpha=1.0)  # JOSH

    ###====###

    # Molecule
    nifty.printcool(
        'Building the reactant object with {}'.format(inpfileq['coordinate_type'])
    )
    Form_Hessian = True

    hybrid_indices = None
    # frozen_indices = [0, 1, 6, 10, 11, 22, 44, 46, 49, 51, 53, 54, 55]
    # frozen_indices = [0, 1, 2]  # JOSH - for test example
    prim_indices = None

    nifty.printcool('Building the topology')
    atom_symbols = manage_xyz.get_atoms(geoms[0])
    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]
    xyz1 = manage_xyz.xyz_to_np(geoms[0])
    top1 = Topology.build_topology(
        xyz1,
        atoms,
        hybrid_indices=hybrid_indices,
        prim_idx_start_stop=prim_indices,
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
    initial = Molecule.from_options(
        geom=geoms[0],
        PES=pes,
        coord_obj=coord_obj1,
        Form_Hessian=Form_Hessian,
        # frozen_atoms=frozen_indices,
    )

    optimizer = eigenvector_follow.from_options(
        Linesearch='backtrack',  # a step size algorithm
        OPTTHRESH=0.001,  # The gradrms threshold, this is generally easy to reach for large systems
        DMAX=0.01,  # The initial max step size, will be adjusted if optimizer is doing well. Max is 0.5
        conv_Ediff=1.0,  # convergence of difference energy
        conv_dE=0.1,  # convergence of energy difference between optimization steps
        conv_gmax=0.005,  # convergence of max gradient
        opt_cross=True,  # use difference energy criteria to determine if you are at crossing
    )

    print(' Optimizing last node')
    geoms, energies = optimizer.optimize(
        molecule=initial,
        refE=initial.energy,
        opt_steps=250,  # The max number of optimization steps, use a small number until you have your final sigma
        verbose=True,
        opt_type='UNCONSTRAINED',  # JOSH
        xyzframerate=1,  # JOSH
    )

    print('Final energy is {:5.4f}'.format(initial.energy))
    manage_xyz.write_xyz('meci.xyz', geoms[-1], energies[-1], scale=1.0)


def print_msg():
    msg = """
    __        __   _                            _        
    \ \      / /__| | ___ ___  _ __ ___   ___  | |_ ___  
     \ \ /\ / / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \ 
      \ V  V /  __/ | (_| (_) | | | | | |  __/ | || (_) |
       \_/\_/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/ 
                                    ____ ____  __  __ 
                       _ __  _   _ / ___/ ___||  \/  |
                      | '_ \| | | | |  _\___ \| |\/| |
                      | |_) | |_| | |_| |___) | |  | |
                      | .__/ \__, |\____|____/|_|  |_|
                      |_|    |___/                    
#==========================================================================#
#| If this code has benefited your research, please support us by citing: |#
#|                                                                        |# 
#| Aldaz, C.; Kammeraad J. A.; Zimmerman P. M. "Discovery of conical      |#
#| intersection mediated photochemistry with growing string methods",     |#
#| Phys. Chem. Chem. Phys., 2018, 20, 27394                               |#
#| http://dx.doi.org/10.1039/c8cp04703k                                   |#
#|                                                                        |# 
#| Wang, L.-P.; Song, C.C. (2016) "Geometry optimization made simple with |#
#| translation and rotation coordinates", J. Chem, Phys. 144, 214108.     |#
#| http://dx.doi.org/10.1063/1.4952956                                    |#
#==========================================================================#


    """
    print(msg)


if __name__ == '__main__':
    print_msg()
    main()
