# standard library imports
from coordinate_systems import Dihedral
from utilities import manage_xyz, nifty
import sys
from os import path

# third party
import numpy as np
import simtk.unit as openmm_units
import simtk.openmm.app as openmm_app
import simtk.openmm as openmm
from parmed import load_file

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
try:
    from .base_lot import Lot
except:
    from base_lot import Lot


class OpenMM(Lot):
    def __init__(self, options):

        super(OpenMM, self).__init__(options)

        # get simulation from options if it exists
        # self.options['job_data']['simulation'] = self.options['job_data'].get('simulation',None)
        self.simulation = self.options['job_data'].get('simulation', None)

        if self.lot_inp_file is not None and self.simulation is None:

            # Now go through the logic of determining which FILE options are activated.
            self.file_options.set_active('use_crystal', False, bool, "Use crystal unit parameters")
            self.file_options.set_active('use_pme', False, bool, '', "Use particle mesh ewald-- requires periodic boundary conditions")
            self.file_options.set_active('cutoff', 1.0, float, '', depend=(self.file_options.use_pme), msg="Requires PME")
            self.file_options.set_active('prmtopfile', None, str, "parameter file")
            self.file_options.set_active('inpcrdfile', None, str, "inpcrd file")
            self.file_options.set_active('restrain_bondfile', None, str, 'list of bonds to restrain')
            self.file_options.set_active('restrain_torfile', None, str, "list of torsions to restrain")
            self.file_options.set_active('restrain_tranfile', None, str, "list of translations to restrain")

            for line in self.file_options.record():
                print(line)

            # set all active values to self for easy access
            for key in self.file_options.ActiveOptions:
                setattr(self, key, self.file_options.ActiveOptions[key])

            nifty.printcool(" Options for OpenMM")
            for val in [self.prmtopfile, self.inpcrdfile]:
                assert val is not None, "Missing prmtop or inpcrdfile"

            # Integrator will never be used (Simulation requires one)
            integrator = openmm.VerletIntegrator(1.0)

            # create simulation object
            if self.use_crystal:
                crystal = load_file(self.prmtopfile, self.inpcrdfile)
                if self.use_pme:
                    system = crystal.createSystem(
                        nonbondedMethod=openmm_app.PME,
                        nonbondedCutoff=self.cutoff*openmm_units.nanometer,
                    )
                else:
                    system = crystal.createSystem(
                        nonbondedMethod=openmm_app.NoCutoff,
                    )

                # Add restraints
                self.add_restraints(system)

                self.simulation = openmm_app.Simulation(crystal.topology, system, integrator)
                # set the box vectors
                inpcrd = openmm_app.AmberInpcrdFile(self.inpcrdfile)
                if inpcrd.boxVectors is not None:
                    print(" setting box vectors")
                    print(inpcrd.boxVectors)
                    self.simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
            else:  # Do not use crystal parameters
                prmtop = openmm_app.AmberPrmtopFile(self.prmtopfile)
                if self.use_pme:
                    system = prmtop.createSystem(
                        nonbondedMethod=openmm_app.PME,
                        nonbondedCutoff=self.cutoff*openmm_units.nanometer,
                    )
                else:
                    system = prmtop.createSystem(
                        nonbondedMethod=openmm_app.NoCutoff,
                    )

                # add restraints
                self.add_restraints(system)

                self.simulation = openmm_app.Simulation(
                    prmtop.topology,
                    system,
                    integrator,
                )

    def add_restraints(self, system):
        # Bond Restraints
        if self.restrain_bondfile is not None:
            nifty.printcool(" Adding bonding restraints!")
            # Harmonic constraint

            flat_bottom_force = openmm.CustomBondForce(
                'step(r-r0) * (k/2) * (r-r0)^2')
            flat_bottom_force.addPerBondParameter('r0')
            flat_bottom_force.addPerBondParameter('k')
            system.addForce(flat_bottom_force)

            with open(self.restrain_bondfile, 'r') as input_file:
                for line in input_file:
                    print(line)
                    columns = line.split()
                    atom_index_i = int(columns[0])
                    atom_index_j = int(columns[1])
                    r0 = float(columns[2])
                    k = float(columns[3])
                    flat_bottom_force.addBond(
                        atom_index_i, atom_index_j, [r0, k])

        # Torsion restraint
        if self.restrain_torfile is not None:
            nifty.printcool(" Adding torsional restraints!")

            # Harmonic constraint
            tforce = openmm.CustomTorsionForce("0.5*k*min(dtheta, 2*pi-dtheta)^2; dtheta = abs(theta-theta0); pi = 3.1415926535")
            tforce.addPerTorsionParameter("k")
            tforce.addPerTorsionParameter("theta0")
            system.addForce(tforce)

            xyz = manage_xyz.xyz_to_np(self.geom)
            with open(self.restrain_torfile, 'r') as input_file:
                for line in input_file:
                    columns = line.split()
                    a = int(columns[0])
                    b = int(columns[1])
                    c = int(columns[2])
                    d = int(columns[3])
                    k = float(columns[4])
                    dih = Dihedral(a, b, c, d)
                    theta0 = dih.value(xyz)
                    tforce.addTorsion(a, b, c, d, [k, theta0])

        # Translation restraint
        if self.restrain_tranfile is not None:
            nifty.printcool(" Adding translational restraints!")
            trforce = openmm.CustomExternalForce("k*periodicdistance(x, y, z, x0, y0, z0)^2")
            trforce.addPerParticleParameter("k")
            trforce.addPerParticleParameter("x0")
            trforce.addPerParticleParameter("y0")
            trforce.addPerParticleParameter("z0")
            system.addForce(trforce)

            xyz = manage_xyz.xyz_to_np(self.geom)
            with open(self.restrain_tranfile, 'r') as input_file:
                for line in input_file:
                    columns = line.split()
                    a = int(columns[0])
                    k = float(columns[1])
                    x0 = xyz[a, 0]*0.1  # Units are in nm
                    y0 = xyz[a, 1]*0.1  # Units are in nm
                    z0 = xyz[a, 2]*0.1  # Units are in nm
                    trforce.addParticle(a, [k, x0, y0, z0])

    @property
    def simulation(self):
        return self.options['job_data']['simulation']

    @simulation.setter
    def simulation(self, value):
        self.options['job_data']['simulation'] = value

    def run(self, geom, mult, ad_idx, runtype='gradient'):

        coords = manage_xyz.xyz_to_np(geom)

        # Update coordinates of simulation (shallow-copied object)
        xyz_nm = 0.1 * coords  # coords are in angstrom
        self.simulation.context.setPositions(xyz_nm)

        # actually compute (only applicable to ground-states,singlet mult)
        if mult != 1 or ad_idx > 1:
            raise RuntimeError('MM cant do excited states')

        s = self.simulation.context.getState(
            getEnergy=True,
            getForces=True,
        )
        tmp = s.getPotentialEnergy()
        E = tmp.value_in_unit(openmm_units.kilocalories / openmm_units.moles)
        self._Energies[(mult, ad_idx)] = self.Energy(E, 'kcal/mol')

        F = s.getForces()
        G = -1.0 * np.asarray(F.value_in_unit(openmm_units.kilocalories/openmm_units.moles / openmm_units.angstroms))

        self._Gradients[(mult, ad_idx)] = self.Gradient(G, 'kcal/mol/Angstrom')
        self.hasRanForCurrentCoords = True

        return


if __name__ == "__main__":
    from openbabel import pybel as pb
    # Create and initialize System object from prmtop/inpcrd
    prmtopfile = '../../data/solvated.prmtop'
    inpcrdfile = '../../data/solvated.rst7'
    prmtop = openmm_app.AmberPrmtopFile(prmtopfile)
    inpcrd = openmm_app.AmberInpcrdFile(inpcrdfile)
    system = prmtop.createSystem(
        rigidWater=False,
        removeCMMotion=False,
        nonbondedMethod=openmm_app.PME,
        nonbondedCutoff=1*openmm_units.nanometer  # 10 ang
    )

    # Integrator will never be used (Simulation requires one)
    integrator = openmm.VerletIntegrator(1.0)
    simulation = openmm_app.Simulation(
        prmtop.topology,
        system,
        integrator,
    )
    mol = next(pb.readfile('pdb', '../../data/solvated.pdb'))
    coords = nifty.getAllCoords(mol)
    atoms = nifty.getAtomicSymbols(mol)
    print(coords)
    geom = manage_xyz.combine_atom_xyz(atoms, coords)

    lot = OpenMM.from_options(states=[(1, 0)], job_data={'simulation': simulation}, geom=geom)

    E = lot.get_energy(coords, 1, 0)
    print(E)

    G = lot.get_gradient(coords, 1, 0)
    nifty.pmat2d(G)
