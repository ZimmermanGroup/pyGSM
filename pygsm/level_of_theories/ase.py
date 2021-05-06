"""
Level Of Theory for ASE calculators
https://gitlab.com/ase/ase

Written by Tamas K. Stenczel in 2021
"""
try:
    from ase import Atoms
    from ase.calculators.calculator import Calculator
    from ase.data import atomic_numbers
    from ase import units
except ModuleNotFoundError:
    print("ASE not installed, ASE-based calculators will not work")

from .base_lot import Lot


class ASELoT(Lot):
    """
    Warning:
        multiplicity is not implemented, the calculator ignores it
    """

    def __init__(self, calculator: Calculator, options):
        super(ASELoT, self).__init__(options)

        self.ase_calculator = calculator

    @classmethod
    def from_options(cls, calculator: Calculator, **kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(calculator, cls.default_options().set_values(kwargs))

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        assert isinstance(lot, ASELoT)
        return cls(lot.ase_calculator, lot.options.copy().set_values(options))

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        # run ASE
        self.run_ase_atoms(xyz_to_ase(geom), mult, ad_idx, runtype)

    def run_ase_atoms(self, atoms: Atoms, mult, ad_idx, runtype='gradient'):
        # set the calculator
        atoms.set_calculator(self.ase_calculator)

        # perform gradient calculation if needed
        if runtype == "gradient":
            self._Gradients[(mult, ad_idx)] = self.Gradient(- atoms.get_forces() / units.Ha * units.Bohr,
                                                            'Hartree/Bohr')
        elif runtype == "energy":
            pass
        else:
            raise NotImplementedError(f"Run type {runtype} is not implemented in the ASE calculator interface")

        # energy is always calculated -> cached if force calculation was done
        self._Energies[(mult, ad_idx)] = self.Energy(atoms.get_potential_energy() / units.Ha, 'Hartree')

        # write E to scratch
        self.write_E_to_file()

        self.hasRanForCurrentCoords = True


def xyz_to_ase(xyz):
    """

    Parameters
    ----------
    xyz : np.ndarray, shape=(N, 4)


    Returns
    -------
    atoms : ase.Atoms
        ASE's Atoms object

    """

    # compatible with list-of-list as well
    numbers = [atomic_numbers[x[0]] for x in xyz]
    pos = [x[1:4] for x in xyz]
    return geom_to_ase(numbers, pos)


def geom_to_ase(numbers, positions, **kwargs):
    """Geometry to ASE atoms object

    Parameters
    ----------
    numbers : array_like, shape=(N_atoms,)
        atomic numbers
    positions : array_like, shape=(N_atoms,3)
        positions of atoms in Angstroms
    kwargs

    Returns
    -------
    atoms : ase.Atoms
        ASE's Atoms object
    """

    return Atoms(numbers=numbers, positions=positions, **kwargs)
