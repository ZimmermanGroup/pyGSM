# standard library imports
import sys
from os import path

# third party
import numpy as np

try:
    from tblite.interface import Calculator
except ImportError:
    Calculator = None

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utilities import manage_xyz, units, elements
try:
    from .base_lot import Lot
except:
    from base_lot import Lot


class tblite_lot(Lot):
    """
    GFN1-xTB/GFN2-xTB energies and gradients via tblite.

    tblite is the maintained successor to xtb-python, which has had no release
    since 2022 and ships no wheels beyond Python 3.11. This class is a drop-in
    alternative to xTB_lot and takes the same options.
    """

    def __init__(self, options):
        super(tblite_lot, self).__init__(options)

        numbers = []
        E = elements.ElementData()
        for a in manage_xyz.get_atoms(self.geom):
            elem = E.from_symbol(a)
            numbers.append(elem.atomic_num)
        self.numbers = np.asarray(numbers)

    def run(self, geom, multiplicity, state, verbose=False):

        if Calculator is None:
            raise ImportError(
                "tblite is required for the tblite level of theory but is not installed. "
                "Install it with 'pip install tblite' or 'conda install -c conda-forge tblite-python'."
            )

        coords = manage_xyz.xyz_to_np(geom)

        # convert to bohr
        positions = coords * units.ANGSTROM_TO_AU

        # tblite takes the method name directly as a string, and uhf as the
        # number of unpaired electrons
        calc = Calculator(
            self.xTB_Hamiltonian,
            self.numbers,
            positions,
            charge=float(self.charge),
            uhf=int(multiplicity) - 1,
        )

        calc.set('accuracy', self.xTB_accuracy)

        # tblite expects the electronic temperature in Hartree, whereas pyGSM
        # stores it in Kelvin (as xtb-python expected it)
        calc.set('temperature', self.xTB_electronic_temperature * units.K_TO_AU)

        # keep the solver quiet; tblite has no per-node output file
        calc.set('verbosity', 0)

        if self.solvent is not None:
            calc.add('alpb-solvation', self.solvent)

        res = calc.singlepoint()

        # energy in hartree
        self._Energies[(multiplicity, state)] = self.Energy(res.get('energy'), 'Hartree')

        # grad in Hartree/Bohr
        self._Gradients[(multiplicity, state)] = self.Gradient(res.get('gradient'), 'Hartree/Bohr')

        # write E to scratch
        self.write_E_to_file()

        return res
