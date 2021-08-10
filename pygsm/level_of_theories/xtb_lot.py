# standard library imports
import sys
from os import path

# third party
import numpy as np
import contextlib
from xtb.interface import Calculator
from xtb.utils import get_method, get_solvent
from xtb.interface import Environment
from xtb.libxtb import VERBOSITY_FULL

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
try:
    from .base_lot import Lot
except:
    from base_lot import Lot
from utilities import *

class xTB_lot(Lot):
    def __init__(self,options):
        super(xTB_lot,self).__init__(options)

        numbers = []
        E = elements.ElementData()
        for a in manage_xyz.get_atoms(self.geom):
            elem = E.from_symbol(a)
            numbers.append(elem.atomic_num)
        self.numbers = np.asarray(numbers)

    def run(self,geom,multiplicity,state,verbose=False):
        
        #print('running!')
        #sys.stdout.flush()
        coords = manage_xyz.xyz_to_np(geom) 

        # convert to bohr
        positions = coords* units.ANGSTROM_TO_AU
        calc = Calculator(get_method(self.xTB_Hamiltonian), self.numbers, positions, charge=self.charge)

        calc.set_accuracy(self.xTB_accuracy)
        calc.set_electronic_temperature(self.xTB_electronic_temperature)

        if self.solvent is not None:
            calc.set_solvent(get_solvent(self.solvent))
        
        calc.set_output('lot_jobs_{}.txt'.format(self.node_id))
        res = calc.singlepoint()  # energy printed is only the electronic part
        calc.release_output()
     
        # energy in hartree
        self._Energies[(multiplicity,state)] = self.Energy(res.get_energy(),'Hartree')

        # grad in Hatree/Bohr
        self._Gradients[(multiplicity,state)] = self.Gradient(res.get_gradient(),'Hartree/Bohr')

        # write E to scratch
        self.write_E_to_file()

        self.hasRanForCurrentCoords = True

        return res


if __name__=="__main__":
    

    geom=manage_xyz.read_xyz('../../data/ethylene.xyz')
    #geoms=manage_xyz.read_xyzs('../../data/diels_alder.xyz')
    #geom = geoms[0]
    #geom=manage_xyz.read_xyz('xtbopt.xyz')
    xyz = manage_xyz.xyz_to_np(geom) 
    #xyz *= units.ANGSTROM_TO_AU

    lot  = xTB_lot.from_options(states=[(1,0)],gradient_states=[(1,0)],geom=geom,node_id=0)

    E = lot.get_energy(xyz,1,0)
    print(E)

    g = lot.get_gradient(xyz,1,0)
    print(g)

    #env = Environment()
    #env.set_output("error.log")
    #env.set_verbosity(0)
    #numbers = np.array([8, 1, 1])
    #positions = np.array([
    #[ 0.00000000000000, 0.00000000000000,-0.73578586109551],
    #[ 1.44183152868459, 0.00000000000000, 0.36789293054775],
    #[-1.44183152868459, 0.00000000000000, 0.36789293054775]])
    #
    #calc = Calculator(get_method("GFN2-xTB"), numbers, positions)

    ##with open('lot_jobs2.txt','a') as f:
    ##    with contextlib.redirect_stdout(f):
    ##        res = calc.singlepoint()  # energy printed is only the electronic part

    #res = calc.singlepoint()  # energy printed is only the electronic part

    #E = res.get_energy()
    #print(E)
    #g = res.get_gradient()
    #print(g)
    #
    #c = res.get_charges()
    #print(c)
