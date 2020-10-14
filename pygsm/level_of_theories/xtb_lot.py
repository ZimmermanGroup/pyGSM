# standard library imports
import sys
from os import path

# third party
import numpy as np
import contextlib
from xtb.interface import Calculator
from xtb.utils import get_method
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

    def run(self,coords,multiplicity,state,verbose=False):
        
        #print('running!')
        #sys.stdout.flush()

        self.E=[]
        self.grada=[]

        # convert to angstrom
        positions = coords* units.ANGSTROM_TO_AU

        calc = Calculator(get_method("GFN2-xTB"), self.numbers, positions)
        calc.set_output('lot_jobs_{}.txt'.format(self.node_id))
        res = calc.singlepoint()  # energy printed is only the electronic part
        calc.release_output()
     
        # energy in hartree
        self.E.append((multiplicity,state,res.get_energy()))

        # grad in Hatree/Bohr
        self.grada.append((multiplicity,state,res.get_gradient()))

        # write E to scratch
        with open('scratch/E_{}.txt'.format(self.node_id),'w') as f:
            for E in self.E:
                f.write('{} {:9.7f}\n'.format(E[0],E[2]))
        self.hasRanForCurrentCoords = True

        return res

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(coords,multiplicity,state)
        tmp = self.search_PES_tuple(self.E,multiplicity,state)[0][2]
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(coords,multiplicity,state)
        tmp = self.search_PES_tuple(self.grada,multiplicity,state)[0][2]
        if tmp is not None:
            return np.asarray(tmp)*units.ANGSTROM_TO_AU  #Ha/bohr*bohr/ang=Ha/ang
        else:
            return None


if __name__=="__main__":
    

    geom=manage_xyz.read_xyz('../../data/ethylene.xyz')
    #geoms=manage_xyz.read_xyzs('../../data/diels_alder.xyz')
    #geom = geoms[0]
    #geom=manage_xyz.read_xyz('xtbopt.xyz')
    xyz = manage_xyz.xyz_to_np(geom) 
    #xyz *= units.ANGSTROM_TO_AU

    lot  = xTB.from_options(states=[(1,0)],gradient_states=[(1,0)],geom=geom,node_id=0)

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
