# standard library imports
import sys
from os import path

# third party
import numpy as np
import simtk.unit as openmm_units
import simtk.openmm.app as openmm_app
import simtk.openmm as openmm
import json


from parmed import load_file, unit as u

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .base_lot import Lot
from utilities import *

class OpenMM(Lot):
    def __init__(self,options):

        # BASE LOT READS FILE
        super(OpenMM,self).__init__(options)
        if self.lot_inp_file is not None and self.simulation is None:
            # Now go through the logic of determining which FILE options are activated.
            self.set_active('use_crystal',False,bool,"Use crystal unit parameters")
            self.set_active('use_pme',False,bool,"Use particle mesh ewald-- requires periodic boundary conditions")
            self.set_active('cutoff',1.0,float,depend=(self.use_pme),msg="Requires PME")
            self.set_active('prmtopfile',None,str,"parameter file")
            self.set_active('inpcrdfile',None,str,"inpcrd file")

            nifty.printcool(" Options for OpenMM")
            for val in [self.prmtopfile,self.inpcrdfile]:
                assert val!=None,"Missing prmtop or inpcrdfile"
            for line in self.record():
                print(line)

            # Integrator will never be used (Simulation requires one)
            integrator = openmm.VerletIntegrator(1.0)

            # create simulation object
            if self.use_crystal=='yes':
                crystal = load_file(prmtopfile,inpcrdfile)
                if self.use_pme=='yes':
                    system = crystal.createSystem(
                        nonbondedMethod=openmm_app.PME,
                        nonbondedCutoff=cutoff*openmm_units.nanometer,
                        )
                else:
                    system = crystal.createSystem(
                        nonbondedMethod=openmm_app.NoCutoff,
                        )
                self.simulation = openmm_app.Simulation(crystal.topology, system, integrator)
                # set the box vectors
                inpcrd = openmm_app.AmberInpcrdFile(inpcrdfile)
                if inpcrd.boxVectors is not None:
                    print(" setting box vectors")
                    print(inpcrd.boxVectors)
                    self.simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
            else:  # Do not use crystal parameters
                prmtop = openmm_app.AmberPrmtopFile(prmtopfile)
                if use_pme=='yes':
                    system = prmtop.createSystem(
                        nonbondedMethod=openmm_app.PME,
                        nonbondedCutoff=cutoff*openmm_units.nanometer,
                        )
                else:
                    system = prmtop.createSystem(
                        nonbondedMethod=openmm_app.NoCutoff,
                        )
                self.simulation = openmm_app.Simulation(
                    prmtop.topology,
                    system,
                    integrator,
                    )
   
    @property
    def simulation(self):
        return self.options['job_data']['simulation']

    @simulation.setter
    def simulation(self,value):
        self.options['job_data']['simulation'] = value
  
    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            self.run(coords)
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            self.run(coords)
        return self.search_PES_tuple(self.grada,multiplicity,state)[0][2]


    def run(self,coords):
        self.E=[]
        self.grada=[]

        # Update coordinates of simulation (shallow-copied object)
        xyz_nm = 0.1 * coords  # coords are in angstrom
        self.simulation.context.setPositions(xyz_nm)
    
        # actually compute (only applicable to ground-states,singlet mult)
        for state in self.states:
            multiplicity=state[0]
            ad_idx=state[1]
            s = self.simulation.context.getState(
                    getEnergy=True,
                    getForces=True,
                    )
            tmp = s.getPotentialEnergy()
            E = tmp.value_in_unit(openmm_units.kilocalories / openmm_units.moles)
            E /= units.KCAL_MOL_PER_AU
            self.E.append((multiplicity,ad_idx,E))

            F = s.getForces()
            G = F.value_in_unit(openmm_units.kilocalories/openmm_units.moles / openmm_units.angstroms)
            G = np.asarray(G)
             
            self.grada.append((multiplicity,ad_idx, -1.0 * G * units.KCAL_MOL_TO_AU)) # H/ang
        self.hasRanForCurrentCoords=True

        return 

if __name__=="__main__":
    import pybel as pb
    # Create and initialize System object from prmtop/inpcrd
    prmtopfile='../../data/solvated.prmtop'
    inpcrdfile='../../data/solvated.rst7'
    prmtop = openmm_app.AmberPrmtopFile(prmtopfile)
    inpcrd = openmm_app.AmberInpcrdFile(inpcrdfile)
    system = prmtop.createSystem(
        rigidWater=False, 
        removeCMMotion=False,
        nonbondedMethod=openmm_app.PME,
        nonbondedCutoff=1*openmm_units.nanometer  #10 ang
        )

    # Integrator will never be used (Simulation requires one)
    integrator = openmm.VerletIntegrator(1.0)
    simulation = openmm_app.Simulation(
        prmtop.topology,
        system,
        integrator,
        )
    mol=next(pb.readfile('pdb','../data/solvated.pdb'))
    coords = nifty.getAllCoords(mol)
    atoms = nifty.getAtomicSymbols(mol)
    print(coords)
    geom= manage_xyz.combine_atom_xyz(atoms,coords)

    lot = OpenMM.from_options(states=[(1,0)],job_data={'simulation':simulation},geom=geom)

    E = lot.get_energy(coords,1,0)
    print(E)

    G = lot.get_gradient(coords,1,0)
    nifty.pmat2d(G)

