
# standard library imports
import sys
import os
from os import path
import re
from collections import namedtuple
import copy as cp
# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))

try:
    from .base_lot import Lot
    from .file_options import File_Options
except:
    from base_lot import Lot
    from file_options import File_Options
from utilities import *

'''
'''

class nanoreactor_engine(Lot):

    def __init__(self,options):
        super(nanoreactor_engine,self).__init__(options)
        # can we do a check here?
        self.engine=options['job_data']['engine']
        self.nscffail = 0
        self.save_orbital = True
        if type(self.options['job_data']['orbfile']) != dict:
            self.options['job_data']['orbfile'] = {}
        if 'all_geoms' not in self.options['job_data']:
            self.options['job_data']['all_geoms'] = {}
    @classmethod
    def rmsd(cls, geom1, geom2):
        total = 0

        flat_geom1 = np.array(geom1).flatten()
        flat_geom2 = np.array(geom2).flatten()
        for i in range(len(flat_geom1)):
            total += (flat_geom1[i] - flat_geom2[i]) ** 2

        return total

    @classmethod
    def copy(cls, lot, options={}, copy_wavefunction=True):
        options['job_data'] = lot.options['job_data']
        if copy_wavefunction:
            options['job_data']['orbfile'].update({'copied_orb': lot.node_id})    
        return cls(lot.options.copy().set_values(options))

    def run(self,geom,mult,ad_idx,runtype='gradient'):
        self.Gradients={}
        self.Energies = {}
        xyz = manage_xyz.xyz_to_np(geom)*units.ANGSTROM_TO_AU
        
        if self.engine.options['closed_shell']:
            fields = ('energy', 'gradient', 'orbfile')
        else:
            fields = ('energy', 'gradient', 'orbfile_a', 'orbfile_b')
        #print(self.options['job_data']['orbfile']['propagate'])
        if self.hasRanForCurrentCoords == False and self.node_id in self.options['job_data']['orbfile'].keys():
            print("Recalculating energy")
            del self.options['job_data']['orbfile'][self.node_id]
        if self.node_id not in self.options['job_data']['orbfile'].keys():
            if 'copied_orb' in self.options['job_data']['orbfile'].keys():
                if self.options['job_data']['orbfile']['copied_orb'] in self.options['job_data']['orbfile'].keys():
                    orb_guess = self.options['job_data']['orbfile'][self.options['job_data']['orbfile']['copied_orb']] 
                    del self.options['job_data']['orbfile']['copied_orb']
                else:
                    orb_guess = None
                    del self.options['job_data']['orbfile']['copied_orb']
            else:
                orb_guess = None
            if (self.node_id - 1) in self.options['job_data']['orbfile'].keys() and orb_guess == None:
                orb_guess = self.options['job_data']['orbfile'][self.node_id - 1]
            elif (self.node_id + 1) in self.options['job_data']['orbfile'].keys() and orb_guess == None:
                orb_guess = self.options['job_data']['orbfile'][self.node_id + 1]
        else:
            orb_guess = self.options['job_data']['orbfile'][self.node_id]
        try:
            if orb_guess:
                results = self.engine.compute_blocking(xyz, fields, job_type = 'gradient', guess = orb_guess) #compute__(fields = "energy, gradient, orbfiles")
            #if we're not using a previous orbital as a guess, we want to ensure we find the correct SCF minimum if using guess = generate
            else:
                print("No orbital guess available")
                old_options = cp.deepcopy(self.engine.options)
                #does this work with non-unrestricted methods? #TODO
                if 'fon' in self.engine.options.keys():
                    if self.engine.options['fon'] == 'yes':
                        self.engine.options['fon_coldstart'] = 'no'
                        self.engine.options['fon_converger'] = 'no'
                        self.engine.options['fon_tests'] = 2
                results = self.engine.compute_blocking(xyz, fields, job_type = 'gradient', guess = 'generate') #compute__(fields = "energy, gradient, orbfiles")
                self.engine.options = old_options
        except:
            # The calculation failed
            # set energy to a large number so the optimizer attempts to slow down
            print(" SCF FAILURE")
            self.nscffail+=1
            energy,gradient = 999, 0 
            if self.nscffail>25:
                raise RuntimeError
        #unpacking results and updating orb dictionaries
        energy = results[0]
        gradient = results[1]
        self._Energies[(mult,ad_idx)] = self.Energy(energy,'Hartree')
        self._Gradients[(mult,ad_idx)] = self.Gradient(gradient,'Hartree/Bohr')
        if self.engine.options['closed_shell']:
            orb_path = results[2]
            self.options['job_data']['orbfile'].update({self.node_id: orb_path})
        else:
            orb_a_path = results[2]
            orb_b_path = results[3]
            self.options['job_data']['orbfile'].update({self.node_id: orb_a_path + ' ' + orb_b_path})
        # Store the values in memory


if __name__=="__main__":
    from nanoreactor.engine import get_engine
    from nanoreactor.parsing import load_settings

    # read settings from name
    db, setting_name, settings = load_settings('refine.yaml', host='fire-05-30')

    # Create the nanoreactor TCPB engine
    engine_type=settings['engine'].pop('type')
    engine = get_engine(r.mol, engine_type=engine_type, **settings['engine'])

    
    # read in a geometry
    geom = manage_xyz.read_xyz('../../data/ethylene.xyz')
    xyz = manage_xyz.xyz_to_np(geom)
    
    # create the pygsm level of theory object
    test_lot = nanoreactor_engine(geom,job_data = {'engine',test_engine})

    # Test
    print("getting energy")
    print(test_lot.get_energy(xyz,1,0))

    print("getting grad")
    print(test_lot.get_gradient(xyz,1,0))


    
