import tcc
import json
from base_lot import Lot 
import numpy as np
import manage_xyz
from units import *
import sys

class TCC(Lot):

    @property
    def TC(self):
        return self.options['job_data']['TC']

    @property
    def tcc_options(self):
        return self.options['job_data']['tcc_options']

    @tcc_options.setter
    def tcc_options(self,d):
        self.options['job_data']['tcc_options'] = d
        return

    @property
    def orbfile(self):
        return self.options['job_data']['orbfile']

    @orbfile.setter
    def orbfile(self,value):
        self.options['job_data']['orbfile'] = value

    def __init__(self,options):
        super(TCC,self).__init__(options)
        tcc_options_copy = self.tcc_options.copy()
        tcc_options_copy['atoms'] = self.atoms
        self.tcc_options = tcc_options_copy

    def run(self,coords):
        self.E=[]
        self.grada=[]
    
        for state in self.states:
            #print("on state %d" % state[1])
            multiplicity=state[0]
            ad_idx=state[1]
            grad_options = self.tcc_options.copy()
            grad_options['runtype'] = 'gradient'
            grad_options['castargetmult'] = multiplicity
            grad_options['castarget'] = ad_idx
            if not self.orbfile:
                grad_options['guess'] = self.orbfile
            print(" orbfile is %s" % self.orbfile)
            results = self.TC.compute(coords,grad_options)
            print((json.dumps(results, indent=2, sort_keys=True)))
            self.orbfile = results['orbfile']
            self.E.append((multiplicity,ad_idx,results['energy'][ad_idx]))
            self.grada.append((multiplicity,ad_idx,results['gradient']))
        if self.do_coupling==True:
            state1=self.states[0][1]
            state2=self.states[1][1]
            nac_options = self.tcc_options.copy()
            nac_options['runtype'] = 'coupling'
            nac_options['nacstate1'] = 0
            nac_options['nacstate2'] = 1
            results = self.TC.compute(coords,nac_options)
            self.coup = results['coupling']

        self.hasRanForCurrentCoords=True
        return  

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.run(coords)
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.run(coords)
        tmp = self.search_PES_tuple(self.grada,multiplicity,state)[0][2]
        return np.asarray(tmp)*ANGSTROM_TO_AU

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            self.run(coords)
        return np.reshape(self.coup,(3*len(self.coup),1))*ANGSTROM_TO_AU
