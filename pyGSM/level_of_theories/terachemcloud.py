# standard library imports

import sys
from os import path
import time

# third party
import numpy as np
import tcc
import json

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from .base_lot import Lot
from utilities import nifty


class TeraChemCloud(Lot):

    @property
    def TC(self):
        return self.options['job_data']['TC']

    @property
    def tcc_options(self):
        return self.options['job_data']['tcc_options']

    @tcc_options.setter
    def tcc_options(self, d):
        self.options['job_data']['tcc_options'] = d
        return

    @property
    def orbfile(self):
        return self.options['job_data']['orbfile']

    @orbfile.setter
    def orbfile(self, value):
        self.options['job_data']['orbfile'] = value

    def __init__(self, options):
        super(TeraChemCloud, self).__init__(options)
        self.options['job_data']['tcc_options'] = self.options['job_data'].get(
            'tcc_options', {})
        self.options['job_data']['TC'] = self.options['job_data'].get(
            'TC', None)

        if self.lot_inp_file is not None:
            exec(open(self.lot_inp_file).read())
            print(' done executing lot_inp_file')
            self.options['job_data']['TC'] = TC
            self.options['job_data']['tcc_options'] = tcc_options
            #self.options['job_data']['orbfile']
        tcc_options_copy = self.tcc_options.copy()
        tcc_options_copy['atoms'] = self.atoms
        self.tcc_options = tcc_options_copy

    def run(self, coords):

        E = []
        grada = []
        for state in self.states:
            # print("on state %d" % state[1])
            multiplicity = state[0]
            ad_idx = state[1]
            grad_options = self.tcc_options.copy()
            grad_options['runtype'] = 'gradient'
            grad_options['castargetmult'] = multiplicity
            grad_options['castarget'] = ad_idx
            if self.orbfile:
                grad_options['guess'] = self.orbfile
                print(" orbfile is %s" % self.orbfile)
            else:
                print(" generating orbs from guess")
            job_id = self.TC.submit(coords, grad_options)
            results = self.TC.poll_for_results(job_id)
            while results['message'] == "job not finished":
                results = self.TC.poll_for_results(job_id)
                print(results['message'])
                print("sleeping for 1")
                time.sleep(1)
                sys.stdout.flush()

            # print((json.dumps(results, indent=2, sort_keys=True)))
            self.orbfile = results['orbfile']
            try:
                E.append((multiplicity, ad_idx, results['energy'][ad_idx]))
            except:
                E.append((multiplicity, ad_idx, results['energy']))

            grada.append((multiplicity, ad_idx, results['gradient']))
        if self.do_coupling:
            # state1 = self.states[0][1]
            # state2 = self.states[1][1]
            nac_options = self.tcc_options.copy()
            nac_options['runtype'] = 'coupling'
            nac_options['nacstate1'] = 0
            nac_options['nacstate2'] = 1
            nac_options['guess'] = self.orbfile

            # nifty.printcool_dictionary(nac_options)
            job_id = self.TC.submit(coords, nac_options)
            results = self.TC.poll_for_results(job_id)
            while results['message'] == "job not finished":
                results = self.TC.poll_for_results(job_id)
                print(results['message'])
                print("sleeping for 1")
                time.sleep(1)
                sys.stdout.flush()
            # print((json.dumps(results, indent=2, sort_keys=True)))
            coup = results['nacme']
            self.Couplings[self.coupling_states] = self.Coupling(
                coup, 'Hartree/Bohr')

        for energy, state in zip(E, self.states):
            self._Energies[state] = self.Energy(energy, 'Hartree')
        for grad, state in zip(E, self.gradient_states):
            self._Gradients[state] = self.Gradient(grad, "Hartree/Bohr")

        self.hasRanForCurrentCoords = True
        return
