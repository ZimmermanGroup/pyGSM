
# standard library imports
import sys
from os import path
import copy as cp


# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utilities import manage_xyz, units

try:
    from .base_lot import Lot
except:
    from base_lot import Lot


class nanoreactor_engine(Lot):
    '''
    '''

    def __init__(self, options):
        super(nanoreactor_engine, self).__init__(options)
        # can we do a check here?
        self.engine = options['job_data']['engine']
        self.nscffail = 0

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):
        '''
        '''
        if copy_wavefunction:
            return cls(lot, lot.options.copy().set_values(options))
        else:
            return cls(lot)

    def run(self, geom, mult, ad_idx, runtype='gradient'):
        '''
        '''
        self.Gradients = {}
        self.Energies = {}
        print('This is node id: {}'.format(self.node_id))
        xyz = manage_xyz.xyz_to_np(geom)*units.ANGSTROM_TO_AU

        if self.engine.options['closed_shell']:
            fields = ('energy', 'gradient', 'orbfile')
        else:
            fields = ('energy', 'gradient', 'orbfile_a', 'orbfile_b')

        try:
            if self.options['job_data']['orbfile']:
                orb_guess = self.options['job_data']['orbfile']
                results = self.engine.compute_blocking(xyz, fields, job_type='gradient', guess=orb_guess)  # compute__(fields = "energy, gradient, orbfiles")
            # if we're not using a previous orbital as a guess, we want to ensure we find the correct SCF
            #  minimum if using guess = generate
            else:
                old_options = cp.deepcopy(self.engine.options)
                self.engine.options['fon'] = 'yes'
                self.engine.options['fon_coldstart'] = 'no'
                self.engine.options['fon_converger'] = 'no'
                self.engine.options['fon_tests'] = 3
                results = self.engine.compute_blocking(xyz, fields, job_type='gradient', guess='generate')  # compute__(fields = "energy, gradient, orbfiles")
                self.engine.options = old_options
                print("Options are: {}".format(self.engine.options))
        except:
            # The calculation failed
            # set energy to a large number so the optimizer attempts to slow down
            print(" SCF FAILURE")
            self.nscffail += 1
            energy, gradient = 999, 0

            if self.nscffail > 25:
                raise RuntimeError

        # unpacking results and updating orb dictionaries
        energy = results[0]
        gradient = results[1]
        if self.engine.options['closed_shell']:
            orb_path = results[2]
            self.options['job_data']['orbfile'] = orb_path
        else:
            orb_a_path = results[2]
            orb_b_path = results[3]
            self.options['job_data']['orbfile'] = orb_a_path + ' ' + orb_b_path

        print("this is orbfile: {}".format(self.options['job_data']['orbfile']))

        # Store the values in memory
        self._Energies[(mult, ad_idx)] = self.Energy(energy, 'Hartree')
        self._Gradients[(mult, ad_idx)] = self.Gradient(gradient, 'Hartree/Bohr')


if __name__ == "__main__":
    pass
