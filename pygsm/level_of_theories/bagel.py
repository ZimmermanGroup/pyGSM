# standard library imports
# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from utilities import manage_xyz
import sys
import os
from os import path
import re

# third party
import numpy as np

try:
    from .base_lot import Lot
    from .file_options import File_Options
except:
    from base_lot import Lot
    from file_options import File_Options


class BAGEL(Lot):

    def __init__(self, options):
        super(BAGEL, self).__init__(options)

        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

        # regular options
        self.file_options.set_active('basis', '6-31g', str, '')
        self.file_options.set_active('df_basis', 'svp-jkfit', str, '')
        self.file_options.set_active('maxiter', 200, int, '')

        # CASSCF
        self.file_options.set_active('casscf', None, bool, '')
        self.file_options.set_active('nstate', None, int, '')
        self.file_options.set_active('nact', None, int, '')
        self.file_options.set_active('nclosed', None, int, '')

        # CASPT2 options
        self.file_options.set_active('caspt2', None, bool, '', depend=(
            self.file_options.casscf), msg='casscf must be True')
        self.file_options.set_active(
            'ms', True, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'xms', True, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'sssr', False, bool, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'shift', 0.3, float, '', depend=(self.file_options.caspt2), msg='')
        self.file_options.set_active(
            'frozen', True, bool, '', depend=(self.file_options.caspt2), msg='')

        keys_to_del = []
        for key, value in self.file_options.ActiveOptions.items():
            if value is None:
                keys_to_del.append(key)
        for key in keys_to_del:
            self.file_options.deactivate(key)

        if "casscf" in self.file_options.ActiveOptions:
            for key in ['nstate', 'nact', 'nclosed']:
                if key not in self.file_options.ActiveOptions:
                    raise RuntimeError

        keys_to_del = []
        if "caspt2" not in self.file_options.ActiveOptions:
            for key in ['ms', 'xms', 'sssr', 'shift', 'frozen']:
                print('deactivating key ', key)
                keys_to_del.append(key)
        for key in keys_to_del:
            self.file_options.deactivate(key)

        guess_file = 'scratch/{}/orbs'.format(self.node_id)
        self.file_options.set_active('load_ref', guess_file, str, doc='guess',
                                     depend=(os.path.isfile(guess_file+'.archive')), msg='ref does not exist or not needed, deactivating for now...')

        if self.node_id == 0:
            for line in self.file_options.record():
                print(line)

        # set all active values to self for easy access
        for key in self.file_options.ActiveOptions:
            setattr(self, key, self.file_options.ActiveOptions[key])

    @classmethod
    def copy(cls, lot, options, copy_wavefunction=True):

        # Get node id for new struct
        node_id = options.get('node_id', 1)
        print(" making folder scratch/{}".format(node_id))
        os.system('mkdir -p scratch/{}'.format(node_id))

        file_options = File_Options.copy(lot.file_options)
        options['file_options'] = file_options

        if node_id != lot.node_id and copy_wavefunction:
            old_path = 'scratch/{}/orbs.archive'.format(lot.node_id)
            new_path = 'scratch/{}/orbs.archive'.format(node_id)
            cmd = 'cp -r ' + old_path + ' ' + new_path
            print(" copying scr files\n {}".format(cmd))
            os.system(cmd)
            os.system('wait')
        return cls(lot.options.copy().set_values(options))

    def write_input(self, geom, runtype='gradient'):
        # filenames
        inpfilename = 'scratch/{}/bagel.json'.format(self.node_id)
        inpfile = open(inpfilename, 'w')

        # Header
        inpfile.write('{ "bagel" : [\n\n')

        # molecule section
        inpfile.write(('{\n'
                       ' "title" : "molecule",\n'))

        # basis set
        inpfile.write(' "basis" : "{}",\n'.format(self.basis))
        inpfile.write(' "df_basis" : "{}",\n'.format(self.df_basis))
        inpfile.write('"angstrom" : true,\n')

        # write geometry
        inpfile.write(' "geometry" : [\n')
        for atom in geom[:-1]:
            inpfile.write(' { "atom" : "%s", "xyz" : [ %14.6f, %14.6f, %14.6f ] },\n' % (
                atom[0],
                atom[1],
                atom[2],
                atom[3],
            ))
        inpfile.write(' { "atom" : "%s", "xyz" : [ %14.6f, %14.6f, %14.6f ] }\n' % (
            geom[-1][0],
            geom[-1][1],
            geom[-1][2],
            geom[-1][3],
        ))
        inpfile.write(']\n')
        inpfile.write('},\n')

        # Load reference
        if 'load_ref' in self.file_options.ActiveOptions:
            inpfile.write(('{{ \n'
                           ' "title" : "load_ref", \n'
                           ' "file" : "scratch/{}/orbs", \n'
                           ' "continue_geom" : false \n}},\n'.format(self.node_id)))

        if "casscf" in self.file_options.ActiveOptions:
            inpfile.write(('{{\n'
                           ' "title" : "casscf", \n'
                           ' "nstate" : {}, \n'
                           ' "nact" : {}, \n'
                           ' "nclosed" : {}, \n'
                           ' "maxiter": {} \n'
                           '  }},\n\n'.format(self.nstate, self.nact, self.nclosed, self.maxiter)))
        else:
            print(" only casscf implemented now")
            raise NotImplementedError

        if runtype == "gradient":
            inpfile.write(('{\n'
                           '  "title" : "forces",\n'))

            inpfile.write('     "grads" : [\n')
            num_states = len(self.gradient_states)
            for i, state in enumerate(self.gradient_states):
                if i+1 == num_states:
                    inpfile.write(
                        '     {{ "title" : "force", "target" : {} }}\n'.format(state[1]))
                else:
                    inpfile.write(
                        '     {{ "title" : "force", "target" : {} }},\n'.format(state[1]))
            inpfile.write(' \n],\n')
        elif runtype == "gh":
            inpfile.write(('{\n'
                           '  "title" : "forces",\n'))

            inpfile.write('     "grads" : [\n')
            num_states = len(self.gradient_states)
            for i, state in enumerate(self.gradient_states):
                inpfile.write(
                    '     {{ "title" : "force", "target" : {} }},\n'.format(state[1]))

            inpfile.write('     {{ "title"  : "nacme", "target1" : {}, "tartet2": {}, "nacmtype" : "full" }}\n'.format(
                self.coupling_states[0], self.coupling_states[1]))
            inpfile.write(' ],\n')

        if "caspt2" in self.file_options.ActiveOptions and (runtype == 'gh' or runtype == 'gradient'):
            inpfile.write((' "method" : [ {{\n'
                           '  "title" : "caspt2",\n'
                           '  "smith" : {{\n'
                           '    "method" : "caspt2",\n'
                           '    "ms" : "{}",\n'
                           '    "xms" : "{}",\n'
                           '    "sssr" : "{}",\n'
                           '    "shift" : {},\n'
                           '    "frozen" : "{}"\n'
                           '  }},\n'
                           '  "nstate" : {},\n'
                           '  "nact" : {},\n'
                           '  "nclosed" : {},\n'
                           '  "charge" : 0\n'
                           '  }} ]\n'
                           '}},\n'.format(self.ms, self.xms, self.sssr, self.shift, self.frozen, self.nstate, self.nact, self.nclosed)))
        elif runtype == 'energy':
            pass
        else:
            print("only caspt2 gradient implemented")
            raise NotImplementedError

        inpfile.write(('{{\n'
                       '"title" : "save_ref",\n'
                       '"file" : "scratch/{}/orbs"\n'
                       '}}\n'.format(self.node_id)))

        inpfile.write(']}')
        inpfile.close()

    def parse(self, geom, runtype='gradient'):
        self.Gradients = {}
        self.Energies = {}
        self.Couplings = {}

        # Parse the output for Energies
        tempfileout = 'scratch/{}/output.dat'.format(self.node_id)
        tmp = []
        if "caspt2" in self.file_options.ActiveOptions:
            pattern = re.compile(
                r'\s. MS-CASPT2 energy : state  \d \s* ([-+]?[0-9]*\.?[0-9]+)')
            for line in open(tempfileout):
                for match in re.finditer(pattern, line):
                    tmp.append(float(match.group(1)))
        else:
            print("CASSCF not currently enabled")
            for i in self.states:
                tmp.append(0.0)

        for E, state in zip(tmp, self.states):
            self._Energies[state] = self.Energy(E, 'Hartree')

        # Save to file
        self.write_E_to_file()

        # Parse the output for Gradients
        if runtype == "gradient" or runtype == 'gh':
            tmpgrada = []
            tmpgrad = []
            tmpcoup = []
            count = 1
            with open(tempfileout, "r") as f:
                for line in f:
                    if line.startswith("* Nuclear energy gradient", 2) and count < 3:
                        next(f)
                        for i in range((len(geom)*4)):
                            findline = next(f, '').strip()
                            if findline.startswith("o Atom"):
                                pass
                            else:
                                mobj = re.match(
                                    r'. \s* ([-+]?[0-9]*\.?[0-9]+)', findline)
                                tmpgrad.append(float(mobj.group(1)))
                        tmpgrada.append(np.asarray(tmpgrad))
                        tmpgrad = []
                        count += 1
                    elif line.startswith("* Nuclear energy gradient", 2) and count == 3 and runtype == 'gh':
                        next(f)
                        for i in range((len(geom)*4)):
                            findline = next(f, '').strip()
                            if findline.startswith("o Atom"):
                                pass
                            else:
                                mobj = re.match(
                                    r'. \s* ([-+]?[0-9]*\.?[0-9]+)', findline)
                                tmpcoup.append(float(mobj.group(1)))
                        self.Couplings[self.coupling_states] = self.Coupling(
                            np.asarray(tmpcoup), 'Hartree/Bohr')
                        break

            for tup, tmpgrad in zip(self.gradient_states, tmpgrada):
                self._Gradients[tup] = self.Gradient(tmpgrad, 'Hartree/Bohr')

    def run(self, geom, runtype='gradient'):
        # first write the file, run it, and the read the output

        inpfilename = 'scratch/{}/bagel.json'.format(self.node_id)
        outfilename = 'scratch/{}/output.dat'.format(self.node_id)

        # Write
        self.write_input(geom, runtype)

        # Run BAGEL
        # RUN THE CALCULATION ###
        cmd = "BAGEL {} > {}".format(inpfilename, outfilename)
        os.system(cmd)
        os.system('wait')

        # Parse the output for Energies
        self.parse(geom, runtype)

        # Turn on guess for calculations after running
        if 'load_ref' not in self.file_options.ActiveOptions:
            self.file_options.set_active(
                'load_ref', 'scratch/{}/orbs'.format(self.node_id), str, '')
        return

    def runall(self, geom, runtype=None):
        ''' calculate all states with BAGEL '''

        if (not self.gradient_states and not self.coupling_states) or runtype == 'energy':
            print(" only calculating energies")
            # TODO what about multiple multiplicities?
            tup = self.states[0]
            self.run(geom, 'energy')
            # make grada all None
            for tup in self.states:
                self._Gradients[tup] = self.Gradients(None, None)
        elif self.gradient_states and self.coupling_states or runtype == 'gh':
            self.run(geom, 'gh')
        elif self.gradient_states and not self.coupling_states or runtype == 'gradient':
            self.run(geom, 'gradient')
        else:
            raise RuntimeError

        self.hasRanForCurrentCoords = True
        return


if __name__ == "__main__":
    # ,units.ANGSTROM_TO_AU)
    geom = manage_xyz.read_xyz('../../data/ethylene.xyz')
    B = BAGEL.from_options(states=[(1, 0), (1, 1)], gradient_states=[(
        1, 0), (1, 1)], coupling_states=(0, 1), geom=geom, lot_inp_file='bagel.txt', node_id=0)
    coords = manage_xyz.xyz_to_np(geom)
    E0 = B.get_energy(coords, 1, 0)
    E1 = B.get_energy(coords, 1, 1)
    g0 = B.get_gradient(coords, 1, 0)
    g1 = B.get_gradient(coords, 1, 1)
    c = B.get_coupling(coords, 1, 0, 1)
    print(E0, E1)
    print(g0.T)
    print(g1.T)
    print(c.T)

    # for line in B.file_options.record():
    #    print(line)

    # for key,value in B.file_options.ActiveOptions.items():
    #    print(key,value)
