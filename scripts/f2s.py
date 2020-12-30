# -*- coding: utf-8 -*-
"""
Create QM_region.xyz, MM_region.txt, link.txt, and frozen.txt for GSM job in QMMM system from a given freq output file.

python f2s.py qchem_freq_output.py

"""

import argparse

class QChemLog():
    """
    Represent an output file from QChem. The attribute `path` refers to the
    location on disk of the QChem output file of interest.
    """
    def __init__(self, path):
        self.path = path
        self.check()

    def check(self):
        is_freq = False
        if not self.is_QM_MM_INTERFACE:
            raise ValueError('This file is not for QMMM system')
        with open(self.path) as f:
            line = f.readline()
            while line != '':
                if 'VIBRATIONAL ANALYSIS' in line:
                    is_freq = True
                line = f.readline()
        if not is_freq:
            raise ValueError('This file is not freq job output file')

    def get_QM_ATOMS(self):
        """
        Return the index of QM_atoms.
        """
        QM_atoms = []

        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                if '$QM_ATOMS' in line.upper():
                    line = f.readline()
                    while '$end' not in line and line != '\n':
                        QM_atoms.append(line.strip())
                        line = f.readline()
                    break
                line = f.readline()
            
        return QM_atoms

    def is_QM_MM_INTERFACE(self):
        """
        Return the bool value.
        """
        is_QM_MM_INTERFACE = False

        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                if 'QM_MM_INTERFACE' in line.upper():
                    is_QM_MM_INTERFACE = True
                line = f.readline()

        return is_QM_MM_INTERFACE

    def get_USER_CONNECT(self):
        """
        Return a list of string with "<MM atom type> <Bond 1> <Bond 2> <Bond 3> <Bond 4>" .
        """
        USER_CONNECTS = []

        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                if '$MOLECULE' in line.upper():

                    for i in range(2):
                        line = f.readline()
                    while '$end' not in line and line != '\n':
                        _, _, _, _, MM_atom_type, Bond_1, Bond_2, Bond_3, Bond_4 = line.split()
                        USER_CONNECT = '{}  {}  {}  {}  {}\n'.format(MM_atom_type, Bond_1, Bond_2, Bond_3, Bond_4)
                        USER_CONNECTS.append(USER_CONNECT)
                        line = f.readline()
                    break
                line = f.readline()
        
        return USER_CONNECTS

    def get_fixed_molecule(self):
        """
        Return the string of fixed part in $moledule
        """
        # <Atom> <X> <Y> <Z>
        # O 7.256000 1.298000 9.826000
        # O 6.404000 1.114000 12.310000
        # O 4.077000 1.069000 0.082000
        # H 1.825000 1.405000 12.197000
        # H 2.151000 1.129000 9.563000
        # -----------------------------------

        fixed_molecule_string = ''

        n_atoms = len(self.get_QM_ATOMS())
        with open(self.path, 'r') as f:
            line = f.readline()
            while line != '':
                if '$MOLECULE' in line.upper():
                    for i in range(2):
                        line = f.readline()
                    for i in range(n_atoms):
                        line = f.readline()
                    while '$end' not in line and line != '\n':
                        a, x, y, z, _, _, _, _, _ = line.split()
                        line = '{:<2s}       {:.10f}     {:.10f}     {:.10f}\n'.format(a, float(x), float(y), float(z))
                        fixed_molecule_string += line
                        line = f.readline()
                    break
                line = f.readline()

        return fixed_molecule_string[:-1]
    
    def load_geometry(self):
        with open(self.path) as f:
            log = f.readlines()
        for line in reversed(log):
            if 'VIBRATIONAL ANALYSIS' in line:
                end_ind = log.index(line)
        start_ind = end_ind
        for line in reversed(log[:end_ind]):
            start_ind -= 1
            if 'Standard Nuclear Orientation' in line:
                break
        atom, coord = [], []
        geometry_flag = False
        # Now look for the geometry.
        # Will return the final geometry in the file under Standard Nuclear Orientation.
        for line in log[start_ind+3:end_ind]:
            if '------------' not in line:
                line.strip()
                data = line.split()
                atom.append(data[1])
                coord.append([float(c) for c in data[2:]])
            else:
                break
        self.natom = len(atom)
        self.nHcap = len(atom) - len(self.get_QM_ATOMS())
        return atom, coord

    def getXYZ(self):
        """
        Return a string of the molecule in the XYZ file format.
        """
        atoms, cart_coords = self.load_geometry()
        natom = len(atoms)
        xyz = str(natom) + '\n\n'
        for i in range(natom):
            xyz += '{:<2s}       {:.10f}     {:.10f}     {:.10f}\n'.format(atoms[i],cart_coords[i][0],cart_coords[i][1],cart_coords[i][2])
        return xyz
    
    def write_files(self):
        """
        Create files for GSM simulation in QMMM system.
        """
        geometry = self.getXYZ()
        g = open('QM_region.xyz', 'w')
        g.write(geometry)
        g.close()
        
        MM_xyz = self.get_fixed_molecule()
        m = open('MM_region.txt', 'w')
        m.write(MM_xyz)
        m.close()     
        
        link = self.get_USER_CONNECT()
        f = open('link.txt', 'w')
        f.writelines(link)
        f.close()        
        
        h = open('frozen.txt', 'w')
        for i in range(self.natom-self.nHcap, self.natom):
            h.write(str(i))
            h.write('\n')
        h.close()
        
parser = argparse.ArgumentParser(description='Qchem frequency files to SSM starting files')
parser.add_argument('file', metavar='FILE', type=str, nargs=1, 
                    help='a Q-Chem freq job output file')
args = parser.parse_args()
input_file = args.file[0]
log = QChemLog(path=input_file)
log.write_files()
