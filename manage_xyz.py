import numpy as np
import units 
import re
import openbabel as ob

# => XYZ File Utility <= #

def read_xyz(
    filename, 
    scale=units.ANGSTROM_TO_AU):

    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    
    lines = open(filename).readlines()
    lines = lines[2:]
    geom = []
    for line in lines:
        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
        geom.append((
            mobj.group(1),
            scale*float(mobj.group(2)),
            scale*float(mobj.group(3)),
            scale*float(mobj.group(4)),
            ))
    return geom

def get_atoms(
        geom
        ):
    atoms=[]
    for atom in geom:
        atoms.append(atom[0])
    return atoms

def write_xyz(
    filename, 
    geom, 
    charge=0,
    scale=(1.0/units.ANGSTROM_TO_AU),
    ):

    """ Writes xyz file with single frame

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    fh = open(filename,'w')
    fh.write('%d\n' % len(geom))
    fh.write('%i\n' % charge)
    for atom in geom:
        fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
            atom[0],
            scale*atom[1],
            scale*atom[2],
            scale*atom[3],
            ))

def write_xyzs(
    filename, 
    geoms, 
    scale=(1.0/units.ANGSTROM_TO_AU),
    ):

    """ Writes xyz trajectory file with multiple frames

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:

    """

    fh = open(filename,'w')
    for geom in geoms:
        fh.write('%d\n\n' % len(geom))
        for atom in geom:
            fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
                atom[0],
                scale*atom[1],
                scale*atom[2],
                scale*atom[3],
                ))

def xyz_to_np(
    geom,
    ):

    """ Convert from xyz file format xyz array for editing geometry

    Params:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:
        xyz ((natoms,3) np.ndarray) - system geometry (x,y,z)

    """

    xyz2 = np.zeros((len(geom),3))
    for A, atom in enumerate(geom):
        xyz2[A,0] = atom[1]
        xyz2[A,1] = atom[2]
        xyz2[A,2] = atom[3]
    return xyz2

def np_to_xyz(
    geom, 
    xyz2,
    ):

    """ Convert from xyz array to xyz file format in order to write xyz

    Params:
        geom ((natoms,4) np.ndarray) - system reference geometry (atom symbol, x,y,z) from xyz file
        xyz2 ((natoms,3) np.ndarray) - system geometry (x,y,z)

    Returns:
        geom2 ((natoms,4) np.ndarray) - new system geometry (atom symbol, x,y,z)

    """

    geom2 = []
    for A, atom in enumerate(geom):
        geom2.append((
            atom[0],
            xyz2[A,0],
            xyz2[A,1],
            xyz2[A,2],
            ))
    return geom2

def combine_atom_xyz(
    atoms,
    xyz,
    ):
    """ Combines atom list with xyz array 
    
     Params:
        atom list
        geom ((natoms,3) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:
        geom2 ((natoms,4) np.ndarray) - new system geometry (atom symbol, x,y,z)

    """
    geom2 = []
    for A, atom in enumerate(atoms):
        geom2.append((
            atom,
            xyz[A,0],
            xyz[A,1],
            xyz[A,2],
            ))
    return geom2

