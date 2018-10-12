import numpy as np
import units 
import atom_data
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

def write_xyz(
    filename, 
    geom, 
    scale=(1.0/units.ANGSTROM_TO_AU),
    ):

    """ Writes xyz file with single frame

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    fh = open(filename,'w')
    fh.write('%d\n\n' % len(geom))
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

def write_fms90(
    filename,
    geomx,  
    geomp=None,
    ):

    """ Write fms90 geometry file with position and velocities

    Params:
        filename (str) - name of fms90 geometry file to write
        geomx ((natoms,4) np.ndarray) - system positions (atom symbol, x,y,z)
        geomp ((natoms,4) np.ndarray) - system momenta (atom symbol, px, py, pz)

    """

    fh = open(filename,'w')
    fh.write('UNITS=BOHR\n')
    fh.write('%d\n' % len(geomx))
    for atom in geomx:
        fh.write('%-2s %14.6f %14.6f %14.6f\n' % (
            atom[0],
            atom[1],
            atom[2],
            atom[3],
            ))
    if geomp:
        fh.write('# momenta\n')
        for atom in geomp:
            fh.write('  %14.6f %14.6f %14.6f\n' % (
                atom[1],
                atom[2],
                atom[3],
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
    for N,atom in enumerate(atoms):
        tmp = list(atom) + [x for x in xyz[N]] 
        geom2.append(tmp)
    return geom2

def create_OBMol(
    coordinates,
    ):
    
    mol = ob.OBMol() 
    for entry in coordinates: 
           #print(entry)
           #print atom_data.atom_symbol_table.keys()[atom_data.atom_symbol_table.values().index(entry[0])]
           newAtom = mol.NewAtom() 
           newAtom.SetAtomicNum(atom_data.atom_symbol_table.keys()[atom_data.atom_symbol_table.values().index(entry[0])])
           X=[ x/units.units['au_per_ang'] for x in entry[1:]]
           #print(X)
    #for obatom in ob.OBMolAtomIter(mol):
    #    print(obatom.GetVector())
    #atom=mol.GetAtom(1)
    #print(atom)
    #print(mol.GetCoordinates())
    #print(mol.OBMol.GetCoordinates())
    #pybelmol = pybel.Molecule(mol)
    #pybelmol.write("xyz", "outputfile.xyz")
    return mol

