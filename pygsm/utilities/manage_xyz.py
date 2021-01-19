import numpy as np
try:
    from . import units 
except:
    import units

import re
#import openbabel as ob

# => XYZ File Utility <= #

def read_xyz(
    filename, 
    scale=1.):

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


def read_xyzs(
    filename, 
    scale=1.
    ):

    """ Read xyz file

    Params:
        filename (str) - name of xyz file to read

    Returns:
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    
    lines = open(filename).readlines()
    natoms = int(lines[0])
    total_lines = len(lines)
    num_geoms = total_lines/(natoms+2)

    geoms = []
    sa=2
    for i in range(int(num_geoms)):
        ea=sa+natoms
        geom=[]
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
                ))
        sa=ea+2
        geoms.append(geom)
    return geoms

def read_molden_geoms(
    filename, 
    scale=1.
    ):

    lines = open(filename).readlines()
    natoms=int(lines[2])
    nlines = len(lines)

    #print "number of atoms is ",natoms
    num_geoms = (nlines-6)/ (natoms+5) #this is for three blocks after GEOCON
    num_geoms = int(num_geoms)
    print(num_geoms)
    geoms = []

    sa=4
    for i in range(int(num_geoms)):
        ea=sa+natoms
        geom=[]
        for line in lines[sa:ea]:
            mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            geom.append((
                mobj.group(1),
                scale*float(mobj.group(2)),
                scale*float(mobj.group(3)),
                scale*float(mobj.group(4)),
                ))
        sa=ea+2
        geoms.append(geom)
    return geoms

def read_molden_Energy(
        filename,
        ):
    with open(filename) as f:
        nlines = sum(1 for _ in f)
    #print "number of lines is ", nlines
    with open(filename) as f:
        natoms = int(f.readlines()[2])

    #print "number of atoms is ",natoms
    nstructs = (nlines-6)/ (natoms+5) #this is for three blocks after GEOCON
    nstructs = int(nstructs)
    
    #print "number of structures in restart file is %i" % nstructs
    coords=[]
    E = [0.]*nstructs
    grmss = []
    atomic_symbols=[]
    dE = []
    with open(filename) as f:
        f.readline()
        f.readline() #header lines
        # get coords
        for struct in range(nstructs):
            tmpcoords=np.zeros((natoms,3))
            f.readline() #natoms
            f.readline() #space
            for a in range(natoms):
                line=f.readline()
                tmp = line.split()
                tmpcoords[a,:] = [float(i) for i in tmp[1:]]
                if struct==0:
                    atomic_symbols.append(tmp[0])
        coords.append(tmpcoords)
        # Get energies
        f.readline() # line
        f.readline() #energy
        for struct in range(nstructs):
            E[struct] = float(f.readline())

    return E

        
def write_molden_geoms(
        filename,
        geoms,
        energies,
        gradrms,
        dEs,
        ):
        with open(filename,'w') as f:
            f.write("[Molden Format]\n[Geometries] (XYZ)\n")
            for geom in geoms:
                f.write('%d\n\n' % len(geom))
                for atom in geom:
                    f.write('%-2s %14.6f %14.6f %14.6f\n' % (
                        atom[0],
                        atom[1],
                        atom[2],
                        atom[3],
                        ))
            f.write("[GEOCONV]\n")
            f.write('energy\n')
            V0=energies[0]
            for energy in energies:
                f.write('{}\n'.format(energy-V0))
            f.write("max-force\n")
            for grad in gradrms:
                f.write('{}\n'.format(float(grad)))
            #print(" WARNING: Printing dE as max-step in molden output ")
            f.write("max-step\n")
            for dE in dEs:
                f.write('{}\n'.format(float(dE)))

def get_atoms(
        geom,
        ):

    atoms=[]
    for atom in geom:
        atoms.append(atom[0])
    return atoms

def write_xyz(
    filename, 
    geom, 
    comment=0,
    scale=1.0 #(1.0/units.ANGSTROM_TO_AU),
    ):

    """ Writes xyz file with single frame

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    """
    fh = open(filename,'w')
    fh.write('%d\n' % len(geom))
    fh.write('{}\n'.format(comment))
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
    scale=1.,
    #scale=(1.0/units.ANGSTROM_TO_AU),
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

def write_amber_xyz(
        filename,
        geom,
        ):

    count=0
    fh = open(filename,'w')
    fh.write("default name\n")
    fh.write('  %d\n' % len(geom))
    for line in geom:
        for elem in line[1:]:
            fh.write(" {:11.7f}".format(float(elem)))
            count+=1
        if count % 6 == 0:
            fh.write("\n")


def write_xyzs_w_comments(
    filename, 
    geoms,
    comments,
    scale=1.0 #(1.0/units.ANGSTROM_TO_AU),
    ):

    """ Writes xyz trajectory file with multiple frames

    Params:
        filename (str) - name of xyz file to write
        geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)

    Returns:

    """

    fh = open(filename,'w')
    for geom,comment in zip(geoms,comments):
        fh.write('%d\n' % len(geom))
        fh.write('%s\n' % comment)
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
