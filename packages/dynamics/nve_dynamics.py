import numpy as np
import vv
from nifty import click
from time import time
from sys import exit


def nve_dynamics(
        mol,
        dt,
        tmax,
        xyzframerate=4,
        ):

    #t0=time()
    V0 = mol.energy
    #t1 = time()
    #print(" time to get V0 %.3f" % (t1-t0))
    F0 = mol.gradient
    #t2 = time()
    #print(" time to get F0 %.3f" % (t2-t1))
    X0 = mol.xyz
    #t3 = time()
    #print(" time to get X0 %.3f" % (t3-t2))
    print(X0.shape)
    M =  np.ones(X0.shape)*mol.atomic_mass[:,np.newaxis]
    print(M)
    P0 = np.zeros_like(M)
    exit()

    prop = vv.VV(dt,M,X0,P0,F0,V0)

    stats = {
        't' : [],
        'T' : [],
        'V' : [],
        'E' : [],
    }

    print 'NVE: %5s %14s %24s %24s %24s' % (
        "I",
        "t", 
        "T",
        "V",
        "E",
        )

    while prop.t < tmax:

        print 'NVE: %5d %14.6f %24.16E %24.16E %24.16E' % (
            prop.I,
            prop.t,
            prop.T,
            prop.V,
            prop.E,
            )
        stats['t'].append(prop.t)
        stats['T'].append(prop.T)
        stats['V'].append(prop.V)
        stats['E'].append(prop.E)

        if prop.I % xyzframerate == 0:
            #TODO
            pass
            #lot.molecule.save_xyz_file(xyzfilename,append=(False if prop.I == 0 else True))

        lot = lot.update_xyz(prop.Xnew)
        V = lot.compute_energy(S, index)
        F = lot.compute_gradient(S, index)
        F.scale(-1.0)

        prop.step(F,V)

    stats = { key : np.array(val) for key, val in stats.iteritems() }

    return stats

#def momentum_vector(M, T):
#
#    return ls.Tensor.zeros_like(M) # TODO


if __name__=="__main__":
    import pybel as pb
    from nifty import custom_redirection,getAllCoords,getAtomicSymbols,click,printcool,pvec1d,pmat2d
    from openmm import OpenMM
    import simtk.unit as openmm_units
    import simtk.openmm.app as openmm_app
    import simtk.openmm as openmm
    import manage_xyz
    from pes import PES
    from molecule import Molecule

    # Create and initialize System object from prmtop/inpcrd
    prmtopfile='data/dimer.prmtop'
    prmtop = openmm_app.AmberPrmtopFile(prmtopfile)
    system = prmtop.createSystem(
        rigidWater=False, 
        removeCMMotion=False,
        )

    # Integrator will never be used (Simulation requires one)
    integrator = openmm.VerletIntegrator(1.0)
    simulation = openmm_app.Simulation(
        prmtop.topology,
        system,
        integrator,
        )

    # create lot
    mol=next(pb.readfile('pdb','data/dimer_h2o.pdb'))
    coords = getAllCoords(mol)
    atoms = getAtomicSymbols(mol)
    geom= manage_xyz.combine_atom_xyz(atoms,coords)
    lot = OpenMM.from_options(states=[(1,0)],job_data={'simulation':simulation},geom=geom)

    # create pes
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

    # create molecule
    M = Molecule.from_options(fnm='data/dimer_h2o.pdb',PES=pes,coordinate_type="TRIC",Form_Hessian=False)
    print("done making objects")

    # set up dynamics    
    stats = nve_dynamics(
        M,
        dt=20.0,
        tmax=5000.0,
        ) 
    
