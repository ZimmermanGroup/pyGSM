import lightspeed as ls
import rhf
import dynamics
import sys
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Timestamp
start_time = time.time()
print '*** Started at: %s on %s ***' % (
           time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime()),
           os.uname()[1],
           )

molname = sys.argv[1]
basisname = sys.argv[2]

resources = ls.ResourceList.build(1024,1024)

molecule = ls.Molecule.from_xyz_file(molname)

ref = rhf.RHF.build(
    resources,
    molecule,
    basisname=basisname,
    options = {
        'print_orbitals' : False,
        'g_convergence' : 1.0E-6,
    })

M = dynamics.mass_vector(molecule)

P0 = dynamics.momentum_vector(M,0.0)

stats = dynamics.nve_dynamics(
    ref,
    M,
    P0,
    dt=20.0,
    tmax=5000.0,
    ) 

plt.clf()
plt.plot(stats['t'],stats['T']-stats['T'][0],label='T')
plt.plot(stats['t'],stats['V']-stats['V'][0],label='V')
plt.plot(stats['t'],stats['E']-stats['E'][0],label='E')
plt.xlabel('t')
plt.ylabel('E')
plt.legend()
plt.savefig('E.pdf')

# Timestamp
stop_time = time.time()
print '*** Stopped at: %s on %s ***' % (
           time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime()),
           os.uname()[1],
           )
print '*** Runtime: %.3f [s] ***\n' % (
        stop_time - start_time)
