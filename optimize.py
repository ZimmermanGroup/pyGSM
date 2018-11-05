import numpy as np
import openbabel as ob
import pybel as pb
import os
import StringIO 

def optimize(ic,nsteps,nconstraints=0):
    xyzfile=os.getcwd()+"/xyzfile.xyz"
    output_format = 'xyz'
    obconversion = ob.OBConversion()
    obconversion.SetOutFormat(output_format)
    opt_molecules=[]
    opt_molecules.append(obconversion.WriteString(ic.mol.OBMol))
    ic.V0 = ic.PES.get_energy(ic.geom)
    ic.energy=0
    grmss = []
    steps = []
    energies=[]
    Es =[]
    ic.do_bfgs=False # gets reset after each step
    ic.buf = StringIO.StringIO()

    print "Initial energy is %1.4f\n" % ic.V0
    ic.buf.write("\n Writing convergence:")


    for step in range(nsteps):
        if ic.print_level==1:
            print("\nOpt step: %i" %(step+1)),
        ic.buf.write("\nOpt step: %d" %(step+1))

        # => Opt step <= #
        smag =ic.opt_step(nconstraints)
        grmss.append(ic.gradrms)
        steps.append(smag)
        energies.append(ic.energy)
        opt_molecules.append(obconversion.WriteString(ic.mol.OBMol))

        #write convergence
        largeXyzFile =pb.Outputfile("xyz",xyzfile,overwrite=True)
        for mol in opt_molecules:
            largeXyzFile.write(pb.readstring("xyz",mol))
        with open(xyzfile,'r+') as f:
            content  =f.read()
            f.seek(0,0)
            f.write("[Molden Format]\n[Geometries] (XYZ)\n"+content)
        with open(xyzfile, "a") as f:
            f.write("[GEOCONV]\n")
            f.write("energy\n")
            for energy in energies:
                f.write('{}\n'.format(energy))
            f.write("max-force\n")
            for grms in grmss:
                f.write('{}\n'.format(grms))
            f.write("max-step\n")
            for step in steps:
                f.write('{}\n'.format(step))

        if ic.gradrms<ic.OPTTHRESH:
            break
    print(ic.buf.getvalue())
    print "Final energy is %2.5f" % (ic.V0 + ic.energy)
    return smag

if __name__ == '__main__':
    if 1:
        from pytc import *
        from pes import *
        from deloc_ics import *

        filepath="tests/stretched_fluoroethene.xyz"
        nocc=11
        nactive=2
        lot1=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot1.cas_from_file(filepath)
        pes = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
        mol1=pb.readfile("xyz",filepath).next()
        ic1=DLC.from_options(mol=mol1,PES=pes)
        optimize(ic1,50)
