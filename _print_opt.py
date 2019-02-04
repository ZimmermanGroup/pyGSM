import openbabel as ob
import pybel as pb
from penalty_pes import Penalty_PES
from avg_pes import Avg_PES
import os 


class Print:
    def write_node(self,n,opt_molecules,energies,grmss,steps,deltaEs=[]):
        xyzfile=os.getcwd()+"/scratch/node_{}.xyz".format(n)
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
            if isinstance(self.icoords[0].PES,Penalty_PES) or isinstance(self.icoords[0].PES,Avg_PES):
                f.write('rms-step\n')
                for dE in deltaEs:
                    f.write('{}\n'.format(dE))
                
    def write_xyz_files(self,iters=0,base='xyzgeom',nconstraints=1):
        xyzfile = os.getcwd()+'/scratch/'+base+'_{:03}.xyz'.format(iters)
        stringxyz = pb.Outputfile('xyz',xyzfile,overwrite=True)
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat('xyz')
        opt_nodes = []
        for ico,act in zip(self.icoords,self.active):
            if act:
                mol = obconversion.WriteString(ico.mol.OBMol)
                opt_nodes.append(mol)
            elif ico != 0:
                mol = obconversion.WriteString(ico.mol.OBMol)
                opt_nodes.append(mol)

        for mol in opt_nodes:
            stringxyz.write(pb.readstring('xyz',mol))

        with open(xyzfile,'r+') as f:
            content = f.read()
            f.seek(0,0)
            f.write("[Molden Format]\n[Geometries] (XYZ)\n"+content)
#            print "writing geometries to",xyzfile
        with open(xyzfile, 'a') as f:
            f.write("[GEOCONV]\n")
            f.write('energy\n')
            V0=self.icoords[0].energy
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(ico.energy-V0))
                elif ico!=0:
                    f.write('{}\n'.format(ico.energy-V0))
            f.write("max-force\n")
            for ico,act in zip(self.icoords,self.active):
                if act:
                    f.write('{}\n'.format(float(ico.gradrms)))
                elif ico != 0:
                    f.write('{}\n'.format(float(ico.gradrms)))
            if isinstance(self.icoords[0].PES,Penalty_PES) or isinstance(self.icoords[0].PES,Avg_PES):
                print " WARNING: Printing dE as max-step in molden output "
                f.write("max-step\n")
                for ico,act in zip(self.icoords,self.active):
                    if ico!=0:
                        f.write('{}\n'.format(float(ico.PES.dE)))
        f.close()

    def write_node_xyz(self,xyzfile = "nodes_xyz_file.xyz"):
        xyzfile = os.getcwd()+"/scratch/"+xyzfile
        nodesXYZ = pb.Outputfile("xyz",xyzfile,overwrite=True)
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat('xyz')
        opt_mols = []
        for ico in self.icoords:
            if ico != 0:
                opt_mols.append(obconversion.WriteString(ico.mol.OBMol))
        for mol in opt_mols:
            nodesXYZ.write(pb.readstring("xyz",mol))

