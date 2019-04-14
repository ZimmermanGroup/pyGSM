import os 
import manage_xyz


class Print:
                
    def write_xyz_files(self,iters=0,base='xyzgeom',nconstraints=1):
        xyzfile = os.getcwd()+'/scratch/'+base+'_{:03}.xyz'.format(iters)
        geoms = []
        for ico in self.nodes:
            if ico != None:
                geoms.append(ico.geometry)

        with open(xyzfile,'w') as f:
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
            V0=self.nodes[0].energy
            for n,ico in enumerate(self.nodes):
                if ico!=None:
                    f.write('{}\n'.format(ico.energy-V0))
                    #f.write('{}\n'.format(self.energies[n]))
            f.write("max-force\n")
            for ico in self.nodes:
                if ico != None:
                    f.write('{}\n'.format(float(ico.gradrms)))
            #print(" WARNING: Printing dE as max-step in molden output ")
            f.write("max-step\n")
            for ico,act in zip(self.nodes,self.active):
                if ico!=None:
                    f.write('{}\n'.format(float(ico.difference_energy)))
        f.close()

