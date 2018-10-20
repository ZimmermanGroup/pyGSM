from base_lot import *
import numpy as np
import os
from units import *

class QChem(Base):
    
    def compute_energy(self,S=0,index=0):
        tempfilename = 'tempQCinp'
        tempfile = open(tempfilename,'w')
        tempfile.write(' $rem\n')
        tempfile.write(' JOBTYPE FORCE\n')
        tempfile.write(' EXCHANGE {}\n'.format(self.functional))
        tempfile.write(' SCF_ALGORITHM rca_diis\n')
        tempfile.write(' SCF_MAX_CYCLES 150\n')
        tempfile.write(' BASIS {}\n'.format(self.basis))
        tempfile.write(' WAVEFUNCTION_ANALYSIS FALSE\n')
        tempfile.write(' GEOM_OPT_MAX_CYCLES 300\n')
        tempfile.write('scf_convergence 6\n')
        tempfile.write(' SYM_IGNORE TRUE\n')
        tempfile.write(' SYMMETRY   FALSE\n')
        tempfile.write('molden_format true\n')
        tempfile.write(' $end\n')
        tempfile.write('\n')
        tempfile.write('$molecule\n')
        tempfile.write('0 {}\n'.format(S))
        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords) #updates geom
        for coord in self.geom:
            for i in coord:
                tempfile.write(str(i)+' ')
            tempfile.write('\n')
        tempfile.write('$end')
        tempfile.close()
        cmd = "qchem -np 1 -save {} {}.qchem.out{} {}.{}".format(tempfilename,tempfilename,index,S,index)
        os.system(cmd)
        
        efilepath = os.environ['QCSCRATCH']
        efilepath += '/{}.{}/GRAD'.format(S,index)
        with open(efilepath) as efile:
            elines = efile.readlines()
        
        temp = 0
        for lines in elines:
            if temp == 1:
                energy = float(lines.split()[0])
                break
            if "$" in lines:
                temp += 1
        return energy*KCAL_MOL_PER_AU

    def compute_gradient(self,S=0,index=0,*args):
        """ Assuming grad already computed in compute_energy"""
        gradfilepath = os.environ['QCSCRATCH']
        gradfilepath += '/{}.{}/GRAD'.format(S,index)
        with open(gradfilepath) as gradfile:
            gradlines = gradfile.readlines()
        gradient = []
        temp = 0
        for lines in gradlines:
            if '$' in lines:
                temp+=1
            elif temp == 2:
                tmpline = lines.split()
                gradient.append([float(i) for i in tmpline])
            elif temp == 3:
                break
        return np.asarray(gradient)*ANGSTROM_TO_AU

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return QChem(QChem.default_options().set_values(kwargs))
    
if __name__ == '__main__':

    import icoord as ic
    import pybel as pb    
    import manage_xyz

    cwd = os.getcwd()
    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)   

    lot=QChem.from_options(E_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP')
    e=lot.getEnergy()
    print e
    g=lot.getGrad()
    print g
