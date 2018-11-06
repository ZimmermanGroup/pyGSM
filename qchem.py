from base_lot import *
import numpy as np
import os
from units import *

class QChem(Base):
    
    def run(self,geom,multiplicity):

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
        tempfile.write('{} {}\n'.format(self.charge,multiplicity))
        for coord in geom:
            for i in coord:
                tempfile.write(str(i)+' ')
            tempfile.write('\n')
        tempfile.write('$end')
        tempfile.close()

        cmd = "qchem -nt {} -save {} {}.qchem.out {}".format(self.nproc,tempfilename,tempfilename,multiplicity)
        os.system(cmd)
        
        efilepath = os.environ['QCSCRATCH']
        efilepath += '/{}/GRAD'.format(multiplicity)
        with open(efilepath) as efile:
            elines = efile.readlines()
        
        temp = 0
        for lines in elines:
            if temp == 1:
                self.E.append((multiplicity,float(lines.split()[0])))
                break
            if "$" in lines:
                temp += 1

        gradfilepath = os.environ['QCSCRATCH']
        gradfilepath += '/{}/GRAD'.format(multiplicity)

        with open(gradfilepath) as gradfile:
            gradlines = gradfile.readlines()
        temp = 0
        tmp=[]
        for lines in gradlines:
            if '$' in lines:
                temp+=1
            elif temp == 2:
                tmpline = lines.split()
                tmp.append([float(i) for i in tmpline])
            elif temp == 3:
                break
        self.grada.append((multiplicity,tmp))

        return 

    def runall(self,geom):
        self.E=[]
        self.grada = []
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets) 
        if len_singlets is not 0:
            self.run(geom,1)
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets) 
        if len_triplets is not 0:
            self.run(geom,3)
        self.hasRanForCurrentCoords=True

    def get_energy(self,geom,multiplicity,state):
        if self.has_nelectrons==False:
            for i in self.states:
                self.get_nelec(geom,i[0])
            self.has_nelectrons==True
        if self.hasRanForCurrentCoords==False:
            self.runall(geom)
        return self.getE(state,multiplicity)

    def getE(self,state,multiplicity):
        tmp = self.search_tuple(self.E,multiplicity)
        return np.asarray(tmp[state][1])*KCAL_MOL_PER_AU


    def get_gradient(self,geom,multiplicity,state):
        if self.has_nelectrons==False:
            for i in self.states:
                self.get_nelec(geom,i[0])
            self.has_nelectrons==True
        if self.hasRanForCurrentCoords==False:
            self.runall(geom)
        return self.getgrad(state,multiplicity)

    def getgrad(self,state,multiplicity):
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*ANGSTROM_TO_AU

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return QChem(QChem.default_options().set_values(kwargs))
    
if __name__ == '__main__':

    import dlc as ic
    import pybel as pb    
    import manage_xyz

    cwd = os.getcwd()
    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)   

    lot=QChem.from_options(states=[(1,0),(3,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    e=lot.get_energy(geom,1,0)
    print e
    e=lot.get_energy(geom,3,0)
    print e
    g=lot.get_gradient(geom,1,0)
    print g
    g=lot.get_gradient(geom,3,0)
    print g
