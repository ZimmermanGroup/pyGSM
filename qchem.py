from base_lot import *
import numpy as np
import os

class QChem(Base):
    
    #def getEnergy(self):
    #    print "Getting Energy"
    #    energy =0.
    #    average_over =0
    #    for i in self.calc_states:
    #        energy += self.compute_energy(average_over,S=i[0],index=i[1])
    #        average_over+=1

    #    final_energy = energy/average_over
    #    print "Energy: %.9f"%final_energy
    #    return final_energy

    #def getGrad(self):
    #    print "Getting Gradient"
    #    average_over=0
    #    grad = np.zeros((len(self.geom),3))
    #    for i in self.calc_states:
    #        tmp = self.read_gradients(average_over,S=i[0],index=i[1])
    #        print(np.shape(tmp[...]))
    #        grad += tmp[...] 
    #        average_over+=1
    #    final_grad = grad/average_over

    #    return np.reshape(final_grad,(3*len(self.geom),1))

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
        return energy

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
        return np.asarray(gradient)    

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return QChem(QChem.default_options().set_values(kwargs))
    
if __name__ == '__main__':

    from obutils import *
    import icoords as ic
    import pybel as pb    
    import manage_xyz

    cwd = os.getcwd()
    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)   

    lot=QChem.from_options(calc_states=[(1,0)],geom=geom,basis='6-31g(d)',functional='B3LYP')
    lot.getEnergy()
    lot.getGrad()
