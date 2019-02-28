from base_lot import *
import numpy as np
import os
from units import *

class QChem(Lot):
    
    def run(self,geom,multiplicity):

        tempfilename = 'tempQCinp'
        tempfile = open(tempfilename,'w')
        if self.lot_inp_file == False:
            tempfile.write(' $rem\n')
            tempfile.write(' JOBTYPE FORCE\n')
            tempfile.write(' EXCHANGE {}\n'.format(self.functional))
            tempfile.write(' SCF_ALGORITHM rca_diis\n')
            tempfile.write(' SCF_MAX_CYCLES 300\n')
            tempfile.write(' BASIS {}\n'.format(self.basis))
            #tempfile.write(' ECP LANL2DZ \n')
            tempfile.write(' WAVEFUNCTION_ANALYSIS FALSE\n')
            tempfile.write(' GEOM_OPT_MAX_CYCLES 300\n')
            tempfile.write('scf_convergence 6\n')
            tempfile.write(' SYM_IGNORE TRUE\n')
            tempfile.write(' SYMMETRY   FALSE\n')
            tempfile.write('molden_format true\n')
            tempfile.write(' $end\n')
            tempfile.write('\n')
            tempfile.write('$molecule\n')
        else:
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                tempfile.write(line)

        tempfile.write('{} {}\n'.format(self.charge,multiplicity))
        if os.path.isfile("link.txt"):
            with open("link.txt") as link:
                link_lines = link.readlines()
            tmp_geom = [list(i) for i in geom]
            for i,coord in enumerate(tmp_geom):
                coord.append(link_lines[i].rstrip('\n'))
                for i in coord:
                    tempfile.write(str(i)+' ')
                tempfile.write('\n')
        else:
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
    def copy(QChemA,node_id):
        """ create a copy of this lot object"""
        return QChem(QChemA.options.copy().set_values({
            "node_id" :node_id,
            }))

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return QChem(QChem.default_options().set_values(kwargs))
    
if __name__ == '__main__':

    import hybrid_dlc 
    import pybel as pb    
    import manage_xyz
    import pes

    cwd = os.getcwd()
    #filepath="examples/tests/fluoroethene.xyz"
    filepath="firstnode.pdb"
    geom=manage_xyz.read_xyz(filepath,scale=1)   

    #lot=QChem.from_options(states=[(2,0)],charge=1,basis='6-31g(d)',functional='B3LYP')
    lot = QChem.from_options(states=[(2,0)],lot_inp_file='qstart')
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=2)
    ic = Hybrid_DLC(mol=mol,pes=pes,IC_region=["UNL"])
    #e=lot.get_energy(geom,2,0)
    #print e
    #g=lot.get_gradient(geom,2,0)
    #print g
