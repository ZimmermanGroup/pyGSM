# standard library imports
import sys
import os
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from base_lot import Lot
from utilities import *

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
        
        cmd = "qchem -nt {} -save {} {}.qchem.out {}.{}".format(self.nproc,tempfilename,tempfilename,self.node_id,multiplicity)

        os.system(cmd)
        
        efilepath = os.environ['QCSCRATCH']
        efilepath += '/{}.{}/GRAD'.format(self.node_id,multiplicity)
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
        gradfilepath += '/{}.{}/GRAD'.format(self.node_id,multiplicity)

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

    def get_energy(self,coords,multiplicity,state):
        #if self.has_nelectrons==False:
        #    for i in self.states:
        #        self.get_nelec(geom,i[0])
        #    self.has_nelectrons==True
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.runall(geom)
            self.hasRanForCurrentCoords=True
        tmp = self.search_tuple(self.E,multiplicity)
        return np.asarray(tmp[state][1])*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        #if self.has_nelectrons==False:
        #    for i in self.states:
        #        self.get_nelec(geom,i[0])
        #    self.has_nelectrons==True
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.runall(geom)
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*units.ANGSTROM_TO_AU

    @classmethod
    def copy(cls,lot,**kwargs):
        base = os.environ['QCSCRATCH']
        node_id = kwargs.get('node_id',1)
        for state in self.states:
            multiplicity = state[0]
            efilepath_old=base+ '/{}.{}'.format(lot.node_id,multiplicity)
            efilepath_new =base+ '/{}.{}'.format(node_id,multiplicity)
            os.system('cp -r ' + efilepath_old +' ' + efilepath_new)
        return cls(lot.options.copy().set_values(kwargs))

