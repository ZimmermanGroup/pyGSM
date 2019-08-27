# standard library imports
import sys
import os
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .base_lot import Lot
from utilities import *

class QChem(Lot):
    def __init__(self,options):
        super(QChem,self).__init__(options)

        qcscratch = os.environ['QCSCRATCH']
        for state in self.states:
            tempfolder = qcscratch + '/string_{:03d}/{}.{}/'.format(self.ID,self.node_id,state[0])
            print(" making temp folder {}".format(tempfolder))
            os.system('mkdir -p {}'.format(tempfolder))
    
    def run(self,geom,multiplicity):

        qcscratch = os.environ['QCSCRATCH']
        tempfilename = qcscratch + '/string_{:03d}/{}.{}/tempQCinp'.format(self.ID,self.node_id,multiplicity)

        tempfile = open(tempfilename,'w')
        if not self.lot_inp_file:
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
        
        cmd = "qchem -nt {} -save {} {}.qchem.out string_{:03d}/{}.{}".format(self.nproc,tempfilename,tempfilename,self.ID,self.node_id,multiplicity)
        #print(cmd)

        os.system(cmd)
        
        efilepath = qcscratch + '/string_{:03d}/{}.{}/GRAD'.format(self.ID,self.node_id,multiplicity)
        with open(efilepath) as efile:
            elines = efile.readlines()
        
        temp = 0
        for lines in elines:
            if temp == 1:
                self.E.append((multiplicity,float(lines.split()[0])))
                break
            if "$" in lines:
                temp += 1

        with open(efilepath) as efile:
            gradlines = efile.readlines()
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
    def copy(cls,lot,options,copy_wavefunction=True):
        base = os.environ['QCSCRATCH']
        node_id = options.get('node_id',1)

        if node_id != lot.node_id:  #and copy_wavefunction: # other theories are more sensitive than qchem -- commenting out
            for state in lot.states:
                multiplicity = state[0]
                efilepath_old=base+ '/string_{:03d}/{}.{}'.format(lot.ID,lot.node_id,multiplicity)
                efilepath_new =base+ '/string_{:03d}/{}.{}'.format(lot.ID,node_id,multiplicity)
                cmd = 'cp -r ' + efilepath_old +' ' + efilepath_new
                print(" copying QCSCRATCH files\n {}".format(cmd))
                os.system(cmd)
        return cls(lot.options.copy().set_values(options))

