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
import subprocess 

class QChem(Lot):
    def __init__(self,options):
        super(QChem,self).__init__(options)

        qcscratch = os.environ['QCSCRATCH']
        for state in self.states:
            tempfolder = qcscratch + '/string_{:03d}/{}.{}/'.format(self.ID,self.node_id,state[0])
            print(" making temp folder {}".format(tempfolder))
            os.system('mkdir -p {}'.format(tempfolder))

        copy_input_file = os.getcwd() + "/QChem_input.txt"
        print(copy_input_file)
        self.write_preamble(self.geom,self.states[0][0],copy_input_file)
        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

    def write_preamble(self,geom,multiplicity,tempfilename,jobtype='FORCE'):

        tempfile = open(tempfilename,'w')
        if not self.lot_inp_file:
            tempfile.write(' $rem\n')
            tempfile.write(' JOBTYPE {}\n'.format(jobtype))
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
    
    def run(self,geom,multiplicity,ad_idx,runtype='gradient'):

        assert ad_idx == 0,"pyGSM Q-Chem doesn't currently support ad_idx!=0"

        qcscratch = os.environ['QCSCRATCH']
        tempfilename = qcscratch + '/string_{:03d}/{}.{}/tempQCinp'.format(self.ID,self.node_id,multiplicity)

        if self.calc_grad and runtype!="energy":
           self.write_preamble(geom,multiplicity,tempfilename)
        else:
           self.write_preamble(geom,multiplicity,tempfilename,jobtype='SP')
        
        cmd = ['qchem']
        args = ['-nt',str(self.nproc),
                '-save',
                tempfilename,
                '{}.qchem.out'.format(tempfilename),
                'string_{:03d}/{}.{}'.format(self.ID,self.node_id,multiplicity)
                ]
        cmd.extend(args)

        #  Run the process
        output = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr = subprocess.PIPE).communicate()[0]

        self.parse()
       
        return 

    def parse(self):
        # PARSE OUTPUT #
        if self.calc_grad:
            efilepath = qcscratch + '/string_{:03d}/{}.{}/GRAD'.format(self.ID,self.node_id,multiplicity)
            with open(efilepath) as efile:
                elines = efile.readlines()
            
            temp = 0
            for lines in elines:
                if temp == 1:
                    # defaulting to the ground-state
                    self._Energies[(multiplicity,0)] = self.Energy(float(lines.split()[0],'Hartree'))
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
            self._Gradients[(multiplicity,0)] = self.Gradient(tmp,'Hartree/Bohr')
        else:
            raise NotImplementedError
        self.write_E_to_file()

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

