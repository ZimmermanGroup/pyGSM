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

class Orca(Lot):
   
    def write_input_file(self, geom, multiplicity):
        if self.lot_inp_file == False:
            inpstring = '!'
            inpstring += ' '+self.functional
            inpstring += ' '+self.basis
            inpstring += ' EnGrad\n\n'# SOSCF SlowConv \n\n'
            inpstring += '%scf\nMaxIter 300\nconvergence strong\n sthresh 1e-7\n'
            inpstring += 'thresh 1e-11\n tcut 1e-13 \n directresetfreq 1 \n SOSCFStart 0.00033\nend\n'
            #inpstring += '%scf\nMaxIter 300\nend\n'
            inpstring += '\n%maxcore 1000\n\n'
            inpstring += '%pal\nnproc {}\nend\n\n'.format(self.nproc)
        else:
            inpstring = ''
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                inpstring += line

        inpstring += '\n*xyz {} {}\n'.format(self.charge,multiplicity)
        for coord in geom:
            for i in coord:
                inpstring += str(i)+' '
            inpstring += '\n'
        inpstring += '*'
        tempfilename = 'tempORCAinp_{}'.format(multiplicity)
        tempfile = open(tempfilename,'w')
        tempfile.write(inpstring)
        tempfile.close()
        return tempfilename

    def run(self,geom,multiplicity,ad_idx,runtype='gradient'):

        assert ad_idx == 0,"pyGSM ORCA doesn't currently support ad_idx!=0"
        
        # Write input file
        tempfilename = self.write_input_file(geom, multiplicity)

        path2orca = os.popen('which orca').read().rstrip()
        user = os.environ['USER']
        cwd = os.environ['PWD']
        try:
            slurmID = os.environ['SLURM_ARRAY_JOB_ID']
            try:
                slurmTASK = os.environ['SLURM_ARRAY_TASK_ID']
                runscr = '/tmp/'+user+'/'+slurmID+'/'+slurmTASK
            except:
                runscr = '/tmp/'+user+'/'+slurmID
        except:
            pbsID = os.environ['PBS_JOBID']
            orcascr = 'temporcarun'
            #runscr = '/tmp/'+user+'/'+orcascr
            runscr ='/tmp/'+pbsID+'/'+orcascr

        os.system('mkdir -p {}'.format(runscr))
        os.system('mv {} {}/'.format(tempfilename,runscr))
        cmd = 'cd {}; {} {} > {}/{}.log; cd {}'.format(runscr,path2orca,tempfilename,runscr,tempfilename,cwd)
        os.system(cmd)

        # parse output
        self.parse(multiplicity, runscr, tempfilename)

        return

    def parse(self, multiplicity, runscr, tempfilename):
        engradpath = runscr+'/{}.engrad'.format(tempfilename) 
        with open(engradpath) as engradfile:
            engradlines = engradfile.readlines()

        temp = 100000
        for i,lines in enumerate(engradlines):
            if '# The current total energy in Eh\n' in lines:
                temp = i
            if i > temp+1:
                self._Energies[(multiplicity,0)] = self.Energy(float(lines.split()[0]),'Hartree')
                break

        temp = 100000
        tmp = []
        tmp2 = []
        for i,lines in enumerate(engradlines):
            if '# The current gradient in Eh/bohr\n' in lines:
                temp = i
            if  i> temp+1:
                if "#" in lines:
                    break
                tmp2.append(float(lines.split()[0]))
            if len(tmp2) == 3:
                tmp.append(tmp2)
                tmp2 = []
        self._Gradients[(multiplicity,0)] = self.Gradient(np.asarray(tmp),'Hartree/Bohr')

