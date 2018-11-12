from base_lot import *
import numpy as np
import os
from units import *

class Orca(Lot):
    
    def run(self,geom,multiplicity):
        
        inpstring = '!'
        inpstring += ' '+self.functional
        inpstring += ' '+self.basis
        inpstring += ' EnGrad\n\n'
        inpstring += '%scf\nMaxIter 150\nend\n\n'
        inpstring += '%pal\nnproc {}\nend\n\n'.format(self.nproc)
        inpstring += '*xyz {} {}\n'.format(self.charge,multiplicity)
        for coord in geom:
            for i in coord:
                inpstring += str(i)+' '
            inpstring += '\n'
        inpstring += '*'
        tempfilename = 'tempORCAinp'
        tempfile = open(tempfilename,'w')
        tempfile.write(inpstring)
        tempfile.close()
        
        path2orca = os.popen('which orca').read().rstrip()
#        path2orca = '/export/zimmerman/khyungju/orca_4_0_0_2_linux_x86-64/orca'
        orcascr = 'temporcarun'
        os.system('mkdir -p {}'.format(orcascr))
        os.system('mv {} {}/'.format(tempfilename,orcascr))
        cmd = 'cd {}; {} {} > {}.log; cd ..'.format(orcascr,path2orca,tempfilename,tempfilename)
        os.system(cmd)

        engradpath = 'temporcarun/{}.engrad'.format(tempfilename) 
        with open(engradpath) as engradfile:
            engradlines = engradfile.readlines()

        temp = 1000
        for i,lines in enumerate(engradlines):
            if 'current total energy' in lines:
                temp = i
            if i > temp+1:
                self.E.append((multiplicity,float(lines.split()[0]))) 
                break

        temp = 1000
        tmp = []
        tmp2 = []
        for i,lines in enumerate(engradlines):
            if 'current gradient' in lines:
                temp = i
            if  i> temp+1:
                if "#" in lines:
                    break
                tmp2.append(float(lines.split()[0]))
            if len(tmp2) == 3:
                tmp.append(tmp2)
                tmp2 = []
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
    def copy(OrcaA,node_id):
        """ create a copy of this lot object"""
        return Orca(OrcaA.options.copy().set_values({
            "node_id" :node_id,
            }))
        
    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Orca(Orca.default_options().set_values(kwargs))

