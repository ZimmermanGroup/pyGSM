from base_lot import *
import numpy as np
import os
from units import *

class Orca(Lot):
    
    def run(self,geom,multiplicity):
        
        inpstring = '!'
        inpstring += ' '+self.functional
        inpstring += ' '+self.basis
        inpstring += ' EnGrad\n\n'# SOSCF SlowConv \n\n'
        inpstring += '%scf\nMaxIter 150\nconvergence medium\n sthresh 1e-7\n'
        inpstring += 'thresh 1e-11\n tcut 1e-13 \n directresetfreq 1 \nend\n'
        inpstring += '%pal\nnproc {}\nend\n\n'.format(self.nproc)
        inpstring += '*xyz {} {}\n'.format(self.charge,multiplicity)
        for coord in geom:
            for i in coord:
                inpstring += str(i)+' '
            inpstring += '\n'
        inpstring += '*'
        tempfilename = 'tempORCAinp_{}'.format(multiplicity)
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

        temp = 100000
        for i,lines in enumerate(engradlines):
            if 'current total energy' in lines:
                temp = i
            if i > temp+1:
                self.E.append((multiplicity,float(lines.split()[0]))) 
                break

        temp = 100000
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
        return np.asarray(tmp[state][1])#*ANGSTROM_TO_AU #ORCA grad is given in AU

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

if __name__ == '__main__':
    from se_xing import *
    filepath = 'tests/SiH4.xyz'
    basis = '6-31G*'
    lot = Orca.from_options(states=[(1,0)],charge=0,basis=basis,functional='B3LYP',nproc=4)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    print 'ic1'
    mol=pb.readfile('xyz',filepath).next()
    ic1 = DLC.from_options(mol=mol,PES=pes,print_level=1,resetopt=False)
    ic1.PES.get_energy(ic1.geom)
    print 'getting gradient'
    ic1.PES.get_gradient(ic1.geom)
    print 'printing gradient -----------'
    print ic1.PES.lot.grada
