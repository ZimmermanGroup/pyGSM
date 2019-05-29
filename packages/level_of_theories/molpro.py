# standard library imports
import sys
import os
from os import path
import re

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from base_lot import Lot
from utilities import *

class Molpro(Lot):

    # designed to do multiple multiplicities at once... maybe not such a good idea, that feature is currently broken
    #TODO 
    def run(self,geom):
        if self.has_nelectrons==False:
            for i in self.states:
                self.get_nelec(geom,i[0])
            self.has_nelectrons==True

        #TODO gopro needs a number
        tempfilename = 'scratch/gopro.com'
        tempfile = open(tempfilename,'w')
        tempfile.write(' file,2,mp_0000_{:04d}\n'.format(self.node_id))
        tempfile.write(' memory,800,m\n')
        tempfile.write(' symmetry,nosym\n')
        tempfile.write(' orient,noorient\n\n')
        tempfile.write(' geometry={\n')
        for coord in geom:
            for i in coord:
                tempfile.write(' '+str(i))
            tempfile.write('\n')
        tempfile.write('}\n\n')
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets)
        len_E_singlets=singlets[-1][1] +1 #final adiabat +1 because 0 indexed
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets)

        if self.lot_inp_file == False:
            tempfile.write(' basis={}\n\n'.format(self.basis))
            if len_singlets is not 0:
                tempfile.write(' {multi\n')
                nclosed = self.nocc
                nocc = nclosed+self.nactive
                tempfile.write(' direct\n')
                tempfile.write(' closed,{}\n'.format(nclosed))
                tempfile.write(' occ,{}\n'.format(nocc))
                tempfile.write(' wf,{},1,0\n'.format(self.n_electrons))
                #this can be made the len of singlets
                tempfile.write(' state,{}\n'.format(len_singlets))

                for state in singlets:
                    s=state[1]
                    grad_name="510"+str(s)+".1"
                    tempfile.write(' CPMCSCF,GRAD,{}.1,record={}\n'.format(s+1,grad_name))

                #TODO this can only do coupling if states is 2, want to generalize to 3 states
                if self.do_coupling==True and len(singlets)==2:
                    tempfile.write('CPMCSCF,NACM,{}.1,{}.1,record=5200.1\n'.format(singlets[0][1]+1,singlets[1][1]+1))
                tempfile.write(' }\n')

                for state in singlets:
                    s=state[1]
                    grad_name="510"+str(s)+".1"
                    tempfile.write('Force;SAMC,{};varsav\n'.format(grad_name))
                if self.do_coupling==True and len(singlets)==2:
                    tempfile.write('Force;SAMC,5200.1;varsav\n')
            if len_triplets is not 0:
                tempfile.write(' {multi\n')
                nclosed = self.nocc
                nocc = nclosed+self.nactive
                tempfile.write(' closed,{}\n'.format(nclosed))
                tempfile.write(' occ,{}\n'.format(nocc))
                tempfile.write(' wf,{},1,2\n'.format(self.n_electrons))
                nstates = len(self.states)
                tempfile.write(' state,{}\n'.format(len_triplets))

                for state in triplets:
                    s=state[1]
                    grad_name="511"+str(s)+".1"
                    tempfile.write(' CPMCSCF,GRAD,{}.1,record={}\n'.format(s+1,grad_name))
                tempfile.write(' }\n')

                for s in triplets:
                    s=state[1]
                    grad_name="511"+str(s)+".1"
                    tempfile.write('Force;SAMC,{};varsav\n'.format(grad_name))
        else:
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                #print line
                tempfile.write(line)

        tempfile.close()

        cmd = "molpro -W scratch -n {} {} --no-xml-output".format(self.nproc,tempfilename)
        os.system(cmd)

        tempfileout='scratch/gopro.out'
        pattern = re.compile(r'MCSCF STATE \d.1 Energy \s* ([-+]?[0-9]*\.?[0-9]+)')
        self.E = []
        tmp =[]
        for i,line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern,line):
                tmp.append(float(match.group(1)))

        for i in range(len_E_singlets):
            self.E.append((1,i,tmp[i]))
        for i in range(len_triplets): #triplets broken here
            self.E.append((3,i,tmp[len_singlets+i]))

        tmpgrada=[]
        tmpgrad=[]
        self.grada=[]
        self.coup=[]
        with open(tempfileout,"r") as f:
            for line in f:
                if line.startswith("GRADIENT FOR STATE",7): #will work for SA-MC and RSPT2 HF
                    for i in range(3):
                        next(f)
                    for i in range(len(geom)):
                        findline = next(f,'').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        tmpgrad.append([
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                            ])
                    tmpgrada.append(tmpgrad)
                    tmpgrad = []
                if line.startswith(" SA-MC NACME FOR STATES"):
                    for i in range(3):
                        next(f)
                    for i in range(len(geom)):
                        findline = next(f,'').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        self.coup.append([
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                            ])

        for count,i in enumerate(self.states):
            if i[0]==1:
                self.grada.append((1,i[1],tmpgrada[count]))
            if i[0]==3:
                self.grada.append((3,i[1],tmpgrada[count]))
        self.hasRanForCurrentCoords=True
        return

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_PES_tuple(self.grada,multiplicity,state)[0][2]
        return np.asarray(tmp)*units.ANGSTROM_TO_AU

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return np.reshape(self.coup,(3*len(self.coup),1))*units.ANGSTROM_TO_AU

    #TODO molpro requires extra things when copying. . . can this be done in the base_lot? 
    # e.g. if cls=="Molpro": #do molpro stuff?
    @classmethod
    def copy(cls,lot,**kwargs):
        """ create a copy of this lot object"""
        #print(" creating copy, new node id =",node_id)
        #print(" old node id = ",self.node_id)
        node_id = kwargs.get('node_id',1)
        if node_id != lot.node_id:
            cmd = "cp scratch/mp_0000_{:03d} scratch/mp_0000_{:03d}".format(lot.node_id,node_id)
            print(cmd)
            os.system(cmd)
        return cls(lot.options.copy().set_values(kwargs))

if __name__=='__main__':
    filepath="../../data/ethylene.xyz"
    molpro = Molpro.from_options(states=[(1,0)],fnm=filepath,lot_inp_file='../../data/ethylene_molpro.com')
    geom=manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print molpro.get_energy(xyz,1,0)
    print molpro.get_gradient(xyz,1,0)
