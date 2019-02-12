from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *
import os
import re

class Molpro(Lot):

    def run(self,geom):
        if self.has_nelectrons==False:
            for i in self.states:
                self.get_nelec(geom,i[0])
            self.has_nelectrons==True

        #TODO gopro needs a number
        tempfilename = 'scratch/gopro.com'
        tempfile = open(tempfilename,'w')
        print self.node_id
        tempfile.write(' file,2,mp_0000_{:03}\n').format(self.node_id)
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
                print line
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

        for i in range(len_singlets):
            self.E.append((1,tmp[i]))
        for i in range(len_triplets):
            self.E.append((3,tmp[len_singlets+i]))

        tmpgrada=[]
        tmpgrad=[]
        self.grada=[]
        self.coup=[]
        with open(tempfileout,"r") as f:
            for line in f:
                if line.startswith("GRADIENT FOR STATE",6): #will work for SA-MC and RSPT2 HF
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

        for i in range(len_singlets):
            self.grada.append((1,tmpgrada[i]))
        for i in range(len_triplets):
            self.grada.append((3,tmpgrada[len_singlets+i]))

        self.hasRanForCurrentCoords=True

        return

    def getE(self,state,multiplicity):
        tmp = self.search_tuple(self.E,multiplicity)
        return tmp[state][1]*KCAL_MOL_PER_AU

    def get_energy(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getE(state,multiplicity)

    def getgrad(self,state,multiplicity):
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*ANGSTROM_TO_AU

    def get_gradient(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getgrad(state,multiplicity)

    def getcoup(self,state1,state2,multiplicity):
        #TODO this could be better
        return np.reshape(self.coup,(3*len(self.coup),1))*ANGSTROM_TO_AU

    def get_coupling(self,geom,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getcoup(state1,state2,multiplicity)

    @staticmethod
    def copy(MolproA,node_id):
        """ create a copy of this lot object"""
        return Molpro(MolproA.options.copy().set_values({
            "node_id" :node_id,
            }))

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Molpro(Molpro.default_options().set_values(kwargs))

if __name__ == '__main__':

    import pybel as pb    
    import manage_xyz
    from dlc import *
    filepath="tests/fluoroethene.xyz"
    nocc=11
    nactive=2
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    lot=Molpro.from_options(states=[(1,0),(1,1)],charge=0,nocc=nocc,nactive=nactive,basis='6-31G*',do_coupling=True,nproc=4)
    e=lot.get_energy(geom,1,0)
    print e
    e=lot.get_energy(geom,1,1)
    print e
    g=lot.get_gradient(geom,1,0)
    print g
    g=lot.get_gradient(geom,1,1)
    print g
    d=lot.get_coupling(geom,state1=0,state2=1,multiplicity=1)
    print d
