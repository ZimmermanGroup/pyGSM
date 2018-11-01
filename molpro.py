from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *
import os
import re
import elements 
ELEMENT_TABLE = elements.ElementData()

class Molpro(Base):

    def run(self,geom):

        if self.got_electrons==False:
            self.get_nelec(geom)
        #TODO gopro needs a number
        tempfilename = 'scratch/gopro.com'
        tempfile = open(tempfilename,'w')
        #tempfile.write(' memory,{},m\n'.format(self.memory))
        tempfile.write(' memory,400,m\n')
        #tempfile.write(' file,2,{}\n'.format(self.scratchname))
        tempfile.write(' file,2,mp_0000_0000\n')
        tempfile.write(' symmetry,nosym\n')
        tempfile.write(' orient,noorient\n\n')

        tempfile.write(' geometry={\n')
        for coord in geom:
            for i in coord:
                tempfile.write(' '+str(i))
            tempfile.write('\n')
        tempfile.write('}\n\n')

        tempfile.write(' basis={}\n\n'.format(self.basis))

        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets) 
        if len_singlets is not 0:
            tempfile.write(' {multi\n')
            nclosed = self.nocc
            nocc = nclosed+self.nactive
            tempfile.write(' closed,{}\n'.format(nclosed))
            tempfile.write(' occ,{}\n'.format(nocc))
            tempfile.write(' wf,{},1,0\n'.format(self.n_electrons))
            #this can be made the len of singlets
            tempfile.write(' state,{}\n'.format(len_singlets))

            for state in singlets:
                s=state[1]
                grad_name="510"+str(s)+".1"
                tempfile.write(' CPMCSCF,GRAD,{}.1,record={}\n'.format(s+1,grad_name))
                tempfile.write(' }\n')

            for state in singlets:
                s=state[1]
                grad_name="510"+str(s)+".1"
                tempfile.write('Force;SAMC,{};varsav\n'.format(grad_name))

        triplets=self.search_tuple(self.states,1)
        len_triplets=len(triplets) 
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

        print("finding grad")
        tmpgrada=[]
        tmpgrad=[]
        self.grada=[]
        with open(tempfileout,"r") as f:
            for line in f:
                if line.startswith(" SA-MC GRADIENT FOR STATE"):
                    for i in range(3):
                        next(f)
                    for i in range(len(geom)):
                        findline = next(f,'').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        tmpgrad.append((
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                            ))
                    tmpgrada.append(tmpgrad)

        for i in range(len_singlets):
            self.grada.append((1,tmpgrada[i]))
        for i in range(len_triplets):
            self.grada.append((3,tmpgrada[len_singlets+i]))

        self.hasRanForCurrentCoords=True

        return

    def getE(self,state,multiplicity):
        tmp = self.search_tuple(self.E,multiplicity)
        print tmp
        return tmp[state][1]*KCAL_MOL_PER_AU

    def get_energy(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getE(state,multiplicity)

    def getgrad(self,state,multiplicity):
        tmp = self.search_tuple(self.grada,multiplicity)
        return tmp[state][1]

    def get_gradient(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getgrad(state,multiplicity)


    #    return np.asarray(grad)*ANGSTROM_TO_AU


    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Molpro(Molpro.default_options().set_values(kwargs))

if __name__ == '__main__':

    import icoord as ic
    import pybel as pb    
    import manage_xyz
    filepath="tests/fluoroethene.xyz"
    nocc=11
    nactive=2
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    lot=Molpro.from_options(states=[(1,0),(3,0)],charge=0,nocc=nocc,nactive=nactive,basis='6-31G*')
    e=lot.get_energy(geom,1,0)
    print e
    e=lot.get_energy(geom,3,0)
    print e
    g=lot.get_gradient(geom,1,0)
    print g
    g=lot.get_gradient(geom,3,0)
    print g
