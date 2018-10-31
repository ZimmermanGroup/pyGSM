from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *
import os
import re

class Molpro(Base):

    def compute_energy(self,multiplicity,charge,state):
        #TODO gopro needs a number
        tempfilename = 'scratch/gopro.com'
        tempfile = open(tempfilename,'w')
        #tempfile.write(' memory,{},m\n'.format(self.memory))
        tempfile.write(' memory,400,m\n')
        #tempfile.write(' file,2,{}\n'.format(self.scratchname))
        tempfile.write(' file,2,mp_0000_0000\n')
        tempfile.write(' symmetry,nosym\n')
        tempfile.write(' orient,noorient\n\n')
        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords) #updates geom

        tempfile.write(' geometry={\n')
        for coord in self.geom:
            for i in coord:
                tempfile.write(' '+str(i))
            tempfile.write('\n')
        tempfile.write('}\n\n')

        tempfile.write(' basis={}\n\n'.format(self.basis))
        tempfile.write(' {multi\n')
        nclosed = self.nocc
        nocc = nclosed+self.nactive
        print nclosed
        print nocc
        tempfile.write(' closed,{}\n'.format(nclosed))
        tempfile.write(' occ,{}\n'.format(nocc))
        tempfile.write(' wf,{},1,0\n'.format(self.n_electrons))
        nstates = len(self.states)
        tempfile.write(' state,{}\n'.format(nstates))
        #grad_name='510+{}+".1'.format(state)
        grad_name="510"+str(state)+".1"
        tempfile.write(' CPMCSCF,GRAD,{}.1,record={}\n'.format(state+1,grad_name))
        tempfile.write(' }\n')
        tempfile.write('Force;SAMC,{};varsav\n'.format(grad_name))

        tempfile.close()

        cmd = "molpro -W scratch -n {} {} --no-xml-output".format(self.nproc,tempfilename)
        os.system(cmd)

        tempfileout='scratch/gopro.out'
        pattern = re.compile(r'MCSCF STATE \d.1 Energy \s* ([-+]?[0-9]*\.?[0-9]+)')
        for i,line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern,line):
                energy = float(match.group(1))

        return energy*KCAL_MOL_PER_AU


    def compute_gradient(self,multiplicity,charge,state):
        """ Assuming grad already computed in compute_energy"""
        gradfilepath = "scratch/gopro.out"
        print("finding grad")
        grad=[]
        with open(gradfilepath,"r") as f:
            for line in f:
                if line.startswith(" SA-MC GRADIENT FOR STATE"):
                    for i in range(3):
                        next(f)
                    for i in range(len(self.geom)):
                        findline = next(f,'').strip()
                        mobj = re.match(r'^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                        grad.append((
                            float(mobj.group(2)),
                            float(mobj.group(3)),
                            float(mobj.group(4)),
                            ))
                    break

        return np.asarray(grad)*ANGSTROM_TO_AU


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
    lot=Molpro.from_options(states=[0],multiplicity=[1],charge=0,geom=geom,nocc=nocc,nactive=nactive,basis='6-31G*')
    e=lot.getEnergy()
    print e
    g=lot.getGrad()
    print g
