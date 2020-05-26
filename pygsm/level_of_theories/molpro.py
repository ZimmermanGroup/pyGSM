# standard library imports
import sys
import os
from os import path
import re

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
try:
    from .base_lot import Lot
except:
    from base_lot import Lot
from utilities import *
import subprocess 

class Molpro(Lot):

    def __init__(self,options):
        super(Molpro,self).__init__(options)

        self.file_options.set_active('basis','6-31g',str,'')
        self.file_options.set_active('closed',None,int,'')
        self.file_options.set_active('occ',None,int,'')
        self.file_options.set_active('n_electrons',None,int,'')
        self.file_options.set_active('memory',800,int,'')
        self.file_options.set_active('n_states',2,int,'')

        # set all active values to self for easy access
        for key in self.file_options.ActiveOptions:
            setattr(self, key, self.file_options.ActiveOptions[key])

        if self.has_nelectrons==False:
            for i in self.states:
                self.get_nelec(self.geom,i[0])


    # designed to do multiple multiplicities at once... maybe not such a good idea, that feature is currently broken
    #TODO 
    def run(self,geom):

        #TODO gopro needs a number
        tempfilename = 'scratch/gopro.com'
        tempfile = open(tempfilename,'w')
        tempfile.write(' file,2,mp_{:04d}_{:04d}\n'.format(self.ID,self.node_id))
        tempfile.write(' memory,{},m\n'.format(self.memory))
        tempfile.write(' symmetry,nosym\n')
        tempfile.write(' orient,noorient\n\n')
        tempfile.write(' geometry={\n')
        for coord in geom:
            for i in coord:
                tempfile.write(' '+str(i))
            tempfile.write('\n')
        tempfile.write('}\n\n')

        # get states
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets)
        len_E_singlets=singlets[-1][1] +1 #final adiabat +1 because 0 indexed
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets)

        tempfile.write(' basis={}\n\n'.format(self.basis))

        # do singlets
        if len_singlets is not 0:
            tempfile.write(' {multi\n')
            tempfile.write(' direct\n')
            tempfile.write(' closed,{}\n'.format(self.closed))
            tempfile.write(' occ,{}\n'.format(self.occ))
            tempfile.write(' wf,{},1,0\n'.format(self.n_electrons))
            #this can be made the len of singlets
            tempfile.write(' state,{}\n'.format(self.n_states))

            for state in self.gradient_states:
                s=state[1]
                print("running grad state ",s)
                grad_name="510"+str(s)+".1"
                tempfile.write(' CPMCSCF,GRAD,{}.1,record={}\n'.format(s+1,grad_name))

            #TODO this can only do coupling if states is 2, want to generalize to 3 states
            if self.coupling_states:
                tempfile.write('CPMCSCF,NACM,{}.1,{}.1,record=5200.1\n'.format(self.coupling_states[0],self.coupling_states[1]))
            tempfile.write(' }\n')

            for state in self.gradient_states:
                s=state[1]
                grad_name="510"+str(s)+".1"
                tempfile.write('Force;SAMC,{};varsav\n'.format(grad_name))
            if self.coupling_states:
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

        tempfile.close()
        scratch = os.environ['SLURM_LOCAL_SCRATCH']

        #cmd = "molpro -W scratch -n {} {} -d {} --no-xml-output".format(self.nproc,tempfilename,scratch)
        args = ['-W','scratch','-n',str(self.nproc),tempfilename,'-d',scratch]
        command=['molpro']
        command.extend(args)
        output = subprocess.Popen(command, stdout=subprocess.PIPE, stderr = subprocess.PIPE).communicate()[0]

        #with open('molpro.stdout','a') as out:
        #    with nifty.custom_redirection2(out):
        #        os.system(cmd)
        #process = Popen(['cat', 'test.py'], stdout=PIPE, stderr=PIPE)
        #process = Popen([cmd], stdout=PIPE, stderr=PIPE)
        #stdout, stderr = process.communicate()

        # Now read the output
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

        for count,i in enumerate(self.gradient_states):
            if i[0]==1:
                self.grada.append((1,i[1],tmpgrada[count]))
            if i[0]==3:
                self.grada.append((3,i[1],tmpgrada[count]))

        with open('scratch/E_{}.txt'.format(self.node_id),'w') as f:
            for E in self.E:
                f.write('{} {} {:9.7f}\n'.format(E[0],E[1],E[2]))
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
        return np.asarray(tmp)*units.ANGSTROM_TO_AU  #hartree/ang

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return np.reshape(self.coup,(3*len(self.coup),1))*units.ANGSTROM_TO_AU

    @classmethod
    def copy(cls,lot,options,copy_wavefunction=True):
        """ create a copy of this lot object"""
        #print(" creating copy, new node id =",node_id)
        #print(" old node id = ",self.node_id)
        node_id = options.get('node_id',1)
        if node_id != lot.node_id and copy_wavefunction:
            cmd = "cp scratch/mp_{:04d}_{:04d} scratch/mp_{:04d}_{:04d}".format(lot.ID,lot.node_id,lot.ID,node_id)
            print(cmd)
            os.system(cmd)
        return cls(lot.options.copy().set_values(options))

if __name__=='__main__':
    filepath="../../data/ethylene.xyz"
    molpro = Molpro.from_options(states=[(1,0),(1,1)],fnm=filepath,lot_inp_file='../../data/ethylene_molpro.com',coupling_states=[0,1])
    geom=manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print(molpro.get_energy(xyz,1,0))
    print(molpro.get_gradient(xyz,1,0))
    print(molpro.get_coupling(xyz,1,0,1))
