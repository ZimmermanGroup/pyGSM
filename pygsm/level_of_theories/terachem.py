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


class TeraChem(Lot):
    def __init__(self,options):
        super(TeraChem,self).__init__(options)
        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

    @classmethod
    def copy(cls,lot,options):
        node_id = options.get('node_id',1)

        print(" making folder scratch/{}".format(node_id))
        os.system('mkdir -p scratch/{}'.format(node_id))

        if node_id != lot.node_id:
            old_path = 'scratch/{}/c0.casscf'.format(lot.node_id)
            new_path = 'scratch/{}/'.format(node_id)
            cmd = 'cp -r ' + old_path +' ' + new_path
            print(" copying scr files\n {}".format(cmd))
            os.system(cmd)
        return cls(lot.options.copy().set_values(options))

    def run(self,geom):

        print(" In run")
        # count number of states
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets)
        len_E_singlets=singlets[-1][1] +1 #final adiabat +1 because 0 indexed
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets)

        def go(mult,ad_idx):
            # write the current geometry in the scratch folder
            manage_xyz.write_xyz('scratch/{}/tmp.xyz'.format(self.node_id),geom,scale=1.0)

            # write the input file
            inpfilename = 'scratch/{}/{}'.format(self.node_id,self.lot_inp_file)
            outfilename = 'scratch/{}/output.dat'.format(self.node_id)

            inpfile = open(inpfilename,'w')
            inpfile.write('coordinates     scratch/{}/tmp.xyz\n'.format(self.node_id))
            inpfile.write('scrdir     scratch/{}/scr\n'.format(self.node_id))
            inpfile.write('casguess     scratch/{}/c0.casscf\n'.format(self.node_id))
            inpfile.write('castarget     {}\n'.format(ad_idx))
            inpfile.write('castargetmult     {}\n'.format(mult))
            with open(self.lot_inp_file) as lot_inp:
                lot_inp_lines = lot_inp.readlines()
            for line in lot_inp_lines:
                inpfile.write(line)
            inpfile.close()
    
            cmd = "terachem {} > {}".format(inpfilename,outfilename)
            print(cmd)
            os.system(cmd)

        
        for s in range(len_singlets):
            print(" getting {} for node {}".format(s,self.node_id))
            go(1,s)
            # copy the wavefunction file
            cp_cmd = 'cp scratch/{}/scr/c0.casscf scratch/{}/'.format(self.node_id,self.node_id)
            print(cp_cmd)
            os.system(cp_cmd)
            cp_cmd = 'cp scratch/{}/scr/casscf.molden scratch/{}/'.format(self.node_id,self.node_id)
            print(cp_cmd)
            os.system(cp_cmd)
            cp_grad = 'cp scratch/{}/scr/grad.xyz scratch/{}/grad_{}_{}.xyz'.format(self.node_id,self.node_id,1,s)
            print(cp_grad)
            os.system(cp_grad)
            # rm the old scrach folder

            #tmp
            cp_out = 'cp scratch/{}/output.dat scratch/{}/scr/'.format(self.node_id,self.node_id)
            os.system(cp_out)

            #rm_cmd = 'rm -rf scratch/{}/scr'.format(self.node_id)
            #print(rm_cmd)
            #os.system(rm_cmd)

        # parse the output
        tempfileout='scratch/{}/output.dat'.format(self.node_id)
        pattern = re.compile(r'Singlet state  \d energy: \s* ([-+]?[0-9]*\.?[0-9]+)')
        self.E = []
        tmp =[]
        for i,line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern,line):
                tmp.append(float(match.group(1)))
        for i in range(len_singlets):
            self.E.append((1,i,tmp[i]))
        for i in range(len_triplets): #triplets broken here
            self.E.append((3,i,tmp[len_singlets+i]))

        self.grada=[]
        for s in range(len_singlets):
            grad = manage_xyz.read_xyz('scratch/{}/grad_{}_{}.xyz'.format(self.node_id,1,s))
            grad = manage_xyz.xyz_to_np(grad)
            self.grada.append((1,s,grad))


        self.hasRanForCurrentCoords=True
        

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_PES_tuple(self.E,multiplicity,state)[0][2]
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_PES_tuple(self.grada,multiplicity,state)[0][2]
        return np.asarray(tmp)*-1.*units.ANGSTROM_TO_AU



if __name__=="__main__":

    filepath="../../data/ethylene.xyz"
    TC = TeraChem.from_options(states=[(1,0),(1,1)],fnm=filepath,lot_inp_file='tc_input.com')
    geom=manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print(TC.get_energy(xyz,1,0))
    print(TC.get_gradient(xyz,1,0))

    xyz = xyz+ np.random.rand(xyz.shape[0],xyz.shape[1])*0.1
    print(TC.get_energy(xyz,1,0))
    print(TC.get_gradient(xyz,1,0))
