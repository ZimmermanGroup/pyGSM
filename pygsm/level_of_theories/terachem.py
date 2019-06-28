# standard library imports
import sys
import os
from os import path
import re

# third party
import numpy as np
import json

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
        self.tc_options = self.options['job_data'].get('tc_options',None)

        if not self.tc_options:
            self.build_lot_from_dictionary()

        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

    @property
    def tc_options(self):
        return self.options['job_data']['tc_options']

    @tc_options.setter
    def tc_options(self,d):
        self.options['job_data']['tc_options'] = d

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

    def build_lot_from_dictionary(self):
        d = {}
        print(self.lot_inp_file)
        d = json.load(open(self.lot_inp_file))
        nifty.printcool_dictionary(d)

        # QM
        basis = d.get('basis',None)
        method = d.get('method','HF')
        charge = d.get('charge',0)

        #SCF
        convthre = d.get('convthre',1.e-7)
        threall  = d.get('threall',1e-14)
        scf      = d.get('scf','diis+a')
        maxit    = d.get('maxit',200)
        purify   = d.get('purify',"no")
        precision = d.get('precision','double')

        # active space
        active = d.get('active',0)
        closed = d.get('closed',0)
        nalpha = d.get('nalpha',int(active/2))
        nbeta = d.get('nbeta',nalpha)
        casscfmaxiter = d.get('casscfmaxiter',200)
        casscf = d.get('casscf', 'no')
        cassinglets = d.get('cassinglets',2)

        doCAS = False
        if casscf == "yes":
            doCAS = True


        # TODO fix fomo for terachem ...
        # FOMO
        fomo = d.get('fomo',False)
        fomo_temp = d.get('fomo_temp',0.3)
        fomo_nocc=d.get('fomo_nocc',closed)
        fomo_nact = d.get('fomo_nact',active)
        fomo_method = d.get('fomo_method','gaussian')
        doFOMO = False
        if fomo:
            doFOMO=True

        # QMMM
        prmtop = d.get('prmtop',None)
        qmindices = d.get('qmindices',None)
        doQMMM=False
        if prmtop:
            doQMMM=True

        # GPUs
        gpus = d.get('gpus',1)
        runtype = d.get('runtype','gradient')


        # make terachem options
        self.tc_options = {
                'basis'     :   basis,
                'method'    :   method,
                'charge'    :   charge,
                'convthre'  :   convthre,
                'threall'   :   threall,
                'scf'       :   scf,
                'maxit'     :   maxit,
                'purify'    :   purify,
                'precision' :   precision,
                'doCAS'     :   doCAS,
                'active'    :   active,
                'closed'    :   closed,
                'nalpha'    :   nalpha,
                'nbeta'     :   nbeta,
                'casscf'    :   casscf,
                'casscfmaxiter': casscfmaxiter,
                'cassinglets':  cassinglets,

                'doFOMO'    :   doFOMO,
                'fomo'      :   fomo,
                'fomo_temp' :   fomo_temp,
                'fomo_nocc' :   fomo_nocc,
                'fomo_nact' :   fomo_nact,
                'fomo_method':  fomo_method,
                'doQMMM'    :   doQMMM,
                'prmtop'    :   prmtop,
                'qmindices'    :   qmindices,
                'runtype'   :   runtype,
                'gpus'      :   gpus,
                }


        nifty.printcool_dictionary(self.tc_options)

    def run(self,geom):

        # count number of states
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets)
        len_E_singlets=singlets[-1][1] +1 #final adiabat +1 because 0 indexed
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets)

        def go(mult,ad_idx):
            # write the current geometry in the scratch folder
            if self.tc_options['doQMMM']: 
                manage_xyz.write_amber_xyz('scratch/{}/tmp.inpcrd'.format(self.node_id,geom),geom)
            else:
                manage_xyz.write_xyz('scratch/{}/tmp.xyz'.format(self.node_id),geom,scale=1.0)

            # filenames
            inpfilename = 'scratch/{}/{}'.format(self.node_id,self.lot_inp_file)
            outfilename = 'scratch/{}/output.dat'.format(self.node_id)

            # write the input file
            inpfile = open(inpfilename,'w')
            if self.tc_options['doQMMM']: 
                inpfile.write('coordinates      scratch/{}/tmp.inpcrd\n'.format(self.node_id))
                inpfile.write('prmtop          {}\n'.format(self.tc_options['prmtop'])) 
                inpfile.write('qmindices          {}\n\n'.format(self.tc_options['qmindices'])) 
            else:
                inpfile.write('coordinates     scratch/{}/tmp.xyz\n\n'.format(self.node_id))

            inpfile.write('scrdir               scratch/{}/scr\n\n'.format(self.node_id))
            
            # method
            inpfile.write("# METHOD\n")
            inpfile.write("method               {}\n".format(self.tc_options['method']))
            inpfile.write("basis                {}\n".format(self.tc_options['basis']))
            inpfile.write("convthre             {}\n".format(self.tc_options['convthre']))
            inpfile.write("threall              {}\n".format(self.tc_options['threall']))
            inpfile.write("scf                  {}\n".format(self.tc_options['scf']))
            inpfile.write("maxit                {}\n".format(self.tc_options['maxit']))
            inpfile.write("charge               {}\n".format(self.tc_options['charge']))
            inpfile.write('spinmult             {}\n\n'.format(mult))

            # guess
            inpfile.write(" # GUESS\n")
            if self.tc_options['doCAS']: 
                inpfile.write('casguess             scratch/{}/c0.casscf\n\n'.format(self.node_id))
            else:
                inpfile.write('guess     scratch/{}/c0\n\n'.format(self.node_id))

            # CASSCF
            inpfile.write(" # CASSCF\n")
            if self.tc_options['doCAS']: 
                inpfile.write('casscf             {}\n'.format(self.tc_options['casscf']))
                inpfile.write('castarget            {}\n'.format(ad_idx))
                inpfile.write('castargetmult        {}\n'.format(mult))
                inpfile.write('casscfmaxiter        {}\n'.format(self.tc_options['casscfmaxiter']))
                inpfile.write('cassinglets          {}\n'.format(self.tc_options['cassinglets']))
                inpfile.write('closed              {}\n'.format(self.tc_options['closed']))
                inpfile.write('active               {}\n\n'.format(self.tc_options['active']))

            # RUN
            inpfile.write("run         {}\n\n".format(self.tc_options['runtype']))

            # GPUS
            inpfile.write("gpus            {}".format(self.tc_options['gpus']))
                
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
