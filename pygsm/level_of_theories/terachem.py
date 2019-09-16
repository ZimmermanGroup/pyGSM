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

'''
Unfortunately TC calculates one gradient at time. THis makes it difficult to calculate multiple states since two calculations need to be done per state. 
When doing excited-state calculations e.g. s1 RP, this becomes a pain. 

08/26 -- Note to self. there is something significantly wrong with how the energies and gradients are calculated -- some refactoring is necessary to make this work as desired.
Don't forget to go back and try to fix this . . .
'''

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
    def copy(cls,lot,options,copy_wavefunction=True):
        node_id = options.get('node_id',1)

        print(" making folder scratch/{}".format(node_id))
        os.system('mkdir -p scratch/{}'.format(node_id))

        if node_id != lot.node_id and copy_wavefunction:
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

        # make terachem options
        self.tc_options={}

        # QM
        self.tc_options['basis']= d.get('basis',None)
        self.tc_options['method'] = d.get('method','HF')
        self.tc_options['charge'] = d.get('charge',0)
        self.tc_options['dftd']= d.get('dftd',None)

        #SCF
        self.tc_options['convthre'] = d.get('convthre',1.e-7)
        self.tc_options['threall']  = d.get('threall',1e-14)
        self.tc_options['scf']      = d.get('scf','diis+a')
        self.tc_options['maxit']    = d.get('maxit',200)
        self.tc_options['purify']   = d.get('purify',"no")
        self.tc_options['precision'] = d.get('precision','double')

        # active space
        self.tc_options['active'] = d.get('active',0)
        self.tc_options['closed'] = d.get('closed',0)
        self.tc_options['nalpha'] = d.get('nalpha',int(self.tc_options['active']/2))
        self.tc_options['nbeta'] = d.get('nbeta',self.tc_options['nalpha'])
        self.tc_options['casscfmaxiter'] = d.get('casscfmaxiter',200)
        self.tc_options['casscf'] = d.get('casscf', 'no')
        self.tc_options['cassinglets'] = d.get('cassinglets',2)
        self.tc_options['activeorb']   = d.get('activeorb',None)

        # TODO fix fomo for terachem ...
        # FOMO
        self.tc_options['fomo'] = d.get('fomo',False)
        self.tc_options['fomo_temp'] = d.get('fomo_temp',0.3)
        self.tc_options['fomo_nocc']=d.get('fomo_nocc',self.tc_options['closed'])
        self.tc_options['fomo_nact'] = d.get('fomo_nact',self.tc_options['active'])
        self.tc_options['fomo_method'] = d.get('fomo_method','gaussian')

        # QMMM
        self.tc_options['prmtop'] = d.get('prmtop',None)
        self.tc_options['qmindices'] = d.get('qmindices',None)

        #TDDFT
        self.tc_options['cis'] = d.get('cis',None)
        self.tc_options['cisnumstates']=d.get('cisnumstates',4)
        self.tc_options['cisguessvecs']=d.get('cisguessvecs',self.tc_options['cisnumstates'])
        self.tc_options['cismaxiter'] = d.get('cismaxiter',20)
        self.tc_options['rc_w'] = d.get('rc_w',None)

        # GPUs
        self.tc_options['gpus'] = d.get('gpus',1)
        self.tc_options['gpumem']=d.get('gpumem',None)

        nifty.printcool_dictionary(self.tc_options)

    def go(self,geom,mult,ad_idx,runtype='gradient'):
        # write the current geometry in the scratch folder
        if self.tc_options['prmtop']:
            manage_xyz.write_amber_xyz('scratch/{}/tmp.inpcrd'.format(self.node_id,geom),geom)
        else:
            manage_xyz.write_xyz('scratch/{}/tmp.xyz'.format(self.node_id),geom,scale=1.0)

        # filenames
        inpfilename = 'scratch/{}/{}'.format(self.node_id,self.lot_inp_file)
        outfilename = 'scratch/{}/output.dat'.format(self.node_id)

        # write the input file
        inpfile = open(inpfilename,'w')
        if self.tc_options['prmtop']:
            inpfile.write('coordinates      scratch/{}/tmp.inpcrd\n'.format(self.node_id))
            inpfile.write('prmtop          {}\n'.format(self.tc_options['prmtop'])) 
            inpfile.write('qmindices          {}\n\n'.format(self.tc_options['qmindices'])) 
        else:
            inpfile.write('coordinates     scratch/{}/tmp.xyz\n\n'.format(self.node_id))
        inpfile.write('scrdir               scratch/{}/scr\n\n'.format(self.node_id))
        
        # method
        inpfile.write("# METHOD\n")
        inpfile.write("method               {}\n".format(self.tc_options['method']))
        if self.tc_options['rc_w'] is not None:
            inpfile.write('rc_w             {}\n'.format(self.tc_options['rc_w']))
        if self.tc_options['dftd'] is not None:
            inpfile.write('dftd             {}\n'.format(self.tc_options['dftd']))
        inpfile.write("basis                {}\n".format(self.tc_options['basis']))
        inpfile.write("convthre             {}\n".format(self.tc_options['convthre']))
        inpfile.write("threall              {}\n".format(self.tc_options['threall']))
        inpfile.write("scf                  {}\n".format(self.tc_options['scf']))
        inpfile.write("purify               {}\n".format(self.tc_options["purify"]))
        inpfile.write("precision            {}\n".format(self.tc_options['precision']))
        inpfile.write("maxit                {}\n".format(self.tc_options['maxit']))
        inpfile.write("charge               {}\n".format(self.tc_options['charge']))
        inpfile.write('spinmult             {}\n\n'.format(mult))

        # guess
        inpfile.write(" # GUESS\n")
        if self.tc_options['casscf']=='yes':
            inpfile.write('casguess             scratch/{}/c0.casscf\n\n'.format(self.node_id))
        else:
            inpfile.write('guess     scratch/{}/c0\n\n'.format(self.node_id))

        # CASSCF
        if self.tc_options['casscf']=='yes':
            inpfile.write(" # CASSCF\n")
            inpfile.write('casscf             {}\n'.format(self.tc_options['casscf']))
            if runtype != "coupling":
                inpfile.write('castarget            {}\n'.format(ad_idx))
            else:

                inpfile.write('nacstate1 {}\n'.format(self.coupling_states[0]))
                inpfile.write('nacstate2 {}\n'.format(self.coupling_states[1]))

            inpfile.write('castargetmult        {}\n'.format(mult))
            inpfile.write('casscfmaxiter        {}\n'.format(self.tc_options['casscfmaxiter']))
            inpfile.write('cassinglets          {}\n'.format(self.tc_options['cassinglets']))
            inpfile.write('closed              {}\n'.format(self.tc_options['closed']))
            if self.tc_options['activeorb']:
                inpfile.write('activeorb        {}\n'.format(self.tc_options['activeorb']))
            inpfile.write('active               {}\n\n'.format(self.tc_options['active']))

            # PUT FOMO HERE?

        elif self.tc_options['cis']=='yes':
            inpfile.write('cis                  {}\n'.format(self.tc_options['cis']))
            inpfile.write('cisnumstates         {}\n'.format(self.tc_options['cisnumstates']))
            inpfile.write('cisguessvecs         {}\n'.format(self.tc_options['cisguessvecs']))
            inpfile.write('cismaxiter           {}\n'.format(self.tc_options['cismaxiter']))
            inpfile.write('cistarget            {}\n'.format(ad_idx))

        # RUN
        inpfile.write("run         {}\n\n".format(runtype))

        # GPUS
        inpfile.write("gpus            {}\n".format(self.tc_options['gpus']))
        if self.tc_options['gpumem'] is not None:
            inpfile.write("gpumem            {}\n".format(self.tc_options['gpumem']))
            
        inpfile.close()
    
        # ACTUALLY RUN THE CALCULATION
        cmd = "terachem {} > {}".format(inpfilename,outfilename)
        os.system(cmd)

        ## POST PROCESSING  ##
        # copy the wavefunction file
        if self.tc_options['casscf']=='yes':
            cp_cmd = 'cp scratch/{}/scr/c0.casscf scratch/{}/'.format(self.node_id,self.node_id)
        else:
            cp_cmd = 'cp scratch/{}/scr/c0 scratch/{}/'.format(self.node_id,self.node_id)

        os.system(cp_cmd)

        if self.tc_options['casscf']=='yes':
            cp_cmd = 'cp scratch/{}/scr/casscf.molden scratch/{}/'.format(self.node_id,self.node_id)
            #print(cp_cmd)
            os.system(cp_cmd)

        if not self.tc_options['prmtop'] and self.calc_grad:
            if runtype!='coupling':
                cp_grad = 'cp scratch/{}/scr/grad.xyz scratch/{}/grad_{}_{}.xyz'.format(self.node_id,self.node_id,1,ad_idx)
                #print(cp_grad)
                os.system(cp_grad)
            elif runtype == "coupling":
                cp_coup = 'cp scratch/{}/scr/grad.xyz scratch/{}/coup_{}_{}.xyz'.format(self.node_id,self.node_id,self.coupling_states[0],self.coupling_states[1])
                #print(cp_grad)
                os.system(cp_coup)


        #tmp
        cp_out = 'cp scratch/{}/output.dat scratch/{}/scr/'.format(self.node_id,self.node_id)
        os.system(cp_out)

        rm_cmd = 'rm -rf scratch/{}/scr'.format(self.node_id)
        os.system(rm_cmd)

    def run(self,geom):

        # count number of states
        singlets=self.search_tuple(self.states,1)
        len_singlets=len(singlets)
        #TODO len_E_singlets is not perfect because
        # often one wants to calculate multiple states without calculating
        # gradients, it's confusing that one would then need to set states
        # to be equal to the final state
        # it would be better if the program can detect number of states
        # from the output or something
        len_E_singlets=singlets[-1][1] +1 #final adiabat +1 because 0 indexed
        triplets=self.search_tuple(self.states,3)
        len_triplets=len(triplets)

        #TODO run grads should be states 
        run_grads = [ singlets[i][1] for i in range(len_singlets)]
        
        tempfileout='scratch/{}/output.dat'.format(self.node_id)
        self.grada=[]

        #TODO
        # The following runs all the singlets. Other multiplicities are broken!!!
        for s in range(len_E_singlets):
            #print(" getting {} for node {}".format(s,self.node_id))

            # RUN 
            runtype='gradient' if self.calc_grad else 'energy'
            if s in run_grads:  # matches lot state
                #print(" running state {} {}".format(1,s))
                self.go(geom,1,s,runtype)

            # GET GRADIENT FOR QMMMM -- REGULAR GRADIENT IS PARSED THROUGH grad.xyz
            # QMMM is done differently :(
            if self.tc_options['prmtop']:
                if s in run_grads and self.calc_grad:

                    # parse qmindices
                    with open(self.tc_options['qmindices']) as f:
                        qmindices = f.read().splitlines()
                    qmindices = [int(i) for i in qmindices]
                    all_indices = range(len(geom))
                    mm_indices = list(set(all_indices) - set(qmindices))

                    QM_atoms = len(qmindices)

                    tmpgrad=[]
                    pattern = re.compile(r'Number of (\S+) atoms:\s+(\d+)')
                    for i,line in enumerate(open(tempfileout)):
                        for match in re.finditer(pattern,line):
                            if match.group(1) == "QM":
                                actual_QM_atoms = int(match.group(2))
                            elif match.group(1) =="MM":
                                MM_atoms = int(match.group(2))

                    link_atoms = actual_QM_atoms - QM_atoms
                    #print("Link atoms = ",link_atoms)

                    with open(tempfileout,"r") as f:
                        for line in f:
                            if line.startswith("dE/dX",8):
                                for i in range(QM_atoms):
                                    findline = next(f,'').strip()
                                    mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                    tmpgrad.append([
                                        float(mobj.group(1)),
                                        float(mobj.group(2)),
                                        float(mobj.group(3)),
                                        ])
                                for i in range(link_atoms):
                                    next(f)
                                # read two lines seperating the QM and MM regions
                                next(f)
                                next(f) 
                                for i in range(MM_atoms):
                                    findline = next(f,'').strip()
                                    mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                    tmpgrad.append([
                                        float(mobj.group(1)),
                                        float(mobj.group(2)),
                                        float(mobj.group(3)),
                                        ])

                    tmpgrad = np.asarray(tmpgrad)
                    grad = np.zeros_like(tmpgrad)
                    grad[qmindices] = tmpgrad[:len(qmindices)]
                    grad[mm_indices] = tmpgrad[len(qmindices):]
                    self.grada.append((1,s,grad))

                else:  
                    # APPEND None for grad if adiabatic index exists but state is not in list
                    self.grada.append((1,s,None))
            
        #### FINALLY DONE WITH RUN ###
    
        # parse the output  --> This can be done on any of the files since they should be the same
        if self.tc_options['casscf']=='yes':
            pattern = re.compile(r'Singlet state  \d energy: \s* ([-+]?[0-9]*\.?[0-9]+)')
        else:
            pattern = re.compile(r'FINAL ENERGY: ([-+]?[0-9]*\.?[0-9]+) a.u.')

        self.E = []
        tmp =[]
        
        # grep energy
        for i,line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern,line):
                tmp.append(float(match.group(1)))
        
        # Terachem has weird printout for td-dft energy
        if self.tc_options['cis']=='yes':
            lines=open(tempfileout).readlines()
            lines = lines[([x for x,y in enumerate(lines) if re.match(r'^\s+Final Excited State Results:', y)][0]+4):]
            for line in lines:
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
                if mobj:
                    tmp.append(float(mobj.group(2)))

        for i in range(len_E_singlets):
            self.E.append((1,i,tmp[i]))
        for i in range(len_triplets): #triplets broken here
            self.E.append((3,i,tmp[len_singlets+i]))
        
        if not self.tc_options['prmtop'] and self.calc_grad:
            for s in range(len_E_singlets):
                if s == singlets[s][1]:  # matches lot state
                    gradfile='scratch/{}/grad_{}_{}.xyz'.format(self.node_id,1,s)
                    grad = manage_xyz.read_xyz(gradfile,scale=1.0)
                    grad = manage_xyz.xyz_to_np(grad)
                    self.grada.append((1,s,grad))
                else:   # doesn't match so append None
                    self.grada.append((1,s,None))

        #### Also need to get the derivative coupling ###
        self.coup=[]
        if self.do_coupling:
            self.go(geom,1,s,runtype="coupling")

            if self.tc_options['prmtop']:
                tmpcoup = []
                with open(tempfileout,"r") as f:
                    for line in f:
                        if line.startswith("dE/dX",8): #will work for SA-MC and RSPT2 HF
                            for i in range(QM_atoms):
                                findline = next(f,'').strip()
                                mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                tmpcoup.append([
                                    float(mobj.group(1)),
                                    float(mobj.group(2)),
                                    float(mobj.group(3)),
                                    ])
                            # read link atoms
                            for i in range(link_atoms):
                                next(f)
                            next(f)
                            next(f) # read two lines seperating the QM and MM regions
                            for i in range(MM_atoms):
                                findline = next(f,'').strip()
                                mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                tmpcoup.append([
                                    float(mobj.group(1)),
                                    float(mobj.group(2)),
                                    float(mobj.group(3)),
                                    ])
                self.coup = np.zeros_like(tmpgrad)
                self.coup[qmindices] = tmpcoup[:len(qmindices)]
                self.coup[mm_indices] = tmpcoup[len(qmindices):]
            else:

                coupfile='scratch/{}/coup_{}_{}.xyz'.format(self.node_id,self.coupling_states[0],self.coupling_states[1])
                coup = manage_xyz.read_xyz(coupfile,scale=1.0)
                self.coup = manage_xyz.xyz_to_np(coup)

        self.hasRanForCurrentCoords=True
        return

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
        return np.asarray(tmp)*units.ANGSTROM_TO_AU

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return np.reshape(self.coup,(3*len(self.coup),1))*units.ANGSTROM_TO_AU

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
