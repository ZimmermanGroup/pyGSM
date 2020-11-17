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
    from .file_options import File_Options
except:
    from base_lot import Lot
    from file_options import File_Options
from utilities import *

'''
Unfortunately TC calculates one gradient at time. THis makes it difficult to calculate multiple states since two calculations need to be done per state. 
When doing excited-state calculations e.g. s1 RP, this becomes a pain. 

08/26 -- Note to self. there is something significantly wrong with how the energies and gradients are calculated -- some refactoring is necessary to make this work as desired.
Don't forget to go back and try to fix this . . .
4/20 I'm pretty sure this has been resolved for some time
'''

class TeraChem(Lot):
    def __init__(self,options):
        super(TeraChem,self).__init__(options)

    
        # Now go through the logic of determining which FILE options are activated.
        # DO NOT DUPLICATE OPTIONS WHICH ARE ALREADY PART OF LOT OPTIONS (e.g. charge)

        # QM
        self.file_options.set_active('basis','6-31g',str,'')
        self.file_options.set_active('method','HF',str,'')
        self.file_options.set_active('convthre',1e-7,float,'')
        self.file_options.set_active('threall',1e-14,float,'')
        self.file_options.set_active('xtol',None,float,'')
        self.file_options.set_active('scf','diis+a',str,'')
        self.file_options.set_active('maxit',200,int,'')
        self.file_options.set_active('purify','no',str,'')
        self.file_options.set_active('precision','double',str,'')

        # CASSCF
        self.file_options.set_active('casscf','no',str,'')
        self.file_options.set_active('casscfmaxiter',200,int,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('alphacas','no',str,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('alpha',0.4,float,'',depend=(self.file_options.alphacas=="yes"),msg='')
        self.file_options.set_active('casscfmacroiter',10,int,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('casscfmicroconvthre',0.1,float,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('casscfmacroconvthre',1e-3,float,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('casscfconvthre',1e-6,float,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('casscfenergyconvthre',1e-6,float,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('cpsacasscfmaxiter',20,int,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('cpsacasscfconvthre',1e-7,float,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('dci_explicit_h','no',str,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('directci','no',str,'',depend=(self.file_options.casscf=="yes"),msg='')
        self.file_options.set_active('ci_solver','no',str,'',depend=(self.file_options.casscf=="yes"),msg='')

        # CASCI
        self.file_options.set_active('casci','no',str,'',
                clash=(self.file_options.casscf=="yes"),
                msg = 'Cant activate FOMO with CASSCF')

        # active space (some options below depend on this)
        self.file_options.set_active('active',0,int,'',depend=(self.file_options.casscf=="yes" or self.file_options.casci=="yes")
                ,msg='')
        self.file_options.set_active('closed',0,int,'',depend=(self.file_options.casscf=="yes" or self.file_options.casci=="yes"),msg='')
        self.file_options.set_active('nalpha',0,int,'',depend=(self.file_options.casscf=="yes" or self.file_options.casci=="yes"),msg='')
        self.file_options.set_active('nbeta',0,int,'',depend=(self.file_options.casscf=="yes" or self.file_options.casci=="yes"),msg='')
        self.file_options.set_active('cassinglets',2,int,'',depend=(self.file_options.casscf=="yes" or self.file_options.casci=="yes"),msg='')

        # FON
        self.file_options.set_active('fon','no',str,'',
                clash=(self.file_options.casscf=="yes"),
                depend=(self.file_options.casci=="yes"),
                msg = 'Cant activate FOMO with CASSCF')
        self.file_options.set_active('fon_temperature',0.3,float,'',depend=(self.file_options.fon=="yes"),msg='')

        # FOMO
        #self.file_options.set_active('fon_nocc',self.file_options.closed,int,'',depend=(self.file_options.casci=="yes"),msg='')
        self.file_options.set_active('fomo','no',str,'',
                clash=(self.file_options.casscf=="yes"),
                msg = 'Cant activate FOMO with CASSCF')
        self.file_options.set_active('fomo_temp',0.3,float,'',depend=(self.file_options.fomo=="yes"),msg='')
        self.file_options.set_active('fomo_nocc',self.file_options.closed,int,'',depend=(self.file_options.fomo=="yes"),msg='')
        self.file_options.set_active('fomo_nact',self.file_options.active,int,'',depend=(self.file_options.fomo=="yes"),msg='')
        self.file_options.set_active('fomo_method','gaussian',str,'',depend=(self.file_options.fomo=="yes"),msg='')


        # QMMM
        self.file_options.set_active('prmtop',None,str,'')
        self.file_options.set_active('qmindices',None,str,'')

        # CIS
        self.file_options.set_active('cis','no',str,'',
                clash=(self.file_options.casscf=='yes' or self.file_options.fomo=='yes'),
                msg = 'Cant activate CIS with FOMO or CASSCF')
        self.file_options.set_active('cisnumstates',4,int,'',depend=(self.file_options.cis=="yes"))
        self.file_options.set_active('cisguessvecs',self.file_options.cisnumstates,int,'',depend=(self.file_options.cis=="yes"))
        self.file_options.set_active('cismaxiter',20,int,'',depend=(self.file_options.cis=="yes"))

        # DFT
        self.file_options.set_active('rc_w',None,float,
                doc='',
                clash=(self.file_options.casscf=="yes" or self.file_options.fomo=="yes"),
                msg='',
                )
        self.file_options.set_active('dftd','no',str,'',
                clash=(self.file_options.casscf=="yes" or self.file_options.fomo=="yes"),
                msg = 'Cant activate dftd with FOMO or CASSCF')

        # GPU
        self.file_options.set_active('gpus',1,int,'')
        self.file_options.set_active('gpumem',None,int,'')

        self.file_options.set_active('coordinates','scratch/{}/tmp.xyz'.format(self.node_id),str,doc='tmp coordinate file for running TeraChem')
        self.file_options.set_active('scrdir','scratch/{}/scr/'.format(self.node_id),str,doc='')
        self.file_options.set_active('charge',0,int,doc='')
       
        # Deactivate useless keys 
        #casscf
        keys_to_del=[]
        for key,value in self.file_options.ActiveOptions.items():
            if value=="no":
                keys_to_del.append(key)
        for key in keys_to_del:
            self.file_options.deactivate(key)

        # TODO can make all these "None" and then 
        # deactivate all Nones
        if self.file_options.nalpha==0:
            self.file_options.deactivate('nalpha')
        if self.file_options.nbeta==0:
            self.file_options.deactivate('nbeta')
        if self.file_options.prmtop==None:
            self.file_options.deactivate('prmtop')
        else:
            self.file_options.force_active('coordinates','scratch/{}/tmp.inpcrd'.format(self.node_id))
        if self.file_options.xtol==None:
            self.file_options.deactivate('xtol')
        if self.file_options.qmindices==None:
            self.file_options.deactivate('qmindices')
        if self.file_options.gpumem==None:
            self.file_options.deactivate('gpumem')
        # dft
        if self.file_options.rc_w==None:
            self.file_options.deactivate('rc_w')
        if self.file_options.dftd==None:
            self.file_options.deactivate('dftd')

        self.file_options.set_active('casguess','scratch/{}/c0.casscf'.format(self.node_id),str,doc='guess for casscf',depend=(self.file_options.casscf=="yes"),msg='')

        guess_file='scratch/{}/c0'.format(self.node_id)
        self.file_options.set_active('guess',guess_file,str,doc='guess for dft/HF',
                clash=(self.file_options.casscf or self.file_options.fomo),
                depend=(os.path.isfile(guess_file)),msg='guess does not exist deactivating for now')

        ## DONE setting values ##
        
        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

        self.link_atoms=None

        if self.node_id==0:
            for line in self.file_options.record():
                print(line)
    
    @classmethod
    def copy(cls,lot,options,copy_wavefunction=True):

        node_id = options.get('node_id',1)

        print(" making folder scratch/{}".format(node_id))
        os.system('mkdir -p scratch/{}'.format(node_id))

        file_options = File_Options.copy(lot.file_options)
        options['file_options'] =file_options

        if node_id != lot.node_id and copy_wavefunction:
            if "casscf" in lot.file_options.ActiveOptions:
                old_path = 'scratch/{}/c0.casscf'.format(lot.node_id)
                new_path = 'scratch/{}/'.format(node_id)
            else:
                old_path = 'scratch/{}/c0'.format(lot.node_id)
                new_path = 'scratch/{}/'.format(node_id)
            cmd = 'cp -r ' + old_path +' ' + new_path
            print(" copying scr files\n {}".format(cmd))
            os.system(cmd)
            os.system('wait')
        return cls(lot.options.copy().set_values(options))

    def write_input(self,inpfilename,geom,mult,ad_idx,runtype='gradient'):
        # first write the file, run it, and the read the output
        # filenames
        inpfile = open(inpfilename,'w')

        # Write the file
        for key,value in self.file_options.ActiveOptions.items():
            inpfile.write('{0:<25}{1:<25}\n'.format(key,value))
        inpfile.write('spinmult             {}\n'.format(mult))

        if "casscf" in self.file_options.ActiveOptions or "casci" or self.file_options.ActiveOptions:
            if runtype == "gradient":
                inpfile.write('castarget            {}\n'.format(ad_idx))
                inpfile.write('castargetmult        {}\n'.format(mult))
            elif runtype=="coupling":
                inpfile.write('nacstate1 {}\n'.format(self.coupling_states[0]))
                inpfile.write('nacstate2 {}\n'.format(self.coupling_states[1]))
                inpfile.write('castargetmult        {}\n'.format(mult))
        elif "cis" in self.file_options.ActiveOptions:
            if runtype == "gradient":
                inpfile.write('cistarget            {}\n'.format(ad_idx))
            elif runtype=="coupling":
                inpfile.write('nacstate1 {}\n'.format(self.coupling_states[0]))
                inpfile.write('nacstate2 {}\n'.format(self.coupling_states[1]))


        # Runtype
        inpfile.write("run         {}\n\n".format(runtype))
        inpfile.close()

        # Write the temporary geometry files
        if "prmtop" in self.file_options.ActiveOptions:
            manage_xyz.write_amber_xyz('scratch/{}/tmp.inpcrd'.format(self.node_id),geom)
        else:
            manage_xyz.write_xyz('scratch/{}/tmp.xyz'.format(self.node_id),geom,scale=1.0)

        return
    
    def go(self,geom,mult,ad_idx,runtype='gradient'):
        ''' compute an individual gradient or NACME '''

        inpfilename = 'scratch/{}/{}'.format(self.node_id,self.lot_inp_file)
        outfilename = 'scratch/{}/output.dat'.format(self.node_id)

        # Write input file
        self.write_input(inpfilename,geom,mult,ad_idx,runtype)

        ### RUN THE CALCULATION ###
        cmd = "terachem {} > {}".format(inpfilename,outfilename)
        os.system(cmd)

        # Turn on C0 for non-CASSCF calculations after running
        if 'guess' not in self.file_options.ActiveOptions and 'casscf' not in self.file_options.ActiveOptions:
            if mult == 2:
                self.file_options.set_active('guess','scratch/{}/ca0 scratch/{}/cb0'.format(self.node_id,self.node_id),str,'')
            else:
                self.file_options.set_active('guess','scratch/{}/c0'.format(self.node_id),str,'')

        # if QM/MM get link atoms
        if "prmtop" in self.file_options.ActiveOptions and self.link_atoms is None:
            # parse qmindices
            with open(self.file_options.qmindices) as f:
                qmindices = f.read().splitlines()
            self.qmindices = [int(i) for i in qmindices]
            all_indices = range(len(geom))
            self.mm_indices = list(set(all_indices) - set(self.qmindices))
            self.QM_atoms = len(self.qmindices)
            pattern = re.compile(r'Number of (\S+) atoms:\s+(\d+)')
            for i,line in enumerate(open(outfilename)):
                for match in re.finditer(pattern,line):
                    if match.group(1) == "QM":
                        actual_QM_atoms = int(match.group(2))
                    elif match.group(1) =="MM":
                        self.MM_atoms = int(match.group(2))
            self.link_atoms = actual_QM_atoms - self.QM_atoms

        ## POST PROCESSING  ##
        # copy the wavefunction file
        if "casscf" in self.file_options.ActiveOptions:
            cp_cmd = 'cp scratch/{}/scr/c0.casscf scratch/{}/'.format(self.node_id,self.node_id)
            os.system(cp_cmd)
        else:
            if mult==2:
                cp_cmd = 'cp scratch/{}/scr/ca0 scratch/{}/'.format(self.node_id,self.node_id)
                os.system(cp_cmd)
                cp_cmd = 'cp scratch/{}/scr/cb0 scratch/{}/'.format(self.node_id,self.node_id)
                os.system(cp_cmd)
            else:
                cp_cmd = 'cp scratch/{}/scr/c0 scratch/{}/'.format(self.node_id,self.node_id)
                os.system(cp_cmd)

        if "casscf" in self.file_options.ActiveOptions:
            cp_cmd = 'cp scratch/{}/scr/casscf.molden scratch/{}/'.format(self.node_id,self.node_id)
            os.system(cp_cmd)

        # Get the gradient and coupling
        if "prmtop" not in self.file_options.ActiveOptions:
            if runtype=='gradient':
                cp_grad = 'cp scratch/{}/scr/grad.xyz scratch/{}/grad_{}_{}.xyz'.format(self.node_id,self.node_id,mult,ad_idx)
                os.system(cp_grad)
            elif runtype == "coupling":
                cp_coup = 'cp scratch/{}/scr/grad.xyz scratch/{}/coup_{}_{}.xyz'.format(self.node_id,self.node_id,self.coupling_states[0],self.coupling_states[1])
                os.system(cp_coup)

        # clean up
        #rm_cmd = 'rm -rf scratch/{}/scr'.format(self.node_id)
        #os.system(rm_cmd)

        return
        #Done go

    def run(self,geom,runtype=None):
        ''' calculate all states '''

        tempfileout='scratch/{}/output.dat'.format(self.node_id)
        self.grada=[]
        self.coup=[]
        self.E = []
        if not self.gradient_states and not self.coupling_states or runtype=="energy":
            print(" only calculating energies")
            # TODO what about multiple multiplicities? 
            tup = self.states[0]
            self.go(geom,tup[0],None,'energy')
            # make grada all None
            for tup in self.states:
                self.grada.append((tup[0],tup[1],None))
        elif self.gradient_states:

            ### Calculate gradient(s) ###
            for tup in self.states:
                if tup in self.gradient_states:
                    ## RUN ##
                    self.go(geom,tup[0],tup[1],'gradient')

                    # GET GRADIENT FOR QMMMM -- REGULAR GRADIENT IS PARSED THROUGH grad.xyz
                    # QMMM is done differently :(
                    if "prmtop" in self.file_options.ActiveOptions:
                        tmpgrad=[]
                        with open(tempfileout,"r") as f:
                            for line in f:
                                if line.startswith("dE/dX",8):
                                    for i in range(self.QM_atoms):
                                        findline = next(f,'').strip()
                                        mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                        tmpgrad.append([
                                            float(mobj.group(1)),
                                            float(mobj.group(2)),
                                            float(mobj.group(3)),
                                            ])
                                    for i in range(self.link_atoms):
                                        next(f)
                                    # read two lines seperating the QM and MM regions
                                    next(f)
                                    next(f) 
                                    for i in range(self.MM_atoms):
                                        findline = next(f,'').strip()
                                        mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                        tmpgrad.append([
                                            float(mobj.group(1)),
                                            float(mobj.group(2)),
                                            float(mobj.group(3)),
                                            ])

                        tmpgrad = np.asarray(tmpgrad)
                        grad = np.zeros_like(tmpgrad)
                        grad[self.qmindices] = tmpgrad[:len(self.qmindices)]
                        grad[self.mm_indices] = tmpgrad[len(self.qmindices):]
                        self.grada.append((tup[0],tup[1],grad))

                    else:  
                        # getting gradient of non-prmtop job
                        gradfile='scratch/{}/grad_{}_{}.xyz'.format(self.node_id,tup[0],tup[1])
                        grad = manage_xyz.read_xyz(gradfile,scale=1.0)
                        grad = manage_xyz.xyz_to_np(grad)
                        self.grada.append((tup[0],tup[1],grad))

                else:  # not in gradient states
                    self.grada.append((tup[0],tup[1],None))
        if self.coupling_states: 
            #TODO  Warning only allowing one coupling state, with singlet multiplicities
            self.go(geom,1,None,runtype="coupling")
            if "prmtop" in self.file_options.ActiveOptions:
                tmpcoup = []
                with open(tempfileout,"r") as f:
                    for line in f:
                        if line.startswith("dE/dX",8): #will work for SA-MC and RSPT2 HF
                            for i in range(self.QM_atoms):
                                findline = next(f,'').strip()
                                mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                tmpcoup.append([
                                    float(mobj.group(1)),
                                    float(mobj.group(2)),
                                    float(mobj.group(3)),
                                    ])
                            # read link atoms
                            for i in range(self.link_atoms):
                                next(f)
                            next(f)
                            next(f) # read two lines seperating the QM and MM regions
                            for i in range(self.MM_atoms):
                                findline = next(f,'').strip()
                                mobj = re.match(r'(\S+)\s+(\S+)\s+(\S+)\s*$', findline)
                                tmpcoup.append([
                                    float(mobj.group(1)),
                                    float(mobj.group(2)),
                                    float(mobj.group(3)),
                                    ])
                self.coup = np.zeros_like(tmpgrad)
                self.coup[self.qmindices] = tmpcoup[:len(self.qmindices)]
                self.coup[self.mm_indices] = tmpcoup[len(self.qmindices):]
            else:
                coupfile='scratch/{}/coup_{}_{}.xyz'.format(self.node_id,self.coupling_states[0],self.coupling_states[1])
                coup = manage_xyz.read_xyz(coupfile,scale=1.0)
                self.coup = manage_xyz.xyz_to_np(coup)
        #### FINALLY DONE WITH RUN Energy/Gradients ###

        # parse the output for Energies  --> This can be done on any of the files since they should be the same
        #TODO Parse other multiplicities is broken here
        if "casscf" in self.file_options.ActiveOptions or "casci" in self.file_options.ActiveOptions:
            pattern = re.compile(r'Singlet state  \d energy: \s* ([-+]?[0-9]*\.?[0-9]+)')
        else:
            pattern = re.compile(r'FINAL ENERGY: ([-+]?[0-9]*\.?[0-9]+) a.u.')
        tmp =[]
        for i,line in enumerate(open(tempfileout)):
            for match in re.finditer(pattern,line):
                tmp.append(float(match.group(1)))
        
        # Terachem has weird printout for td-dft energy
        if 'cis' in self.file_options.ActiveOptions:
            lines=open(tempfileout).readlines()
            lines = lines[([x for x,y in enumerate(lines) if re.match(r'^\s+Final Excited State Results:', y)][0]+4):]
            for line in lines:
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
                if mobj:
                    tmp.append(float(mobj.group(2)))

        # Finalize Energy list
        for i,tup in enumerate(self.states):
            self.E.append((tup[0],tup[1],tmp[i]))
        with open('scratch/E_{}.txt'.format(self.node_id),'w') as f:
            for E in self.E:
                f.write('{} {} {:9.7f}\n'.format(E[0],E[1],E[2]))
                
        self.hasRanForCurrentCoords=True
        return

    def get_energy(self,coords,multiplicity,state,runtype=None):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom,runtype)
        tmp = self.search_PES_tuple(self.E,multiplicity,state)[0][2]
        return self.search_PES_tuple(self.E,multiplicity,state)[0][2]*units.KCAL_MOL_PER_AU

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_PES_tuple(self.grada,multiplicity,state)[0][2]
        if tmp is not None:
            return np.asarray(tmp)*units.ANGSTROM_TO_AU  #Ha/bohr*bohr/ang=Ha/ang
        else:
            return None

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).any():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return np.reshape(self.coup,(3*len(self.coup),1))*units.ANGSTROM_TO_AU

if __name__=="__main__":

    #filepath="../../data/ethylene.xyz"
    filepath='tmp.inpcrd'
    geom=manage_xyz.read_xyz('geom.xyz')
    #TC = TeraChem.from_options(states=[(1,1)],fnm=filepath,lot_inp_file='tc_options.txt')
    #TC = TeraChem.from_options(states=[(1,0),(1,1)],fnm=filepath,lot_inp_file='tc_options.txt')
    TC = TeraChem.from_options(states=[(1,0),(1,1)],gradient_states=[(1,1)],geom=geom,lot_inp_file='tc_options.txt',node_id=3)

    for line in TC.file_options.record():
        print(line)
    
    #print(TC.casscf)
    #print(TC.rc_w)

    #for key in TC.file_options.ActiveOptions:
    #    print(key,TC.file_options.ActiveOptions[key])
    for key,value in TC.file_options.ActiveOptions.items():
        print(key,value)

    #c = deepcopy(TC.file_options)
    TC2 = TeraChem.copy(TC,options={'node_id':3})
    print(id(TC.file_options))
    print(id(TC2))

    #geom=manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print("getting energy")
    print(TC.get_energy(xyz,1,0))
    #print(TC.get_energy(xyz,1,1))
    print("getting grad")
    print(TC.get_gradient(xyz,1,0))
    print(TC.get_gradient(xyz,1,1))

    #xyz = xyz+ np.random.rand(xyz.shape[0],xyz.shape[1])*0.1
    #print(TC.get_energy(xyz,1,0))
    #print(TC.get_gradient(xyz,1,0))
