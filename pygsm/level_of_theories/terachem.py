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
    from .base_lot import Lot,copy_file
    from .file_options import File_Options
except:
    from base_lot import Lot,copy_file
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
        self.file_options.set_active('threspdp',None,float,'')  # threspdp 0.0001

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

        # hh-tda
        self.file_options.set_active('hhtda','no',str,'',
                clash=(self.file_options.casscf=="yes"),
                msg='')
        self.file_options.set_active('hhtdasinglets',3,int,'',depend=(self.file_options.hhtda=="yes"),msg='')
        self.file_options.set_active('hhtdatriplets',0,int,'',depend=(self.file_options.hhtda=="yes"),msg='')
        self.file_options.set_active('cphftol',1e-6,float,'',depend=(self.file_options.hhtda=="yes"),msg='')
        self.file_options.set_active('cphfiter',None,int,'',depend=(self.file_options.hhtda=="yes"),msg='')  # a good value could be 1000
        self.file_options.set_active('cphfalgorithm',None,str,'',depend=(self.file_options.hhtda=="yes"),msg='') # a possible value is inc_diis
        self.file_options.set_active('sphericalbasis','no',str,'',depend=(self.file_options.hhtda=="yes"),msg='')
        self.file_options.set_active('max_l_in_basis',None,int,'',depend=(self.file_options.sphericalbasis=="yes"),msg='')

        # FON
        self.file_options.set_active('fon','no',str,'',
                clash=(self.file_options.casscf=="yes"),
                depend=(self.file_options.casci=="yes" or self.file_options.hhtda=="yes"),
                msg = 'Unactivated, or cant activate FOMO with CASSCF')
        self.file_options.set_active('fon_method',None,str,'',depend=(self.file_options.fon=="yes"),msg='')
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
        self.file_options.set_active('cisalgorithm',None,str,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cisnumstates',4,int,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cisguessvecs',self.file_options.cisnumstates,int,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cismaxiter',20,int,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cismax',100,int,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cisconvtol',1.0e-6,float,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))
        self.file_options.set_active('cisincremental','no',str,'',depend=(self.file_options.cis=="yes" or self.file_options.hhtda=="yes"))

        # DFT
        self.file_options.set_active('rc_w',None,float,
                doc='',
                clash=(self.file_options.casscf=="yes" or self.file_options.fomo=="yes"),
                msg='',
                )
        self.file_options.set_active('dftd','no',str,'',
                clash=(self.file_options.casscf=="yes" or self.file_options.fomo=="yes"),
                msg = 'Cant activate dftd with FOMO(?) or CASSCF')
        self.file_options.set_active('dftgrid',None,int,'',clash=(self.file_options.casscf=="yes"))

        # GPU
        self.file_options.set_active('gpus',1,int,'')
        self.file_options.set_active('gpumem',None,int,'')

   
        self.file_options.set_active('scrdir','scratch/{:03}/{}/scr/'.format(self.ID,self.node_id),str,doc='')
        self.file_options.force_active('scrdir','scratch/{:03}/{}/scr/'.format(self.ID,self.node_id))
        self.file_options.set_active('coordinates','scratch/{:03}/{}/tmp.xyz'.format(self.ID,self.node_id),str,doc='',)
        self.file_options.force_active('coordinates','scratch/{:03}/{}/tmp.xyz'.format(self.ID,self.node_id))
        self.file_options.set_active('charge',0,int,doc='')
       
        # Deactivate useless keys 
        #casscf
        keys_to_del=[]
        for key,value in self.file_options.ActiveOptions.items():
            if value=="no" or value==None:
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
            self.file_options.force_active('coordinates','scratch/{:03}/{}/tmp.inpcrd'.format(self.ID,self.node_id))
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

        self.file_options.set_active('casguess','scratch/{:03}/{}/c0.casscf'.format(self.ID,self.node_id),str,doc='guess for casscf',depend=(self.file_options.casscf=="yes"),msg='')

        guess_file='scratch/{:03}/{}/c0'.format(self.ID,self.node_id)
        if os.path.isfile(guess_file):
            self.file_options.set_active('guess',guess_file,str,doc='guess for dft/HF',
                    clash=(self.file_options.casscf or self.file_options.fomo),
                    depend=(os.path.isfile(guess_file)),msg='guess does not exist deactivating for now')
        else:
                self.file_options.set_active('guess','sad',str,doc='',
                    clash=(self.file_options.casscf or self.file_options.fomo),
                    msg=' deactivating guess for CASSCF and FOMO')

        ## DONE setting values ##
        

        self.link_atoms=None

        if self.node_id==0:
            for line in self.file_options.record():
                print(line)
    
    @classmethod
    def copy(cls,lot,options,copy_wavefunction=True):

        node_id = options.get('node_id',1)

        #print(" making folder scratch/{:03}/{}".format(lot.ID,node_id))
        #os.system('mkdir -p scratch/{:03}/{}'.format(lot.ID,node_id))

        file_options = File_Options.copy(lot.file_options)
        options['file_options'] =file_options

        if node_id != lot.node_id and copy_wavefunction:
            if "casscf" in lot.file_options.ActiveOptions:
                old_path = 'scratch/{:03}/{}/c0.casscf'.format(lot.ID,lot.node_id)
                new_path = 'scratch/{:03}/{}/c0.casscf'.format(lot.ID,node_id)
                copy_file(old_path,new_path)
            else:
                use_alpha=False
                for state in lot.states:
                  if state[0] == 2:
                      use_alpha=True
                if use_alpha:
                  old_path = 'scratch/{:03}/{}/ca0'.format(lot.ID,lot.node_id)
                  new_path = 'scratch/{:03}/{}/'.format(lot.ID,node_id)
                  copy_file(old_path,new_path)
                  old_path = 'scratch/{:03}/{}/cb0'.format(lot.ID,lot.node_id)
                  new_path = 'scratch/{:03}/{}/'.format(lot.ID,node_id)
                  copy_file(old_path,new_path)
                else:
                  old_path = 'scratch/{:03}/{}/c0'.format(lot.ID,lot.node_id)
                  new_path = 'scratch/{:03}/{}/'.format(lot.ID,node_id)
                  copy_file(old_path,new_path)
            #cmd = 'cp -r ' + old_path +' ' + new_path
            #print(" copying scr files\n {}".format(cmd))
            #os.system(cmd)
            #os.system('wait')
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
        elif "cis" in self.file_options.ActiveOptions or "hhtda" in self.file_options.ActiveOptions:
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
            manage_xyz.write_amber_xyz('scratch/{:03}/{}/tmp.inpcrd'.format(self.ID,self.node_id),geom)
        else:
            manage_xyz.write_xyz('scratch/{:03}/{}/tmp.xyz'.format(self.ID,self.node_id),geom,scale=1.0)

        return
    
    def run(self,geom,mult,ad_idx,runtype='gradient'):
        ''' compute an individual gradient or NACME '''

        inpfilename = 'scratch/{:03}/{}/{}'.format(self.ID,self.node_id,self.lot_inp_file)
        outfilename = 'scratch/{:03}/{}/output.dat'.format(self.ID,self.node_id)

        # Write input file
        self.write_input(inpfilename,geom,mult,ad_idx,runtype)

        ### RUN THE CALCULATION ###
        cmd = "terachem {} > {}".format(inpfilename,outfilename)
        os.system(cmd)

        # Turn on C0 for non-CASSCF calculations after running
        if 'guess' not in self.file_options.ActiveOptions and 'casscf' not in self.file_options.ActiveOptions or self.file_options.guess in ["sad","generate"]:
            if mult == 2:
                self.file_options.force_active('guess','scratch/{:03}/{}/ca0 scratch/{:03}/{}/cb0'.format(self.ID,self.node_id,self.ID,self.node_id))
            else:
                self.file_options.force_active('guess','scratch/{:03}/{}/c0'.format(self.ID,self.node_id))

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
            cp_cmd = 'cp scratch/{:03}/{}/scr/c0.casscf scratch/{:03}/{}/'.format(self.ID,self.node_id,self.ID,self.node_id)
            os.system(cp_cmd)
        else:
            if mult==2:
                cp_cmd = 'cp scratch/{:03}/{}/scr/ca0 scratch/{:03}/{}/'.format(self.ID,self.node_id,self.ID,self.node_id)
                os.system(cp_cmd)
                cp_cmd = 'cp scratch/{:03}/{}/scr/cb0 scratch/{:03}/{}/'.format(self.ID,self.node_id,self.ID,self.node_id)
                os.system(cp_cmd)
            else:
                cp_cmd = 'cp scratch/{:03}/{}/scr/c0 scratch/{:03}/{}/'.format(self.ID,self.node_id,self.ID,self.node_id)
                os.system(cp_cmd)

        if "casscf" in self.file_options.ActiveOptions:
            cp_cmd = 'cp scratch/{:03}/{}/scr/casscf.molden scratch/{:03}/{}/'.format(self.ID,self.node_id,self.ID,self.node_id)
            os.system(cp_cmd)

        # Get the gradient and coupling
        if "prmtop" not in self.file_options.ActiveOptions:
            if runtype=='gradient':
                cp_grad = 'cp scratch/{:03}/{}/scr/grad.xyz scratch/{:03}/{}/grad_{}_{}.xyz'.format(self.ID,self.node_id,self.ID,self.node_id,mult,ad_idx)
                os.system(cp_grad)
            elif runtype == "coupling":
                cp_coup = 'cp scratch/{:03}/{}/scr/grad.xyz scratch/{:03}/{}/coup_{}_{}.xyz'.format(self.ID,self.node_id,self.ID,self.node_id,self.coupling_states[0],self.coupling_states[1])
                os.system(cp_coup)

        # clean up
        #rm_cmd = 'rm -rf scratch/{}/scr'.format(self.node_id)
        #os.system(rm_cmd)

        return
        #Done go

    def runall(self,geom,runtype=None):
        ''' calculate all states '''

        self.Gradients={}
        self.Energies = {}
        self.Couplings = {}

        tempfileout='scratch/{:03}/{}/output.dat'.format(self.ID,self.node_id)
        if not self.gradient_states and not self.coupling_states or runtype=="energy":
            print(" only calculating energies")
            # TODO what about multiple multiplicities? 
            tup = self.states[0]
            self.run(geom,tup[0],None,'energy')
            # make grada all None
            for tup in self.states:
                self._Gradients[tup] = self.Gradient(None,None)
        elif self.gradient_states:
            ### Calculate gradient(s) ###
            for tup in self.states:
                if tup in self.gradient_states:
                    ## RUN ##
                    self.run(geom,tup[0],tup[1],'gradient')
                    self.parse_grad(tup)

                else:  # not in gradient states
                    self._Gradients[tup] = self.Gradient(None,None)

        if self.coupling_states: 
            #TODO  Warning only allowing one coupling state, with singlet multiplicities
            self.run(geom,1,None,runtype="coupling")
            self.parse_coup()
        #### FINALLY DONE WITH RUN Energy/Gradients ###

        self.parse_E()       
        self.hasRanForCurrentCoords=True
        return
        
    def parse_E(self):
        # parse the output for Energies  --> This can be done on any of the files since they should be the same

        tempfileout='scratch/{:03}/{}/output.dat'.format(self.ID,self.node_id)

        #TODO Parse other multiplicities is broken here
        if "casscf" in self.file_options.ActiveOptions or "casci" in self.file_options.ActiveOptions:
            pattern = re.compile(r'Singlet state  \d energy: \s* ([-+]?[0-9]*\.?[0-9]+)')
            tmp =[]
            for i,line in enumerate(open(tempfileout)):
                for match in re.finditer(pattern,line):
                    tmp.append(float(match.group(1)))
        # Terachem has weird printout for td-dft energy
        elif 'cis' in self.file_options.ActiveOptions:
            tmp =[]
            pattern = re.compile(r'FINAL ENERGY: ([-+]?[0-9]*\.?[0-9]+) a.u.')
            for i,line in enumerate(open(tempfileout)):
                for match in re.finditer(pattern,line):
                    tmp.append(float(match.group(1)))
            lines=open(tempfileout).readlines()
            lines = lines[([x for x,y in enumerate(lines) if re.match(r'^\s+Final Excited State Results:', y)][0]+4):]
            for line in lines:
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)', line)
                if mobj:
                    tmp.append(float(mobj.group(2)))

        elif 'hhtda' in self.file_options.ActiveOptions:
            tmp=[]
            lines=open(tempfileout).readlines()
            lines = lines[([x for x,y in enumerate(lines) if re.match(r'^\s+ Root   Mult.', y)][0]+1):]
            for line in lines:
                mobj = re.match(r'^\s*(\d+)\s+(\S+)\s+(\S+)', line)
                if mobj:
                    if mobj.group(2)=="singlet":
                        tmp.append(float(mobj.group(3)))
        else:
            pattern = re.compile(r'FINAL ENERGY: ([-+]?[0-9]*\.?[0-9]+) a.u.')
            tmp =[]
            for i,line in enumerate(open(tempfileout)):
                for match in re.finditer(pattern,line):
                    tmp.append(float(match.group(1)))
        

        for E,state in zip(tmp,self.states):
            self._Energies[state] = self.Energy(E,'Hartree')

        self.write_E_to_file()
                
        return
    
    def parse_grad(self,
            state,
            tempfileout=None,
            ):

        if tempfileout==None:
            tempfileout ='scratch/{:03}/{}/output.dat'.format(self.ID,self.node_id)
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
        else:  
            # getting gradient of non-prmtop job
            gradfile='scratch/{:03}/{}/grad_{}_{}.xyz'.format(self.ID,self.node_id,state[0],state[1])
            grad = manage_xyz.read_xyz(gradfile,scale=1.0)
            grad = manage_xyz.xyz_to_np(grad)
        self._Gradients[state] = self.Gradient(grad,"Hartree/Bohr")

    def parse_coup(self):
        tempfileout ='scratch/{:03}/{}/output.dat'.format(self.ID,self.node_id)
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
            coup = np.zeros_like(tmpcoup)
            coup[self.qmindices] = tmpcoup[:len(self.qmindices)]
            coup[self.mm_indices] = tmpcoup[len(self.qmindices):]
            coup = np.asarray(coup)
        else:
            coupfile='scratch/{:03}/{}/coup_{}_{}.xyz'.format(self.ID,self.node_id,self.coupling_states[0],self.coupling_states[1])
            coup = manage_xyz.read_xyz(coupfile,scale=1.0)
            coup = manage_xyz.xyz_to_np(coup)
        self.Couplings[self.coupling_states] = self.Coupling(coup,'Hartree/Bohr') 
    

if __name__=="__main__":

    filepath="../../data/ethylene.xyz"
    #filepath='tmp.inpcrd'
    geom=manage_xyz.read_xyz(filepath)
    #TC = TeraChem.from_options(states=[(1,1)],fnm=filepath,lot_inp_file='tc_options.txt')
    #TC = TeraChem.from_options(states=[(1,0),(1,1)],fnm=filepath,lot_inp_file='tc_options.txt')
    TC = TeraChem.from_options(states=[(1,0),(1,1)],gradient_states=[(1,1)],geom=geom,lot_inp_file='tc_options.txt',node_id=0)

    #for line in TC.file_options.record():
    #    print(line)
    
    #print(TC.casscf)
    #print(TC.rc_w)

    #for key in TC.file_options.ActiveOptions:
    #    print(key,TC.file_options.ActiveOptions[key])
    #for key,value in TC.file_options.ActiveOptions.items():
    #    print(key,value)

    #c = deepcopy(TC.file_options)
    #TC2 = TeraChem.copy(TC,options={'node_id':3})
    #print(id(TC.file_options))
    #print(id(TC2))

    #geom=manage_xyz.read_xyz(filepath)
    xyz = manage_xyz.xyz_to_np(geom)
    print("getting energy")
    print(TC.get_energy(xyz,1,0))
    print(TC.get_energy(xyz,1,1))
    print("getting grad")
    print(TC.get_gradient(xyz,1,0))
    print(TC.get_gradient(xyz,1,1))

    #xyz = xyz+ np.random.rand(xyz.shape[0],xyz.shape[1])*0.1
    #print(TC.get_energy(xyz,1,0))
    #print(TC.get_gradient(xyz,1,0))
