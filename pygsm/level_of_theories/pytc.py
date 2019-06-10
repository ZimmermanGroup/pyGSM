# standard library imports
import sys
from os import path

# third party 
import numpy as np
import lightspeed as ls

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .base_lot import Lot
from utilities import *
from rhf_lot import RHF_LOT
from casci_lot_svd import CASCI_LOT_SVD

#TODO get rid of get_energy, get_gradient
class PyTC(Lot):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Lot. Requires a PSIW object
    """

    def __init__(self,options):
        super(PyTC,self).__init__(options)
        if self.lot_inp_file is not None:
           exec(open(self.lot_inp_file).read()) 
           print(' done executing lot_inp_file')
           self.options['job_data']['psiw'] = psiw

    @property
    def psiw(self):
        return self.options['job_data']['psiw']

    @psiw.setter
    def psiw(self,value):
        self.options['job_data']['psiw']=value

    def get_energy(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_tuple(self.E,multiplicity)
        return tmp[state][1]*units.KCAL_MOL_PER_AU

    def get_mm_energy(self,coords):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.psiw.update_qmmm(coords*units.ANGSTROM_TO_AU)
        if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
            return self.psiw.casci.ref.geometry.qmmm.mm_energy
        else:
            return self.psiw.rhf.geometry.qmmm.mm_energy

    def get_mm_gradient(self,coords):
        #TODO need diff variable for hasRan MM energy
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.psiw.update_qmmm(coords*units.ANGSTROM_TO_AU)
        if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
            return self.psiw.casci.ref.geometry.qmmm.mm_gradient
        else:
            return self.psiw.rhf.geometry.qmmm.mm_gradient

    def run_code(self,T):
        self.psiw = self.psiw.update_xyz(T)
        for state in self.states:
            multiplicity=state[0]
            ad_idx=state[1]
            S=multiplicity-1
            if self.psiw.__class__.__name__=="CASCI_LOT" or self.psiw.__class__.__name__=="CASCI_LOT_SVD":
                self.E.append((multiplicity,self.psiw.compute_energy(S=S,index=ad_idx)))
                tmp = self.psiw.compute_gradient(S=S,index=ad_idx)
            elif self.psiw.__class__.__name__=="RHF_LOT": 
                self.E.append((multiplicity,self.psiw.compute_energy()))
                tmp = self.psiw.compute_gradient()
            self.grada.append((multiplicity,tmp[...]))
        if self.do_coupling==True:
            state1=self.states[0][1]
            state2=self.states[1][1]
            tmp = self.psiw.compute_coupling(S=S,indexA=state1,indexB=state2)
            self.coup = tmp[...]

    def run(self,geom,verbose=False):
        self.E=[]
        self.grada=[]
        #normal update
        coords = manage_xyz.xyz_to_np(geom)
        T = ls.Tensor.array(coords*units.ANGSTROM_TO_AU)

        if not verbose:
            with open('psiw_jobs.txt','a') as out:
                with nifty.custom_redirection(out):
                    self.run_code(T)
        else:
            self.run_code(T)
                #filename="{}_rhf_update.molden".format(self.node_id)
                #self.psiw.casci.reference.save_molden_file(filename)

        self.hasRanForCurrentCoords=True
        return

    def get_gradient(self,coords,multiplicity,state):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*units.ANGSTROM_TO_AU

    def get_coupling(self,coords,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            geom = manage_xyz.np_to_xyz(self.geom,self.currentCoords)
            self.run(geom)
        return np.reshape(self.coup,(3*len(self.coup),1))*units.ANGSTROM_TO_AU


if __name__=="__main__":
    import psiw
    from utilities import nifty

    ##### => Job Data <= #####
    states = [(1,0),(1,1)]
    charge=0
    nocc=7
    nactive=2
    basis='6-31gs'
    filepath='../../data/ethylene.xyz'

    #### => PSIW Obj <= ######
    nifty.printcool("Build resources")
    resources = ls.ResourceList.build()
    nifty.printcool('{}'.format(resources))
    
    molecule = ls.Molecule.from_xyz_file(filepath)
    geom = psiw.geometry.Geometry.build(
        resources=resources,
        molecule=molecule,
        basisname=basis,
        )
    nifty.printcool('{}'.format(geom))
    
    ref = psiw.RHF.from_options(
         geometry= geom, 
         g_convergence=1.0E-6,
         fomo=True,
         fomo_method='gaussian',
         fomo_temp=0.3,
         fomo_nocc=nocc,
         fomo_nact=nactive,
         print_level=1,
        )
    ref.compute_energy()
    casci = psiw.CASCI.from_options(
        reference=ref,
        nocc=nocc,
        nact=nactive,
        nalpha=nactive/2,
        nbeta=nactive/2,
        S_inds=[0],
        S_nstates=[2],
        print_level=1,
        )
    casci.compute_energy()
    psiw = psiw.CASCI_LOT.from_options(
        casci=casci,
        rhf_guess=True,
        rhf_mom=True,
        orbital_coincidence='core',
        state_coincidence='full',
        )

    nifty.printcool("Build the pyGSM Level of Theory object (LOT)")
    lot=PyTC.from_options(states=[(1,0),(1,1)],job_data={'psiw':psiw},do_coupling=False,fnm=filepath) 

    geoms = manage_xyz.read_xyz(filepath,scale=1.)
    coords= manage_xyz.xyz_to_np(geoms)
    e=lot.get_energy(coords,1,0)
    print(e)

    g=lot.get_gradient(coords,1,0)
    print(g)
