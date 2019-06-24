# standard library imports
import sys
from os import path

# third party 
import numpy as np
import lightspeed as ls 
#import psiw
import est
import json

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .base_lot import Lot
from utilities import *
from .rhf_lot import RHF_LOT
from .casci_lot_svd import CASCI_LOT_SVD

#TODO get rid of get_energy, get_gradient
class PyTC(Lot):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Lot. Requires a PSIW object
    """

    def __init__(self,options):
        super(PyTC,self).__init__(options)
        if self.lot_inp_file is not None and self.lot is None:
            self.build_lot_from_dictionary()
            #print(self.lot)
            #print(' done executing lot_inp_file')
            #exec(open(self.lot_inp_file).read()) 
            #print(lot)
            #self.options['job_data']['lot'] = lot

    def build_lot_from_dictionary(self):

        d = {}
        #with open(self.lot_inp_file) as f:
        #    for line in f:
        #        (key, val) = line.split()
        #        d[str(key)] = val
        d = json.load(open(self.lot_inp_file))
        print(d)

        filepath = d.get('filepath',None)

        # QM
        basis = d.get('basis',None)
        charge = d.get('charge',0)
        S_inds = d.get('S_inds',[0])
        S_nstates = d.get('S_nstates',[1])

        # SCF
        diis_max_vecs = d.get('diis_max_vecs',6)
        maxiter = d.get('maxiter',200)
        cphf_diis_max_vecs=d.get('cphf_diis_max_vecs',6)
        diis_use_disk = d.get('diis_use_disk',False)
        rhf_guess=d.get('rhf_guess',True)
        rhf_mom=d.get('rhf_mom',True)
    
        # active space
        doCASCI = d.get('doCASCI',False)
        nactive = d.get('nactive',0)
        nocc = d.get('nocc',0)
        nalpha = d.get('nalpha',int(nactive/2))
        nbeta = d.get('nbeta',nalpha)

        # FOMO
        doFOMO = d.get('doFOMO',False)
        fomo = d.get('fomo',True)
        fomo_temp = d.get('fomo_temp',0.3)
        fomo_nocc=d.get('fomo_nocc',nocc)
        fomo_nact = d.get('fomo_nact',nactive)
        fomo_method = d.get('fomo_method','gaussian')

        # QMMM
        doQMMM = d.get('doQMMM',False)
        prmtopfile = d.get('prmtopfile',None)
        inpcrdfile = d.get('inpcrdfile',None)
        qmindsfile = d.get('qmindsfile',None)
        
        # DFT
        doDFT = d.get('doDFT',False)
        dft_functional=d.get('dft_functional','None')
        dft_grid_name = d.get('dft_grid_name','SG0')
   
        nifty.printcool("Building Resources")
        resources = ls.ResourceList.build()
        nifty.printcool("{}".format(resources))

        if not doQMMM:
            nifty.printcool("Building Molecule and Geom")
            molecule = ls.Molecule.from_xyz_file(filepath)    
            geom =est.Geometry.build(
                resources=resources,
                molecule=molecule,
                basisname=basis,
                )
        else:
            nifty.printcool("Building QMMM Molecule and Geom")
            qmmm = QMMM.from_prmtop(
                prmtopfile=prmtopfile,
                inpcrdfile=inpcrdfile,
                qmindsfile=qmindsfile,
                charge=charge,
                )
            geom = geometry.Geometry.build(
                resources=resources,
                qmmm=qmmm,
                basisname=basis,
                )
        nifty.printcool("{}".format(geom))

        if doFOMO:
            nifty.printcool("Building FOMO RHF")
            ref = est.RHF.from_options(
                    geometry=geom,
                    diis_max_vecs=diis_max_vecs,
                    maxiter=maxiter,
                    cphf_diis_max_vecs=cphf_diis_max_vecs,
                    diis_use_disk=diis_use_disk,
                    fomo=fomo,
                    fomo_method=fomo_method,
                    fomo_temp=fomo_temp,
                    fomo_nocc=nocc,
                    fomo_nact=nactive,
                    )
            ref.compute_energy()
        elif doDFT: 
            nifty.printcool("Building DFT LOT")
            ref = est.RHF.from_options(
                    geometry=geom,
                    diis_max_vecs=diis_max_vecs,
                    maxiter=maxiter,
                    cphf_diis_max_vecs=cphf_diis_max_vecs,
                    diis_use_disk=diis_use_disk,
                    dft_functional=dft_functional,
                    dft_grid_name=dft_grid_name
                    )
            self.lot = RHF_LOT.from_options(rhf=ref)
        else:
            raise NotImplementedError

        if doCASCI:
            nifty.printcool("Building CASCI LOT")
            casci = est.CASCI.from_options(
                reference=ref,
                nocc=nocc,
                nact=nactive,
                nalpha=nalpha,
                nbeta=nbeta,
                S_inds=S_inds,
                S_nstates=S_nstates,
                print_level=1,
                )
            casci.compute_energy()
            self.lot = est.CASCI_LOT.from_options(
                casci=casci,
                print_level=1,
                rhf_guess=rhf_guess,
                rhf_mom=rhf_mom,
                )


    @property
    def lot(self):
        return self.options['job_data']['lot']

    @lot.setter
    def lot(self,value):
        self.options['job_data']['lot']=value

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
            self.lot.update_qmmm(coords*units.ANGSTROM_TO_AU)
        if self.lot.__class__.__name__=="CASCI_LOT" or self.lot.__class__.__name__=="CASCI_LOT_SVD":
            return self.lot.casci.ref.geometry.qmmm.mm_energy
        else:
            return self.lot.rhf.geometry.qmmm.mm_energy

    def get_mm_gradient(self,coords):
        #TODO need diff variable for hasRan MM energy
        if self.hasRanForCurrentCoords==False or (coords != self.currentCoords).all():
            self.currentCoords = coords.copy()
            self.lot.update_qmmm(coords*units.ANGSTROM_TO_AU)
        if self.lot.__class__.__name__=="CASCI_LOT" or self.lot.__class__.__name__=="CASCI_LOT_SVD":
            return self.lot.casci.ref.geometry.qmmm.mm_gradient
        else:
            return self.lot.rhf.geometry.qmmm.mm_gradient

    def run_code(self,T):
        self.lot = self.lot.update_xyz(T)
        for state in self.states:
            multiplicity=state[0]
            ad_idx=state[1]
            S=multiplicity-1
            if self.lot.__class__.__name__=="CASCI_LOT" or self.lot.__class__.__name__=="CASCI_LOT_SVD":
                self.E.append((multiplicity,self.lot.compute_energy(S=S,index=ad_idx)))
                tmp = self.lot.compute_gradient(S=S,index=ad_idx)
            elif self.lot.__class__.__name__=="RHF_LOT": 
                self.E.append((multiplicity,self.lot.compute_energy()))
                tmp = self.lot.compute_gradient()
            self.grada.append((multiplicity,tmp[...]))
        if self.do_coupling==True:
            state1=self.states[0][1]
            state2=self.states[1][1]
            tmp = self.lot.compute_coupling(S=S,indexA=state1,indexB=state2)
            self.coup = tmp[...]

    def run(self,geom,verbose=False):
        self.E=[]
        self.grada=[]
        #normal update
        coords = manage_xyz.xyz_to_np(geom)
        T = ls.Tensor.array(coords*units.ANGSTROM_TO_AU)

        if not verbose:
            with open('lot_jobs.txt','a') as out:
                with nifty.custom_redirection(out):
                    self.run_code(T)
                filename="{}.molden".format(self.node_id)
                self.lot.casci.reference.save_molden_file(filename)
        else:
            self.run_code(T)
                #filename="{}_rhf_update.molden".format(self.node_id)
                #self.lot.casci.reference.save_molden_file(filename)

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
