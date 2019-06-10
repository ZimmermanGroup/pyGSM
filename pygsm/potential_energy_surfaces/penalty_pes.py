# standard library imports
import sys
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .pes import PES
from utilities import *

class Penalty_PES(PES):
    """ penalty potential energy surface calculators """

    def __init__(self,
            PES1,
            PES2,
            lot,
            sigma=1.0,
            alpha=0.02*units.KCAL_MOL_PER_AU,
            ):
        self.PES1 = PES(PES1.options.copy().set_values({
            "lot": lot,
            }))
        self.PES2 = PES(PES2.options.copy().set_values({
            "lot": lot,
            }))
        self.lot = lot
        self.alpha = alpha
        self.dE = 1000.
        self.sigma = sigma
        print(' PES1 multiplicity: {} PES2 multiplicity: {}'.format(self.PES1.multiplicity,self.PES2.multiplicity))

    @classmethod
    def create_pes_from(cls,PES,options={}):
        lot = type(PES.lot).copy(PES.lot,options)
        return cls(PES.PES1,PES.PES2,lot,PES.sigma,PES.alpha)

    def get_energy(self,geom):
        E1 = self.PES1.get_energy(geom)
        E2 = self.PES2.get_energy(geom)
        #avgE = 0.5*(self.PES1.get_energy(geom) + self.PES2.get_energy(geom))
        avgE = 0.5*(E1+E2)
        #self.dE = self.PES2.get_energy(geom) - self.PES1.get_energy(geom)
        self.dE = E2-E1
        #print "E1: %1.4f E2: %1.4f"%(E1,E2),
        #print "delta E = %1.4f" %self.dE,
        #TODO what to do if PES2 is or goes lower than PES1?
        G = (self.dE*self.dE)/(abs(self.dE) + self.alpha)
        #if self.dE < 0:
        #    G*=-1
        #print "G = %1.4f" % G
        #print "alpha: %1.4f sigma: %1.4f"%(self.alpha,self.sigma),
        #print "F: %1.4f"%(avgE+self.sigma*G)
        sys.stdout.flush()
        return avgE+self.sigma*G

    def get_gradient(self,geom):
        self.grad1 = self.PES1.get_gradient(geom)
        self.grad2 = self.PES2.get_gradient(geom)
        avg_grad = 0.5*(self.grad1 + self.grad2)
        dgrad = self.grad2 - self.grad1
        if self.dE < 0:
            dgrad *= -1
        factor = self.sigma*((self.dE*self.dE) + 2.*self.alpha*abs(self.dE))/((abs(self.dE) + self.alpha)**2)
        #print "factor is %1.4f" % factor
        return avg_grad + factor*dgrad

if __name__ == '__main__':

    from level_of_theories.pytc import PyTC 
    import psiw
    import lightspeed as ls

    filepath='../../data/ethylene.xyz'
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    ##### => Job Data <= #####
    states = [(1,0),(1,1)]
    charge=0
    nocc=7
    nactive=2
    basis='6-31gs'

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

    pes1 = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot,ad_idx=1,multiplicity=1)
    pes = Penalty_PES(PES1=pes1,PES2=pes2,lot=lot)
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    coords= manage_xyz.xyz_to_np(geom)
    print(pes.get_energy(coords))
    print(pes.get_gradient(coords))

