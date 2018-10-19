import lightspeed as ls
import psiw
from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *


class PyTC(Base):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Base
    """

    def compute_energy(self,S,index):
        T = ls.Tensor.array(self.coords*ANGSTROM_TO_AU)
        self.lot = self.lot.update_xyz(T)
        tmp = self.lot.compute_energy(S=S,index=index)
        return tmp*KCAL_MOL_PER_AU

    def compute_gradient(self,S,index):
        tmp=self.lot.compute_gradient(S=S,index=index)
        return tmp[...]*ANGSTROM_TO_AU

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PyTC(PyTC.default_options().set_values(kwargs))

    def dft_from_geom(
            self,
            ):
        raise NotImplementedError()
        return

    def cas_from_file(
            self,
            filepath,
            ):
  
        molecule = ls.Molecule.from_xyz_file(filepath)    
        resources = ls.ResourceList.build()

        nalpha = nbeta = self.nactive/2
        fomo_method = 'gaussian'
        fomo_temp = 0.3
        
        S_inds = [0]
        S_nstates = [2]

        geom1 = psiw.Geometry.build(
            resources=resources,
            molecule=molecule,
            basisname=self.basis,
            )
        
        ref1 = psiw.RHF.from_options(
            geometry=geom1,
            g_convergence=1.0E-6,
            fomo=True,
            fomo_method=fomo_method,
            fomo_temp=fomo_temp,
            fomo_nocc=self.nocc,
            fomo_nact=self.nactive,
            print_level=0,
            )
        ref1.compute_energy()
        
        casci1 = psiw.CASCI.from_options(
            reference=ref1,
            nocc=self.nocc,
            nact=self.nactive,
            nalpha=nalpha,
            nbeta=nbeta,
            S_inds=S_inds,
            S_nstates=S_nstates,
            print_level=0,
            )
        casci1.compute_energy()

        self.lot = psiw.CASCI_LOT.from_options(
            casci=casci1,
            print_level=0,
            rhf_guess=True,
            rhf_mom=True,
            )


if __name__ == '__main__':

    from pytc import *
    import manage_xyz
    nocc=11
    nactive=2

    lot=PyTC.from_options(calc_states=[(0,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
    x="tests/fluoroethene.xyz"
    lot.cas_from_file(x)

    #e=lot.getEnergy()
    #print e
    #g=lot.getGrad()
    #print g

