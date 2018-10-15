import lightspeed as ls
import psiw
from base_lot import * 
import numpy as np
import manage_xyz as mx


class PyTC(Base):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Base
    """

    def compute_energy(self,S,index):
        return self.lot.compute_energy(S=S,index=index)

    def compute_gradient(self,S,index):
        tmp=self.lot.compute_gradient(S=S,index=index)
        return tmp[...]

    def update_xyz(self):
        raise NotImplementedError()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PyTC(PyTC.default_options().set_values(kwargs))

    def cas_from_geom(
        self,
        ):

        mx.write_xyz("ls.xyz",self.geom,0,1)
        self.molecule = ls.Molecule.from_xyz_file("ls.xyz")    
        self.resources = ls.ResourceList.build()

        nalpha = nbeta = self.nactive/2
        fomo_method = 'gaussian'
        fomo_temp = 0.3
        
        S_inds = [0]
        S_nstates = [2]

        geom1 = psiw.Geometry.build(
            resources=self.resources,
            molecule=self.molecule,
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
    
    filepath="tests/fluoroethene.xyz"
    nocc=23
    nactive=2
    #geom=manage_xyz.read_xyz(filepath,scale=1)

    #lot=PyTC.from_options(calc_states=[(0,0)],geom=geom,nocc=nocc,nactive=nactive,basis='6-31gs')
    lot=PyTC.from_options(calc_states=[(0,0)],filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
    lot.cas_from_geom()

    e=lot.getEnergy()
    print e
    g=lot.getGrad()
    print g

