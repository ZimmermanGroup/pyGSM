import lightspeed as ls
import psiw
from base import * 
import numpy as np
import manage_xyz as mx


class PyTC(Base):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Base
    """

    def getEnergy(self):
        energy =0.
        average_over =0
        print(" in getEnergy")
        for i in self.calc_states:
            energy += self.lot.compute_energy(S=i[0],index=i[1])
            average_over+=1
        return energy/average_over

    def getGrad(self):
        average_over=0
        grad = np.zeros((self.molecule.natom,3))
        print(" in getGrad")
        print self.lot.casci.print_level
        for i in self.calc_states:
            tmp = self.lot.compute_gradient(S=i[0],index=i[1])
            grad += tmp[...] 
            average_over+=1
        final_grad = grad/average_over

        return np.reshape(final_grad,(3*self.molecule.natom,1))

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

    from obutils import *
    from pytc import *
    import manage_xyz
    
    filepath="tests/fluoroethene.xyz"
    nocc=23
    nactive=2
    geom=manage_xyz.read_xyz(filepath,scale=1)

    lot=PyTC.from_options(calc_states=[(0,0)],geom=geom,nocc=nocc,nactive=nactive,basis='6-31gs')
    lot.cas_from_geom()

