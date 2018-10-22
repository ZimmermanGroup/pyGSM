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
        #self.lot = self.lot.update_xyz(T)
        geom = self.lot.casci.geometry.update_xyz(T)
        self.casci_from_template(geom,self.nocc)
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

    def casci_from_template(
            self,
            geometry2,
            fomo_nocc2,
            ):
    
        # => Setup RHF of  the new molecule? <= #
        ref2 = psiw.RHF(self.lot.casci.reference.options.copy().set_values(dict(
            geometry=geometry2,
            fomo_nocc=fomo_nocc2,
            )))
        ref2.initialize()
    
        Cact1 = self.lot.casci.reference.tensors['Cact']
        #print Cact1
    
        pairlist12 = ls.PairList.build_schwarz(
            self.lot.casci.reference.basis,
            ref2.basis,
            False,
            self.lot.casci.reference.pairlist.thre)
        S12 = ls.IntBox.overlap(
            self.lot.casci.resources,
            pairlist12)
        X2 = ref2.tensors['X']
    
        M = ls.Tensor.chain([Cact1, S12, X2], [True, False, False])
        
        U, s, V = ls.Tensor.svd(M, full_matrices=False)
        #print s
    
        Cact_mom2 = ls.Tensor.chain([X2, V], [False, True])
        
        O = ls.Tensor.chain([Cact1, S12, Cact_mom2], [True, False, False])
        #print O
        
        # => Calculate RHF of  the new molecule? <= #
        ref2.compute_energy(Cact_mom=Cact_mom2)
        ref2.save_molden_file('rhf2.molden')
    
        casci2 = psiw.CASCI(self.lot.casci.options.copy().set_values(dict(
            reference=ref2,
            nocc=fomo_nocc2,
            )))
    
        casci2.compute_energy()

        self.lot = psiw.CASCI_LOT.from_options(
        	  casci=casci2,
            print_level=0,
        		)


    def cas_from_file(
            self,
            filepath,
            ):
 
        geom=manage_xyz.read_xyz(filepath,scale=1)
        self.coords=manage_xyz.xyz_to_np(geom)
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

    lot=PyTC.from_options(E_states=[(0,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
    x="tests/fluoroethene.xyz"
    lot.cas_from_file(x)

    e=lot.getEnergy()
    print e
    g=lot.getGrad()
    print g

