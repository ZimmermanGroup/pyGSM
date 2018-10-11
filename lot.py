import lightspeed as ls
import psiw
from pes import * 


class LOT(Base):
    """
    Level of theory is a wrapper for the Psiwinder and lightspeed EST to do DFT and CASCI calculations 
    Inherits from Base
    """

    def getEnergy(self):
        return self.lot.compute_energy(S=self.wspin,index=self.wstate)

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return LOT(LOT.default_options().set_values(kwargs))

    def cas_from_geom(
        self,
        ):

        molecule1 = ls.Molecule.from_xyz_file(self.filepath)      
        self.resources = ls.ResourceList.build()

        nalpha = nbeta = self.nactive/2
        fomo_method = 'gaussian'
        fomo_temp = 0.3
        
        S_inds = [0]
        S_nstates = [2]

        geom1 = psiw.Geometry.build(
            resources=self.resources,
            molecule=molecule1,
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
            )
        casci1.compute_energy()

        self.lot = psiw.CASCI_LOT.from_options(
            casci=casci1,
            print_level=0,
            rhf_guess=True,
            rhf_mom=True,
            )


