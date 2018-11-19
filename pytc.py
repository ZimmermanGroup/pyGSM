import lightspeed as ls
import psiw
from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *


class PyTC(Lot):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Lot
    """

    def get_energy(self,geom,multiplicity,state):
        #normal update
        coords = mx.xyz_to_np(geom)
        T = ls.Tensor.array(coords*ANGSTROM_TO_AU)
        self.lot = self.lot.update_xyz(T)
        # from template
        #geom = self.lot.casci.geometry.update_xyz(T)
        #self.casci_from_template(geom,self.nocc)
        S=multiplicity-1
        tmp = self.lot.compute_energy(S=S,index=state)
        return tmp*KCAL_MOL_PER_AU

    def get_gradient(self,geom,multiplicity,state):
        S=multiplicity-1
        tmp=self.lot.compute_gradient(S=S,index=state)
        #return np.reshape(tmp[...],(3*len(tmp[...]),1))*ANGSTROM_TO_AU
        return tmp[...]*ANGSTROM_TO_AU

    def get_coupling(self,geom,multiplicity,state1,state2):
        S=multiplicity-1
        tmp = self.lot.compute_coupling(S=S,indexA=state1,indexB=state2)
        return np.reshape(tmp[...],(3*len(tmp[...]),1))*ANGSTROM_TO_AU

    @staticmethod
    def copy(PyTCA,node_id):
        """ create a copy of this lot object"""
        obj = PyTC(PyTCA.options.copy().set_values({
            "node_id" :node_id,
            }))
        obj.lot = PyTCA.lot
        obj.casci1 = PyTCA.casci1
        return obj

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

        # => Setup RHF of  the new molecule <= #
        ref2 = psiw.RHF(self.casci1.reference.options.copy().set_values(dict(
            geometry=geometry2,
            fomo_nocc=fomo_nocc2,
            )))
        ref2.initialize()
    
        Cact1 = self.casci1.reference.tensors['Cact']
        #print Cact1
    
        pairlist12 = ls.PairList.build_schwarz(
            self.casci1.reference.basis,
            ref2.basis,
            False,
            self.casci1.reference.pairlist.thre)
        S12 = ls.IntBox.overlap(
            self.casci1.resources,
            pairlist12)
        X2 = ref2.tensors['X']
    
        M = ls.Tensor.chain([Cact1, S12, X2], [True, False, False])
        
        U, s, V = ls.Tensor.svd(M, full_matrices=False)
        #print s
    
        Cact_mom2 = ls.Tensor.chain([X2, V], [False, True])
        
        O = ls.Tensor.chain([Cact1, S12, Cact_mom2], [True, False, False])
        #print O
        
        # => Calculate RHF of  the new molecule <= #
        ref2.compute_energy(Cact_mom=Cact_mom2)
        ref2.save_molden_file('rhf2.molden')
   

        casci2 = psiw.CASCI(self.casci1.options.copy().set_values(dict(
            reference=ref2,
            nocc=fomo_nocc2,
            grad_thre_dp = 1.0E-8,
            )))
    
        casci2.compute_energy()

        self.lot = psiw.CASCI_LOT.from_options(
            casci=casci2,
            print_level=0,
        		)


    def casci_from_file_from_template(
            self,
            filepath1,
            filepath2,
            nocc1,
            nocc2,
            ):
        """ creates casci1 object and geom2 and then calls casci_from_template"""

        molecule1 = ls.Molecule.from_xyz_file(filepath1)    
        molecule2 = ls.Molecule.from_xyz_file(filepath2)
        # stash coords
        geom=manage_xyz.read_xyz(filepath2,scale=1)
        self.coords=manage_xyz.xyz_to_np(geom)
        resources = ls.ResourceList.build()

        nalpha = nbeta = self.nactive/2
        fomo_method = 'gaussian'
        fomo_temp = 0.3
        
        singlets=self.search_tuple(self.states,1)
        len_singlets = len(singlets)
        triplets=self.search_tuple(self.states,3)
        len_triplets = len(triplets)
        singlet_inds = [i for i in range(len(singlets))]
        triplet_inds = [i for i in range(len(triplets))]
        singlet_states = [len_singlets]
        triplet_states = [len_triplets]
        S_inds = singlet_inds+triplet_inds
        S_nstates = singlet_states+triplet_states

        geom1 = psiw.Geometry.build(
            resources=resources,
            molecule=molecule1,
            basisname=self.basis,
            )
        
        ref1 = psiw.RHF.from_options(
            geometry=geom1,
            g_convergence=1.0E-6,
            fomo=True,
            fomo_method=fomo_method,
            fomo_temp=fomo_temp,
            fomo_nocc=nocc1,
            fomo_nact=self.nactive,
            print_level=0,
            )
        ref1.compute_energy()

        # => save casci1 to memory <= #
        self.casci1 = psiw.CASCI.from_options(
            reference=ref1,
            nocc=nocc1,
            nact=self.nactive,
            nalpha=nalpha,
            nbeta=nbeta,
            S_inds=S_inds,
            S_nstates=S_nstates,
            print_level=0,
            )

        self.casci1.compute_energy()
        print "saving casci1 to memory"

        geom2 = psiw.Geometry.build(
            resources=resources,
            molecule=molecule2,
            basisname=self.basis,
            )

        self.casci_from_template(geom2,nocc2)


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
        
        singlets=self.search_tuple(self.states,1)
        len_singlets = len(singlets)
        triplets=self.search_tuple(self.states,3)
        len_triplets = len(triplets)
        singlet_inds =  [len_singlets]
        triplet_inds =  [len_triplets]
        S_nstates = singlet_inds+triplet_inds
        S_inds=[]
        if len_singlets>0:
            S_inds.append(0)
        if len_triplets>0:
            S_inds.append(2)
       
        print S_nstates
        print S_inds
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
        
        self.casci1 = psiw.CASCI.from_options(
            reference=ref1,
            nocc=self.nocc,
            nact=self.nactive,
            nalpha=nalpha,
            nbeta=nbeta,
            S_inds=S_inds,
            S_nstates=S_nstates,
            print_level=0,
            #g_convergence=1.0E-6, #work?
            grad_thre_dp = 1.0E-8,
            )

        self.casci1.compute_energy()
        self.lot = psiw.CASCI_LOT.from_options(
            casci=self.casci1,
            print_level=0,
            rhf_guess=True,
            rhf_mom=True,
            )


if __name__ == '__main__':

    from pytc import *
    import manage_xyz

    if 1:
        nocc=11
        nactive=2

        lot=PyTC.from_options(states=[(1,0),(3,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        x="tests/fluoroethene.xyz"
        #lot.cas_from_file(x)
        lot.casci_from_file_from_template(x,x,nocc,nocc) # hack to get it to work,need casci1

        geom=manage_xyz.read_xyz(x,scale=1)   
        e=lot.get_energy(geom,1,0)
        print e
        g=lot.get_gradient(geom,1,0)
        print g

    # from reference
    #filepath1="tests/pent-4-enylbenzene.xyz"
    #nocc1=37
    #nactive=6
    #filepath2="tests/pent-4-enylbenzene_pos1_11DICHLOROETHANE.xyz"
    #nocc2=61
    #nactive=6
    #lot1=PyTC.from_options(states=[(0,0)],nocc=nocc2,nactive=nactive,basis='6-31gs')
    #lot1.casci_from_file_from_template(filepath1,filepath2,nocc1,nocc2)

    #e=lot1.getEnergy()
    #print e
    #g=lot1.getGrad()
    #print g
