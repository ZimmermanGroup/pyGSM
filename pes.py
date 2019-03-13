import options
import manage_xyz
import numpy as np
from units import *
import elements 
ELEMENT_TABLE = elements.ElementData()

class PES(object):
    """ PES object """

    @staticmethod
    def default_options():

        if hasattr(PES, '_default_options'): return PES._default_options.copy()
        opt = options.Options() 

        opt.add_option(
                key='lot',
                value=None,
                required=True,
                doc='Level of theory object')

        opt.add_option(
                key='ad_idx',
                value=0,
                required=True,
                doc='adiabatic index')

        opt.add_option(
                key='multiplicity',
                value=1,
                required=True,
                doc='multiplicity')

        opt.add_option(
                key="FORCE",
                value=None,
                required=False,
                doc='Apply a spring force between atoms in units of AU, e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive',
                )

        PES._default_options = opt
        return PES._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PES(PES.default_options().set_values(kwargs))

    @classmethod
    def create_pes_from(cls,PES,lot):
        return cls(PES.options.copy().set_values({
            "lot":lot,
            }))

    def __init__(self,
            options,
            ):
        """ Constructor """
        self.options = options

        self.lot = self.options['lot']
        self.ad_idx = self.options['ad_idx']
        self.multiplicity = self.options['multiplicity']
        self.FORCE = self.options['FORCE']
        self.dE=1000.
        #print ' PES object parameters:'
        #print ' Multiplicity:',self.multiplicity,'ad_idx:',self.ad_idx

    def get_energy(self,xyz):
        #if self.checked_input == False:
        #    self.check_input(geom)
        fdE=0.
        if self.FORCE is not None:
            for i in self.FORCE:
                atoms=[i[0],i[1]]
                force=i[2]
                diff = (xyz[i[0]]- xyz[i[1]])*ANGSTROM_TO_AU
                d = np.linalg.norm(diff)
                fdE +=  force*d*KCAL_MOL_PER_AU
        return self.lot.get_energy(xyz,self.multiplicity,self.ad_idx) +fdE

    #TODO this needs to be fixed
    def get_finite_difference_hessian(self,geom):
        hess = np.zeros((len(geom)*3,len(geom)*3))
        I = np.eye(hess.shape[0])
        for n,row in enumerate(I):
            print "on hessian product ",n
            hess[n] = np.squeeze(self.get_finite_difference_hessian_product(geom,row))
        return hess

    #TODO this needs to be fixed
    def get_finite_difference_hessian_product(self,geom,direction):
        FD_STEP_LENGTH=0.001
        direction = direction/np.linalg.norm(direction)
        direction = direction.reshape((len(geom),3))
        fdstep = direction*FD_STEP_LENGTH
        coords = manage_xyz.xyz_to_np(geom)
        fwd_coords = coords+fdstep
        bwd_coords = coords-fdstep

        fwd_geom = manage_xyz.np_to_xyz(geom,fwd_coords)
        self.lot.hasRanForCurrentCoords=False
        grad_fwd = self.get_gradient(fwd_geom)

        bwd_geom = manage_xyz.np_to_xyz(geom,bwd_coords)
        self.lot.hasRanForCurrentCoords=False
        grad_bwd = self.get_gradient(bwd_geom)
    
        return (grad_fwd-grad_bwd)/(FD_STEP_LENGTH*2)
    
    def get_gradient(self,xyz):
        tmp =self.lot.get_gradient(xyz,self.multiplicity,self.ad_idx)
        grad = np.reshape(tmp,(3*len(tmp),1))
        if self.FORCE is not None:
            for i in self.FORCE:
                atoms=[i[0],i[1]]
                force=i[2]
                diff = (xyz[i[0]]- xyz[i[1]])*ANGSTROM_TO_AU
                t = (force/d/2.)  # Hartree/Ang
                savegrad = np.copy(grad)
                sign=1
                for a in [3*(i-1) for i in atoms]:
                    grad[a:a+3] += sign*t*diff.T
                    sign*=-1
        return grad

    def check_input(self,geom):
        atoms = manage_xyz.get_atoms(self.geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.checked_input =True

if __name__ == '__main__':

    QCHEM=True
    PYTC=False
    if QCHEM:
        from qchem import *
    elif PYTC:
        from pytc import *

    #filepath="tests/fluoroethene.xyz"
    #nocc=11
    #nactive=2
    #geom=manage_xyz.read_xyz(filepath,scale=1)   
    #if QCHEM:
    #    lot=QChem.from_options(states=[(1,0),(3,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    #elif PYTC:
    #    lot=PyTC.from_options(states=[(1,0),(3,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
    #    lot.casci_from_file_from_template(x,x,nocc,nocc) # hack to get it to work,need casci1

    #pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    #print pes.get_energy(geom)
    #print pes.get_gradient(geom)

    filepath="firstnode.pdb"
    #geom=manage_xyz.read_xyz(filepath,scale=1)   

    #lot=QChem.from_options(states=[(2,0)],charge=1,basis='6-31g(d)',functional='B3LYP')
    lot = QChem.from_options(states=[(2,0)],lot_inp_file='qstart',nproc=4)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=2)
    print pes.get_energy(geom)
    print pes.get_gradient(geom)
    #ic = Hybrid_DLC(mol=mol,pes=pes,IC_region=["UNL"])
