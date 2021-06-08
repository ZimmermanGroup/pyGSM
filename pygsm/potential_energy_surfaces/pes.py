# standard library imports
import sys
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from coordinate_systems import rotate

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
                doc='Apply a constant force between atoms in units of AU, e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive',
                )

        opt.add_option(
                key='RESTRAINTS',
                value=None,
                required=False,
                doc='Translational harmonic constraint'
                )

        opt.add_option(
                key='mass',
                value=None,
                required=False,
                doc='Mass is sometimes required'
                )

        PES._default_options = opt
        return PES._default_options.copy()

    @classmethod
    def from_options(cls,**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    #TODO make kwargs
    @classmethod
    def create_pes_from(cls,PES,options={},copy_wavefunction=True):
        lot = type(PES.lot).copy(PES.lot,options,copy_wavefunction)

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
        self.RESTRAINTS = self.options['RESTRAINTS']
        self._dE=1000.
        #print ' PES object parameters:'
        #print ' Multiplicity:',self.multiplicity,'ad_idx:',self.ad_idx

    @property
    def dE(self):
        return self._dE

    @dE.setter
    def dE(self,value):
        self._dE = value

    @property
    def energy(self):
        return self.get_energy(self.lot.currentCoords)
    #def energy(self):
    #    if self.lot.Energies:
    #        # if E is property and a dictionary
    #        return self.lot.Energies[(self.multiplicity,self.ad_idx)].value
    #    else:
    #        return 0.

    def create_2dgrid(self,
            xyz,
            xvec,
            yvec,
            nx,
            ny,
            xmin=-1.,
            ymin=-1.,
            xmax=1.,
            ymax=1.
            ):
        xyz = xyz.flatten()
        # create the scalar values in the grid from 0 to mag
        x=np.linspace(xmin,xmax,nx)
        y=np.linspace(ymin,ymax,ny)
        xv,yv = np.meshgrid(x,y)
        # create the xyz coordinates and save as a tensor
        xyz_grid = np.zeros((xv.shape[0],xv.shape[1],xvec.shape[0]))
        rc=0
        for xrow,yrow in zip(xv,yv):
            cc=0
            for xx,yy in zip(xrow,yrow):
                idx = (rc,cc)
                xyz_grid[rc,cc,:] = xx*xvec + yy*yvec + xyz
                cc+=1
            rc+=1
        return xyz_grid,xv,yv


    #def fill_energy_grid2d(self,xyz_grid):
    def fill_energy_grid2d(self,
            xyz_grid
            ):

        assert xyz_grid.shape[-1] == len(self.lot.geom)*3, "xyz nneds to be 3*natoms long"
        assert xyz_grid.ndim == 3, " xyzgrid needs to be a tensor with 3 dimensions"

        energies = np.zeros((nx,ny))
        rc=0
        for mat in xyz_grid:
            cc=0
            for row in mat:
                xyz = np.reshape(row,(-1,3))
                energies[rc,cc] = self.lot.get_energy(xyz,self.multiplicity,self.ad_idx,runtype='energy')
                cc+=1
            rc+=1
         
        return energies

    def get_energy(self,xyz):
        fdE=0.
        if self.FORCE is not None:
            for i in self.FORCE:
                force=i[2]
                diff = (xyz[i[0]]- xyz[i[1]])
                d = np.linalg.norm(diff)*units.ANGSTROM_TO_AU  # AU
                fdE +=  force*d*units.KCAL_MOL_PER_AU   
                #print(" Force energy: {} kcal/mol".format(fdE))
        kdE=0.
        if self.RESTRAINTS is not None:
            for i in self.RESTRAINTS:
                a=i[0]
                force=i[1]   # In kcal/mol/Ang^2?
                kdE += 0.5*force*(xyz[a] - self.reference_xyz[a])**2
        return self.lot.get_energy(xyz,self.multiplicity,self.ad_idx) +fdE +kdE   # Kcal/mol


    def get_finite_difference_hessian(self,coords,qm_region=None):
        ''' Calculate Finite Differnce Hessian

        Params:
            coords ((natoms,3) np.ndarray - system coordinates  (x,y,z)
            qm_region list of QM atoms in a QMMM simulation to obtain environment perturbed Hessian with the size of the QM region

        Returns:
            Hessian (N1,N1) np.ndarray 

        '''
        if qm_region is None:
            hess = np.zeros((len(coords)*3,len(coords)*3))
            n1_region = [ x for x in range(3*len(coords)) ]
        else:
            n1_region =[]
            for n in qm_region:
                for j in range(3):
                    n1_region.append(n*3+j)
            print('n1_region')
            print(n1_region)
            #hess = np.zeros((len(coords[qm_region])*3,len(coords[qm_region])*3))
            hess = np.zeros((len(n1_region),len(n1_region)))
        print("hess shape", hess.shape)

        def gen_row(n):
            vec = np.zeros(coords.shape[0]*3)
            vec[n] = 1.
            return vec
       
        n_actual = 0
        for n in range(len(coords)*3):
            if n in n1_region:
                print("on hessian product ",n)
                row = gen_row(n)
                ans = self.get_finite_difference_hessian_product(coords,row)
                #hess[n_actual] = np.squeeze(self.get_finite_difference_hessian_product(coords,row))
                hess[n_actual] = np.squeeze(ans[n1_region])
                n_actual +=1
        return hess

    def get_finite_difference_hessian_product(self,coords,direction,FD_STEP_LENGTH=0.001):

        # format the direction
        direction = direction/np.linalg.norm(direction)
        direction = direction.reshape((len(coords),3))

        # fd step
        fdstep = direction*FD_STEP_LENGTH
        fwd_coords = coords+fdstep
        bwd_coords = coords-fdstep

        # calculate grad fwd and bwd in a.u. (Bohr/Ha)
        grad_fwd = self.get_gradient(fwd_coords)/units.ANGSTROM_TO_AU
        grad_bwd = self.get_gradient(bwd_coords)/units.ANGSTROM_TO_AU
    
        return (grad_fwd-grad_bwd)/(FD_STEP_LENGTH*2)

    @staticmethod
    def normal_modes(
            geom,       # Optimized geometry in au
            hess,       # Hessian matrix in au
            masses,     # Masses in au 
            ):
    
        """
        Params:
            geom ((natoms,4) np.ndarray) - atoms symbols and xyz coordinates
            hess ((natoms*3,natoms*3) np.ndarray) - molecule hessian
            masses ((natoms) np.ndarray) - masses
    
        Returns:
            w ((natoms*3 - 6) np.ndarray)  - normal frequencies
            Q ((natoms*3, natoms*3 - 6) np.ndarray)  - normal modes
    
        """
    
        # masses repeated 3x for each atom (unravels)
        m = np.ravel(np.outer(masses,[1.0]*3))
    
        # mass-weight hessian
        hess2 = hess / np.sqrt(np.outer(m,m))
    
        # Find normal modes (project translation/rotations before)
        #B = 3N,3N-6
        B = rotate.vibrational_basis(geom, masses)
        h, U3 = np.linalg.eigh(np.dot(B.T,np.dot(hess2,B)))
        # U3 = (3N-6,3N)(3N,3N)(3N-6,3N) = 3N-6,3N
        U = np.dot(B, U3)
        # U = (3N,3N-6),(3N-6,3N)
    
        ## TEST: Find normal modes (without projection translations/rotations)
        ## RMP: Matches TC output for PYP - same differences before/after projection
        #h2, U2 = np.linalg.eigh(hess2)
        #h3 = h2[:6]
        #for hval in h3:
        #    print(hval)

        #h2 = h2[6:]
        #U2 = U2[:,6:]
        #for hval, hval2 in zip(h,h2):
        #    #wval = np.sqrt(hval) / units['au_per_cminv']
        #    wval = np.sqrt(hval) * units.INV_CM_PER_AU
        #    #wval2 = np.sqrt(hval2) / units['au_per_cminv']
        #    wval2 = np.sqrt(hval2) *units.INV_CM_PER_AU
        #    print('%10.6E %10.6E %11.3E' % (wval, wval2, np.abs(wval - wval2)))
    
        # Normal frequencies
        w = np.sqrt(h)
        # Imaginary frequencies
        w[h < 0.0] = -np.sqrt(-h[h < 0.0]) 
    
        # Normal modes
        Q = U / np.outer(np.sqrt(m), np.ones((U.shape[1],)))
    
        return w, Q 
    
    def get_gradient(self,xyz,frozen_atoms=None):

        grad = self.lot.get_gradient(xyz,self.multiplicity,self.ad_idx,frozen_atoms=frozen_atoms)
        if self.FORCE is not None:
            for i in self.FORCE:
                atoms=[i[0],i[1]]
                force=i[2]
                diff = (xyz[i[0]]- xyz[i[1]])
                d = np.linalg.norm(diff)*units.ANGSTROM_TO_AU  # Bohr

                # Constant force  
                # grad=\nabla E + FORCE
                t = (force/d/2.)  # Hartree/bohr
                t*= units.ANGSTROM_TO_AU   # Ha/bohr * bohr/ang = Ha/ang
                sign=1
                for a in atoms:
                    grad[a] += sign*t*diff.T
                    sign*=-1
        if self.RESTRAINTS is not None:
            for i in self.RESTRAINTS:
                a= i[0]
                force=i[1]
                grad[a] += force*(xyz[a] - self.reference_xyz[a])

        grad = np.reshape(grad,(-1,1))
        return grad  #Ha/ang

    def check_input(self,geom):
        atoms = manage_xyz.get_atoms(self.geom)
        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]
        self.checked_input =True

if __name__ == '__main__':

    QCHEM=True
    PYTC=False
    if QCHEM:
        #from .qchem import QChem
        from level_of_theories.qchem import QChem
    elif PYTC:
        from level_of_theories.pytc import PyTC 
        import psiw
        import lightspeed as ls

    filepath='../../data/ethylene.xyz'
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    if QCHEM:
        lot=QChem.from_options(states=[(1,0),(1,1)],charge=0,basis='6-31g(d)',functional='B3LYP',fnm=filepath)
    elif PYTC:
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

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    coords= manage_xyz.xyz_to_np(geom)
    print(pes.get_energy(coords))
    print(pes.get_gradient(coords))

