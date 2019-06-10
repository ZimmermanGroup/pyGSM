# standard library imports
import sys
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from .pes import PES
from utilities import *

class Avg_PES(PES):
    """ Avg potential energy surface calculators """

    #TODO can fix this up so it automatically initializes PES1 and PES2?
    def __init__(self,
            PES1,
            PES2,
            lot,
            ):
        self.options = PES1.options
        self.PES1 = PES(PES1.options.copy().set_values({
            "lot": lot,
            }))
        self.PES2 = PES(PES2.options.copy().set_values({
            "lot": lot,
            }))
        self._dE=1000.
        self.lot = lot

    @classmethod
    def create_pes_from(cls,PES,options={}):
        lot = type(PES.lot).copy(PES.lot,options)
        return cls(PES.PES1,PES.PES2,lot)

    @property
    def dE(self):
        return self._dE

    @dE.setter
    def dE(self,value):
        self._dE = value

    def get_energy(self,xyz):
        if self.PES1.multiplicity==self.PES2.multiplicity:
            assert self.PES2.ad_idx>self.PES1.ad_idx,"dgrad wrong direction"
        self.dE = self.PES2.get_energy(xyz) - self.PES1.get_energy(xyz)
        return 0.5*(self.PES1.get_energy(xyz) + self.PES2.get_energy(xyz))

    def get_gradient(self,xyz):
        return 0.5*(self.PES1.get_gradient(xyz) + self.PES2.get_gradient(xyz))

    def get_coupling(self,xyz):
        assert self.PES1.multiplicity==self.PES2.multiplicity,"coupling is 0"
        assert self.PES1.ad_idx!=self.PES2.ad_idx,"coupling is 0"
        return self.lot.get_coupling(xyz,self.PES1.multiplicity,self.PES1.ad_idx,self.PES2.ad_idx)

    def get_dgrad(self,xyz):
        if self.PES1.multiplicity==self.PES2.multiplicity:
            assert self.PES2.ad_idx>self.PES1.ad_idx,"dgrad wrong direction"
        return (self.PES2.get_gradient(xyz) - self.PES1.get_gradient(xyz))

    def get_average_gradient(self,xyz):
        if self.PES1.multiplicity==self.PES2.multiplicity:
            assert self.PES2.ad_idx>self.PES1.ad_idx,"dgrad wrong direction"
        return 0.5*(self.PES2.get_gradient(xyz) + self.PES1.get_gradient(xyz))

    def critical_points_bp(self,xyz,radius=0.2,num_slices=40):
        def get_beta(g,h):
            gdoth = np.dot(g.T,h)
            print(" g*h = %1.4f" % gdoth)
            dotg = np.dot(g.T,g)
            doth = np.dot(h.T,h)
            arg = (2*gdoth)/(dotg - doth)
            return 0.5*np.arctan(arg)
        def calc_pitch(g,h):
            dotg = np.dot(g.T,g)
            doth = np.dot(h.T,h)
            return np.sqrt(0.5*(dotg+doth))
        def calc_asymmetry(g,h):
            dotg = np.dot(g.T,g)
            doth = np.dot(h.T,h)
            return (dotg - doth)/(dotg + doth)

        # get vectors
        energy = self.get_energy(xyz)
        dgrad = self.get_dgrad(xyz)
        dvec = self.get_coupling(xyz)
        sab = self.get_average_gradient(xyz)
        print(" dE is %5.4f" % self.dE)

        # non-adiabatic coupling is derivative coupling times dE
        nac = dvec*self.dE
        beta = get_beta(dgrad,nac)
        print(" beta = %1.6f" % beta)
        
        # rotate dvec and dgrad to be orthonormal
        nac = nac*np.cos(beta) - dgrad*np.sin(beta)
        dgrad = dgrad*np.cos(beta) - nac*np.sin(beta)

        norm_nac = np.linalg.norm(nac)
        norm_dg = np.linalg.norm(dgrad)
        print(" norm x %2.5f" % norm_dg)
        print(" norm y %2.5f" % norm_nac)

        pitch = calc_pitch(dgrad,nac)
        print(" pitch %1.6f" % pitch)
        asymmetry = calc_asymmetry(dgrad,nac)
        if asymmetry<0.:
            # swap dvec and dgrad
            dgrad_copy = dgrad.copy()
            dgrad = nac.copy()
            nac = dgrad_copy

        # discretize branching plane and find critical points
        #theta = np.linspace(0,2*np.pi,num_slices)
        theta = [ i*2*np.pi/num_slices for i in xrange(num_slices)]
       
        dotx = np.dot(sab.T,dgrad)/norm_dg
        doty = np.dot(sab.T,nac)/norm_nac
        sab_x = dotx/pitch
        sab_y = doty/pitch

        # flip dgrad,dvec
        if sab_x < 0. and sab_y > 0.:
            dgrad = -dgrad
            dotx = np.dot(sab.T,dgrad)/norm_dg
            sab_x = dotx/pitch
        elif sab_x > 0. and sab_y < 0.:
            dvec = -dvec
            doty = np.dot(sab.T,nac)/norm_nac
            sab_y = doty/pitch

        sigma = np.sqrt(sab_x*sab_x + sab_y*sab_y)
        theta_s = np.arctan(sab_y/sab_x)
        print(" asymmetry %1.5f, sigma %1.5f theta_s %1.5f sx %1.2f sy %1.2f" %(asymmetry,sigma,theta_s,sab_x,sab_y))

        EA=[]
        EB=[]
        for n in xrange(num_slices):
            factorA = sigma*np.cos(theta[n] - theta_s) + np.sqrt(1.+asymmetry*np.cos(2*theta[n]))
            factorB = sigma*np.cos(theta[n] - theta_s) - np.sqrt(1.+asymmetry*np.cos(2*theta[n]))
            EA.append( float(pitch*radius*factorA)) #energy +
            EB.append( float(pitch*radius*factorB)) # energy +
       
        #EA = np.asarray(EA)
        #EB = np.asarray(EB)
        for n in xrange(num_slices):
            print(" EA[%2i] = %2.8f" %(n,EA[n]))
        for n in xrange(num_slices):
            print(" EB[%2i] = %2.8f" %(n,EB[n]))

        #plot(EA,theta)

        # determine minima
        print(EB)
        theta_min = []
        for i in range(1,num_slices-1):
            if EB[i] < EB[i-1] and EB[i+1]>EB[i]:
                theta_min.append(i)
        # check endpoints
        if EB[0] < EB[-1] and EB[1] > EB[0]: 
            print("Min at 0")
            theta_min.append(0)
        if EB[-1]<EB[-2] and EB[0]>EB[-1]:
            print("Min at -1")
            theta_min.append(num_slices-1)

        # determine maxima
        theta_max=[]
        for i in range(1,num_slices-1):
            if EB[i] > EB[i-1] and EB[i+1]<EB[i]:
                theta_max.append(i)
        # check endpoints
        if EB[0] > EB[-1] and EB[1] < EB[0]: 
            print(" Max at 0")
            theta_max.append(0)
        if EB[-1]> EB[-2] and EB[0]<EB[-1]:
            print(" Max at 39")
            theta_max.append(num_slices-1)
     
        print("mins")
        print(theta_min)
        print("max")
        print(theta_max)
        # geometries of max and min
        mxyz = []
        theta_list = theta_min+theta_max
        dgrad = np.reshape(dgrad,(-1,3))
        nac = np.reshape(nac,(-1,3))
        for n in theta_list:
            mxyz.append(xyz + radius*np.cos(theta[n])*dgrad/norm_dg + radius*np.cos(theta[n])*nac/norm_nac)
        
        return mxyz

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
    pes = Avg_PES(PES1=pes1,PES2=pes2,lot=lot)
    geom=manage_xyz.read_xyz(filepath,scale=1)   
    coords= manage_xyz.xyz_to_np(geom)
    print(pes.get_energy(coords))
    print(pes.get_gradient(coords))

