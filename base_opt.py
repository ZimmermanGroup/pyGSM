import options
import numpy as np
import os
import openbabel as ob
import pybel as pb
from dlc import *
from copy import deepcopy
import StringIO
from _print_opt import *

global DQMAG_SSM_SCALE
DQMAG_SSM_SCALE=1.5
global DQMAG_SSM_MAX
DQMAG_SSM_MAX=0.8
global DQMAG_SSM_MIN
DQMAG_SSM_MIN=0.2

class Base_Method(object,Print):
    
    @staticmethod
    def default_options():
        if hasattr(Base_Method, '_default_options'): return Base_Method._default_options.copy()

        opt = options.Options() 
        
        opt.add_option(
            key='ICoord1',
            required=True,
            allowed_types=[DLC],
            doc='')

        opt.add_option(
            key='ICoord2',
            required=False,
            allowed_types=[DLC],
            doc='')

        opt.add_option(
            key='nnodes',
            required=False,
            value=1,
            allowed_types=[int],
            doc='number of string nodes')
        
        opt.add_option(
            key='isSSM',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify SSM or DSM')

        opt.add_option(
            key='isomers',
            required=False,
            value=[],
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify atoms\
                 indexed at 1')

        opt.add_option(
            key='isMAP_SE',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify isMAP_SE')

        opt.add_option(
            key='nconstraints',
            required=False,
            value=0,
            allowed_types=[int])

        opt.add_option(
            key='CONV_TOL',
            value=0.001,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
            key='ADD_NODE_TOL',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')


        Base_Method._default_options = opt
        return Base_Method._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return Base_Method(Base_Method.default_options().set_values(kwargs))

#    def restart_string(self,xyzbase='restart'):#,nR,nP):
#        with open(xyzfile) as xyzcoords:
#            xyzlines = xyzcoords.readlines()
#        
#    def write_restart(self,xyzbase='restart'):
#        rxyzfile = os.getcwd()+"/"+xyzbase+'_r.xyz'
#        pxyzfile = os.getcwd()+'/'+xyzbase+'_p.xyz'
#        rxyz = pb.Outputfile('xyz',rxyzfile,overwrite=True)
#        pxyz = pb.Outputfile('xyz',pxyzfile,overwrite=True)
#        obconversion = ob.OBConversion()
#        obconversion.SetOutFormat('xyz')
#        r_mols = []
#        for i in range(self.nR):
#            r_mols.append(obconversion.WriteString(self.icoords[i]
        

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes

        #TODO What is optCG Ask Paul
        self.optCG = False
        self.isTSnode =False
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        if self.nnodes>1:
            tmp = self.options['ICoord2']
            self.icoords[0] = DLC.union_ic(self.icoords[0],tmp)
            print "after union"
            lot1 = tmp.PES.lot.copy(
                    tmp.PES.lot, 
                    self.nnodes-1)
            PES1 = PES(tmp.PES.options.copy().set_values({
                "lot": lot1,
                }))
            self.icoords[-1] = DLC(self.icoords[0].options.copy().set_values(dict(
                mol= tmp.mol,
                PES=PES1,
                ))
                )
        
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.isSSM = self.options['isSSM']
        self.isMAP_SE = self.options['isMAP_SE']
        self.active = [False] * self.nnodes
        self.active[0] = False
        self.active[-1] = False
        self.isomers = self.options['isomers']
        #self.isomer_init()
        self.nconstraints = self.options['nconstraints']
        self.CONV_TOL = self.options['CONV_TOL']
        self.ADD_NODE_TOL = self.options['ADD_NODE_TOL']

        self.energies = np.asarray([-1e8]*self.nnodes)
        self.emax = float(max(self.energies))
        self.nmax = 0
        self.climb = False
        self.find = False

        self.rn3m6 = np.sqrt(3.*self.icoords[0].natoms-6.);
        self.gaddmax = self.ADD_NODE_TOL/self.rn3m6;
        print " gaddmax:",self.gaddmax

    def store_energies(self):
        for i,ico in enumerate(self.icoords):
            if ico != 0:
                self.energies[i] = ico.energy

    def optimize(self,n=0,nsteps=100,nconstraints=0):
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules=[]
        #opt_molecules.append(obconversion.WriteString(self.icoords[n].mol.OBMol))
        self.icoords[n].V0 = self.icoords[n].PES.get_energy(self.icoords[n].geom)
        self.icoords[n].energy=0
        grmss = []
        steps = []
        energies=[]
        Es =[]
        self.icoords[n].do_bfgs=False # gets reset after each step
        self.icoords[n].buf = StringIO.StringIO()
        self.icoords[n].bmatp = self.icoords[n].bmatp_create()
        self.icoords[n].bmatp_to_U()
        self.icoords[n].bmat_create()
        # set node id
        self.icoords[n].node_id = n
    
        print "Initial energy is %1.4f\n" % self.icoords[n].V0
        self.icoords[n].buf.write("\n Writing convergence:")
    
        for step in range(nsteps):
            if self.icoords[n].print_level>0:
                print("\nOpt step: %i" %(step+1)),
            self.icoords[n].buf.write("\nOpt step: %d" %(step+1))
   
            # => update DLCs <= #
            self.icoords[n].bmatp = self.icoords[n].bmatp_create()
            self.icoords[n].bmatp_to_U()
            self.icoords[n].bmat_create()
            if self.icoords[n].PES.lot.do_coupling is False:
                if nconstraints>0:
                    constraints=self.ictan[n]
            else:
                if nconstraints==2:
                    dvec = self.icoords[n].PES.get_coupling(self.icoords[n].geom)
                    dgrad = self.icoords[n].PES.get_dgrad(self.icoords[n].geom)
                    dvecq = self.icoords[n].grad_to_q(dvec)
                    dgradq = self.icoords[n].grad_to_q(dgrad)
                    dvecq_U = self.icoords[n].fromDLC_to_ICbasis(dvecq)
                    dgradq_U = self.icoords[n].fromDLC_to_ICbasis(dgradq)
                    constraints = np.zeros((len(dvecq_U),2),dtype=float)
                    constraints[:,0] = dvecq_U[:,0]
                    constraints[:,1] = dgradq_U[:,0]
                elif nconstraints==3:
                    raise NotImplemented

            if nconstraints>0:
                self.icoords[n].opt_constraint(constraints)
                self.icoords[n].bmat_create()
            #print self.icoords[n].bmatti
            self.icoords[n].Hint = self.icoords[n].Hintp_to_Hint()

            # => Opt step <= #
            if self.icoords[n].PES.lot.do_coupling is False:
                smag =self.icoords[n].opt_step(nconstraints)
            else:
                smag =self.icoords[n].combined_step(nconstraints)

            # convergence quantities
            grmss.append(float(self.icoords[n].gradrms))
            steps.append(smag)
            energies.append(self.icoords[n].energy-self.icoords[n].V0)
            opt_molecules.append(obconversion.WriteString(self.icoords[n].mol.OBMol))
    
            #write convergence
            self.write_node(n,opt_molecules,energies,grmss,steps)
    
            if self.icoords[n].gradrms<self.CONV_TOL:
                break
        print(self.icoords[n].buf.getvalue())
        print "Final energy is %2.5f" % (self.icoords[n].energy)
        return smag

if __name__ == '__main__':
    filepath="tests/stretched_fluoroethene.xyz"
    if False:
        from pytc import *
        nocc=11
        nactive=2
        lot1=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        #lot1.cas_from_file(filepath)
    if False:
        from pytc import *
        nocc=7
        nactive=2
        filepath="tests/twisted_ethene.xyz"
        lot1=PyTC.from_options(states=[(1,0),(1,1)],nocc=nocc,nactive=nactive,basis='6-31gs',do_coupling=True)
        lot1.cas_from_file(filepath)
    if False:
        from qchem import *
        lot1=QChem.from_options(states=[(1,0),(3,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    if False:
        from qchem import *
        lot1=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP')
    if False:
        from molpro import *
        filepath="tests/twisted_ethene.xyz"
        nocc=6
        nactive=4
        lot1=Molpro.from_options(states=[(1,0),(1,1)],charge=0,nocc=nocc,nactive=nactive,basis='6-31G*',do_coupling=True,nproc=4)


    from pes import *
    from penalty_pes import *
    from avg_pes import *
    from dlc import *

    if False:
        pes = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
        mol1=pb.readfile("xyz",filepath).next()
        #isOkay = mol1.OBMol.AddBond(6,4,1)
        ic1=DLC.from_options(mol=mol1,PES=pes)
        opt = Base_Method.from_options(ICoord1=ic1,CONV_TOL=0.0005)
        opt.optimize(0,50,0)
    if False:
        pes1 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
        pes2 = PES.from_options(lot=lot1,ad_idx=1,multiplicity=1)
        p = Avg_PES(pes1,pes2)
        mol1=pb.readfile("xyz",filepath).next()
        isOkay = mol1.OBMol.AddBond(6,4,1)
        ic1=DLC.from_options(mol=mol1,PES=p,print_level=1)
        opt = Base_Method.from_options(ICoord1=ic1,CONV_TOL=0.0005)
        opt.optimize(0,50,2)
    if True:
        from qchem import *
        lot = QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=8)
        mol=pb.readfile('xyz','tests/SiH2H2.xyz').next()
        pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
        ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
        opt = Base_Method.from_options(ICoord1=ic1,CONV_TOL=0.001)
        opt.optimize(0,50,0)
    #pes2 = PES.from_options(lot=lot1,ad_idx=0,multiplicity=3)
    #penalty_pes = Penalty_PES(pes,pes2)
    #mol1=pb.readfile("xyz",filepath).next()
    #ic1=DLC.from_options(mol=mol1,PES=penalty_pes)
    #opt = Base_Method.from_options(ICoord1=ic1)
    #opt.optimize(0,50,0)
    #ic1.draw()
