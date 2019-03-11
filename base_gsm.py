import options
import numpy as np
import os
import openbabel as ob
import pybel as pb
from dlc import DLC
from pes import PES
from penalty_pes import Penalty_PES
from avg_pes import Avg_PES
from copy import deepcopy
import StringIO
from _print_opt import *
from _analyze_string import *
from units import *
import sys
import base_optimizer
import eigenvector_follow
from dlc_new import DelocalizedInternalCoordinates
from molecule import Molecule

class Base_Method(object,Print,Analyze):

    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))
    
    @staticmethod
    def default_options():
        if hasattr(Base_Method, '_default_options'): return Base_Method._default_options.copy()

        opt = options.Options() 
        
        opt.add_option(
            key='reactant',
            required=True,
            allowed_types=[Molecule],
            doc='Molecule object as the initial reactant structure')

        opt.add_option(
            key='product',
            required=False,
            allowed_types=[Molecule],
            doc='Molecule object for the product structure (not required for single-ended methods.')

        opt.add_option(
            key='nnodes',
            required=False,
            value=1,
            allowed_types=[int],
            #TODO I don't want nnodes to include the endpoints!
            doc="number of string nodes"
            )

        opt.add_option(
                key='optimizer',
                required=True,
                doc='Optimzer object  to use e.g. eigenvector_follow, conjugate_gradient,etc. \
                        most of the default options are okay for here since GSM will change them anyway',
                )

        opt.add_option(
            key='driving_coords',
            required=False,
            value=[],
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify atoms\
                 indexed at 1')

        opt.add_option(
            key='CONV_TOL',
            value=0.0005,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
            key='ADD_NODE_TOL',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold')

        opt.add_option(
                key="product_geom_fixed",
                value=True,
                required=False,
                doc="Fix last node?"
                )

        opt.add_option(
                key="growth_direction",
                value=0,
                required=False,
                doc="how to grow string,0=Normal,1=from reactant"
                )

        opt.add_option(
                key="DQMAG_MAX",
                value=0.8,
                required=False,
                doc="max step along tangent direction for SSM"
                )
        opt.add_option(
                key="DQMAG_MIN",
                value=0.2,
                required=False,
                doc=""
                )
    
        opt.add_option(
                key="BDIST_RATIO",
                value=0.5,
                required=False,
                doc="SE-Crossing uses this \
                        bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
                        to be considered grown.",
                        )
        Base_Method._default_options = opt
        return Base_Method._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return Base_Method(Base_Method.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        self.print_msg()

        # Cache attributes
        self.nnodes = self.options['nnodes']
        self.nodes = [None]*self.nnodes
        self.nodes[0] = self.options['reactant']
        self.nodes[-1] = self.options['product']
        self.driving_coords = self.options['driving_coords']
        self.product_geom_fixed = self.options['product_geom_fixed']
        self.growth_direction=self.options['growth_direction']
        self.isRestarted=False
        self.DQMAG_MAX=self.options['DQMAG_MAX']
        self.DQMAG_MIN=self.options['DQMAG_MIN']
        self.BDIST_RATIO=self.options['BDIST_RATIO']
        self.optimizer = options['optimizer']

        # Set initial values
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.energies = np.asarray([0.]*self.nnodes)
        self.emax = 0.0
        self.TSnode = 0 
        self.climb = False 
        self.find = False  
        self.n0 = 0 # something to do with added nodes? "first node along current block"
        self.end_early=False
        self.tscontinue=True # whether to continue with TS opt or not
        self.rn3m6 = np.sqrt(3.*self.nodes[0].natoms-6.);
        self.gaddmax = self.options['ADD_NODE_TOL']/self.rn3m6;
        print " gaddmax:",self.gaddmax
        self.ictan = [[]]*self.nnodes
        self.active = [False] * self.nnodes
        self.climber=False  #is this string a climber?
        self.finder=False   # is this string a finder?
        self.done_growing = False

        #self.nodes[0].Hessian = self.DLC.Prims.guess_hessian(self.nodes[0].xyz)

    def restart_string(self,xyzbase='restart'):#,nR,nP):
        self.growth_direction=0
        xyzfile=xyzbase+".xyz"
        with open(xyzfile) as f:
            nlines = sum(1 for _ in f)
        print "number of lines is ", nlines
        with open(xyzfile) as f:
            natoms = int(f.readlines()[2])

        print "number of atoms is ",natoms
        nstructs = (nlines-5)/ (natoms+4) #this is for two blocks after GEOCON
        
        print "number of structures in restart file is %i" % nstructs
        coords=[]
        energy = []
        grmss = []
        atomic_symbols=[]
        with open(xyzfile) as f:
            f.readline()
            f.readline() #header lines
            for struct in range(nstructs):
                tmpcoords=np.zeros((natoms,3))
                f.readline() #natoms
                f.readline() #space
                for a in range(natoms):
                    line=f.readline()
                    tmp = line.split()
                    #coords.append([float(i) for i in tmp[1:]])
                    tmpcoords[a,:] = [float(i) for i in tmp[1:]]
                    if struct==0:
                        atomic_symbols.append(tmp[0])
                coords.append(tmpcoords)
            
            # Get energies
            f.readline() # line
            f.readline() #energy
            for struct in range(nstructs):
                energy.append(float(f.readline()))
            f.readline() # max-force
            for struct in range(nstructs):
                grmss.append(float(f.readline()))

        print "copying node and setting values"
        mol = DLC.make_mol_from_coords(coords[-1],atomic_symbols)
        tmp=DLC(self.nodes[0].options.copy().set_values({
            'mol':mol,
            }))

        print "doing union"
        self.nodes[0] = DLC.union_ic(self.nodes[0],tmp)
        self.newic = DLC.copy_node(self.nodes[0],0,-1)
        self.nodes[0].energy = self.nodes[0].V0 = self.nodes[0].PES.get_energy(self.nodes[0].geom)
        self.nodes[0].gradrms=grmss[0]

        print "initial energy is %3.4f" % self.nodes[0].energy
        for struct in range(1,nstructs):
            if struct==nstructs-1:
                self.nodes[struct] = DLC.copy_node(self.nodes[struct-1],struct,0)
            else:
                self.nodes[struct] = DLC.copy_node(self.nodes[struct-1],struct,1)
            self.nodes[struct].set_xyz(coords[struct])
            self.nodes[struct].update_ics()
            self.nodes[struct].setup()
            self.nodes[struct].Hintp = self.nodes[struct].make_Hint()
            self.nodes[struct].DMAX=0.05 #half of default value
            self.nodes[struct].energy =self.nodes[struct].V0 = self.nodes[0].energy +energy[struct]
            #self.nodes[struct].energy = self.nodes[struct].PES.get_energy(self.nodes[struct].geom)
            self.nodes[struct].gradrms=grmss[struct]

        self.store_energies() 
        self.nnodes=self.nR=nstructs
        self.isRestarted=True
        self.done_growing=True
        print "setting all interior nodes to active"
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.optimizer[n].options['OPTTHRESH']=self.options['CONV_TOL']*2
        print " V_profile: ",
        for n in range(self.nnodes):
            print " {:7.3f}".format(float(self.energies[n])),
        print

    def store_energies(self):
        for i,ico in enumerate(self.nodes):
            if ico != 0:
                self.energies[i] = ico.energy - self.nodes[0].energy

    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1,rtype=2):
        print "*********************************************************************"
        print "************************** in opt_iters *****************************"
        print "*********************************************************************"

        self.nclimb=0
        self.nhessreset=10  # are these used??? TODO 
        self.hessrcount=0   # are these used?!  TODO
        self.newclimbscale=2.

        self.set_finder(rtype)

        for oi in range(max_iter):
            sys.stdout.flush()

            # stash previous TSnode  
            self.pTSnode = self.TSnode

            # => get TS node <=
            self.emaxp = self.emax
            self.TSnode = np.argmax(self.energies)
            self.nodes[self.TSnode].isTSnode=True

            # => Get all tangents 3-way <= #
            self.get_tangents_1e()
            
            # => do opt steps <= #
            self.opt_steps(optsteps)
            self.store_energies()
            self.emax = float(max(self.energies[1:-1]))
            print " V_profile: ",
            for n in range(self.nnodes):
                print " {:7.3f}".format(float(self.energies[n])),
            print

            if self.climb and not self.find: print("c")
            elif self.find: print("x")

            #TODO stuff with path_overlap/path_overlapn #TODO need to save path_overlap
            

            #TODO resetting
            #TODO special SSM criteria if TSNode is second to last node
            #TODO special SSM criteria if first opt'd node is too high?

            # => calculate totalgrad <= #
            totalgrad,gradrms = self.calc_grad()
            print " opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(oi,float(totalgrad),float(gradrms),self.TSnode,float(self.emax))

            # => set stage <= #
            fp = self.find_peaks(2)
            print " fp = ",fp
            ts_cgradq=abs(self.nodes[self.TSnode].gradq[self.nodes[self.TSnode].nicd-1])
            ts_gradrms=self.nodes[self.TSnode].gradrms
            self.dE_iter=abs(self.emax-self.emaxp)
            print " dE_iter ={:2.2f}".format(self.dE_iter)
            self.set_stage(totalgrad,ts_cgradq,ts_gradrms,fp)

            # => Check Convergence <= #
            isDone = self.check_opt(totalgrad,fp,rtype)
            if isDone:
                break
            if not self.climber and not self.finder and totalgrad<0.025: #Break even if not climb/find
                break

            # => write Convergence to file <= #
            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)

            # => Reparam the String <= #
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)

            if self.pTSnode!=self.TSnode and self.climb and not self.find:
                print " slowing down climb optimization"
                self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale
                if self.newclimbscale<5.0:
                    self.newclimbscale +=1.


            #also prints tgrads and jobGradCount

        print " Printing string to opt_converged_000.xyz"
        self.write_xyz_files(base='opt_converged',iters=0,nconstraints=nconstraints)
        sys.stdout.flush()
        return

    def get_tangents_1(self,n0=0):
        size_ic = self.nodes[0].num_ics
        nbonds = self.nodes[0].BObj.nbonds
        nangles = self.nodes[0].AObj.nangles
        ntor = self.nodes[0].TObj.ntor
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes+1,self.nnodes))
        ictan = [[]]*self.nnodes
        #print "getting tangents for nodes 0 to ",self.nnodes
        for n in range(n0+1,self.nnodes):
            #print "getting tangent between %i %i" % (n,n-1)
            assert self.nodes[n]!=0,"n is bad"
            assert self.nodes[n-1]!=0,"n-1 is bad"
            ictan[n] = DLC.tangent_1(self.nodes[n],self.nodes[n-1])
            dqmaga[n] = 0.
            ictan0= np.copy(ictan[n])

            ictan[n] /= np.linalg.norm(ictan[n])

            self.newic.set_xyz(self.nodes[n].coords)
            self.newic.update_ics()
            self.newic.bmatp = self.newic.bmatp_create()
            self.newic.bmatp_to_U()
            self.newic.opt_constraint(ictan0)
            dqmaga[n] += np.dot(ictan0[:nbonds].T,self.newic.Ut[-1,:nbonds])*2.5
            dqmaga[n] += np.dot(ictan0[nbonds:].T,self.newic.Ut[-1,nbonds:])
            dqmaga[n] = float(np.sqrt(dqmaga[n]))
        
        self.dqmaga = dqmaga
        self.ictan = ictan

        if self.newic.print_level>1:
            print '------------printing ictan[:]-------------'
            for n in range(n0+1,self.nnodes):
                print "ictan[%i]" %n
                for i in range(self.newic.BObj.nbonds):
                    print "%1.2f " % self.ictan[n][i],
                print 
                for i in range(self.newic.BObj.nbonds,self.newic.AObj.nangles+self.newic.BObj.nbonds):
                    print "%1.2f " % self.ictan[n][i],
                for i in range(self.newic.BObj.nbonds+self.newic.AObj.nangles,self.newic.AObj.nangles+self.newic.BObj.nbonds+self.newic.TObj.ntor):
                    print "%1.2f " % self.ictan[n][i],
                print "\n"


    # for some reason this fxn doesn't work when called outside gsm
    def get_tangents_1e(self,n0=0):
        size_ic = self.nodes[0].num_ics
        nbonds = self.nodes[0].BObj.nbonds
        nangles = self.nodes[0].AObj.nangles
        ntor = self.nodes[0].TObj.ntor
        ictan0 = np.zeros((size_ic,1))
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes,self.nnodes))
        self.store_energies()

        for n in range(n0+1,self.nnodes-1):
            do3 = False
            if not self.find:
                if self.energies[n+1] > self.energies[n] and self.energies[n] > self.energies[n-1]:
                    intic_n = n
                    newic_n = n+1
                elif self.energies[n-1] > self.energies[n] and self.energies[n] > self.energies[n+1]:
                    intic_n = n-1
                    newic_n = n
                else:
                    do3 = True
                    newic_n = n
                    intic_n = n+1
                    int2ic_n = n-1
            else:
                if n < self.TSnode:
                    intic_n = n
                    newic_n = n+1
                elif n> self.TSnode:
                    intic_n = n-1
                    newic_n = n
                else:
                    do3 = True
                    newic_n = n
                    intic_n = n+1
                    int2ic_n = n-1
            if not do3:
                ictan0 = self.tangent(newic_n,intic_n)
            else:
                f1 = 0.
                dE1 = abs(self.energies[n+1]-self.energies[n])
                dE2 = abs(self.energies[n] - self.energies[n-1])
                dEmax = max(dE1,dE2)
                dEmin = min(dE1,dE2)
                if self.energies[n+1]>self.energies[n-1]:
                    f1 = dEmax/(dEmax+dEmin+0.00000001)
                else:
                    f1 = 1 - dEmax/(dEmax+dEmin+0.00000001)

                print ' 3 way tangent ({}): f1:{:3.2}'.format(n,f1)

                t1 = np.zeros(size_ic)
                t2 = np.zeros(size_ic)

                for i in range(nbonds):
                    t1[i] = self.nodes[intic_n].BObj.bondd[i] - self.nodes[newic_n].BObj.bondd[i]
                    t2[i] = self.nodes[newic_n].BObj.bondd[i] - self.nodes[int2ic_n].BObj.bondd[i]
                for i in range(nangles):
                    t1[nbonds+i] = (self.nodes[intic_n].AObj.anglev[i] - self.nodes[newic_n].AObj.anglev[i])*np.pi/180.
                    t2[nbonds+i] = (self.nodes[newic_n].AObj.anglev[i] - self.nodes[int2ic_n].AObj.anglev[i])*np.pi/180.
                for i in range(ntor):
                    tmp1 = (self.nodes[intic_n].TObj.torv[i] - self.nodes[newic_n].TObj.torv[i])*np.pi/180.
                    tmp2 = (self.nodes[newic_n].TObj.torv[i] - self.nodes[int2ic_n].TObj.torv[i])*np.pi/180.
                    if tmp1 > np.pi:
                        tmp1 = -1*(2*np.pi-tmp1)
                    if tmp1 < -np.pi:
                        tmp1 = 2*np.pi + tmp1
                    if tmp2 > np.pi:
                        tmp2 = -1*(2*np.pi - tmp2)
                    if tmp2 < -np.pi:
                        tmp2 = 2*np.pi + tmp2
                    t1[nbonds+nangles+i] = tmp1
                    t2[nbonds+nangles+i] = tmp2
                ictan0 = f1*t1 +(1-f1)*t2
                ictan0 = ictan0.reshape((size_ic,1))
                self.ictan[n]=ictan0
            
            dqmaga[n]=0.0
            ictan0 = np.reshape(np.copy(self.ictan[n]),(size_ic,1))
            self.newic.set_xyz(self.nodes[newic_n].coords)
            self.newic.bmatp = self.newic.bmatp_create()
            self.newic.bmatp_to_U()
            self.newic.opt_constraint(ictan0)
            dqmaga[n] += np.dot(ictan0[:nbonds].T,self.newic.Ut[-1,:nbonds])*2.5
            dqmaga[n] += np.dot(ictan0[nbonds:].T,self.newic.Ut[-1,nbonds:])
            dqmaga[n] = float(dqmaga[n])
        

        #print '------------printing ictan[:]-------------'
        #for row in ictan:
        #    print row
        if self.newic.print_level>0:
            print '------------printing dqmaga---------------'
            print dqmaga
        self.dqmaga = dqmaga

    def get_tangents_1g(self):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase
        """
        #ictan = [[]]*self.nnodes
        dqmaga = [0.]*self.nnodes
        dqa = [[],]*self.nnodes

        ncurrent,nlist = self.make_nlist()

        for n in range(ncurrent):
            self.ictan[nlist[2*n]] = self.tangent(nlist[2*n],nlist[2*n+1])

            #save copy to get dqmaga
            ictan0 = np.copy(self.ictan[nlist[2*n]])
            if self.nodes[nlist[2*n+1]].print_level>1:
                print "forming space for", nlist[2*n+1]

            opt_type=self.set_opt_type(nlist[2*n+1],quiet=True)  #TODO why is this here? 2/2019
            
            cVec = self.nodes[nlist[2*n+1]].form_Cvec_from_prim_Vecs(self.ictan[nlist[2*n]])

            self.nodes[nlist[2*n+1]].get_coords(constraints=cVec)
            exit()
            #form_constrained_DLC(self.ictan[nlist[2*n]])

            #normalize ictan
            self.ictan[nlist[2*n]] /= np.linalg.norm(self.ictan[nlist[2*n]])
            
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,self.nodes[nlist[2*n+1]].Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))

        self.dqmaga = dqmaga
       
        if False:
            for n in range(ncurrent):
                print "dqmag[%i] =%1.2f" %(nlist[2*n],self.dqmaga[nlist[2*n]])
                print "printing ictan[%i]" %nlist[2*n]       
                for i in range(self.nodes[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % self.ictan[nlist[2*n]][i],
                print 
                for i in range(self.nodes[nlist[2*n]].BObj.nbonds,self.nodes[nlist[2*n]].AObj.nangles+self.nodes[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % self.ictan[nlist[2*n]][i],
                for i in range(self.nodes[nlist[2*n]].BObj.nbonds+self.nodes[nlist[2*n]].AObj.nangles,self.nodes[nlist[2*n]].AObj.nangles+self.nodes[nlist[2*n]].BObj.nbonds+self.nodes[nlist[2*n]].TObj.ntor):
                    print "%1.2f " % self.ictan[nlist[2*n]][i],
                print "\n"
#           #     print np.transpose(ictan[nlist[2*n]])
        for i,tan in enumerate(self.ictan):
            if np.all(tan==0.0):
                print "tan %i of the tangents is 0" %i
                raise RuntimeError


    def growth_iters(self,iters=1,maxopt=1,nconstraints=1,current=0):
        print ''
        print "*********************************************************************"
        print "************************ in growth_iters ****************************"
        print "*********************************************************************"

        for n in range(iters):
            sys.stdout.flush()
            success = self.check_add_node()
            if not success:
                print "can't add anymore nodes, bdist too small"
                break
            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.get_tangents_1g()
            self.opt_steps(maxopt)
            self.store_energies()
            totalgrad,gradrms = self.calc_grad()
            self.emax = float(max(self.energies[1:-1]))
            self.TSnode = np.where(self.energies==self.emax)[0][0]
            print " gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n".format(n,float(totalgrad),float(gradrms),float(self.emax))
            self.write_xyz_files(iters=n,base='growth_iters',nconstraints=nconstraints)
            if self.check_if_grown(): 
                break
            self.ic_reparam_g()
        self.newic = DLC.copy_node(self.nodes[0],0,-1)
        return n

    def opt_steps(self,opt_steps):

        print "in opt_steps"
        # these can be put in a function
        for n in range(self.nnodes):
            if self.nodes[n]!=0:
                self.nodes[n].isTSnode=False
        fp=0
        if self.done_growing:
            fp = self.find_peaks(2)
        if fp>0:
            self.TSnode = np.argmax(self.energies)
            self.nodes[self.TSnode].isTSnode=True

        # this can be put in a function
        optlastnode=False
        if self.product_geom_fixed==False:
            if self.energies[self.nnodes-1]>self.energies[self.nnodes-2] and fp>0:
                optlastnode=True
            if self.nodes[self.nnodes-1].gradrms>self.options['CONV_TOL']:
                optlastnode=True

        for n in range(self.nnodes):
            if self.nodes[n] != 0 and self.active[n]==True:

                print "\n Optimizing node %i" % n
                # => set opt type <= #
                opt_type = self.set_opt_type(n)
                self.optimizer[n].options['opt_type']=opt_type

                exsteps=1 #multiplier for nodes near the TS node
                if self.find and self.energies[n]+1.5 > self.energies[self.TSnode] and n!=self.TSnode:  # should this be for climb too?
                    exsteps=2
                    print " multiplying steps for node %i by %i" % (n,exsteps)
                if self.find and n==self.TSnode: #multiplier for TS node during
                    exsteps=2
                    print " multiplying steps for node %i by %i" % (n,exsteps)
                
                # => do constrained optimization
                self.nodes[n].energy = self.optimizer[n].optimize(
                        c_obj=self.nodes[n],
                        refE=self.nodes[0].V0,
                        opt_steps=opt_steps*exsteps,
                        ictan=self.ictan[n]
                        )

            if optlastnode==True and n==self.nnodes-1 and not self.nodes[n].PES.lot.do_coupling:
                print " optimizing last node"
                self.optimizer[n].options['opt_type']='UNCONSTRAINED'
                self.nodes[n].energy = self.optimizer[n].optimize(
                        c_obj=self.nodes[n],
                        refE=self.nodes[0].V0,
                        opt_steps=opt_steps
                        )

    def set_stage(self,totalgrad,ts_cgradq,ts_gradrms,fp):
        if totalgrad < 0.3 and fp>0: # extra criterion in og-gsm for added
            if not self.climb and self.climber:
                print(" ** starting climb **")
                self.climb=True
                print " totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq)
            elif (self.climb and not self.find and self.finder and self.dE_iter<4. and self.nclimb<1 and
                    ((totalgrad<0.2 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.01) or
                    (totalgrad<0.1 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.02) or
                    (ts_gradrms<self.options['CONV_TOL']*5.))
                    ):
                print(" ** starting exact climb **")
                print " totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq)
                self.find=True
                self.get_tangents_1e()
                self.get_eigenv_finite(self.TSnode)
                self.nhessreset=10  # are these used??? TODO 
                self.hessrcount=0   # are these used?!  TODO
            if self.climb: 
                self.nclimb-=1

            for n in range(1,self.nnodes-1):
                self.active[n]=True
                self.optimizer[n].options['OPTTHRESH']=self.options['CONV_TOL']*2

        if self.find and self.optimizer[self.TSnode].nneg > 3 and ts_cgradq>self.options['CONV_TOL']:
            #if self.hessrcount<1 and self.pTSnode == self.TSnode:
            if self.pTSnode == self.TSnode:
                print " resetting TS node coords Ut (and Hessian)"
                self.get_tangents_1e()
                self.get_eigenv_finite(self.TSnode)
                self.nhessreset=10
                self.hessrcount=1
            else:
                print " Hessian consistently bad, going back to climb (for 3 iterations)"
                self.find=0
                self.nclimb=3
        elif self.find and self.optimizer[self.TSnode].nneg <= 3:
            self.hessrcount-=1
        self.nhessreset-=1

    def interpolateR(self,newnodes=1):
        print " Adding reactant node"
        success= True
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.nodes[self.nR] =self.add_node(self.nR-1,self.nR,self.nnodes-self.nP)
            if self.nodes[self.nR]==0:
                success= False
                break
            print " getting energy for  node ",self.nR
            print self.nodes[self.nR].energy-self.nodes[0].energy
            self.nn+=1
            self.nR+=1
            print " nn=%i,nR=%i" %(self.nn,self.nR)
            self.active[self.nR-1] = True


        return success

    def interpolateP(self,newnodes=1):
        print " Adding product node"
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            #self.nodes[-self.nP-1] = self.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nnodes-self.nP)
            n1=self.nnodes-self.nP
            n2=self.nnodes-self.nP-1
            n3=self.nR-1
            self.nodes[-self.nP-1] = self.add_node(n1,n2,n3)
            if self.nodes[-self.nP-1]==0:
                success= False
                break
            print " getting energy for  node ",self.nnodes-self.nP-1
            print self.nodes[-self.nP-1].energy-self.nodes[0].energy
            self.nn+=1
            self.nP+=1
            print " nn=%i,nP=%i" %(self.nn,self.nP)
            self.active[-self.nP] = True

    def ic_reparam(self,ic_reparam_steps=8,n0=0,nconstraints=1,rtype=0):
        num_ics = self.nodes[0].num_ics
        len_d = self.nodes[0].nicd
        ictalloc = self.nnodes+1
        
        rpmove = np.zeros(ictalloc)
        rpart = np.zeros(ictalloc)
        totaldqmag = 0.0
        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(ictalloc)
        edist = np.zeros(ictalloc)
        
        for i in range(ic_reparam_steps):
            self.get_tangents_1(n0=n0)

            # copies of original ictan
            ictan0 = np.copy(self.ictan)
            ictan = np.copy(self.ictan)

            if self.newic.print_level>1:
                print " printing spacings dqmaga:"
                for n in range(1,self.nnodes):
                    print " %1.2f" % self.dqmaga[n], 
                print 

            totaldqmag = 0.
            totaldqmag = np.sum(self.dqmaga[n0+1:self.nnodes])
            #print " totaldqmag = %1.3f" %totaldqmag
            dqavg = totaldqmag/(self.nnodes-1)

            #if climb:
            if self.climb or rtype==2:
                h1dqmag = np.sum(self.dqmaga[1:self.TSnode+1])
                h2dqmag = np.sum(self.dqmaga[self.TSnode+1:self.nnodes])
                if self.newic.print_level>1:
                    print " h1dqmag, h2dqmag: %1.1f %1.1f" % (h1dqmag,h2dqmag)
           
            # => Using average <= #
            if i==0 and rtype==0:
                print " using average"
                if not self.climb:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
                else:
                    for n in range(n0+1,self.TSnode):
                        rpart[n] = 1./(self.TSnode-n0)
                    for n in range(self.TSnode+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-self.TSnode-1)
                    rpart[self.TSnode]=0.

            if rtype==1 and i==0:
                dEmax = 0.
                for n in range(n0+1,self.nnodes):
                    dE[n] = abs(self.energies[n]-self.energies[n-1])
                dEmax = max(dE)
                for n in range(n0+1,self.nnodes):
                    edist[n] = dE[n]*self.dqmaga[n]

                print " edist: ",
                for n in range(n0+1,self.nnodes):
                    print " {:1.1}".format(edist[n]),
                print 
                
                totaledq = np.sum(edist[n0+1:self.nnodes])
                edqavg = totaledq/(self.nnodes-1)

            if i==0:
                print " rpart: ",
                for n in range(1,self.nnodes):
                    print " {:1.2}".format(rpart[n]),
                print

            if not self.climb and rtype!=2:
                for n in range(n0+1,self.nnodes-1):
                    deltadq = self.dqmaga[n] - totaldqmag * rpart[n]
                    if n==self.nnodes-2:
                        deltadq += totaldqmag * rpart[n] - self.dqmaga[n+1] # so zero?
                    rpmove[n] = -deltadq
            else:
                deltadq = 0.
                rpmove[self.TSnode] = 0.
                for n in range(n0+1,self.TSnode):
                    deltadq = self.dqmaga[n] - h1dqmag * rpart[n]
                    if n==self.nnodes-2:
                        deltadq += h2dqmag * rpart[n] - self.dqmaga[n+1]
                    rpmove[n] = -deltadq
                for n in range(self.TSnode+1,self.nnodes-1):
                    deltadq = self.dqmaga[n] - h2dqmag * rpart[n]
                    if n==self.nnodes-2:
                        deltadq += h2dqmag * rpart[n] - self.dqmaga[n+1]
                    rpmove[n] = -deltadq

            MAXRE = 0.5
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            for n in range(n0+1,self.nnodes-2):
                if n+1 != self.TSnode or self.climb:
                    rpmove[n+1] += rpmove[n]
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            if self.climb or rtype==2:
                rpmove[self.TSnode] = 0.


            disprms = np.linalg.norm(rpmove[n0+1:self.nnodes-1])
            lastdispr = disprms

            if self.newic.print_level>1:
                for n in range(n0+1,self.nnodes-1):
                    print " disp[{}]: {:1.2}".format(n,rpmove[n]),
                print
                print " disprms: {:1.3}\n".format(disprms)

            if disprms < 0.02:
                break

            for n in range(n0+1,self.nnodes-1):
                #print "moving node %i %1.3f" % (n,rpmove[n])
                self.newic.set_xyz(self.nodes[n].coords) 
                self.newic.update_ics()
                opt_type=self.set_opt_type(n,quiet=True)

                if rpmove[n] < 0.:
                    ictan[n] = np.copy(ictan0[n]) 
                else:
                    ictan[n] = np.copy(ictan0[n+1]) 

                dq = np.zeros((self.newic.nicd,1),dtype=float)
                dq[-1] = rpmove[n]

                self.newic.form_constrained_DLC(ictan[n])
                self.newic.ic_to_xyz(dq)
                self.nodes[n].set_xyz(self.newic.coords)
                self.nodes[n].update_ics()

                #TODO might need to recalculate energy here for seam? 

        print ' spacings (end ic_reparam, steps: {}:'.format(ic_reparam_steps)
        for n in range(1,self.nnodes):
            print " {:1.2}".format(self.dqmaga[n]),
        print
        print "  disprms: {:1.3}\n".format(disprms)

    def ic_reparam_g(self,ic_reparam_steps=8,n0=0):  #see line 3863 of gstring.cpp
        """size_ic = self.nodes[0].num_ics; len_d = self.nodes[0].nicd"""

        #close_dist_fix(0) #done here in GString line 3427.

        #print '**************************************************'
        #print '***************in ic_reparam_g********************'
        #print '**************************************************'

        num_ics = self.nodes[0].num_ics
        len_d = self.nodes[0].nicd
        
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)

        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(self.nnodes)
        edist = np.zeros(self.nnodes)
        
        self.TSnode = -1 
        emax = -1000 # And this?

        for i in range(ic_reparam_steps):
            self.get_tangents_1g()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
            if self.nodes[0].print_level>1:
                if i==0:
                    print " totaldqmag (without inner): {:1.2}\n".format(totaldqmag)
                print " printing spacings dqmaga: "
                for n in range(self.nnodes):
                    print " {:1.2}".format(self.dqmaga[n]),
                    if (n+1)%5==0:
                        print
                print 
            
            if i == 0:
                if self.nn!=self.nnodes:
                    rpart = np.zeros(self.nnodes)
                    for n in range(n0+1,self.nR):
                        rpart[n] = 1.0/(self.nn-2)
                    for n in range(self.nnodes-self.nP,self.nnodes-1):
                        rpart[n] = 1.0/(self.nn-2)
                    if self.nodes[0].print_level>1:
                        if i==0:
                            print " rpart: "
                            for n in range(1,self.nnodes):
                                print " {:1.2}".format(rpart[n]),
                                if (n)%5==0:
                                    print
                            print
                else:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
            nR0 = self.nR
            nP0 = self.nP

            if False:
                if self.nnodes-self.nn > 2:
                    nR0 -= 1
                    nP0 -= 1
            
            deltadq = 0.0
            for n in range(n0+1,nR0):
                deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes-nP0,self.nnodes-1):
                deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n]) > MAXRE:
                    rpmove[n] = float(np.sign(rpmove[n])*MAXRE)


            disprms = float(np.linalg.norm(rpmove[n0+1:self.nnodes-1]))
            lastdispr = disprms
            if self.nodes[0].print_level>1:
                for n in range(n0+1,self.nnodes-1):
                    print " disp[{}]: {:1.2f}".format(n,rpmove[n]),
                print
                print " disprms: {:1.3}\n".format(disprms)

            if disprms < 1e-2:
                break

            #TODO check how range is defined in gstring, uses n0...
            for n in range(n0+1,self.nnodes-1):
                if isinstance(self.nodes[n],DLC):
                    if rpmove[n] > 0:
    
                        dq0 = np.zeros((self.nodes[n].nicd,1))
                        self.nodes[n].form_constrained_DLC(self.ictan[n])

                        dq0[self.nodes[n].nicd-1] = rpmove[n]  # ictan is always last vector
                        if self.nodes[0].print_level>1:
                            print " dq0[constraint]: {:1.3}".format(float(dq0[self.nodes[n].nicd-1]))
                        self.nodes[n].ic_to_xyz(dq0)
                    else:
                        pass
        print " spacings (end ic_reparam, steps: {}):".format(ic_reparam_steps),
        for n in range(self.nnodes):
            print " {:1.2}".format(self.dqmaga[n]),
        print "  disprms: {:1.3}".format(disprms)

        #Failed = check_array(self.nnodes,self.dqmaga)
        #If failed, do exit 1

    def get_eigenv_finite(self,en):
        ''' Modifies Hessian using RP direction'''

        print "modifying Hessian with RP"
        #self.nodes[en].form_constrained_DLC(self.ictan[en])
        self.nodes[en].form_unconstrained_DLC()

        self.newic.set_xyz(self.nodes[en].coords)
        self.newic.update_ics()
        self.newic.form_constrained_DLC(self.ictan[en])
        nicd = self.newic.nicd  #len
        num_ics = self.newic.num_ics #size_ic

        E0 = self.energies[en]/KCAL_MOL_PER_AU
        Em1 = self.energies[en-1]/KCAL_MOL_PER_AU
        if en+1<self.nnodes:
            Ep1 = self.energies[en+1]/KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        q0 =  self.newic.q[nicd-1]
        #print "q0 is %1.3f" % q0
        tan0 = self.newic.Ut[nicd-1,:]
        #print "tan0"
        #print tan0

        self.newic.set_xyz(self.nodes[en-1].coords)
        self.newic.bmatp_create()
        self.newic.bmat_create()
        qm1 = self.newic.q[nicd-1]
        #print "qm1 is %1.3f " %qm1

        if en+1<self.nnodes:
            self.newic.set_xyz(self.nodes[en+1].coords)
            self.newic.bmatp_create()
            self.newic.bmat_create()
            qp1 = self.newic.q[nicd-1]
        else:
            qp1 = qm1

        #print "qp1 is %1.3f" % qp1

        if self.nodes[en].isTSnode:
            print " TS Hess init'd w/ existing Hintp"

        self.newic.set_xyz(self.nodes[en].coords)
        self.newic.form_unconstrained_DLC()
        self.newic.Hintp=np.copy(self.nodes[en].Hintp)
        self.newic.Hint = self.newic.Hintp_to_Hint()

        tan = np.dot(self.newic.Ut,tan0.T)  #(nicd,numic)(num_ic,1)=nicd,1 
        #print "tan"
        #print tan

        Ht = np.dot(self.newic.Hint,tan) #nicd,1
        tHt = np.dot(tan.T,Ht) 

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        print " tHt %1.3f a: %1.1f b: %1.1f c: %1.3f" % (tHt,a[0],b[0],c[0])

        ttt = np.outer(tan,tan)
        #print "Hint before"
        #with np.printoptions(threshold=np.inf):
        #    print self.newic.Hint
        eig,tmph = np.linalg.eigh(self.newic.Hint)
        #print "initial eigenvalues"
        #print eig
       
        self.newic.Hint += (c-tHt)*ttt
        #self.nodes[en].Hint = np.copy(self.newic.Hint)
        self.optimizer[en].Hint = np.copy(self.newic.Hint)
        #print "Hint"
        #with np.printoptions(threshold=np.inf):
        #    print self.nodes[en].Hint
        #    print self.optimizer[en].Hint
        #print "shape of Hint is %s" % (np.shape(self.nodes[en].Hint),)

        #this can also work on non-TS nodes?
        self.nodes[en].newHess = 2

        if False:
            eigen,tmph = np.linalg.eigh(self.optimizer[en].Hint) #nicd,nicd
            #print "eigenvalues of new Hess"
            #print eigen

        # reset pgradrms ? 

    def set_V0(self):
        raise NotImplementedError 

    def set_opt_type(self,n,quiet=False):
        #TODO
        opt_type='ICTAN' 
        if self.climb and self.nodes[n].isTSnode==True:
            opt_type='CLIMB'
        if self.find and self.nodes[n].isTSnode==True:
            opt_type='TS'
        if self.nodes[n].PES.lot.do_coupling==True:
            opt_type='SEAM'
        if self.climb and self.nodes[n].isTSnode==True and opt_type=='SEAM':
            opt_type='TS-SEAM'
        if not quiet:
            print(" setting node %i opt_type to %s" %(n,opt_type))

        return opt_type

    def set_finder(self,rtype):
        assert rtype in [0,1,2], "rtype not defined"
        print ''
        print "*********************************************************************"
        if rtype==2:
            print "****************** set climber and finder to True*****************"
            self.climber=True
            self.finder=True
        elif rtype==1:
            print("***************** setting climber to True*************************")
            self.climber=True
        else:
            print("******** Turning off climbing image and exact TS search **********")
        print "*********************************************************************"
   



if __name__=='__main__':
    from qchem import QChem
    from pes import PES
    #from hybrid_dlc import Hybrid_DLC
    #filepath="firstnode.pdb"
    #mol=pb.readfile("pdb",filepath).next()
    #lot = QChem.from_options(states=[(2,0)],lot_inp_file='qstart',nproc=1)
    #pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=2)
    #ic = Hybrid_DLC.from_options(mol=mol,PES=pes,IC_region=["UNL"],print_level=2)
   
    from dlc import DLC
    basis="sto-3g"
    nproc=1
    filepath="examples/tests/bent_benzene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    
    # => DLC constructor <= #
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
    param = parameters.from_options(opt_type='UNCONSTRAINED')
    #gsm = Base_Method.from_options(ICoord1=ic1,param,optimize=eigenvector_follow)

    #gsm.optimize(nsteps=20)


