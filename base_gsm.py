import options
import numpy as np
import os
import openbabel as ob
import pybel as pb
from dlc import *
from copy import deepcopy
import StringIO
from _print_opt import *
from _analyze_string import *
import sys

class Base_Method(object,Print,Analyze):

    @staticmethod
    def from_options(**kwargs):
        return GSM(GSM.default_options().set_values(kwargs))
    
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
                key="last_node_fixed",
                value=True,
                required=False,
                doc="Fix last node?"
                )

        opt.add_option(
                key="growth_direction",
                value=0,
                required=False,
                doc="how to grow string,0=Normal,1=from reactant,2=from product"
                )

        opt.add_option(
                key="DQMAG_MAX",
                value=0.8,
                required=False,
                doc="for SSM"
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
                doc="bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
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

        # Cache attributes
        self.optCG = False #TODO
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        self.active = [False] * self.nnodes
        self.driving_coords = self.options['driving_coords']
        self.CONV_TOL = self.options['CONV_TOL']
        self.ADD_NODE_TOL = self.options['ADD_NODE_TOL']
        self.last_node_fixed = self.options['last_node_fixed']
        self.climber=False  #is this string a climber?
        self.finder=False   # is this string a finder?
        self.growth_direction=self.options['growth_direction']
        self.isRestarted=False
        self.DQMAG_MAX=self.options['DQMAG_MAX']
        self.DQMAG_MIN=self.options['DQMAG_MIN']
        self.BDIST_RATIO=self.options['BDIST_RATIO']

        # Set initial values
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.energies = np.asarray([0.]*self.nnodes)
        self.emax = 0.0
        self.TSnode = 0 
        self.climb = False #TODO
        self.find = False  #TODO
        self.n0 = 0 # something to do with added nodes? "first node along current block"
        self.end_early=False
        self.tscontinue=True # whether to continue with TS opt or not
        self.rn3m6 = np.sqrt(3.*self.icoords[0].natoms-6.);
        self.gaddmax = self.ADD_NODE_TOL/self.rn3m6;
        print " gaddmax:",self.gaddmax
        self.stage=0 #growing
        self.ictan = [[]]*self.nnodes

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
        tmp=DLC(self.icoords[0].options.copy().set_values({
            'mol':mol,
            }))

        print "doing union"
        self.icoords[0] = DLC.union_ic(self.icoords[0],tmp)
        self.newic = DLC.copy_node(self.icoords[0],0,-1)
        self.icoords[0].energy = self.icoords[0].V0 = self.icoords[0].PES.get_energy(self.icoords[0].geom)
        self.icoords[0].gradrms=grmss[0]

        print "initial energy is %3.4f" % self.icoords[0].energy
        for struct in range(1,nstructs):
            if struct==nstructs-1:
                self.icoords[struct] = DLC.copy_node(self.icoords[struct-1],struct,0)
            else:
                self.icoords[struct] = DLC.copy_node(self.icoords[struct-1],struct,1)
            self.icoords[struct].set_xyz(coords[struct])
            self.icoords[struct].update_ics()
            self.icoords[struct].setup()
            self.icoords[struct].DMAX=0.05 #half of default value
            #self.icoords[struct].energy =self.icoords[struct].V0 = self.icoords[0].energy +energy[struct]
            self.icoords[struct].energy = self.icoords[struct].PES.get_energy(self.icoords[struct].geom)
            self.icoords[struct].gradrms=grmss[struct]

        self.store_energies() 
        self.nnodes=self.nR=nstructs
        self.isRestarted=True
        print "setting all interior nodes to active"
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.icoords[n].OPTTHRESH=self.CONV_TOL
        print " V_profile: ",
        for n in range(self.nnodes):
            print " {:7.3f}".format(float(self.energies[n])),
        print

    def store_energies(self):
        for i,ico in enumerate(self.icoords):
            if ico != 0:
                self.energies[i] = ico.energy - self.icoords[0].energy

    def optimize(self,n=0,nsteps=100,opt_type=0,ictan=None):
        assert self.icoords[n]!=0,"icoord not set"
        assert opt_type in range(-1,8), "opt_type {} not defined".format(opt_type)
        if opt_type==0:
            assert ictan==None
        if opt_type in [5,6,7]:
            assert self.icoords[n].PES.lot.do_coupling==True,"Turn do_coupling on."
        elif opt_type not in [5,6,7]:
            assert self.icoords[n].PES.lot.do_coupling==False,"Turn do_coupling off."
        if opt_type in [1,2,6,7] and ictan.any()==None:
            raise RuntimeError, "Need ictan"
        if opt_type in [2,4]:
            assert self.icoords[n].isTSnode,"only run climb and eigenvector follow on TSnode."
            
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat('xyz')
        opt_molecules=[]
        self.icoords[n].V0 = self.icoords[n].PES.get_energy(self.icoords[n].geom)
        grad = self.icoords[n].PES.get_gradient(self.icoords[n].geom)
        self.icoords[n].gradq = self.icoords[n].grad_to_q(grad)
        self.icoords[n].energy=0
        grmss = []
        steps = []
        energies=[]
        deltaEs=[]
        Es =[]
        self.icoords[n].update_hess=False # gets reset after each step
        self.icoords[n].buf = StringIO.StringIO()

        if self.icoords[n].print_level>0:
            print " Initial energy is %1.4f" % self.icoords[n].V0

        for step in range(nsteps):
            if self.icoords[n].print_level>0:
                print(" \n Opt step: %i" %(step+1)),
            self.icoords[n].buf.write("\n Opt step: %d" %(step+1))

            ########### => Opt step <= ############
            smag =self.icoords[n].opt_step(opt_type=opt_type,ictan=ictan,refE=self.icoords[0].V0)

            ## => modify hessian if overlap is becoming too small <= #
            if opt_type==4:
                if self.icoords[n].check_overlap_good()==False:
                    self.get_eigenv_finite(n)

            # convergence quantities
            grmss.append(float(self.icoords[n].gradrms))
            steps.append(smag)
            energies.append(self.icoords[n].energy-self.icoords[n].V0)
            opt_molecules.append(obconversion.WriteString(self.icoords[n].mol.OBMol))
            if isinstance(self.icoords[n].PES,Penalty_PES) or isinstance(self.icoords[n].PES,Avg_PES):
                deltaEs.append(self.icoords[n].PES.dE)
    
            #write convergence
            self.write_node(n,opt_molecules,energies,grmss,steps,deltaEs)
   
            #TODO convergence 
            sys.stdout.flush()
            if self.converged(n,opt_type):
                print 'TRIUMPH! Optimization converged!'
                break

        #TODO if gradrms is greater than gradrmsl than further reduce DMAX
        #TODO change how revertopt is done is opt_step?

        print(self.icoords[n].buf.getvalue())
        if self.icoords[n].print_level>0:
            print " Final energy is %2.5f" % (self.icoords[n].energy)
        return smag

    def converged(self,n,opt_type):
        if self.icoords[n].gradrms<self.icoords[n].OPTTHRESH:
            return True
        else:
            return False

    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1,rtype=2):
        print "*********************************************************************"
        print "************************** in opt_iters *****************************"
        print "*********************************************************************"

        nclimb=0
        self.set_finder(rtype)

        for oi in range(max_iter):
            sys.stdout.flush()

            # => get TS node <=
            self.emaxp = self.emax
            self.TSnode = np.argmax(self.energies)
            self.icoords[self.TSnode].isTSnode=True

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
            # => calculate totalgrad <= #
            totalgrad,gradrms = self.calc_grad()

            if self.stage==1: print("c")
            elif self.stage==2: print("x")

            #TODO stuff with path_overlap/path_overlapn #TODO need to save path_overlap
            
            print " opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(oi,float(totalgrad),float(gradrms),self.TSnode,float(self.emax))

            fp = self.find_peaks(2)
            print " fp = ",fp

            #TODO resetting
            #TODO special SSM criteria if TSNode is second to last node
            #TODO special SSM criteria if first opt'd node is too high?

            # => Check Convergence <= #
            isDone = self.check_opt(totalgrad,fp,rtype)
            if isDone:
                break
            if not self.climber and not self.finder and totalgrad<0.025: #Break even if not climb/find
                break

            # => set stage <= #
            ts_cgradq=abs(self.icoords[self.TSnode].gradq[self.icoords[self.TSnode].nicd-1])
            ts_gradrms=self.icoords[self.TSnode].gradrms
            self.dE_iter=abs(self.emax-self.emaxp)
            print " dE_iter ={:1.2}".format(self.dE_iter)
            nclimb = self.set_stage(totalgrad,ts_cgradq,ts_gradrms,fp,self.dE_iter,nclimb)

            # => write Convergence to file <= #
            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)

            # => Reparam the String <= #
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)

            #also prints tgrads and jobGradCount

        print " Printing string to opt_converged_000.xyz"
        self.write_xyz_files(base='opt_converged',iters=0,nconstraints=nconstraints)
        sys.stdout.flush()
        return

    def get_tangents_1(self,n0=0):
        size_ic = self.icoords[0].num_ics
        nbonds = self.icoords[0].BObj.nbonds
        nangles = self.icoords[0].AObj.nangles
        ntor = self.icoords[0].TObj.ntor
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes+1,self.nnodes))
        ictan = [[]]*self.nnodes
        #print "getting tangents for nodes 0 to ",self.nnodes
        for n in range(n0+1,self.nnodes):
            #print "getting tangent between %i %i" % (n,n-1)
            assert self.icoords[n]!=0,"n is bad"
            assert self.icoords[n-1]!=0,"n-1 is bad"
            ictan[n] = DLC.tangent_1(self.icoords[n],self.icoords[n-1])
            dqmaga[n] = 0.
            ictan0= np.copy(ictan[n])

            ictan[n] /= np.linalg.norm(ictan[n])

            self.newic.set_xyz(self.icoords[n].coords)
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


    def get_tangents_1e(self,n0=0):
        size_ic = self.icoords[0].num_ics
        nbonds = self.icoords[0].BObj.nbonds
        nangles = self.icoords[0].AObj.nangles
        ntor = self.icoords[0].TObj.ntor
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
                    t1[i] = self.icoords[intic_n].BObj.bondd[i] - self.icoords[newic_n].BObj.bondd[i]
                    t2[i] = self.icoords[newic_n].BObj.bondd[i] - self.icoords[int2ic_n].BObj.bondd[i]
                for i in range(nangles):
                    t1[nbonds+i] = (self.icoords[intic_n].AObj.anglev[i] - self.icoords[newic_n].AObj.anglev[i])*np.pi/180.
                    t2[nbonds+i] = (self.icoords[newic_n].AObj.anglev[i] - self.icoords[int2ic_n].AObj.anglev[i])*np.pi/180.
                for i in range(ntor):
                    tmp1 = (self.icoords[intic_n].TObj.torv[i] - self.icoords[newic_n].TObj.torv[i])*np.pi/180.
                    tmp2 = (self.icoords[newic_n].TObj.torv[i] - self.icoords[int2ic_n].TObj.torv[i])*np.pi/180.
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
            self.newic.set_xyz(self.icoords[newic_n].coords)
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

            if self.icoords[nlist[2*n+1]].print_level>1:
                print "forming space for", nlist[2*n+1]
            self.icoords[nlist[2*n+1]].form_constrained_DLC(self.ictan[nlist[2*n]])

            #normalize ictan
            self.ictan[nlist[2*n]] /= np.linalg.norm(self.ictan[nlist[2*n]])
            
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,self.icoords[nlist[2*n+1]].Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))

        self.dqmaga = dqmaga
       
        if False:
            for n in range(ncurrent):
                print "dqmag[%i] =%1.2f" %(nlist[2*n],self.dqmaga[nlist[2*n]])
                print "printing ictan[%i]" %nlist[2*n]       
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % self.ictan[nlist[2*n]][i],
                print 
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds,self.icoords[nlist[2*n]].AObj.nangles+self.icoords[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % self.ictan[nlist[2*n]][i],
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds+self.icoords[nlist[2*n]].AObj.nangles,self.icoords[nlist[2*n]].AObj.nangles+self.icoords[nlist[2*n]].BObj.nbonds+self.icoords[nlist[2*n]].TObj.ntor):
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
        self.newic = DLC.copy_node(self.icoords[0],0,-1)
        return n

    def opt_steps(self,opt_steps):
        # these can be put in a function
        for n in range(self.nnodes):
            if self.icoords[n]!=0:
                self.icoords[n].isTSnode=False
        fp=0
        if self.stage>0:
            fp = self.find_peaks(2)
        if fp>0:
            self.TSnode = np.argmax(self.energies)
            self.icoords[self.TSnode].isTSnode=True

        # this can be put in a function
        optlastnode=False
        if self.last_node_fixed==False:
            if self.energies[self.nnodes-1]>self.energies[self.nnodes-2] and fp>0:
                optlastnode=True
            if self.icoords[self.nnodes-1].gradrms<self.CONV_TOL:
                optlastnode=True

        for n in range(self.nnodes):
            if self.icoords[n] != 0 and self.active[n]==True:

                print "\n Optimizing node %i" % n
                # => set opt type <= #
                opt_type = self.set_opt_type(n)

                exsteps=1 #multiplier for nodes near the TS node
                if self.stage==2 and self.energies[n]+1.5 > self.energies[self.TSnode] and n!=self.TSnode:
                    exsteps=2
                    print " multiplying steps for node %i by %i" % (n,exsteps)
                if self.stage>0 and n==self.TSnode: #and opt_type==4:
                    exsteps=2
                    print " multiplying steps for node %i by %i" % (n,exsteps)
                
                # => do constrained optimization
                self.icoords[n].smag = self.optimize(n=n,nsteps=opt_steps*exsteps,opt_type=opt_type,ictan=self.ictan[n])

            if optlastnode==True and n==self.nnodes-1 and not self.icoords[n].PES.lot.do_coupling:
                print " optimizing last node"
                self.icoords[n].smag = self.optimize(n,opt_steps,opt_type=0) #non-constrained optimization  

    @property
    def stage(self):
        return self._stage

    @stage.setter
    def stage(self,value):
        if value>2:
            raise ValueError("Stage can only be 0==Growing,1==Climbing, or 2==Finding")
        print("setting stage value to %i" %value)
        self._stage=value

    def set_stage(self,totalgrad,ts_cgradq,ts_gradrms,fp,dE_iter,nclimb):
        if (fp>0 and ((totalgrad<0.2 and ts_gradrms<self.CONV_TOL*10) or
                (totalgrad<0.3 and dE_iter<0.1 and ts_gradrms<self.CONV_TOL*2.5))):
            if self.stage==0 and self.climber:
                print(" ** starting climb **")
                self.stage=1
                return nclimb
        if (self.stage==1 and self.finder and dE_iter<4. and nclimb<1 and
                ((totalgrad<0.2 and ts_gradrms<self.CONV_TOL*10. and ts_cgradq<0.01) or
                (totalgrad<0.1 and ts_gradrms<self.CONV_TOL*10. and ts_cgradq<0.02) or
                (ts_gradrms<self.CONV_TOL*5.))
                ):
            print(" ** starting exact climb **")
            print " totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq)
            self.stage=2
            self.get_eigenv_finite(self.TSnode)
        if self.stage==1: #TODO this doesn't do anything
            nclimb-=1
        return nclimb

    def interpolateR(self,newnodes=1):
        print " Adding reactant node"
        success= True
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.icoords[self.nR] =self.add_node(self.nR-1,self.nR,self.nnodes-self.nP)
            if self.icoords[self.nR]==0:
                success= False
                break
            print " getting energy for  node ",self.nR
            self.icoords[self.nR].energy = self.icoords[self.nR].PES.get_energy(self.icoords[self.nR].geom)
            print self.icoords[self.nR].energy
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
            self.icoords[-self.nP-1] = self.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nR-1)
            self.nn+=1
            self.nP+=1
            print " nn=%i,nR=%i" %(self.nn,self.nR)
            self.active[-self.nP] = True

    def ic_reparam(self,ic_reparam_steps=8,n0=0,nconstraints=1,rtype=0):
        num_ics = self.icoords[0].num_ics
        len_d = self.icoords[0].nicd
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
            print " totaldqmag = %1.3f" %totaldqmag
            dqavg = totaldqmag/(self.nnodes-1)

            #if climb:
            if self.stage>0 or rtype==2:
                h1dqmag = np.sum(self.dqmaga[1:self.TSnode+1])
                h2dqmag = np.sum(self.dqmaga[self.TSnode+1:self.nnodes])
                if self.newic.print_level>1:
                    print " h1dqmag, h2dqmag: %1.1f %1.1f" % (h1dqmag,h2dqmag)
           
            # => Using average <= #
            if i==0 and rtype==0:
                print " using average"
                if self.stage==0:
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

            if self.stage==0 and rtype!=2:
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
                    rpmove[n]
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
                if n+1 != self.TSnode or self.stage==0:
                    rpmove[n+1] += rpmove[n]
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            if self.stage>0 or rtype==2:
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
                self.newic.set_xyz(self.icoords[n].coords) 
                self.newic.update_ics()

                if rpmove[n] < 0.:
                    ictan[n] = np.copy(ictan0[n]) 
                else:
                    ictan[n] = np.copy(ictan0[n+1]) 

                self.newic.form_constrained_DLC(ictan[n])
                dq = np.zeros((self.newic.nicd,1),dtype=float)
                dq[-1] = rpmove[n]
                self.newic.ic_to_xyz(dq)
                self.icoords[n].set_xyz(self.newic.coords)
                self.icoords[n].update_ics()

                #TODO might need to recalculate energy here for seam? 

        print ' spacings (end ic_reparam, steps: {}:'.format(ic_reparam_steps)
        for n in range(1,self.nnodes):
            print " {:1.2}".format(self.dqmaga[n]),
        print
        print "  disprms: {:1.3}\n".format(disprms)

    def ic_reparam_g(self,ic_reparam_steps=8,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        """size_ic = self.icoords[0].num_ics; len_d = self.icoords[0].nicd"""

        #close_dist_fix(0) #done here in GString line 3427.

        #print '**************************************************'
        #print '***************in ic_reparam_g********************'
        #print '**************************************************'

        num_ics = self.icoords[0].num_ics
        len_d = self.icoords[0].nicd
        
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
            #print 'on ic_reparam step',i
            self.get_tangents_1g()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
            if self.icoords[0].print_level>1:
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
                    if self.icoords[0].print_level>1:
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
            if self.icoords[0].print_level>1:
                for n in range(n0+1,self.nnodes-1):
                    print " disp[{}]: {:1.2f}".format(n,rpmove[n]),
                print
                print " disprms: {:1.3}\n".format(disprms)

            if disprms < 1e-2:
                break

            #TODO check how range is defined in gstring, uses n0...
            for n in range(n0+1,self.nnodes-1):
                if isinstance(self.icoords[n],DLC):
                    if rpmove[n] > 0:
                        #print "May need to make copy_CI"
                        #This does something to ictan0
                        self.icoords[n].update_ics()
                        self.icoords[n].bmatp = self.icoords[n].bmatp_create()
                        self.icoords[n].bmatp_to_U()
                        self.icoords[n].opt_constraint(self.ictan[n])
                        self.icoords[n].bmat_create()
                        dq0 = np.zeros((self.icoords[n].nicd,1))
                        dq0[self.icoords[n].nicd-nconstraints] = rpmove[n]
                        if self.icoords[0].print_level>1:
                            print " dq0[constraint]: {:1.3}".format(float(dq0[self.icoords[n].nicd-nconstraints]))
                        self.icoords[n].ic_to_xyz(dq0)
                        self.icoords[n].update_ics()
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
        #self.icoords[en].form_constrained_DLC(self.ictan[en])
        self.icoords[en].form_unconstrained_DLC()

        self.newic.set_xyz(self.icoords[en].coords)
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

        self.newic.set_xyz(self.icoords[en-1].coords)
        self.newic.bmatp_create()
        self.newic.bmat_create()
        qm1 = self.newic.q[nicd-1]
        #print "qm1 is %1.3f " %qm1

        if en+1<self.nnodes:
            self.newic.set_xyz(self.icoords[en+1].coords)
            self.newic.bmatp_create()
            self.newic.bmat_create()
            qp1 = self.newic.q[nicd-1]
        else:
            qp1 = qm1

        #print "qp1 is %1.3f" % qp1

        if self.icoords[en].isTSnode:
            print " TS Hess init'd w/ existing Hintp"

        self.newic.set_xyz(self.icoords[en].coords)
        self.newic.form_unconstrained_DLC()
        self.newic.Hintp=np.copy(self.icoords[en].Hintp)
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
        self.icoords[en].Hint = np.copy(self.newic.Hint)
        #print "Hint"
        #with np.printoptions(threshold=np.inf):
        #    print self.icoords[en].Hint
        #print "shape of Hint is %s" % (np.shape(self.icoords[en].Hint),)

        #this can also work on non-TS nodes?
        self.icoords[en].newHess = 5

        eigen,tmph = np.linalg.eigh(self.icoords[en].Hint) #nicd,nicd
        #print "eigenvalues of new Hess"
        #print eigen

    def set_V0(self):
        raise NotImplementedError 

    def set_opt_type(self,n):
        #TODO
        opts={
        -1 : 'no optimization',
         0 : 'non-constrained optimization',
         1 : 'ictan constraint optimization',
         2 : 'ictan constrained opt with climb',
         3 : 'non-TS eigenvector follow',
         4 : 'TS eigenvector follow',
         5 : 'MECI optimization',
         6 : 'constrained CI optimization',
         7 : 'constrained CI optimization with climb',
         }
        opt_type=1 
        if self.stage==1 and self.icoords[n].isTSnode==True:
            opt_type=2 #climb
        #if self.stage==1 and n!=self.nmax: #turning off, OG gsm does, but it's useleess
        #    opt_type==3
        if self.stage>1 and self.icoords[n].isTSnode==True:
            opt_type=4 #eigenvector follow
        if self.icoords[n].PES.lot.do_coupling==True:
            opt_type=6
        if self.stage==1 and self.icoords[n].isTSnode==True and opt_type==5:
            opt_type=7
        print(" setting node %i opt_type to %s (%i)" %(n,opts[opt_type],opt_type))

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

