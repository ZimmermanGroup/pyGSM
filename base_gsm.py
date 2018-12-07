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


        opt.add_option(
                key='tstype',
                value=0,
                required=False,
                allowed_types=[int],
                doc='0==Find and Climb TS,1 Climb with no exact find, 2==turning of climbing image and TS search'
                )

        opt.add_option(
                key="last_node_fixed",
                value=True,
                required=False,
                doc="Fix last node?"
                )

        Base_Method._default_options = opt
        return Base_Method._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return Base_Method(Base_Method.default_options().set_values(kwargs))

    def restart_string(self,xyzbase='restart'):#,nR,nP):
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
        self.icoords[0].energy = self.icoords[0].V0 = self.icoords[0].PES.get_energy(self.icoords[0].geom)
        self.icoords[0].gradrms=grmss[0]

        print "initial energy is %3.4f" % self.icoords[0].energy
        for struct in range(1,nstructs):
            self.icoords[struct] = DLC.copy_node(self.icoords[0],struct)
            self.icoords[struct].set_xyz(coords[struct])
            self.icoords[struct].update_ics()
            self.icoords[struct].setup()
            self.icoords[struct].DMAX=0.05 #half of default value
            self.icoords[struct].energy =self.icoords[struct].V0 = self.icoords[0].energy +energy[struct]
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

        # Cache attributes
        self.optCG = False #TODO
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        self.active = [False] * self.nnodes
        #self.active[0] = False
        #self.active[-1] = False
        self.driving_coords = self.options['driving_coords']
        self.nconstraints = self.options['nconstraints']
        self.CONV_TOL = self.options['CONV_TOL']
        self.ADD_NODE_TOL = self.options['ADD_NODE_TOL']
        self.last_node_fixed = self.options['last_node_fixed']
        self.tstype=self.options['tstype']
        self.climber=False  #is this string a climber?
        self.finder=False   # is this string a finder?
        self.isRestarted=False
        if self.tstype==0:
            print("set climber and finder to True")
            self.climber=True
            self.finder=True
        elif self.tstype==1:
            print("setting climber to True")
            self.climber=True
        else:
            print(" Turning off climbing image and exact TS search")

        # Set initial values
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.energies = np.asarray([0.]*self.nnodes)
        self.emax = float(max(self.energies))
        self.nmax = 0 
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

    def store_energies(self):
        for i,ico in enumerate(self.icoords):
            if ico != 0:
                self.energies[i] = ico.energy - self.icoords[0].energy

    
    def update_DLC_obj(self,n,nconstraints):
         # => update DLCs <= #
         if self.icoords[n].PES.lot.do_coupling is False:
             if nconstraints==0:
                 self.icoords[n].form_unconstrained_DLC()
             else:
                 constraints=self.ictan[n]
                 self.icoords[n].form_constrained_DLC(constraints)
         else:
             if nconstraints==2:
                 self.icoords[n].form_CI_DLC()
             elif nconstraints==3:
                 raise NotImplemented

    def optimize(self,n=0,nsteps=100,nconstraints=0,ictan=None,fixed_DLCs=[],follow_overlap=False):
        assert len(fixed_DLCs)==nconstraints, "nconstraints != fixed_DLC"
        assert self.icoords[n]!=0,"icoord not set"
        if nconstraints==1 and ictan.any()==None:
            print "warning no ictan"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
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
        self.icoords[n].node_id = n  # set node id is this necessary?
        if self.icoords[n].print_level>0:
            print "Initial energy is %1.4f" % self.icoords[n].V0

        for step in range(nsteps):
            if self.icoords[n].print_level>1:
                print(" \nOpt step: %i" %(step+1)),
            if step==0:
                self.icoords[n].buf.write(" \nOpt step: %d" %(step+1))
            else:
                self.icoords[n].buf.write("\n Opt step: %d" %(step+1))

            # => Update DLC obj <= #
            self.update_DLC_obj(n,nconstraints)

            ######### => form constraint steps <= ###########
            constraint_steps=[0]*nconstraints
            # => step no climb
            if all(dlc_fix==True for dlc_fix in fixed_DLCs):
                pass
            # => step with climb
            elif self.icoords[n].PES.lot.do_coupling is False and fixed_DLCs==[False] and follow_overlap==False:
                constraint_steps[0]=self.icoords[n].walk_up(self.icoords[n].nicd-1)
            # => MECI step
            elif self.icoords[n].PES.lot.do_coupling is True and fixed_DLCs==[True,False]:
                constraint_steps[1] = self.icoords[n].dgrad_step()
            # => seam step
            elif self.icoords[n].PES.lot.do_coupling is True and fixed_DLCs==[True,True,False]:
                constraint_steps[2] = self.icoords[n].dgrad_step()
            # => seam climb
            elif self.icoords[n].PES.lot.do_coupling is True and fixed_DLCs==[True,False,False]:
                constraint_steps[1] = self.icoords[n].dgrad_step()
                constraint_steps[2]=self.icoords[n].walk_up(self.icoords[n].nicd-1)
            else:
                raise ValueError(" Optimize doesn't know what to do ")

            ########### => Opt step <= ############
            smag =self.icoords[n].opt_step(nconstraints,constraint_steps,ictan,follow_overlap)

            # convergence quantities
            grmss.append(float(self.icoords[n].gradrms))
            steps.append(smag)
            energies.append(self.icoords[n].energy-self.icoords[n].V0)
            opt_molecules.append(obconversion.WriteString(self.icoords[n].mol.OBMol))
            if isinstance(self.icoords[n].PES,Penalty_PES):
                deltaEs.append(self.icoords[n].PES.dE)
    
            #write convergence
            self.write_node(n,opt_molecules,energies,grmss,steps,deltaEs)
   
            #TODO convergence 
            sys.stdout.flush()
            if self.icoords[n].gradrms<self.CONV_TOL:
                break

        #TODO if gradrms is greater than gradrmsl than further reduce DMAX
        #TODO change how revertopt is done is opt_step?

        print(self.icoords[n].buf.getvalue())
        if self.icoords[n].print_level>0:
            print "Final energy is %2.5f" % (self.icoords[n].energy)
        return smag


    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1):
        print "*********************************************************************"
        print "************************** in opt_iters *****************************"
        print "*********************************************************************"

        print "beginning opt iters" 
        print "convergence criteria 1: totalgrad < ",self.CONV_TOL*(self.nnodes-2)*self.rn3m6*5
        nclimb=0

        #for i in range(1,self.nnodes-1):
        #    self.active[i] = True

        for oi in range(max_iter):
            sys.stdout.flush()

            # => Get all tangents 3-way <= #
            self.get_tangents_1e()

            # => do opt steps <= #
            self.opt_steps(optsteps,nconstraints)
            self.store_energies()
            print " V_profile: ",
            for n in range(self.nnodes):
                print " {:7.3f}".format(float(self.energies[n])),
            print
        

            # => Turn off converged nodes for next round? <= #
            #for i,ico in zip(range(1,self.nnodes-1),self.icoords[1:self.nnodes-1]):
            #    if ico.gradrms < self.CONV_TOL:
            #        self.active[i] = False

            # => calculate totalgrad <= #
            totalgrad = 0.0
            gradrms = 0.0
            for i,ico in zip(range(1,self.nnodes-1),self.icoords[1:self.nnodes-1]):
                print " node: {:2} gradrms: {:1.4}".format(i,float(ico.gradrms)),
                if i%5 == 0:
                    print
                totalgrad += ico.gradrms*self.rn3m6
                gradrms += ico.gradrms*ico.gradrms
            print

            gradrms = np.sqrt(gradrms/(self.nnodes-2))
            self.emaxp = self.emax
            self.emax = float(max(self.energies[1:-1]))
            self.nmax = np.where(self.energies==self.emax)[0][0]

            if self.stage==1: print("c")
            elif self.stage==2: print("x")

            #TODO stuff with path_overlap/path_overlapn #TODO need to save path_overlap
            
            print " opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}".format(oi,float(totalgrad),float(gradrms),float(self.emax))

            fp = self.find_peaks(2)

            # => set stage <= #
            ts_cgradq=abs(self.icoords[self.nmax].gradq[self.icoords[self.nmax].nicd-1])
            ts_gradrms=self.icoords[self.nmax].gradrms
            dE_iter=abs(self.emax-self.emaxp)
            nclimb,form_eigenv_finite = self.set_stage(totalgrad,ts_cgradq,ts_gradrms,fp,dE_iter,nclimb)

            #TODO resetting

            #TODO special SSM criteria if TSNode is second to last node
            #TODO special SSM criteria if first opt'd node is too high?
            isDone = self.check_opt(totalgrad,fp)


            #TODO put in de-gsm
            ##if totalgrad < self.CONV_TOL*(self.nnodes-2)*self.rn3m6*5:
            #if self.icoords[self.TSnode].gradrms<self.CONV_TOL: #TODO should check totalgrad
            #    break
            #if totalgrad<0.1 and self.icoords[self.TSnode].gradrms<2.5*self.CONV_TOL: #TODO extra crit here
            #    break

            if isDone:
                break
            if not self.climber and not self.finder and totalgrad<0.025: #Break even if not climb/find
                break

            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)
            if form_eigenv_finite==True:
                self.get_eigenv_finite(self.nmax)

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
        print "getting tangents for nodes 0 to ",self.nnodes
        for n in range(n0+1,self.nnodes):
            #ictan[n] = np.transpose(DLC.tangent_1(self.icoords[n],self.icoords[n-1]))
            #print "getting tangent between %i %i" % (n,n-1)
            assert self.icoords[n]!=0,"n is bad"
            assert self.icoords[n-1]!=0,"n-1 is bad"
            ictan[n] = DLC.tangent_1(self.icoords[n],self.icoords[n-1])
            dqmaga[n] = 0.
            ictan0= np.copy(ictan[n])
            self.icoords[n].bmatp = self.icoords[n].bmatp_create()
            self.icoords[n].bmatp_to_U()
            self.icoords[n].opt_constraint(ictan0)
            dqmaga[n] += np.dot(ictan0[:nbonds].T,self.icoords[n].Ut[-1,:nbonds])*2.5
            dqmaga[n] += np.dot(ictan0[nbonds:].T,self.icoords[n].Ut[-1,nbonds:])
            dqmaga[n] = float(np.sqrt(dqmaga[n]))
        
        self.dqmaga = dqmaga
        self.ictan = ictan

    def get_tangents_1e(self,n0=0):
        size_ic = self.icoords[0].num_ics
        nbonds = self.icoords[0].BObj.nbonds
        nangles = self.icoords[0].AObj.nangles
        ntor = self.icoords[0].TObj.ntor
        #ictan = np.zeros((self.nnodes,size_ic))

        ictan = [[]]*self.nnodes
        #ictan0 = np.copy(ictan[0])
        ictan0 = np.zeros((size_ic,1))
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes,self.nnodes))

        self.store_energies()
        
        self.TSnode = self.nmax
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
                #ictan[n,:]=np.transpose(self.tangent(newic_n,intic_n))
                ictan[n] = self.tangent(newic_n,intic_n)
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
                #ictan[n,:] = f1*t1 + (1-f1)*t2
                ictan[n] = f1*t1 +(1-f1)*t2
                ictan[n] = ictan[n].reshape((size_ic,1))
            
            dqmaga[n]=0.0
            ictan0 = np.reshape(np.copy(ictan[n]),(size_ic,1))
            self.icoords[newic_n].bmatp = self.icoords[newic_n].bmatp_create()
            self.icoords[newic_n].bmatp_to_U()
            self.icoords[newic_n].opt_constraint(ictan0)
            dqmaga[n] += np.dot(ictan0[:nbonds].T,self.icoords[newic_n].Ut[-1,:nbonds])*2.5
            dqmaga[n] += np.dot(ictan0[nbonds:].T,self.icoords[newic_n].Ut[-1,nbonds:])
            dqmaga[n] = float(dqmaga[n])
        

        #print '------------printing ictan[:]-------------'
        #for row in ictan:
        #    print row
        print '------------printing dqmaga---------------'
        print dqmaga
        self.dqmaga = dqmaga
        self.ictan = ictan

    def get_tangents_1g(self):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase
        """
        ictan = [[]]*self.nnodes
        dqmaga = [0.]*self.nnodes
        dqa = [[],]*self.nnodes

        ncurrent,nlist = self.make_nlist()

        for n in range(ncurrent):
            #ictan[nlist[2*n]] = DLC.tangent_1(self.icoords[nlist[2*n+1]],self.icoords[nlist[2*n+0]])
            ictan[nlist[2*n]] = self.tangent(nlist[2*n],nlist[2*n+1])
            ictan0 = np.copy(ictan[nlist[2*n]])

            if self.icoords[nlist[2*n+1]].print_level>1:
                print "forming space for", nlist[2*n+1]
            self.icoords[nlist[2*n+1]].bmatp = self.icoords[nlist[2*n+1]].bmatp_create()
            self.icoords[nlist[2*n+1]].bmatp_to_U()
            self.icoords[nlist[2*n+1]].opt_constraint(ictan[nlist[2*n]])
            self.icoords[nlist[2*n+1]].bmat_create()
            
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,self.icoords[nlist[2*n+1]].Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))

        self.dqmaga = dqmaga
        self.ictan = ictan
       
        if False:
            for n in range(ncurrent):
                print "dqmag[%i] =%1.2f" %(nlist[2*n],self.dqmaga[nlist[2*n]])
                print "printing ictan[%i]" %nlist[2*n]       
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % ictan[nlist[2*n]][i],
                print 
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds,self.icoords[nlist[2*n]].AObj.nangles+self.icoords[nlist[2*n]].BObj.nbonds):
                    print "%1.2f " % ictan[nlist[2*n]][i],
                for i in range(self.icoords[nlist[2*n]].BObj.nbonds+self.icoords[nlist[2*n]].AObj.nangles,self.icoords[nlist[2*n]].AObj.nangles+self.icoords[nlist[2*n]].BObj.nbonds+self.icoords[nlist[2*n]].TObj.ntor):
                    print "%1.2f " % ictan[nlist[2*n]][i],
                print "\n"
#           #     print np.transpose(ictan[nlist[2*n]])


    def growth_iters(self,iters=1,maxopt=1,nconstraints=1,current=0):
        print ''
        print "*********************************************************************"
        print "************************ in growth_iters ****************************"
        print "*********************************************************************"
        for n in range(iters):
            sys.stdout.flush()
            self.check_add_node()
            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.get_tangents_1g()
            self.ic_reparam_g()
            self.get_tangents_1g()
            self.opt_steps(maxopt,nconstraints)
            self.store_energies()

            isDone = self.check_if_grown()
            if isDone:
                print "is Done growing"
                break

            totalgrad = 0.0
            gradrms = 0.0
            self.emaxp = self.emax            
            for ico in self.icoords:
                if ico != 0:
                    totalgrad += ico.gradrms*self.rn3m6
                    gradrms += ico.gradrms*ico.gradrms
            gradrms = np.sqrt(gradrms/(self.nnodes-2))
            self.emax = float(max(self.energies[1:-1]))
            self.nmax = np.where(self.energies==self.emax)[0][0]
            
            print " gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n".format(n,float(totalgrad),float(gradrms),float(self.emax))
            self.write_xyz_files(iters=n,base='growth_iters',nconstraints=nconstraints)

    def opt_steps(self,opt_steps,nconstraints):
        assert nconstraints>0,"opt steps doesn't work without constraints"
        if self.icoords[0].PES.lot.do_coupling==True: 
            assert nconstraint>2,"nconstraints wrong size"

        for n in range(self.nnodes):
            self.icoords[n].isTSnode=False
        fp=0
        if self.stage>0:
            fp = self.find_peaks(2)
        if fp>0:
            nmax = np.argmax(self.energies)
            self.icoords[nmax].isTSnode=True

        optlastnode=False
        if self.last_node_fixed==False:
            if self.energies[self.nnodes-1]>self.energies[self.nnodes-2] and fp>0:
                optlastnode=True

        for n in range(self.nnodes):
            if self.icoords[n] != 0 and self.active[n]==True:
                print " Optimizing node %i" % n
                fixed_DLC = [True]*nconstraints

                exsteps=1 #multiplier for nodes near the TS node
                if self.stage==2 and self.energies[n]+1.5 > self.energies[self.TSnode] and n!=self.TSnode:
                    exsteps=2
                
                # => do constrained optimization
                if self.stage<2 and not self.icoords[n].isTSnode:
                    # => do CI constrained optimization
                    if self.icoords[n].PES.lot.do_coupling:
                        fixed_DLC=[True,False,True]
                    self.icoords[n].smag = self.optimize(n,opt_steps*exsteps,nconstraints,fixed_DLC)

                # => do constrained optimization with climb
                elif self.stage==1 and self.icoords[n].isTSnode:
                    fixed_DLC=[False]
                    # => do constrained seam optimization with climb
                    if self.icoords[n].PES.lot.do_coupling==True:
                        fixed_DLC=[True,False,False]
                    self.icoords[n].smag = self.optimize(n,opt_steps*exsteps,nconstraints,fixed_DLC)

                # => follow maximum overlap with Hessian for TS node if find <= #
                elif self.stage==2 and self.icoords[n].isTSnode:
                    #self.optimize_TS_exact(n,opt_steps,nconstraints)
                    self.optimize(n,opt_steps,nconstraints,fixed_DLC,follow_overlap=True)

            if optlastnode==True and n==self.nnodes-1 and not self.icoords[n].PES.lot.do_coupling:
                self.icoords[n].smag = self.optimize(n,opt_steps,nconstraints=0)

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
        form_eigenv_finite=False
        if totalgrad<0.3 and fp>0:
            if self.stage==0 and self.climber:
                print(" ** starting climb **")
                self.stage=1
                return nclimb,form_eigenv_finite
            if (self.stage==1 and self.finder and dE_iter<4. and nclimb<1 and
                    ((totalgrad<0.2 and ts_gradrms<self.CONV_TOL*10. and ts_cgradq<0.01) or
                    (totalgrad<0.1 and ts_gradrms<self.CONV_TOL*10. and ts_cgradq<0.02) or
                    (ts_gradrms<self.CONV_TOL*5.))
                    ):
                print(" ** starting exact climb **")
                self.stage=2
                form_eigenv_finite=True
            if self.stage==1:
                nclimb-=1
        return nclimb,form_eigenv_finite

    def interpolateR(self,newnodes=1):
        print " Adding reactant node"
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.icoords[self.nR] =self.add_node(self.nR-1,self.nR,self.nnodes-self.nP)
            self.nn+=1
            self.nR+=1
            print " nn=%i,nR=%i" %(self.nn,self.nR)
            self.active[self.nR-1] = True

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

    def ic_reparam(self,ic_reparam_steps=4,n0=0,nconstraints=1,rtype=0):
        num_ics = self.icoords[0].num_ics
        len_d = self.icoords[0].nicd
        ictalloc = self.nnodes+1

        ictan0 = np.zeros((ictalloc,num_ics))
        ictan = np.zeros((ictalloc,num_ics))
        
        rpmove = np.zeros(ictalloc)
        rpart = np.zeros(ictalloc)
        totaldqmag = 0.0
        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(ictalloc)
        edist = np.zeros(ictalloc)
        
        self.emax = float(max(self.energies[1:-1]))
        self.nmax = np.where(self.energies==self.emax)[0][0]
        self.TSnode = self.nmax
        print "TSnode: {} Emax: {:4.5}".format(self.TSnode,float(self.emax))
        
        for i in range(ic_reparam_steps):
            self.get_tangents_1(n0=n0)
            ictan0 = np.copy(self.ictan)

            print " printing spacings dqmaga:"
            for n in range(1,self.nnodes):
                print " %1.2f" % self.dqmaga[n], 
            print 

            totaldqmag = 0.
            for n in range(n0+1,self.nnodes):
                totaldqmag += self.dqmaga[n]
            print " totaldqmag = %1.3f" %totaldqmag
            dqavg = totaldqmag/(self.nnodes-1)

            #if climb:
            if self.stage==1 or rtype==2:
                for n in range(n0+1,self.TSnode+1):
                    h1dqmag += self.dqmaga[n]
                for n in range(self.TSnode+1,self.nnodes):
                    h2dqmag += self.dqmaga[n]
           
            # => Using average <= #
            if i==0 and rtype==0:
                print " using average"
                if self.stage!=1:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes)
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

            if self.stage!=1 and rtype!=2:
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
            for n in range(n0+1,self.nnodes):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            for n in range(n0+1,self.nnodes-2):
                if n+1 != self.TSnode or self.stage!=1:
                    rpmove[n+1] += rpmove[n]
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            if self.stage==1 or rtype==2:
                rpmove[self.TSnode] = 0.
            for n in range(n0+1,self.nnodes-1):
                print " disp[{}]: {:1.2}".format(n,rpmove[n]),
            print
            
            disprms = np.linalg.norm(rpmove[n0+1:self.nnodes-1])
            print " disprms: {:1.3}\n".format(disprms)
            lastdispr = disprms

            if disprms < 0.02:
                break
            for n in range(n0+1,self.nnodes-1):

                self.icoords[n].update_ics()
                self.icoords[n].bmatp=self.icoords[n].bmatp_create()
                self.icoords[n].bmatp_to_U()

                if rpmove[n] < 0.:
                    pass
                else:
                    self.ictan[n] = np.copy(ictan0[n+1]) 
                self.icoords[n].opt_constraint(self.ictan[n]) #3815
                self.icoords[n].bmat_create()
                dq = np.zeros((self.icoords[n].nicd,1),dtype=float)
                dq[-1] = rpmove[n]
                self.icoords[n].ic_to_xyz(dq)

                #TODO might need to recalculate energy here for seam? 

        print ' spacings (end ic_reparam, steps: {}:'.format(ic_reparam_steps)
        for n in range(self.nnodes):
            print " {:1.2}".format(self.dqmaga[n]),
        print
        print "  disprms: {:1.3}".format(disprms)

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
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
            if self.icoords[0].print_level>0:
                print " totaldqmag (without inner): {:1.2}\n".format(totaldqmag)
                print " printing spacings dqmaga: "
                for n in range(self.nnodes):
                    print " {:1.2}".format(self.dqmaga[n]),
                    if (n+1)%5==0:
                        print
                print 
            
            if i == 0:
                rpart = np.zeros(self.nnodes)
                for n in range(n0+1,self.nR):
                    rpart[n] = 1.0/(self.nn-2)
                for n in range(self.nnodes-self.nP,self.nnodes-1):
                    rpart[n] = 1.0/(self.nn-2)
#                rpart[0] = 0.0
#                rpart[-1] = 0.0
                if self.icoords[0].print_level>0:
                    print " rpart: "
                    for n in range(1,self.nnodes):
                        print " {:1.2}".format(rpart[n]),
                        if (n)%5==0:
                            print
                    print
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
            if self.icoords[0].print_level>0:
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
                        if len(self.ictan[n]) != 0:# and self.active[n]: #may need to change active requirement TODO
                            #print "May need to make copy_CI"
                            #This does something to ictan0
                            #self.icoords[n].update_ics()
                            self.icoords[n].bmatp = self.icoords[n].bmatp_create()
                            self.icoords[n].bmatp_to_U()
                            self.icoords[n].opt_constraint(self.ictan[n])
                            self.icoords[n].bmat_create()
                            dq0 = np.zeros((self.icoords[n].nicd,1))
                            dq0[self.icoords[n].nicd-nconstraints] = rpmove[n]
                            #print ' dq0:',
                            #for iii in dq0:
                            #    print iii,
                            #print
                            #print " dq0[constraint]: {:1.3}".format(float(dq0[self.icoords[n].nicd-nconstraints]))
                            self.icoords[n].ic_to_xyz(dq0)
                            self.icoords[n].update_ics()
                        else:
                            pass
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

        #self.icoords[en].form_constrained_DLC(self.ictan[en])
        self.icoords[en].form_unconstrained_DLC()

        newic = DLC.copy_node(self.icoords[en],1)
        newic.form_constrained_DLC(self.ictan[en])
        nicd = newic.nicd  #len
        num_ics = newic.num_ics #size_ic

        E0 = self.energies[en]/KCAL_MOL_PER_AU
        Em1 = self.energies[en-1]/KCAL_MOL_PER_AU
        if en+1<self.nnodes:
            Ep1 = self.energies[en+1]/KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        q0 =  newic.q[nicd-1]
        #print "q0 is %1.3f" % q0
        tan0 = newic.Ut[nicd-1,:]
        #print "tan0"
        #print tan0

        newic.set_xyz(self.icoords[en-1].coords)
        newic.bmatp_create()
        newic.bmat_create()
        qm1 = newic.q[nicd-1]
        #print "qm1 is %1.3f " %qm1

        if en+1<self.nnodes:
            newic.set_xyz(self.icoords[en+1].coords)
            newic.bmatp_create()
            newic.bmat_create()
            qp1 = newic.q[nicd-1]
        else:
            qp1 = qm1

        #print "qp1 is %1.3f" % qp1

        if self.icoords[en].isTSnode:
            print " TS Hess init'd w/ existing Hintp"

        newic.set_xyz(self.icoords[en].coords)
        newic.form_unconstrained_DLC()
        newic.Hintp=np.copy(self.icoords[en].Hintp)
        newic.Hint = newic.Hintp_to_Hint()

        tan = np.dot(newic.Ut,tan0.T)  #(nicd,numic)(num_ic,1)=nicd,1 
        #print "tan"
        #print tan

        Ht = np.dot(newic.Hint,tan) #nicd,1
        tHt = np.dot(tan.T,Ht) 

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        print " tHt %1.3f a: %1.1f b: %1.1f c: %1.3f" % (tHt,a[0],b[0],c[0])


        ttt = np.outer(tan,tan)
        #print "Hint before"
        #with np.printoptions(threshold=np.inf):
        #    print newic.Hint
        eig,tmph = np.linalg.eigh(newic.Hint)
        #print "initial eigenvalues"
        #print eig
       
        newic.Hint += (c-tHt)*ttt
        self.icoords[en].Hint = np.copy(newic.Hint)
        #print "Hint"
        #with np.printoptions(threshold=np.inf):
        #    print self.icoords[en].Hint
        #print "shape of Hint is %s" % (np.shape(self.icoords[en].Hint),)
        self.icoords[en].newHess = 5

        eigen,tmph = np.linalg.eigh(self.icoords[en].Hint) #nicd,nicd
        #print "eigenvalues of new Hess"
        #print eigen


if __name__ == '__main__':
#    from icoord import *
    ORCA=False
    QCHEM=True
    PYTC=False
    nproc=8

    if QCHEM:
        from qchem import *
    if ORCA:
        from orca import *
    if PYTC:
        from pytc import *
    import manage_xyz

    if True:
        filepath="tests/fluoroethene.xyz"
        filepath2="tests/stretched_fluoroethene.xyz"

    if True:
        filepath2="tests/SiH2H2.xyz"
        filepath="tests/SiH4.xyz"

    mol=pb.readfile("xyz",filepath).next()
    mol2=pb.readfile("xyz",filepath2).next()
    if QCHEM:
        lot=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
        lot2=QChem.from_options(states=[(1,0)],charge=0,basis='6-31g(d)',functional='B3LYP',nproc=nproc)
    
    if ORCA:
        lot=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
        lot2=Orca.from_options(states=[(1,0)],charge=0,basis='6-31+g(d)',functional='wB97X-D3',nproc=nproc)
    if PYTC:
        nocc=8
        nactive=2
        lot=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot.cas_from_file(filepath)
        lot2=PyTC.from_options(states=[(1,0)],nocc=nocc,nactive=nactive,basis='6-31gs')
        lot2.cas_from_file(filepath2)

    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    pes2 = PES.from_options(lot=lot2,ad_idx=0,multiplicity=1)

    print "\n IC1 \n"
    ic1=DLC.from_options(mol=mol,PES=pes)
    print "\n IC2 \n"
    ic2=DLC.from_options(mol=mol2,PES=pes2)

    if True:
        print "\n Starting GSM \n"
        gsm=GSM.from_options(ICoord1=ic1,ICoord2=ic2,nnodes=9,nconstraints=1)
        gsm.icoords[0].gradrms = 0.
#        gsm.icoords[-1].gradrms = 0.
        gsm.icoords[0].energy = gsm.icoords[0].PES.get_energy(gsm.icoords[0].geom)
#        gsm.icoords[-1].energy = gsm.icoords[-1].PES.get_energy(gsm.icoords[-1].geom)
        print 'gsm.icoords[0] E:',gsm.icoords[0].energy
        print 'gsm.icoords[-1]E:',gsm.icoords[-1].energy
        gsm.interpolate(2)

    if False:
        print DLC.tangent_1(gsm.icoords[0],gsm.icoords[-1])
    
    if False:
        gsm.get_tangents_1(n0=0)

    if True:
        gsm.ic_reparam_g()


    if True:
        gsm.grow_string(50)
        gsm.opt_iters()
        if ORCA:
            os.system('rm temporcarun/*')

    gsm.write_node_xyz("nodes_xyz_file1.xyz")
