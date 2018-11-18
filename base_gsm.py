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
            key='isSSM',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify SSM or DSM')

        opt.add_option(
            key='driving_coords',
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


        opt.add_option(
                key='tstype',
                value=1,
                required=False,
                allowed_types=[int],
                doc='0==any,1 delta bond, 2==turning of climbing image and TS search'
                )

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

        # Cache attributes
        self.optCG = False #TODO
        self.isTSnode =False
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        self.isSSM = self.options['isSSM']
        self.isMAP_SE = self.options['isMAP_SE']
        self.active = [False] * self.nnodes
        self.active[0] = False
        self.active[-1] = False
        self.driving_coords = self.options['driving_coords']
        self.nconstraints = self.options['nconstraints']
        self.CONV_TOL = self.options['CONV_TOL']
        self.ADD_NODE_TOL = self.options['ADD_NODE_TOL']

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

    def store_energies(self):
        for i,ico in enumerate(self.icoords):
            if ico != 0:
                self.energies[i] = ico.energy - self.icoords[0].energy

    def optimize(self,n=0,nsteps=100,nconstraints=0):
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules=[]
        #opt_molecules.append(obconversion.WriteString(self.icoords[n].mol.OBMol))
        assert self.icoords[n]!=0,"icoord not set"
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
   
        if self.icoords[n].print_level>0:
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
        #if self.icoords[n].print_level>0:
        if True:
            print "Final energy is %2.5f" % (self.icoords[n].energy)
        return smag

    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1):
        print "*********************************************************************"
        print "************************** in opt_iters *****************************"
        print "*********************************************************************"
        for i in range(1,self.nnodes-1):
            self.active[i] = True
        for oi in range(max_iter):
            sys.stdout.flush()
            self.get_tangents_1e() #Try get_tangents_1e here
            self.opt_steps(optsteps,nconstraints)
            for i,ico in zip(range(1,self.nnodes-1),self.icoords[1:self.nnodes-1]):
                if ico.gradrms < self.CONV_TOL:
                    self.active[i] = False
            totalgrad = 0.0
            gradrms = 0.0
            print " rn3m6: {:1.4}".format(self.rn3m6)
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

            #TODO stuff with path_overlap/path_overlapn
            #TODO Stuff with find and climb
            #TODO Stuff with find_peaks            
            print "convergence criteria: totalgrad < ",self.CONV_TOL*(self.nnodes-2)*self.rn3m6*5
            print " opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.1}".format(oi,float(totalgrad),float(gradrms),float(self.emax))
            #also prints tgrads and jobGradCount
            if totalgrad < self.CONV_TOL*(self.nnodes-2)*self.rn3m6*5:
                print " String is converged"
                print " String convergence criteria is {:1.5}".format(self.CONV_TOL*(self.nnodes-2)*self.rn3m6*5)
                print " Printing string to opt_converged_000.xyz"
                self.write_xyz_files(base='opt_converged',iters=0,nconstraints=nconstraints)
                sys.stdout.flush()
                return
            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)

    def get_tangents_1(self,n0=0):
        size_ic = self.icoords[0].num_ics
        nbonds = self.icoords[0].BObj.nbonds
        nangles = self.icoords[0].AObj.nangles
        ntor = self.icoords[0].TObj.ntor
        ictan = np.zeros((self.nnodes+1,size_ic))
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes+1,self.nnodes))
        for n in range(n0+1,self.nnodes):
            ictan[n] = np.transpose(DLC.tangent_1(self.icoords[n],self.icoords[n-1]))
            dqmaga[n] = 0.
            ictan0 = np.reshape(np.copy(ictan[n]),(size_ic,1))
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
        ictan = np.zeros((self.nnodes,size_ic))
        ictan0 = np.copy(ictan[0])
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes,self.nnodes))

        self.store_energies()
        
        print " V_profile: ",
        for n in range(self.nnodes):
            print " {:7.3f}".format(float(self.energies[n])),
        print
        
        TSnode = self.nmax
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
                if n < TSnode:
                    intic_n = n
                    newic_n = n+1
                elif n> TSnode:
                    intic_n = n-1
                    newic_n = n
                else:
                    do3 = True
                    newic_n = n
                    intic_n = n+1
                    int2ic_n = n-1
            if not do3:
                ictan[n,:]=np.transpose(self.tangent(newic_n,intic_n))
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
                ictan[n,:] = f1*t1 + (1-f1)*t2
            
            dqmaga[n]=0.0
            ictan0 = np.reshape(np.copy(ictan[n]),(size_ic,1))
            self.icoords[newic_n].bmatp = self.icoords[newic_n].bmatp_create()
            self.icoords[newic_n].bmatp_to_U()
            self.icoords[newic_n].opt_constraint(ictan0)
            dqmaga[n] += np.dot(ictan0[:nbonds].T,self.icoords[newic_n].Ut[-1,:nbonds])*2.5
            dqmaga[n] += np.dot(ictan0[nbonds:].T,self.icoords[newic_n].Ut[-1,nbonds:])
            dqmaga[n] = float(dqmaga[n])
        

        print '------------printing ictan[:]-------------'
        for row in ictan:
            print row
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
            
            print " gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.1}\n".format(n,float(totalgrad),float(gradrms),float(self.emax))
            self.write_xyz_files(iters=n,base='growth_iters',nconstraints=nconstraints)

    def opt_steps(self,maxopt,nconstraints):
        for i in range(1):
            for n in range(self.nnodes):
                if self.icoords[n] != 0 and self.active[n]==True:
                    print "optimizing node %i" % n
                    self.icoords[n].opt_constraint(self.ictan[n])
                    self.icoords[n].smag = self.optimize(n,maxopt,nconstraints)

    #def grow_string(self,maxiters=20):
    #    self.write_node_xyz("nodes_xyz_file0.xyz")
    #    iters = 1
    #    while True:
    #        print "beginning iteration:",iters
    #        sys.stdout.flush()
    #        do_growth = False
    #        for act in self.active:
    #            if act:
    #                do_growth = True
    #                break
    #        if do_growth:
    #            self.growth_iters(nconstraints=1,current=iters)
    #            sys.stdout.flush()
    #        else:
    #            print 'All nodes added. String done growing'
    #            break
    #        iters += 1
    #        if iters > maxiters:
    #            raise ValueError("reached max number of growth iterations")
    #    self.write_node_xyz()

    def interpolateR(self,newnodes=1):
        print "Adding reactant node"
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.icoords[self.nR] =self.add_node(self.nR-1,self.nR,self.nnodes-self.nP)
            self.nn+=1
            self.nR+=1
            print "nn=%i,nR=%i" %(self.nn,self.nR)
            self.active[self.nR-1] = True

    def interpolateP(self,newnodes=1):
        print "interpolateP"
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            self.icoords[-self.nP-1] = self.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nR-1)
            self.nn+=1
            self.nP+=1
            print "nn=%i,nR=%i" %(self.nn,self.nR)
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
        TSnode = self.nmax
        print "TSnode: {} Emax: {:4.1}".format(TSnode,float(self.emax))
        
        for i in range(ic_reparam_steps):
            self.get_tangents_1(n0=n0)
            totaldqmag = 0.
            for n in range(n0+1,self.nnodes):
                totaldqmag += self.dqmaga[n]
            dqavg = totaldqmag/(self.nnodes-1)
            if self.climb:
                for n in range(n0+1,TSnode+1):
                    h1dqmag += self.dqmaga[n]
                for n in range(TSnode+1,self.nnodes):
                    h2dqmag += self.dqmaga[n]
            
            if i==0 and rtype==0:
                if not self.climb:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(TSnode-n0)
                else:
                    for n in range(n0+1,TSnode):
                        rpart[n] = 1./(TSnode-n0)
                    for n in range(TSnode+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-TSnode-1)
                    rpart[TSnode]=0.

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

            if not self.climb and rtype !=2:
                for n in range(n0+1,self.nnodes-1):
                    deltadq = self.dqmaga[n] - totaldqmag * rpart[n]
                    if n==self.nnodes-2:
                        deltadq += totaldqmag * rpart[n] - self.dqmaga[n+1]
                    rpmove[n] = -deltadq
            else:
                    deltadq = 0.
                    rpmove[TSnode] = 0.
                    for n in range(n0+1,TSnode):
                        deltadq = self.dqmaga[n] - h1dqmag * rpart[n]
                        if n==self.nnodes-2:
                            deltadq += h2dqmag * rpart[n] - self.dqmaga[n+1]
                        rpmove[n]
                    for n in range(TSnode+1,self.nnodes-1):
                        deltadq = self.dqmaga[n] - h2dqmag * rpart[n]
                        if n==self.nnodes-2:
                            deltadq += h2dqmag * rpart[n] - self.dqmaga[n+1]
                        rpmove[n] = -deltadq
            MAXRE = 0.5

            for n in range(n0+1,self.nnodes):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            for n in range(n0+1,self.nnodes-2):
                if n+1 != TSnode or not self.climb:
                    rpmove[n+1] += rpmove[n]
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            if self.climb or rtype==2:
                rpmove[TSnode] = 0.
            for n in range(n0+1,self.nnodes-1):
                print " disp[{}]: {:1.2}".format(n,rpmove[n]),
            print
            
            disprms = np.linalg.norm(rpmove[n0+1:self.nnodes-1])
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
                ictan0 = np.reshape(np.copy(ictan[n]),(num_ics,1))
                self.icoords[n].opt_constraint(ictan0) #3815
            
            print ' spacings (end ic_reparam, steps: {}:'.format(ic_reparam_steps)
            for n in range(self.nnodes):
                print " {:1.2}".format(self.dqmaga[n]),
            print


    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,nconstraints=1):  #see line 3863 of gstring.cpp
        """size_ic = self.icoords[0].num_ics; len_d = self.icoords[0].nicd"""

        #close_dist_fix(0) #done here in GString line 3427.

        print '**************************************************'
        print '***************in ic_reparam_g********************'
        print '**************************************************'

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
        
        TSnode = -1 # What is this?
        emax = -1000 # And this?

        for i in range(ic_reparam_steps):
            print 'on ic_reparam step',i
            self.get_tangents_1g()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
#            totaldqmag = 0.
#            for n in range(n0,self.nR-1):
#                totaldqmag += self.dqmaga[n]
#            for n in range(self.nnodes-self.nP+1,self.nnodes):
#                totaldqmag += self.dqmaga[n]
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

            for n in range(n0+1,self.nnodes-1):
                print " disp[{}]: {:1.2f}".format(n,rpmove[n]),
            print

            disprms = float(np.linalg.norm(rpmove[n0+1:self.nnodes-1]))
            lastdispr = disprms
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
                            print ' dq0:',
                            for iii in dq0:
                                print iii,
                            print
                            print " dq0[constraint]: {:1.3}".format(float(dq0[self.icoords[n].nicd-nconstraints]))
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
