import options
import numpy as np
import os
import openbabel as ob
import pybel as pb
from dlc import *
from copy import deepcopy
import StringIO
from _print_opt import *

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
        
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.isSSM = self.options['isSSM']
        self.isMAP_SE = self.options['isMAP_SE']
        self.active = [False] * self.nnodes
        self.active[0] = False
        self.active[-1] = False
        self.driving_coords = self.options['driving_coords']
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
            dqmaga[n] = np.sqrt(dqmaga[n])
        
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
        ncurrent = 0
        nlist = [0]*(2*self.nnodes)
        for n in range(self.nR-1):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n+1
            ncurrent += 1

        for n in range(self.nnodes-self.nP+1,self.nnodes):
            nlist[2*ncurrent] = n
            nlist[2*ncurrent+1] = n-1
            ncurrent += 1

        nlist[2*ncurrent] = self.nR -1
        nlist[2*ncurrent+1] = self.nnodes - self.nP
        if False:
            nlist[2*ncurrent+1] = self.nR - 2 #for isMAP_SE

        #TODO is this actually used?
        if self.nR == 0: nlist[2*ncurrent+1] += 1
        if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1
        nlist[2*ncurrent] = self.nnodes -self.nP
        nlist[2*ncurrent+1] = self.nR-1
        #TODO is this actually used?
        if self.nR == 0: nlist[2*ncurrent+1] += 1
        if self.nP == 0: nlist[2*ncurrent] -= 1
        ncurrent += 1

        for n in range(ncurrent):
            #ictan[nlist[2*n]] = DLC.tangent_1(self.icoords[nlist[2*n+1]],self.icoords[nlist[2*n+0]])
            ictan[nlist[2*n]] = self.tangent(nlist[2*n+1],nlist[2*n])
            ictan0 = np.copy(ictan[nlist[2*n]])

            if True:
                self.icoords[nlist[2*n+1]].bmatp = self.icoords[nlist[2*n+1]].bmatp_create()
                self.icoords[nlist[2*n+1]].bmatp_to_U()
                self.icoords[nlist[2*n+1]].opt_constraint(ictan[nlist[2*n]])
                self.icoords[nlist[2*n+1]].bmat_create()
        
            dqmaga[nlist[2*n]] = np.dot(ictan0.T,self.icoords[nlist[2*n+1]].Ut[-1,:])
            dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))

        self.dqmaga = dqmaga
        self.ictan = ictan
       
        if True:
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
            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.check_add_node()
            self.get_tangents_1g()
            #self.ic_reparam_g()
            self.opt_steps(maxopt,nconstraints)
            self.store_energies()

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
            
            print " gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.1}".format(current,float(totalgrad),float(gradrms),float(self.emax))
 
            self.write_xyz_files(iters=current,base='growth_iters',nconstraints=nconstraints)


    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1):
        print "*********************************************************************"
        print "************************** in opt_iters *****************************"
        print "*********************************************************************"
        for i in range(1,self.nnodes-1):
            self.active[i] = True
        for oi in range(max_iter):
            self.get_tangents_1g() #Try get_tangents_1e here
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
                return
            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)


    def interpolateR(self,newnodes=1):
        print "interpolateR"
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
                self.icoords[n].opt_constraint(ictan0) #TODO at line 3815
                        
