from __future__ import print_function
# standard library imports
import sys
import os
from os import path

# third party
import numpy as np
import multiprocessing as mp
from multiprocessing import Process 
from collections import Counter

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
import wrappers
from wrappers import Molecule
from coordinate_systems import DelocalizedInternalCoordinates
from coordinate_systems import rotate
from ._print_opt import Print
from ._analyze_string import Analyze
from optimizers import beales_cg,eigenvector_follow
from optimizers._linesearch import double_golden_section
from coordinate_systems import Distance,Angle,Dihedral,OutOfPlane,TranslationX,TranslationY,TranslationZ,RotationA,RotationB,RotationC
from coordinate_systems.rotate import get_quat,calc_fac_dfac

# TODO interpolate is still sloppy. It shouldn't create a new molecule node itself 
# but should create the xyz. GSM should create the new molecule based off that xyz.

# TODO nconstraints in ic_reparam and write_iters is irrelevant

# THis can help us parallelize things
def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])

# e.g. list_of_results = pool.map(worker, ((obj, "my_process", 100, 1) for obj in list_of_objects))



class Base_Method(Print,Analyze,object):

    @staticmethod
    def default_options():
        if hasattr(Base_Method, '_default_options'): return Base_Method._default_options.copy()

        opt = options.Options() 
        
        opt.add_option(
            key='reactant',
            required=True,
            #allowed_types=[Molecule,wrappers.Molecule],
            doc='Molecule object as the initial reactant structure')

        opt.add_option(
            key='product',
            required=False,
            #allowed_types=[Molecule,wrappers.Molecule],
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
                key='print_level',
                value=1,
                required=False
                )

        opt.add_option(
                key='use_multiprocessing',
                value=False,
                doc='Use python multiprocessing module, an OpenMP like implementation \
                        that parallelizes optimization cycles on a single compute node'
                )

#BDIST_RATIO controls when string will terminate, good when know exactly what you want
#DQMAG_MAX controls max step size for adding node
        opt.add_option(
                key="BDIST_RATIO",
                value=0.5,
                required=False,
                doc="SE-Crossing uses this \
                        bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
                        to be considered grown.",
                        )

        opt.add_option(
                key='ID',
                value=0,
                required=False,
                doc='A number for identification'
                )

        Base_Method._default_options = opt
        return Base_Method._default_options.copy()


    @classmethod
    def from_options(cls,**kwargs):
        return cls(cls.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        os.system('mkdir -p scratch')

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
        self.ID = self.options['ID']
        self.use_multiprocessing = self.options['use_multiprocessing']
        self.optimizer=[]
        optimizer = options['optimizer']
        for count in range(self.nnodes):
            self.optimizer.append(optimizer.__class__(optimizer.options.copy()))
        self.print_level = options['print_level']

        # Set initial values
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.energies = np.asarray([0.]*self.nnodes)
        self.emax = 0.0

        # TSnode is now a property
        self.climb = False 
        self.find = False 
        self.ts_exsteps = 3 # multiplier for ts node
        self.n0 = 1 # something to do with added nodes? "first node along current block"
        self.end_early=False
        self.tscontinue=True # whether to continue with TS opt or not
        self.rn3m6 = np.sqrt(3.*self.nodes[0].natoms-6.);
        self.gaddmax = self.options['ADD_NODE_TOL'] #self.options['ADD_NODE_TOL']/self.rn3m6;
        print(" gaddmax:",self.gaddmax)
        self.ictan = [None]*self.nnodes
        self.active = [False] * self.nnodes
        self.climber=False  #is this string a climber?
        self.finder=False   # is this string a finder?
        self.done_growing = False
        self.reference_xyz = None

    @property
    def TSnode(self):
        return np.argmax(self.energies[:self.nnodes-1])
    @property
    def npeaks(self):
        minnodes=[]
        maxnodes=[]
        if self.energies[1]>self.energies[0]:
            minnodes.append(0)
        if self.energies[self.nnodes-1]<self.energies[self.nnodes-2]:
            minnodes.append(self.nnodes-1)
        for n in range(self.n0,self.nnodes-1):
            if self.energies[n+1]>self.energies[n]:
                if self.energies[n]<self.energies[n-1]:
                    minnodes.append(n)
            if self.energies[n+1]<self.energies[n]:
                if self.energies[n]>self.energies[n-1]:
                    maxnodes.append(n)

        return len(maxnodes)


    def store_energies(self):
        for i,ico in enumerate(self.nodes):
            if ico != None:
                self.energies[i] = ico.energy - self.nodes[0].energy

    def opt_iters(self,max_iter=30,nconstraints=1,optsteps=1,rtype=2):
        nifty.printcool("In opt_iters")

        self.nclimb=0
        self.nhessreset=10  # are these used??? TODO 
        self.hessrcount=0   # are these used?!  TODO
        self.newclimbscale=2.

        self.set_finder(rtype)
        self.store_energies()
        self.emax = self.energies[self.TSnode]

        # set convergence for nodes
        if (self.climber or self.finder):
            factor = 2.5
        else: 
            factor = 1.
        for i in range(self.nnodes):
            if self.nodes[i] !=None:
                self.optimizer[i].conv_grms = self.options['CONV_TOL']*factor

        # enter loop
        for oi in range(max_iter):

            nifty.printcool("Starting opt iter %i" % oi)
            if self.climb and not self.find: print(" CLIMBING")
            elif self.find: print(" TS SEARCHING")

            sys.stdout.flush()

            # stash previous TSnode  
            self.pTSnode = self.TSnode
            self.emaxp = self.emax

            # => Get all tangents 3-way <= #
            self.get_tangents_1e()
           
            # => do opt steps <= #
            self.opt_steps(optsteps)
            self.store_energies()

            print()
            print(" V_profile: ", end=' ')
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(self.energies[n])), end=' ')
            print()

            #TODO resetting
            #TODO special SSM criteria if TSNode is second to last node
            #TODO special SSM criteria if first opt'd node is too high?

            # => find peaks <= #
            fp = self.find_peaks(2)

            # => get TS node <=
            self.emax= self.energies[self.TSnode]
            print(" max E({}) {:5.4}".format(self.TSnode,self.emax))

            ts_cgradq = 0.
            if not self.find:
                ts_cgradq = np.linalg.norm(np.dot(self.nodes[self.TSnode].gradient.T,self.nodes[self.TSnode].constraints[:,0])*self.nodes[self.TSnode].constraints[:,0])
                print(" ts_cgradq %5.4f" % ts_cgradq)

            ts_gradrms=self.nodes[self.TSnode].gradrms
            self.dE_iter=abs(self.emax-self.emaxp)
            print(" dE_iter ={:2.2f}".format(self.dE_iter))

            # => calculate totalgrad <= #
            totalgrad,gradrms,sum_gradrms = self.calc_grad()

            # => Check Convergence <= #
            isDone = self.check_opt(totalgrad,fp,rtype)
            if isDone:
                break

            sum_conv_tol = (self.nnodes-2)*self.options['CONV_TOL'] 
            if not self.climber and not self.finder:
                print(" CONV_TOL=%.4f" %self.options['CONV_TOL'])
                print(" convergence criteria is %.5f, current convergence %.5f" % (sum_conv_tol,sum_gradrms))
                all_conv=True
                for  n in range(1,self.nnodes-1):
                    if not self.optimizer[n].converged:
                        all_conv=False
                        break

                if sum_gradrms<sum_conv_tol and all_conv: #Break even if not climb/find
                    break

            # => set stage <= #
            form_TS_hess = self.set_stage(totalgrad,sum_gradrms,ts_cgradq,ts_gradrms,fp)

            # Modify TS Hess if necessary
            if form_TS_hess:
                self.get_tangents_1e()
                self.get_eigenv_finite(self.TSnode)
                if self.optimizer[self.TSnode].options['DMAX']>0.05:
                    self.optimizer[self.TSnode].options['DMAX']=0.05
            elif self.find and not self.optimizer[n].maxol_good:
                # reform Hess for TS if not good
                self.get_tangents_1e()
                self.get_eigenv_finite(self.TSnode)
            elif self.find and (self.optimizer[self.TSnode].nneg > 3 or self.optimizer[self.TSnode].nneg==0) and ts_gradrms >self.options['CONV_TOL']:
                if self.hessrcount<1 and self.pTSnode == self.TSnode:
                    print(" resetting TS node coords Ut (and Hessian)")
                    self.get_tangents_1e()
                    self.get_eigenv_finite(self.TSnode)
                    self.nhessreset=10
                    self.hessrcount=1
                else:
                    print(" Hessian consistently bad, going back to climb (for 3 iterations)")
                    self.find=False
                    #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                    self.nclimb=3

            elif self.find and self.optimizer[self.TSnode].nneg > 1 and ts_gradrms < self.options['CONV_TOL']:
                 print(" nneg > 1 and close to converging -- reforming Hessian")                
                 self.get_tangents_1e()                                                         
                 self.get_eigenv_finite(self.TSnode)                    
            elif self.find and self.optimizer[self.TSnode].nneg <= 3:
                self.hessrcount-=1

            # => Reparam the String <= #
            if oi!=max_iter-1:
                self.ic_reparam(nconstraints=nconstraints)

            # store reparam energies
            self.store_energies()
            self.emax= self.energies[self.TSnode]
            print(" V_profile (after reparam): ", end=' ')
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(self.energies[n])), end=' ')
            print()

            if self.pTSnode!=self.TSnode and self.climb:
                #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy())
                #self.optimizer[self.pTSnode] = self.optimizer[0].__class__(self.optimizer[self.TSnode].options.copy())

                if self.climb and not self.find:
                    print(" slowing down climb optimization")
                    self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale
                    self.optimizer[self.TSnode].options['SCALEQN'] = 2.
                    self.optimizer[self.pTSnode].options['SCALEQN'] = 1.
                    self.ts_exsteps=1
                    if self.newclimbscale<5.0:
                        self.newclimbscale +=1.
                elif self.find:
                    self.find = False
                    self.nclimb=1
                    print(" Find bad, going back to climb")
                    #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.pTSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                    #self.optimizer[self.pTSnode] = self.optimizer[0].__class__(self.optimizer[self.TSnode].options.copy())
                    #self.get_tangents_1e()
                    #self.get_eigenv_finite(self.TSnode)

            # => write Convergence to file <= #
            self.write_xyz_files(base='opt_iters',iters=oi,nconstraints=nconstraints)

            #TODO prints tgrads and jobGradCount
            print("opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(oi,float(totalgrad),float(gradrms),self.TSnode,float(self.emax)))
            print('\n')

        print(" Printing string to opt_converged_000.xyz")
        self.write_xyz_files(base='opt_converged',iters=0,nconstraints=nconstraints)
        sys.stdout.flush()
        return

    def get_tangents_1(self,n0=0):
        dqmaga = [0.]*self.nnodes
        dqa = np.zeros((self.nnodes+1,self.nnodes))
        ictan = [[]]*self.nnodes

        def get_tangent(n):
            assert self.nodes[n]!=None,"n is bad"
            assert self.nodes[n-1]!=None,"n-1 is bad"
            tangent,_ = Base_Method.tangent(self.nodes[n-1],self.nodes[n])
            return tangent

        def get_dqmag(ictan,n):
            dqmag = 0.
            ictan0= np.copy(ictan)
            ictan /= np.linalg.norm(ictan)

            self.newic.xyz = self.nodes[n].xyz
            Vecs = self.newic.update_coordinate_basis(ictan0)

            constraint = self.newic.constraints[:,0]
            prim_constraint = block_matrix.dot(Vecs,constraint)

            for prim in self.newic.primitive_internal_coordinates:
                if type(prim) is Distance:
                    index = self.newic.coord_obj.Prims.dof_index(prim)
                    prim_constraint[index] *= 2.5

            dqmag = np.dot(prim_constraint.T,ictan0) 
            if dqmag<0.:
                raise RuntimeError
            dqmag = float(np.sqrt(dqmag))
            return dqmag


        #print "getting tangents for nodes 0 to ",self.nnodes
        for n in range(n0+1,self.nnodes):
            #print "getting tangent between %i %i" % (n,n-1)
            assert self.nodes[n]!=None,"n is bad"
            assert self.nodes[n-1]!=None,"n-1 is bad"
            ictan[n],_ = Base_Method.tangent(self.nodes[n-1],self.nodes[n])
            dqmaga[n] = 0.
            ictan0= np.copy(ictan[n])
            
            ictan[n] /= np.linalg.norm(ictan[n])
             
            self.newic.xyz = self.nodes[n].xyz
            Vecs = self.newic.update_coordinate_basis(ictan[n])

            # cnorms
            constraint = self.newic.constraints[:,0]
            # just a fancy way to get the normalized tangent vector
            prim_constraint = block_matrix.dot(Vecs,constraint)

            #assert np.isclose([np.linalg.norm(prim_constraint)],[1.0]), "This should be 1"

            for prim in self.newic.primitive_internal_coordinates:
                if type(prim) is Distance:
                    index = self.newic.coord_obj.Prims.dof_index(prim)
                    prim_constraint[index] *= 2.5
           
            dqmaga[n] = float(np.dot(prim_constraint.T,ictan0))
            if dqmaga[n]<0.:
                raise RuntimeError

            # should I do this?  TURN off 11/22/19
            dqmaga[n] = float(np.sqrt(dqmaga[n]))

            # note: C++ gsm modifies tangent here
            #nbonds=self.nodes[0].num_bonds
            #dqmaga[n] += np.dot(Vecs[:nbonds,0],ictan0[:nbonds])*2.5
            #dqmaga[n] += np.dot(Vecs[nbonds:,0],ictan0[nbonds:])
            #print('dqmaga[n] %.3f' % dqmaga[n])
      
        #ictan = [0.]
        #ictan += [ Process(target=get_tangent,args=(n,)) for n in range(n0+1,self.nnodes)]
        #dqmaga = [ Process(target=get_dqmag,args=(n,ictan[n])) for n in range(n0+1,self.nnodes)]

        self.dqmaga = dqmaga
        self.ictan = ictan

        if self.print_level>1:
            print('------------printing ictan[:]-------------')
            for n in range(n0+1,self.nnodes):
                print("ictan[%i]" %n)
                print(ictan[n].T)
        #if self.print_level>0:
        #    print('------------printing dqmaga---------------')
        #    for n in range(n0+1,self.nnodes-1):
        #        print(" {:5.4}".format(dqmaga[n]),end='')
        #    print()


    # for some reason this fxn doesn't work when called outside gsm
    def get_tangents_1e(self,n0=0):
        ictan0 = np.zeros((self.newic.num_primitives,1))
        dqmaga = [0.]*self.nnodes

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
                ictan0,_ = Base_Method.tangent(self.nodes[newic_n],self.nodes[intic_n])
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

                print(' 3 way tangent ({}): f1:{:3.2}'.format(n,f1))

                t1,_ = Base_Method.tangent(self.nodes[intic_n],self.nodes[newic_n])
                t2,_ = Base_Method.tangent(self.nodes[newic_n],self.nodes[int2ic_n])
                print(" done 3 way tangent")
                ictan0 = f1*t1 +(1.-f1)*t2
                #self.ictan[n]=ictan0
            self.ictan[n] = ictan0/np.linalg.norm(ictan0)
            
            dqmaga[n]=0.0

            # update newic coordinate basis
            self.newic.xyz = self.nodes[newic_n].xyz
            Vecs = self.newic.update_coordinate_basis(self.ictan[n])
            nbonds=self.nodes[0].num_bonds
            # cnorms
            constraint = self.newic.constraints[:,0]
            # just a fancy way to get the tangent vector
            prim_constraint = block_matrix.dot(Vecs,constraint)

            # mult bonds
            for prim in self.newic.primitive_internal_coordinates:
                if type(prim) is Distance:
                    index = self.newic.coord_obj.Prims.dof_index(prim)
                    prim_constraint[index] *= 2.5

            dqmaga[n] = np.dot(prim_constraint.T,ictan0) 
            #dqmaga[n] += np.dot(Vecs[:nbonds,0],ictan0[:nbonds])*2.5
            #dqmaga[n] += np.dot(Vecs[nbonds:,0],ictan0[nbonds:])
            if dqmaga[n]<0.:
                raise RuntimeError
        
            dqmaga[n] = float(np.sqrt(dqmaga[n]))

        if self.print_level>1:
            print('------------printing ictan[:]-------------')
            for n in range(n0+1,self.nnodes):
                print(self.ictan[n].T)
        if self.print_level>0:
            print('------------printing dqmaga---------------')
            for n in range(n0+1,self.nnodes):
                print(" {:5.4}".format(dqmaga[n]),end='')
            print()
            #print(dqmaga)
        self.dqmaga = dqmaga

    def get_tangents_1g(self):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase.
        Also updates coordinates
        """
        dqmaga = [0.]*self.nnodes
        ncurrent,nlist = self.make_nlist()

        if self.print_level>1:
            print("ncurrent, nlist")
            print(ncurrent)
            print(nlist)

        for n in range(ncurrent):
            self.ictan[nlist[2*n]],_ = Base_Method.tangent(
                    node1=self.nodes[nlist[2*n]],
                    node2=self.nodes[nlist[2*n+1]],
                    driving_coords=self.driving_coords,
                    reference_xyz = self.reference_xyz)

            #save copy to get dqmaga
            ictan0 = np.copy(self.ictan[nlist[2*n]])
            if self.print_level>1:
                print("forming space for", nlist[2*n+1])
            if self.print_level>1:
                print("forming tangent for ",nlist[2*n])

            if (ictan0[:]==0.).all():
                print(nlist[2*n])
                print(nlist[2*n+1])
                print(self.nodes[nlist[2*n]])
                print(self.nodes[nlist[2*n+1]])
                raise RuntimeError

            #normalize ictan
            norm = np.linalg.norm(ictan0)  
            self.ictan[nlist[2*n]] /= norm
           
            Vecs = self.nodes[nlist[2*n+1]].update_coordinate_basis(constraints=self.ictan[nlist[2*n]])
            #constraint = self.nodes[nlist[2*n+1]].constraints
            #prim_constraint = block_matrix.dot(Vecs,constraint)
            #print(" norm of ictan %5.4f" % norm)
            #dqmaga[nlist[2*n]] = np.dot(prim_constraint.T,ictan0) 
            #dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))

            # for some reason the sqrt norm matches up 
            dqmaga[nlist[2*n]] = np.sqrt(norm)

            #print(" dqmaga %5.4f" %dqmaga[nlist[2*n]])

        self.dqmaga = dqmaga

        if self.print_level>0:
            print('------------printing dqmaga---------------')
            for n in range(1,ncurrent):
                print(" {:5.4}".format(dqmaga[n]),end='')
            print()
       
        if False:
            for n in range(ncurrent):
                print("dqmag[%i] =%1.2f" %(nlist[2*n],self.dqmaga[nlist[2*n]]))
                print("printing ictan[%i]" %nlist[2*n])       
                print(self.ictan[nlist[2*n]].T)
        for i,tan in enumerate(self.ictan):
            if np.all(tan==0.0):
                print("tan %i of the tangents is 0" %i)
                raise RuntimeError

    def growth_iters(self,iters=1,maxopt=1,nconstraints=1,current=0):
        nifty.printcool("In growth_iters")

        self.get_tangents_1g()
        self.set_active(self.nR-1, self.nnodes-self.nP)

        for n in range(iters):
            nifty.printcool("Starting growth iter %i" % n)
            sys.stdout.flush()
            self.opt_steps(maxopt)
            self.store_energies()
            totalgrad,gradrms,sum_gradrms = self.calc_grad()
            self.emax = self.energies[self.TSnode]
            self.write_xyz_files(iters=n,base='growth_iters',nconstraints=nconstraints)
            if self.check_if_grown(): 
                break

            success = self.check_add_node()
            if not success:
                print("can't add anymore nodes, bdist too small")

                # why not for SE_Cross too?
                if self.__class__.__name__=="SE_GSM":
                    if self.nodes[self.nR-1].PES.lot.do_coupling:
                        opt_type='MECI'
                    else:
                       opt_type='UNCONSTRAINED'

                    ictan,_ = Base_Method.tangent(self.nodes[self.nR-1],self.nodes[self.nR-2])

                    print(" optimizing last node")
                    self.optimizer[self.nR-1].conv_grms = self.options['CONV_TOL']
                    print(self.optimizer[self.nR-1].conv_grms)
                    self.optimizer[self.nR-1].optimize(
                            molecule=self.nodes[self.nR-1],
                            refE=self.nodes[0].V0,
                            opt_steps=50,
                            #opt_type=opt_type,
                            opt_type='ICTAN',
                            ictan=ictan,
                            )
                else:
                    raise RuntimeError

                self.check_if_grown()
                break

            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.ic_reparam_g()
            print(" gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n".format(n,float(totalgrad),float(gradrms),float(self.emax)))

        # create newic object
        print(" creating newic molecule--used for ic_reparam")
        self.newic  = Molecule.copy_from_options(self.nodes[0])
        return n

    @staticmethod
    def interpolate_xyz(nodeR,nodeP,stepsize):
        ictan,_ =  Base_Method.tangent(nodeR,nodeP)
        Vecs = nodeR.update_coordinate_basis(constraints=ictan)
        constraint = nodeR.constraints[:,0]
        prim_constraint = block_matrix.dot(Vecs,constraint)
        dqmag = np.dot(prim_constraint.T,ictan)
        print(" dqmag: %1.3f"%dqmag)
        #sign=-1
        sign=1.
        dqmag *= (sign*stepsize)
        print(" scaled dqmag: %1.3f"%dqmag)

        dq0 = dqmag*constraint
        old_xyz = nodeR.xyz.copy()
        new_xyz = nodeR.coord_obj.newCartesian(old_xyz,dq0)

        return new_xyz 

    def opt_steps(self,opt_steps):

        refE=self.nodes[0].energy
        if self.climb and not self.find:
            pass
            #nm1 = self.TSnode-1
            #np1 = self.TSnode+1

            ## interpolate closer to the TS node for the GS search
            #xyz = self.nodes[self.TSnode].xyz 
            ##xyz1 = Base_Method.interpolate_xyz(self.nodes[nm1],self.nodes[self.TSnode],0.9)
            ##xyz7 = Base_Method.interpolate_xyz(self.nodes[np1],self.nodes[self.TSnode],0.9)
            #xyz1 = self.nodes[nm1].xyz
            #xyz7 = self.nodes[np1].xyz

            ## linear approximation
            ##f1 = 0.9* self.nodes[self.TSnode].energy + 0.1*self.nodes[nm1].energy 
            ##f7 = 0.9* self.nodes[self.TSnode].energy + 0.1*self.nodes[np1].energy 
            ##f1=self.nodes[nm1].PES.get_energy(xyz1)
            ##f7=self.nodes[np1].PES.get_energy(xyz7)
            #f1=self.nodes[nm1].energy
            #f7=self.nodes[np1].energy

            #gp_prim = block_matrix.dot(self.nodes[self.TSnode].coord_basis,self.nodes[self.TSnode].gradient)
            #result = double_golden_section(self.nodes[self.TSnode].coordinates,xyz1,xyz7,f1,f7,self.nodes[self.TSnode])
            #status = result['status']

            #if not status:
            #    print("same geometry from golden section")
            #else:
            #    self.nodes[self.TSnode].xyz = result['xyz']
            #    self.get_tangents_1e()
        
            ## NEW 12/2019
            #s0_prim = self.ictan[self.TSnode]
            #s = result['step']
        else:
            #s = None
            s0_prim = None
            gp_prim=None

        if self.use_multiprocessing:
            cpus = mp.cpu_count()/self.nodes[0].PES.lot.nproc                                  
            print(" Parallelizing over {} processes".format(cpus))                             
            out_queue = mp.Queue()

            workers = [ mp.Process(target=mod, args=(self.nodes[n],self.optimizer[n],self.ictan[n],self.mult_steps(n,opt_steps),self.set_opt_type(n),refE,n,s,gp_prim, out_queue) ) for n in range(self.nnodes) if self.nodes[n] and self.active[n] ]
            
            for work in workers: work.start()                                                  
            
            for work in workers: work.join()                                                   
            
            res_lst = []
            for j in range(len(workers)):
                res_lst.append(out_queue.get())                                                
                                             
            #pool = mp.Pool(processes=cpus)
            #print("Created the pool")
            #results=pool.map(
            #        run, 
            #        [[self.nodes[n],self.optimizer[n],self.ictan[n],self.mult_steps(n,opt_steps),self.set_opt_type(n),refE,n,s,gp_prim] for n in range(self.nnodes) if (self.nodes[n] and self.active[n])],
            #        )

            #pool.close()
            #print("Calling join()...")
            #sys.stdout.flush()
            #pool.join()
            #print("Joined")

            #for (node,optimizer,n) in results:
            #    self.nodes[n]=node
            #    self.optimizer[n]=optimizer
        else:
            #results=[]
            #run_list = [n for n in range(self.nnodes) if (self.nodes[n] and self.active[n])]
            #for n in run_list:
            #    args = [self.nodes[n],self.optimizer[n],self.ictan[n],self.mult_steps(n,opt_steps),self.set_opt_type(n),refE,n,s,gp_prim]
            #    results.append(run(args))

            for n in range(self.nnodes):
                if self.nodes[n] and self.active[n]:
                    print()
                    nifty.printcool("Optimizing node {}".format(n))
                    opt_type = self.set_opt_type(n)
                    osteps = self.mult_steps(n,opt_steps)

                    if opt_type != "BEALES_CG":
                        self.optimizer[n].optimize(
                                molecule=self.nodes[n],
                                refE=refE,
                                opt_type=opt_type,
                                opt_steps=osteps,
                                ictan=self.ictan[n],
                                )
                    else:
                        self.optimizer[n].optimize(
                                molecule=self.nodes[n],
                                #s0=s,
                                s0_prim=s0_prim,
                                gp_prim=gp_prim,
                                refE=refE,
                                opt_type=opt_type,
                                opt_steps=osteps,
                                ictan=self.ictan[n],
                                )

        optlastnode=False
        if self.product_geom_fixed==False:
            if self.energies[self.nnodes-1]>self.energies[self.nnodes-2] and fp>0 and self.nodes[self.nnodes-1].gradrms>self.options['CONV_TOL']:
                optlastnode=True


    def set_stage(self,totalgrad,sumgradrms, ts_cgradq,ts_gradrms,fp):
        form_TS_hess=False

        sum_conv_tol = np.sum([self.optimizer[n].conv_grms for n in range(1,self.nnodes-1)])+ 0.0005

        #TODO totalgrad is not a good criteria for large systems
        if (totalgrad < 0.3 or sumgradrms<sum_conv_tol)  and fp>0: # extra criterion in og-gsm for added
            if not self.climb and self.climber:
                print(" ** starting climb **")
                self.climb=True
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq))
       
                # set to beales
                #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                #self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale

                # overwrite this here just in case TSnode changed wont cause slow down climb  
                self.pTSnode = self.TSnode

            elif (self.climb and not self.find and self.finder and self.nclimb<1 and self.dE_iter<4. and
                    ((totalgrad<0.2 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.01) or #
                    (totalgrad<0.1 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.02) or  #
                    (sumgradrms< sum_conv_tol) or
                    (ts_gradrms<self.options['CONV_TOL']*2.)  #  used to be 5
                    )):
                print(" ** starting exact climb **")
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq))
                self.find=True
                form_TS_hess=True
                self.optimizer[self.TSnode] = eigenvector_follow(self.optimizer[self.TSnode].options.copy())
                print(type(self.optimizer[self.TSnode]))
                self.optimizer[self.TSnode].options['SCALEQN'] = 1.
                self.nhessreset=10  # are these used??? TODO 
                self.hessrcount=0   # are these used?!  TODO
            if self.climb: 
                self.nclimb-=1

            #for n in range(1,self.nnodes-1):
            #    self.active[n]=True
            #    self.optimizer[n].options['OPTTHRESH']=self.options['CONV_TOL']*2
            self.nhessreset-=1

        return form_TS_hess

    @staticmethod
    def tangent(node1,node2,**kwargs):
        #if self.print_level>1:
        #    print(" getting tangent from between %i %i pointing towards %i"%(n2,n1,n2))
        # this could have been done easier but it is nicer to do it this way

        print_level = 1

        #if len(kwargs)==0:
        #if node1.node_id != node2.node_id:
        if node2 is not None and node1.node_id!=node2.node_id:
            print(" getting tangent from between %i %i pointing towards %i"%(node2.node_id,node1.node_id,node2.node_id))
            assert node2!=None,'node n2 is None'

            Q1 = node1.primitive_internal_values 
            Q2 = node2.primitive_internal_values 
            PMDiff = Q2-Q1
            for k,prim in zip(list(range(len(PMDiff))),node1.primitive_internal_coordinates):
                if prim.isPeriodic:
                    Plus2Pi = PMDiff[k] + 2*np.pi
                    Minus2Pi = PMDiff[k] - 2*np.pi
                    if np.abs(PMDiff[k]) > np.abs(Plus2Pi):
                        PMDiff[k] = Plus2Pi
                    if np.abs(PMDiff[k]) > np.abs(Minus2Pi):
                        PMDiff[k] = Minus2Pi
            return np.reshape(PMDiff,(-1,1)),None
        else:
            print(" getting tangent from node ",node1.node_id)

            driving_coords = kwargs.get('driving_coords',None)
            assert driving_coords is not None, " Driving coord is None!"

            c = Counter(elem[0] for elem in driving_coords)
            nadds = c['ADD']
            nbreaks = c['BREAK']
            nangles = c['nangles']
            ntorsions = c['ntorsions']

            ictan = np.zeros((node1.num_primitives,1),dtype=float)
            breakdq = 0.3
            bdist=0.0
            atoms = node1.atoms
            xyz = node1.xyz

            for i in driving_coords:
                if "ADD" in i:

                    #order indices to avoid duplicate bonds
                    if i[1]<i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]

                    bond = Distance(index[0],index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(bond)
                    if len(i)==3:
                        #TODO why not just use the covalent radii?
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2.8
                    elif len(i)==4:
                        d0=i[3]
                    current_d =  bond.value(xyz)

                    #TODO don't set tangent if value is too small
                    ictan[prim_idx] = -1*(d0-current_d)
                    #if nbreaks>0:
                    #    ictan[prim_idx] *= 2
                    # => calc bdist <=
                    if current_d>d0:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if print_level>0:
                        print(" bond %s target (less than): %4.3f current d: %4.3f diff: %4.3f " % ((i[1],i[2]),d0,current_d,ictan[prim_idx]))

                if "BREAK" in i:
                    #order indices to avoid duplicate bonds
                    if i[1]<i[2]:
                        index = [i[1]-1, i[2]-1]
                    else:
                        index = [i[2]-1, i[1]-1]
                    bond = Distance(index[0],index[1])
                    prim_idx = node1.coord_obj.Prims.dof_index(bond)
                    if len(i)==3:
                        d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)
                    elif len(i)==4:
                        d0=i[3]

                    current_d =  bond.value(xyz)
                    ictan[prim_idx] = -1*(d0-current_d) 

                    # => calc bdist <=
                    if current_d<d0:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])

                    if print_level>0:
                        print(" bond %s target (greater than): %4.3f, current d: %4.3f diff: %4.3f " % ((i[1],i[2]),d0,current_d,ictan[prim_idx]))
                if "ANGLE" in i:

                    if i[1]<i[3]:
                        index = [i[1]-1, i[2]-1,i[3]-1]
                    else:
                        index = [i[3]-1, i[2]-1,i[1]-1]

                    angle = Angle(index[0],index[1],index[2])
                    prim_idx = node1.coord_obj.Prims.dof_index(angle)
                    anglet = i[4]
                    ang_value = angle.value(xyz)
                    ang_diff = anglet*np.pi/180. - ang_value
                    #print(" angle: %s is index %i " %(angle,ang_idx))
                    if print_level>0:
                        print((" anglev: %4.3f align to %4.3f diff(rad): %4.3f" %(ang_value,anglet,ang_diff)))
                    ictan[prim_idx] = -ang_diff
                    #TODO need to come up with an adist
                    #if abs(ang_diff)>0.1:
                    #    bdist+=ictan[ICoord1.BObj.nbonds+ang_idx]*ictan[ICoord1.BObj.nbonds+ang_idx]
                if "TORSION" in i:

                    if i[1]<i[4]:
                        index = [i[1]-1,i[2]-1,i[3]-1,i[4]-1]
                    else:
                        index = [i[4]-1,i[3]-1,i[2]-1,i[1]-1]
                    torsion = Dihedral(index[0],index[1],index[2],index[3])
                    prim_idx = node1.coord_obj.Prims.dof_index(torsion)
                    tort = i[5]
                    torv = torsion.value(xyz)
                    tor_diff = tort - torv*180./np.pi
                    if tor_diff>180.:
                        tor_diff-=360.
                    elif tor_diff<-180.:
                        tor_diff+=360.
                    ictan[prim_idx] = -tor_diff*np.pi/180.

                    if tor_diff*np.pi/180.>0.1 or tor_diff*np.pi/180.<0.1:
                        bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                    if print_level>0:
                        print((" current torv: %4.3f align to %4.3f diff(deg): %4.3f" %(torv*180./np.pi,tort,tor_diff)))


                if "ROTATE" in i:
                    frag = i[1]
                    a1 = i[2]  #atom 1
                    a2 = i[3]  #atom 2
                    theta_target = i[4]*np.pi/180.    # degree

                    # Need to retrieve reference fragment and reference axis of rotation
                    # Need rotational coordinates of fragment
                    # 2) align reference axis of rotation to new axis of rotation with rotation matrix
                    # 3) rotate reference fragment using the same rotation matrix 
                    # 4) Calculate theta of rotation from reference to current
                    # 5) calculate dtheta as target_theta - theta

                    reference_xyz = kwargs.get('reference_xyz',None)
                    #print(reference_xyz)
                    # block
                    info = node1.coord_obj.Prims.block_info[frag]
                    #print(info)
                    sp,ep,sa,ea = info

                    xyz_frag = node1.xyz[sa:ea]
                    axis = node1.xyz[a2] - xyz_frag[a1]
                    axis /= np.linalg.norm(axis)

                    # only want the fragment of interest
                    ref_xyz = reference_xyz[sa:ea]
                    ref_axis = reference_xyz[a2] - reference_xyz[a1]
                    ref_axis /= np.linalg.norm(ref_axis)
                    #print('axis')
                    #print(axis)
                    #print('ref-axis')
                    #print(ref_axis)

                    print(' Rotating reference axis to current axis')
                    I = np.eye(3)
                    v = np.cross(ref_axis,axis)

                    if v.all()==0.:
                        R=I
                    else:
                        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                        c = np.dot(ref_axis,axis)
                        s = np.linalg.norm(v)
                        R = I + vx + np.dot(vx,vx) * (1. - c)/(s**2)
                    new_ref_axis = np.dot(ref_axis,R.T)
                    print(' overlap of ref-axis and axis (should be 1.) %1.2f' % np.dot(new_ref_axis,axis))
                    new_ref_xyz = np.dot(ref_xyz,R.T)
                    
                    # Calculate theta from ref_xyz to current xyz
                    qnew = get_quat(new_ref_xyz,xyz_frag)
                    theta_actual = np.arccos(qnew[0])*2
                    print('theta around u1 from (rotated) reference to current')
                    print(theta_actual)
                    dtheta = theta_target - theta_actual
                    print(' Still need to twist %4.3f' %dtheta)

                    # TODO  Check if rotation is less than 0.9 pi
                    if theta_actual < 0.9*np.pi:
                        
                        # First calculate reference v
                        sel_ref = ref_xyz.reshape(-1,3)
                        sel_ref -= np.mean(sel_ref, axis=0)
                        rg = np.sqrt(np.mean(np.sum(sel_ref**2, axis=1)))
                        c = np.cos(0./2.0)
                        s = np.sin(0./2.0)
                        q_ref = np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])
                        fac, _ = calc_fac_dfac(c)

                        # Create tangent with dtheta 
                        sel = xyz_frag.reshape(-1,3)
                        sel -= np.mean(sel, axis=0)
                        rg = np.sqrt(np.mean(np.sum(sel**2, axis=1)))
                        c = np.cos(dtheta/2.0)
                        s = np.sin(dtheta/2.0)
                        q = np.array([c, axis[0]*s, axis[1]*s, axis[2]*s])
                        fac, _ = calc_fac_dfac(c)


                        # Try difference
                        v1 = fac*rg*(q[1] - q_ref[1])
                        v2 = fac*rg*(q[2] - q_ref[1])
                        v3 = fac*rg*(q[3] - q_ref[1])

                        print(' v: %3.2f %3.2f %3.2f' % (v1,v2,v3))
                        #v1 = fac*q[1]*rg
                        #v2 = fac*q[2]*rg
                        #v3 = fac*q[3]*rg

                        # rotational coordinates should be the third in the list of primitives
                        ictan[sp+3] = v1
                        ictan[sp+4] = v2
                        ictan[sp+5] = v3

                        # THIS DOESN"T WORK
                        bdist += np.dot(ictan[sp+3],ictan[sp+3])
                        bdist += np.dot(ictan[sp+4],ictan[sp+4])
                        bdist += np.dot(ictan[sp+5],ictan[sp+5])
                        #bdist += dtheta


                #trans = ['TranslationX', 'TranslationY', 'TranslationZ']
                #if any(elem in trans for elem in i):
                #    fragid = i[1]
                #    destination = i[2]
                #    indices = node1.get_frag_atomic_index(fragid)
                #    atoms=range(indices[0]-1,indices[1])
                #    #print('indices of frag %i is %s' % (fragid,indices))
                #    T_class = getattr(sys.modules[__name__], i[0])
                #    translation = T_class(atoms,w=np.ones(len(atoms))/len(atoms))
                #    prim_idx = node1.coord_obj.Prims.dof_index(translation)
                #    trans_curr = translation.value(xyz)
                #    trans_diff = destination-trans_curr
                #    ictan[prim_idx] = -trans_diff
                #    bdist += np.dot(ictan[prim_idx],ictan[prim_idx])
                #    if print_level>0:
                #        print((" current trans: %4.3f align to %4.3f diff: %4.3f" %(trans_curr,destination,trans_diff)))


            bdist = np.sqrt(bdist)
            if np.all(ictan==0.0):
                raise RuntimeError(" All elements are zero")
            return ictan,bdist

    @staticmethod
    def interpolate(start_node,end_node,num_interp):
        nifty.printcool(" interpolate")
       
        num_nodes = num_interp + 2
        nodes = [None]*(num_nodes)
        nodes[0] = start_node
        nodes[-1] = end_node
        sign=1
        nR = 1
        nP = 1
        nn = nR + nP
        #E0=start_node.energy

        for n in range(num_interp):
            if num_nodes - nn > 1:
                stepsize = 1./float(num_nodes - nn)
                #stepsize = 1./float(self.nnodes-self.nn)
            else:
                stepsize = 0.5
            if sign == 1:
                iR = nR-1
                iP = num_nodes - nP
                iN = nR
                nodes[nR] = Base_Method.add_node(nodes[iR],nodes[iP],stepsize,iN)
                if nodes[nR] == None:
                    raise RuntimeError

                #print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                nR +=1 
                nn += 1

            else:
                n1 = num_nodes - nP
                n2 = n1 -1
                n3 = nR -1
                nodes[n2] = Base_Method.add_node(nodes[n1],nodes[n3],stepsize,n2)
                if nodes[n2] == None:
                    raise RuntimeError
                #print(" Energy of node {} is {:5.4}".format(nR,nodes[nR].energy-E0))
                nP +=1 
                nn += 1
            sign *= -1

        return nodes

    def add_GSM_nodeR(self,newnodes=1):
        nifty.printcool("Adding reactant node")
        success= True
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            iR = self.nR-1
            iP = self.nnodes-self.nP
            iN = self.nR
            print(" adding node: %i between %i %i from %i" %(iN,iR,iP,iR))
            if self.nnodes - self.nn > 1:
                stepsize = 1./float(self.nnodes-self.nn)
            else:
                stepsize = 0.5

            self.nodes[self.nR] = Base_Method.add_node(
                    self.nodes[iR],
                    self.nodes[iP],
                    stepsize,
                    iN,
                    DQMAG_MAX = self.DQMAG_MAX,
                    DQMAG_MIN = self.DQMAG_MIN,
                    driving_coords = self.driving_coords,
                    reference_xyz = self.reference_xyz,
                    )

            if self.nodes[self.nR]==None:
                success= False
                break

            if self.__class__.__name__!="DE_GSM":
                ictan,bdist =  Base_Method.tangent(self.nodes[self.nR],None,driving_coords=self.driving_coords,reference_xyz=self.reference_xyz)
                self.nodes[self.nR].bdist = bdist

            self.nn+=1
            self.nR+=1
            print(" nn=%i,nR=%i" %(self.nn,self.nR))
            self.active[self.nR-1] = True

            # align center of mass  and rotation
            #print("%i %i %i" %(iR,iP,iN))

            print(" Aligning")
            self.nodes[self.nR-1].xyz = self.com_rotate_move(iR,iP,iN)
            print(" getting energy for node %d: %5.4f" %(self.nR-1,self.nodes[self.nR-1].energy - self.nodes[0].V0))

        return success

    def add_GSM_nodeP(self,newnodes=1):
        nifty.printcool("Adding product node")
        if self.nn+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")

        success=True
        for i in range(newnodes):
            #self.nodes[-self.nP-1] = Base_Method.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nnodes-self.nP)
            n1=self.nnodes-self.nP
            n2=self.nnodes-self.nP-1
            n3=self.nR-1
            print(" adding node: %i between %i %i from %i" %(n2,n1,n3,n1))
            if self.nnodes - self.nn > 1:
                stepsize = 1./float(self.nnodes-self.nn)
            else:
                stepsize = 0.5

            self.nodes[-self.nP-1] = Base_Method.add_node(
                    self.nodes[n1],
                    self.nodes[n3],
                    stepsize,
                    n2
                    )
            if self.nodes[-self.nP-1]==None:
                success= False
                break

            self.nn+=1
            self.nP+=1
            print(" nn=%i,nP=%i" %(self.nn,self.nP))
            self.active[-self.nP] = True

            # align center of mass  and rotation
            #print("%i %i %i" %(n1,n3,n2))
            print(" Aligning")
            self.nodes[-self.nP].xyz = self.com_rotate_move(n1,n3,n2)
            print(" getting energy for node %d: %5.4f" %(self.nnodes-self.nP,self.nodes[-self.nP].energy - self.nodes[0].V0))

        return success

    @staticmethod
    def add_node(
            nodeR,
            nodeP,
            stepsize,
            node_id,
            **kwargs,
            ):

        #get driving coord
        driving_coords  = kwargs.get('driving_coords',None)
        reference_xyz  = kwargs.get('reference_xyz',None)
        DQMAG_MAX       =kwargs.get('DQMAG_MAX',0.8)
        DQMAG_MIN       =kwargs.get('DQMAG_MIN',0.2)

        if nodeP is None:
            # nodeP is not used!
            BDISTMIN=0.05
            ictan,bdist =  Base_Method.tangent(nodeR,None,driving_coords=driving_coords,reference_xyz=reference_xyz)

            if bdist<BDISTMIN:
                print("bdist too small %.3f" % bdist)
                return None
            new_node = Molecule.copy_from_options(nodeR,new_node_id=node_id)
            Vecs = new_node.update_coordinate_basis(constraints=ictan)
            constraint = new_node.constraints[:,0]
            sign=-1.

            dqmag_scale=1.5
            minmax = DQMAG_MAX - DQMAG_MIN
            a = bdist/dqmag_scale
            if a>1.:
                a=1.
            dqmag = sign*(DQMAG_MIN+minmax*a)
            if dqmag > DQMAG_MAX:
                dqmag = DQMAG_MAX
            print(" dqmag: %4.3f from bdist: %4.3f" %(dqmag,bdist))

            dq0 = dqmag*constraint
            print(" dq0[constraint]: %1.3f" % dqmag)

            new_node.update_xyz(dq0)
            new_node.bdist = bdist

            return new_node
        else:
            ictan,_ =  Base_Method.tangent(nodeR,nodeP)
            Vecs = nodeR.update_coordinate_basis(constraints=ictan)
            constraint = nodeR.constraints[:,0]
            prim_constraint = block_matrix.dot(Vecs,constraint)
            dqmag = np.dot(prim_constraint.T,ictan)
            print(" dqmag: %1.3f"%dqmag)
            #sign=-1
            sign=1.
            dqmag *= (sign*stepsize)
            print(" scaled dqmag: %1.3f"%dqmag)

            dq0 = dqmag*constraint
            old_xyz = nodeR.xyz.copy()
            new_xyz = nodeR.coord_obj.newCartesian(old_xyz,dq0)
            new_node = Molecule.copy_from_options(MoleculeA=nodeR,xyz=new_xyz,new_node_id=node_id)

            return new_node


    def ic_reparam(self,ic_reparam_steps=8,n0=0,nconstraints=1,rtype=0):
        nifty.printcool("reparametrizing string nodes")
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
    
        # align first and last nodes
        # assuming they are aligned
        #self.nodes[self.nnodes-1].xyz = self.com_rotate_move(0,self.nnodes-1,self.nnodes-2)

        for n in range(1,self.nnodes-1):
            self.nodes[n].xyz = self.com_rotate_move(n-1,n+1,n)

        for i in range(ic_reparam_steps):
            self.get_tangents_1(n0=n0)

            # copies of original ictan
            ictan0 = np.copy(self.ictan)
            ictan = np.copy(self.ictan)

            if self.print_level>1:
                print(" printing spacings dqmaga:")
                for n in range(1,self.nnodes):
                    print(" %1.2f" % self.dqmaga[n], end=' ') 
                print() 

            totaldqmag = 0.
            totaldqmag = np.sum(self.dqmaga[n0+1:self.nnodes])
            print(" totaldqmag = %1.3f" %totaldqmag)
            dqavg = totaldqmag/(self.nnodes-1)

            #if climb:
            if self.climb or rtype==2:
                h1dqmag = np.sum(self.dqmaga[1:self.TSnode+1])
                h2dqmag = np.sum(self.dqmaga[self.TSnode+1:self.nnodes])
                if self.print_level>1:
                    print(" h1dqmag, h2dqmag: %1.1f %1.1f" % (h1dqmag,h2dqmag))
           
            # => Using average <= #
            if i==0 and rtype==0:
                print(" using average")
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

                print(" edist: ", end=' ')
                for n in range(n0+1,self.nnodes):
                    print(" {:1.1}".format(edist[n]), end=' ')
                print() 
                
                totaledq = np.sum(edist[n0+1:self.nnodes])
                edqavg = totaledq/(self.nnodes-1)

            if i==0:
                print(" rpart: ", end=' ')
                for n in range(1,self.nnodes):
                    print(" {:1.2}".format(rpart[n]), end=' ')
                print()

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

            if self.print_level>0:
                for n in range(n0+1,self.nnodes-1):
                    print(" disp[{}]: {:1.2}".format(n,rpmove[n]), end=' ')
                print()
                print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 0.02:
                break

            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>0.:
                    #print "moving node %i %1.3f" % (n,rpmove[n])
                    self.newic.xyz = self.nodes[n].xyz

                    if rpmove[n] < 0.:
                        ictan[n] = np.copy(ictan0[n]) 
                    else:
                        ictan[n] = np.copy(ictan0[n+1]) 

                    # TODO
                    # it would speed things up a lot if we don't reform  coordinate basis
                    # perhaps check if similarity between old ictan and new ictan is close?
                    self.newic.update_coordinate_basis(ictan[n])

                    constraint = self.newic.constraints[:,0]
                    dq = rpmove[n]*constraint
                    self.newic.update_xyz(dq,verbose=True)
                    self.nodes[n].xyz = self.newic.xyz.copy()

                    # new 6/7/2019
                    if self.nodes[n].newHess==0:
                        self.nodes[n].newHess=2

                #TODO might need to recalculate energy here for seam? 

        for n in range(1,self.nnodes-1):
            self.nodes[n].xyz = self.com_rotate_move(n-1,n+1,n)

        print(' spacings (end ic_reparam, steps: {}/{}):'.format(i+1,ic_reparam_steps))
        for n in range(1,self.nnodes):
            print(" {:1.2}".format(self.dqmaga[n]), end=' ')
        print()
        print("  disprms: {:1.3}\n".format(disprms))

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0):  #see line 3863 of gstring.cpp
        """
        
        """
        nifty.printcool("Reparamerizing string nodes")
        #close_dist_fix(0) #done here in GString line 3427.
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)
        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(self.nnodes)
        edist = np.zeros(self.nnodes)
        emax = -1000 # And this?

        for i in range(ic_reparam_steps):
            self.get_tangents_1g()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
            if self.print_level>1:
                if i==0:
                    print(" totaldqmag (without inner): {:1.2}\n".format(totaldqmag))
                print(" printing spacings dqmaga: ")
                for n in range(self.nnodes):
                    print(" {:1.2}".format(self.dqmaga[n]), end=' ')
                    if (n+1)%5==0:
                        print()
                print() 
            
            if i == 0:
                if self.nn!=self.nnodes:
                    rpart = np.zeros(self.nnodes)
                    for n in range(n0+1,self.nR):
                        rpart[n] = 1.0/(self.nn-2)
                    for n in range(self.nnodes-self.nP,self.nnodes-1):
                        rpart[n] = 1.0/(self.nn-2)
                    if self.print_level>1:
                        if i==0:
                            print(" rpart: ")
                            for n in range(1,self.nnodes):
                                print(" {:1.2}".format(rpart[n]), end=' ')
                                if (n)%5==0:
                                    print()
                            print()
                else:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
            nR0 = self.nR
            nP0 = self.nP

            # TODO CRA 3/2019 why is this here?
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
            if self.print_level>1:
                for n in range(n0+1,self.nnodes-1):
                    print(" disp[{}]: {:1.2f}".format(n,rpmove[n]), end=' ')
                print()
                print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 1e-2:
                break

            for n in range(n0+1,self.nnodes-1):
                if isinstance(self.nodes[n],Molecule):
                    if rpmove[n] > 0:
                        self.nodes[n].update_coordinate_basis(constraints=self.ictan[n])
                        constraint = self.nodes[n].constraints[:,0]
                        dq0 = rpmove[n]*constraint
                        if self.print_level>1:
                            print(" dq0[constraint]: {:1.3}".format(rpmove[n]))
                        self.nodes[n].update_xyz(dq0,verbose=True)
                    else:
                        pass
        print(" spacings (end ic_reparam, steps: {}/{}):".format(i,ic_reparam_steps), end=' ')
        for n in range(self.nnodes):
            print(" {:1.2}".format(self.dqmaga[n]), end=' ')
        print("  disprms: {:1.3}".format(disprms))

        #TODO old GSM does this here
        #Failed = check_array(self.nnodes,self.dqmaga)
        #If failed, do exit 1

    def get_eigenv_finite(self,en):
        ''' Modifies Hessian using RP direction'''
        print("modifying Hessian with RP")

        self.nodes[en].update_coordinate_basis()

        self.newic.xyz = self.nodes[en].xyz
        Vecs = self.newic.update_coordinate_basis(self.ictan[en])
        #nicd = self.newic.num_coordinates
        #num_ics = self.newic.num_primitives 
        print(" number of primitive coordinates: %i" % self.newic.num_primitives)
        print(" number of non-redundant coordinates: %i" % self.newic.num_coordinates)

        E0 = self.energies[en]/units.KCAL_MOL_PER_AU
        Em1 = self.energies[en-1]/units.KCAL_MOL_PER_AU
        if en+1<self.nnodes:
            Ep1 = self.energies[en+1]/units.KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        q0 = self.newic.coordinates[0]
        #print "q0 is %1.3f" % q0
        print(self.newic.coord_basis.shape)
        #tan0 = self.newic.coord_basis[:,0]
        constraint = self.newic.constraints[:,0]
        tan0 = block_matrix.dot(Vecs,constraint)
        #print "tan0"
        #print tan0

        self.newic.xyz = self.nodes[en-1].xyz
        qm1 = self.newic.coordinates[0]
        #print "qm1 is %1.3f " %qm1

        if en+1<self.nnodes:
            self.newic.xyz = self.nodes[en+1].xyz
            qp1 = self.newic.coordinates[0]
        else:
            qp1 = qm1

        #print "qp1 is %1.3f" % qp1

        if en == self.TSnode:
            print(" TS Hess init'd w/ existing Hintp")

        self.newic.xyz = self.nodes[en].xyz
        Vecs =self.newic.update_coordinate_basis()

        self.newic.Primitive_Hessian = self.nodes[en].Primitive_Hessian
        self.newic.form_Hessian_in_basis()

        tan = block_matrix.dot(block_matrix.transpose(Vecs),tan0)   #nicd,1
        #print "tan"
        #print tan

        Ht = np.dot(self.newic.Hessian,tan) #(nicd,nicd)(nicd,1) = nicd,1
        tHt = np.dot(tan.T,Ht) 

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        print(" tHt %1.3f a: %1.1f b: %1.1f c: %1.3f" % (tHt,a[0],b[0],c[0]))

        ttt = np.outer(tan,tan)
        #print "Hint before"
        #with np.printoptions(threshold=np.inf):
        #    print self.newic.Hessian
        #eig,tmph = np.linalg.eigh(self.newic.Hessian)
        #print "initial eigenvalues"
        #print eig
       
        self.newic.Hessian += (c-tHt)*ttt
        self.nodes[en].Hessian = self.newic.Hessian
        #with np.printoptions(threshold=np.inf):
        #    print self.nodes[en].Hessian
        #print "shape of Hessian is %s" % (np.shape(self.nodes[en].Hessian),)

        self.nodes[en].newHess = 5

        if False:
            print("newHess of node %i %i" % (en,self.nodes[en].newHess))
            eigen,tmph = np.linalg.eigh(self.nodes[en].Hessian) #nicd,nicd
            print("eigenvalues of new Hess")
            print(eigen)

        # reset pgradrms ? 

    def set_V0(self):
        raise NotImplementedError 

    def mult_steps(self,n,opt_steps):
        exsteps=1
        tsnode = int(self.TSnode)

        if self.find and self.energies[n]+1.5 > self.energies[self.TSnode] and n!=tsnode:  #
            exsteps=2
            print(" multiplying steps for node %i by %i" % (n,exsteps))
            self.optimizer[n].conv_grms = self.options['CONV_TOL']      # TODO this is not perfect here
        if (self.find or self.climb) and n==tsnode: 
            exsteps=3
            #exsteps = self.ts_exsteps
            print(" multiplying steps for node %i by %i" % (n,exsteps))

        elif not (self.find or self.climb) and self.energies[tsnode] > 1.75*self.energies[tsnode-1] and self.energies[tsnode] > 1.75*self.energies[tsnode+1] and self.done_growing and n==tsnode: 
            exsteps=2
            print(" multiplying steps for node %i by %i" % (n,exsteps))
        return exsteps*opt_steps

    def set_opt_type(self,n,quiet=False):
        #TODO error for seam climb
        opt_type='ICTAN' 
        if self.climb and n==self.TSnode and not self.find:
            opt_type='CLIMB'
            #opt_type='BEALES_CG'
        elif self.find and n==self.TSnode:
            opt_type='TS'
        elif self.nodes[n].PES.lot.do_coupling:
            opt_type='SEAM'
        elif self.climb and n==self.TSnode and opt_type=='SEAM':
            opt_type='TS-SEAM'
        if not quiet:
            print((" setting node %i opt_type to %s" %(n,opt_type)))

        if isinstance(self.optimizer[n],beales_cg) and opt_type!="BEALES_CG":
            raise RuntimeError("This shouldn't happen")

        return opt_type

    def set_finder(self,rtype):
        assert rtype in [0,1,2], "rtype not defined"
        print('')
        print("*********************************************************************")
        if rtype==2:
            print("****************** set climber and finder to True *******************")
            self.climber=True
            self.finder=True
        elif rtype==1:
            print("***************** setting climber to True*************************")
            self.climber=True
        else:
            print("******** Turning off climbing image and exact TS search **********")
        print("*********************************************************************")
   
    def restart_string(self,xyzfile='restart.xyz'):
        nifty.printcool("Restarting string from file")
        self.growth_direction=0
        with open(xyzfile) as f:
            nlines = sum(1 for _ in f)
        #print "number of lines is ", nlines
        with open(xyzfile) as f:
            natoms = int(f.readlines()[2])

        #print "number of atoms is ",natoms
        nstructs = (nlines-6)/ (natoms+5) #this is for three blocks after GEOCON
        nstructs = int(nstructs)
        
        #print "number of structures in restart file is %i" % nstructs
        coords=[]
        grmss = []
        atomic_symbols=[]
        dE = []
        with open(xyzfile) as f:
            f.readline()
            f.readline() #header lines
            # get coords
            for struct in range(nstructs):
                tmpcoords=np.zeros((natoms,3))
                f.readline() #natoms
                f.readline() #space
                for a in range(natoms):
                    line=f.readline()
                    tmp = line.split()
                    tmpcoords[a,:] = [float(i) for i in tmp[1:]]
                    if struct==0:
                        atomic_symbols.append(tmp[0])
                coords.append(tmpcoords)
            ## Get energies
            #f.readline() # line
            #f.readline() #energy
            #for struct in range(nstructs):
            #    self.energies[struct] = float(f.readline())
            ## Get grms
            #f.readline() # max-force
            #for struct in range(nstructs):
            #    grmss.append(float(f.readline()))
            ## Get dE
            #f.readline()
            #for struct in range(nstructs):
            #    dE.append(float(f.readline()))

        # create newic object
        self.newic  = Molecule.copy_from_options(self.nodes[0])

        # initialize lists
        self.energies = [0.]*nstructs
        self.gradrms = [0.]*nstructs
        self.dE = [1000.]*nstructs

        # initial energy
        self.nodes[0].V0 = self.nodes[0].energy 
        #self.nodes[0].gradrms=grmss[0]
        #self.nodes[0].PES.dE = dE[0]
        #self.nodes[-1].gradrms=grmss[-1]
        #self.nodes[-1].PES.dE = dE[-1]
        self.energies[0] = 0.
        print(" initial energy is %3.4f" % self.nodes[0].energy)

        self.nnodes=self.nR=nstructs
        self.isRestarted=True
        self.done_growing=True
        for struct in range(1,nstructs-1):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[struct-1],coords[struct],new_node_id=struct,copy_wavefunction=False)
            #self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
            #self.nodes[struct].gradrms=grmss[struct]
            #self.nodes[struct].PES.dE = dE[struct]
            self.nodes[struct].newHess=5

        print(" doing one-time ic_reparam REMOVE ME")
        self.get_tangents_1()
        self.ic_reparam(ic_reparam_steps=8)
        self.write_xyz_files(iters=1,base='grown_string1',nconstraints=1)

        for struct in range(1,nstructs-1):
            print(" energy of node %i is %5.4f" % (struct,self.nodes[struct].energy))
            self.energies[struct] = self.nodes[struct].energy - self.nodes[0].V0
            print(" Relative energy of node %i is %5.4f" % (struct,self.energies[struct]))

        print(" setting all interior nodes to active")
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.optimizer[n].conv_grms=self.options['CONV_TOL']*2.5
            self.optimizer[n].options['DMAX'] = 0.05
        print(" V_profile: ", end=' ')
        for n in range(self.nnodes):
            print(" {:7.3f}".format(float(self.energies[n])), end=' ')
        print()
        #print(" grms_profile: ", end=' ')
        #for n in range(self.nnodes):
        #    print(" {:7.3f}".format(float(self.nodes[n].gradrms)), end=' ')
        #print()
        print(" dE_profile: ", end=' ')
        for n in range(self.nnodes):
            print(" {:7.3f}".format(float(self.nodes[n].difference_energy)), end=' ')
        print()

        #tmp for testing
        #self.emax = self.energies[self.TSnode]
        #self.climb=True
        #self.optimizer[self.TSnode] = beales_cg(self.optimizer[0].options.copy().set_values({"Linesearch":"backtrack"}))


    def com_rotate_move(self,iR,iP,iN):
        mfrac = 0.5
        if self.nnodes - self.nn+1  != 1:
            mfrac = 1./(self.nnodes - self.nn+1)

        xyz0 = self.nodes[iR].xyz.copy()
        xyz1 = self.nodes[iN].xyz.copy()
        com0 = self.nodes[iR].center_of_mass
        com1 = self.nodes[iN].center_of_mass

        # rotate to be in maximal coincidence with 0
        # assumes iP i.e. 2 is also in maximal coincidence

        # align xyz1 to xyz0?
        U = rotate.get_rot(xyz0,xyz1)
        new_xyz = np.dot(xyz1,U)

        # align 
        if self.nodes[iP] != None:
            xyz2 = self.nodes[iP].xyz.copy()
            com2 = self.nodes[iP].center_of_mass

            if abs(iN-iR) > abs(iN-iP):
                avg_com = mfrac*com2 + (1.-mfrac)*com0
            else:
                avg_com = mfrac*com0 + (1.-mfrac)*com2
            dist = avg_com - com1  #final minus initial
            new_xyz += dist
        else:
            dist = com0 - com1  #final minus initial
            new_xyz += dist

        return new_xyz

def run(args):

    node,optimizer,ictan,opt_steps,opt_type,refE,n,s0,gp_prim = args
    #print(" entering run: {}".format(n))
    sys.stdout.flush()

    print()
    nifty.printcool("Optimizing node {}".format(n))

    if opt_type=="BEALES_CG": 
        print(opt_type)
        print(type(optimizer))
    else:
        print(type(optimizer))

    # => do constrained optimization
    try:

        if opt_type!="BEALES_CG":
            optimizer.optimize(
                    molecule=node,
                    refE=refE,
                    opt_type=opt_type,
                    opt_steps=opt_steps,
                    ictan=ictan
                    )
        else:
            optimizer.optimize(
                    molecule=node,
                    s0=s0,
                    gp_prim=gp_prim,
                    refE=refE,
                    opt_type=opt_type,
                    opt_steps=opt_steps,
                    ictan=ictan,
                    )

    except:
        RuntimeError

#def mod(
#        node,
#        optimizer,
#        ictan,
#        steps,
#        opt_type,
#        refE,
#        n,
#        s,
#        gp_prim,
#        out_queue,
#        )
#
#    optimizer.optimize(
#            molecule = node,
#            s0=s,
#            gp_prim=gp_prim,
#            refE = refE,
#            opt_type=opt_type,
#            opt_steps=steps,
#            ictan=ictan,
#            )
#
#    out_queue.put(test)



def mod(test,nn,out_queue):
    print(test.num)
    test.num = np.random.randn()
    print(test.num)
    test.name = nn
    out_queue.put(test)



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
    mol=next(pb.readfile("xyz",filepath))
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
    
    # => DLC constructor <= #
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
    param = parameters.from_options(opt_type='UNCONSTRAINED')
    #gsm = Base_Method.from_options(ICoord1=ic1,param,optimize=eigenvector_follow)

    #gsm.optimize(nsteps=20)


