# standard library imports
import sys
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from ._linesearch import backtrack,NoLineSearch


#TODO Add primitive constraint e.g. a list of internal coordinates to be left basically frozen throughout optimization
class base_optimizer(object):
    ''' some common functions that the children can use (ef, cg, hybrid ef/cg, etc). 
    e.g. walk_up, dgrad_step, what else?
    '''

    @staticmethod
    def default_options():
        """ default options. """

        if hasattr(base_optimizer, '_default_options'): return base_optimizer._default_options.copy()
        opt = options.Options() 


        opt.add_option(
                key='OPTTHRESH',
                value=0.0005,
                required=False,
                allowed_types=[float],
                doc='Convergence threshold'
                )

        opt.add_option(
                key='DMAX',
                value=0.1,
                doc='max step size',
                )

        opt.add_option(
                key='SCALEQN',
                value=1,
                )

        opt.add_option(
                key='Linesearch',
                value="NoLineSearch",
                allowed_values=["NoLineSearch","backtrack"],
                required=False,
                doc='A function to do a linesearch e.g. bactrack,NoLineSearch, etc.'
                )

        opt.add_option(
                key='MAXAD',
                value=0.075,
                )

        opt.add_option(
                key='print_level',
                value=1,
                doc="control the printout, 0 less, 1 more, 2 too much"
                )

        opt.add_option(
                key='HESS_TANG_TOL_TS',
                value=0.35,
                doc='Hessian  overlap with tangent tolerance for TS node'
                )

        base_optimizer._default_options = opt
        return base_optimizer._default_options.copy()

    @classmethod
    def from_options(cls,**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return cls(cls.default_options().set_values(kwargs))

    def __init__(self,
            options,
            ):

        self.options = options
        if self.options['Linesearch']=="backtrack":
            self.Linesearch=backtrack
        elif self.options['Linesearch']=="NoLineSearch":
            self.Linesearch=NoLineSearch

        
        # additional convergence criterion (default parameters for Q-Chem)
        self.conv_disp = 12e-4 #max atomic displacement
        self.conv_gmax = 3e-4 #max gradient
        self.conv_Ediff = 1e-6 #E diff
        self.conv_grms = options['OPTTHRESH']

        # TS node properties
        self.nneg = 0  # number of negative eigenvalues
        self.DMIN = self.options['DMAX']/20.

        # Hessian
        self.Hint=None
        self.dx=0.
        self.dg=0.

        # additional parameters needed by linesearch
        self.linesearch_parameters = {
                'epsilon':1e-5,
                'ftol':1e-4,
                'wolfe':0.9,
                'max_linesearch':5,
                'min_step':self.DMIN,
                'max_step':0.5,
        }
        return

    @property
    def conv_grms(self):
        return self.options['OPTTHRESH']

    @conv_grms.setter
    def conv_grms(self,value):
        self.options['OPTTHRESH']=value

    def get_nconstraints(self,opt_type):
        if opt_type in ["ICTAN", "CLIMB"]:
            nconstraints = 1
        elif opt_type in ['MECI']:
            nconstraints=2
        elif opt_type in ['SEAM','TS-SEAM']:
            nconstraints=3
        else:
            nconstraints=0
        return nconstraints

    def check_inputs(self,molecule,opt_type,ictan):
        if opt_type in ['MECI','SEAM','TS-SEAM']:
            assert molecule.PES.lot.do_coupling==True,"Turn do_coupling on."                   
        #elif opt_type not in ['MECI','SEAM','TS-SEAM']: 
        #    assert molecule.PES.lot.do_coupling==False,"Turn do_coupling off."                 
        if opt_type=="UCONSTRAINED":  
            assert ictan==None
        if opt_type in ['ICTAN','CLIMB','TS', 'SEAM','TS-SEAM']  and ictan.any()==None:
            raise RuntimeError("Need ictan")
        if opt_type in ['TS','TS-SEAM']:
            assert molecule.isTSnode,"only run climb and eigenvector follow on TSnode."  

    #def converged(self,g,nconstraints):
    #    # check if finished
    #    gradrms = np.sqrt(np.dot(g[nconstraints:].T,g[nconstraints:])/n)
    #    #print "current gradrms= %r au" % gradrms
    #    #print "gnorm =",gnorm
    #    
    #    gmax = np.max(g[nconstraints:])/ANGSTROM_TO_AU
    #    #print "maximum gradient component (au)", gmax

    #    if gradrms <self.conv_grms:
    #        print('[INFO] converged')
    #        return True

    #    #if gradrms <= self.conv_grms  or \
    #    #    (self.disp <= self.conv_disp and self.Ediff <= self.conv_Ediff) or \
    #    #    (gmax <= self.conv_gmax and self.Ediff <= self.conv_Ediff):
    #    #    print '[INFO] converged'
    #    #    return True
    #    return False

    def set_lambda1(self,opt_type,eigen,maxoln=None):
        if opt_type == 'TS':
            leig = eigen[1]  #! this is eigen[0] if update_ic_eigen() ### also diff values 
            if maxoln!=0:
                leig = eigen[0]
            if leig < 0. and maxoln==0:
                lambda1 = -leig
            else:
                lambda1 = 0.01
        else:
            leig = eigen[0]
            if leig < 0:
                lambda1 = -leig+0.015
            else:
                lambda1 = 0.005
        if abs(lambda1)<0.005: lambda1 = 0.005
    
        return lambda1

    def get_constraint_vectors(self,molecule,opt_type,ictan=None):
        nconstraints=self.get_nconstraints(opt_type)

        if opt_type=="UNCONSTRAINED":
            constraints=None
        elif opt_type=='ICTAN' or opt_type=="CLIMB":
            constraints = ictan
        elif opt_type=='MECI':
            print("MECI")
            dgrad_U = np.dot(molecule.coord_basis,molecule.difference_gradient)
            dvec_U = np.dot(molecule.coord_basis,molecule.derivative_coupling)
            constraints = np.hstack((dgrad_U,dvec_U))
        elif opt_type=="SEAM" or opt_type=="TS-SEAM":
            dgrad_U = np.dot(molecule.coord_basis,molecule.difference_gradient)
            dvec_U = np.dot(molecule.coord_basis,molecule.derivative_coupling)
            constraints = np.hstack((ictan,dgrad_U,dvec_U))
        else:
            raise NotImplementedError
        return constraints

    def get_constraint_steps(self,molecule,opt_type,g):
        nconstraints=self.get_nconstraints(opt_type)
        n=len(g)
        #TODO Raise Error for CartesianCoordinates
    
        #TODO 4/24/2019 block matrix/distributed constraints
        constraint_steps = np.zeros((n,1))
        # => ictan climb
        if opt_type=="CLIMB": 
            constraint_steps[0]=self.walk_up(g,0)
        # => MECI
        elif opt_type=='MECI': 
            constraint_steps[0] = self.dgrad_step(molecule) #first vector is x
        elif opt_type=='SEAM':
            constraint_steps[1]=self.dgrad_step(molecule)
        # => seam climb
        elif opt_type=='TS-SEAM':
            constraint_steps[0]=self.walk_up(g,0)
            constraint_steps[1]=self.dgrad_step(molecule)
        return constraint_steps

    def dgrad_step(self,molecule):
        """ takes a linear step along dgrad"""

        norm_dg = np.linalg.norm(molecule.difference_gradient)
        if self.options['print_level']>0:
            print(" norm_dg is %1.4f" % norm_dg)
            print(" dE is %1.4f" % molecule.difference_energy)

        dq = -molecule.difference_energy/units.KCAL_MOL_PER_AU/norm_dg 
        if dq<-0.075:
            dq=-0.075

        return dq

    def walk_up(self,g,n):
        """ walk up the n'th DLC"""
        #assert isinstance(g[n],float), "gradq[n] is not float!"
        #if self.print_level>0:
        #    print(' gts: {:1.4f}'.format(self.gradq[n,0]))
        #self.buf.write(' gts: {:1.4f}'.format(self.gradq[n,0]))
        SCALEW = 1.0
        SCALE = self.options['SCALEQN']
        dq = g[n,0]/SCALE
        print(" walking up the %i coordinate = %1.4f" % (n,dq))
        if abs(dq) > self.options['MAXAD']/SCALEW:
            dq = np.sign(dq)*self.options['MAXAD']/SCALE
        return dq

    #def walk_up_DNR(self,g,n):
    #    SCALE =self.options['SCALEQN']
    #    if c_obj.newHess>0: SCALE = self.options['SCALEQN']*c_obj.newHess  # what to do about newHess?
    #    if self.options['SCALEQN']>10.0: SCALE=10.0
    #    
    #    # convert to eigenvector basis
    #    temph = self.Hint[:n,:n]
    #    e,v_temp = np.linalg.eigh(temph)
    #    v_temp = v_temp.T
    #    gqe = np.dot(v_temp,g[:n])
    #    
    #    lambda1 = self.set_lambda1('NOT-TS',e) 

    #    dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
    #    dqe0 = [ np.sign(i)*self.options['MAXAD'] if abs(i)>self.options['MAXAD'] else i for i in dqe0 ]
    #    
    #    # => Convert step back to DLC basis <= #
    #    dq_tmp = np.dot(v_temp.T,dqe0)
    #    dq_tmp = [ np.sign(i)*self.options['MAXAD'] if abs(i)>self.options['MAXAD'] else i for i in dq_tmp ]
    #    dq = np.zeros((c_obj.nicd_DLC,1))
    #    for i in range(n): dq[i,0] = dq_tmp[i]

    #    return dq


    def step_controller(self,step,ratio,gradrms,pgradrms,dEpre,opt_type,dE_iter):
        # => step controller controls DMAX/DMIN <= #
        if dE_iter > 0.01 and opt_type in ['UNCONSTRAINED','ICTAN']:
            if step<self.options['DMAX']:
                self.options['DMAX'] = step/1.5
            else:
                self.options['DMAX'] = self.options['DMAX']/1.5
        if (ratio<0.25 or ratio>2.0):  #and abs(dEpre)>0.05:
            #or ratio >1.5
            if self.options['print_level']>0:
                print(" decreasing DMAX")
            if step<self.options['DMAX']:
                self.options['DMAX'] = step/1.2
            else:
                self.options['DMAX'] = self.options['DMAX']/1.2
        elif ratio>0.75 and ratio<1.25 and step > self.options['DMAX'] and gradrms<(pgradrms*1.35):
            if self.options['print_level']>0:
                print(" increasing DMAX")
            #self.buf.write(" increasing DMAX")
            #if step > self.options['DMAX']:
            #    if True:
            #        print " wolf criterion increased stepsize... ",
            #        print " setting DMAX to wolf  condition...?"
            #        self.options['DMAX']=step
            self.options['DMAX']=self.options['DMAX']*1.2 + 0.01
            if self.options['DMAX']>0.25:
                self.options['DMAX']=0.25
        
        if self.options['DMAX']<self.DMIN:
            self.options['DMAX']=self.DMIN
        print(" DMAX %1.2f" % self.options['DMAX'])

    def eigenvector_step(self,molecule,g):

        SCALE =self.options['SCALEQN']
        if molecule.newHess>0: SCALE = self.options['SCALEQN']*molecule.newHess
        if self.options['SCALEQN']>10.0: SCALE=10.0

        if self.options['print_level']>1:
            print("new_hess %i" % molecule.newHess)

        new_rows = []
        if self.options['print_level']>1:
            print("constraints")
            print(molecule.constraints.T)

        P =np.eye(len(molecule.constraints),dtype=float) -  np.outer(molecule.constraints,molecule.constraints.T)
        self.Hessian = np.dot(np.dot(P,molecule.Hessian),P)

        e,v_temp = np.linalg.eigh(self.Hessian)
        gqe = np.dot(v_temp.T,g)
        lambda1 = self.set_lambda1('NOT-TS',e) 

        if self.options['print_level']>1:
            print(" eigenvalues ",e)
        if self.options['print_level']>1:
            print(" eigenvectors ",v_temp)

        if self.options['print_level']>1:
            print(" g ",g.T)

        if self.options['print_level']>1:
            print(" gqe ",gqe.T)

        dqe0 = -gqe.flatten()/(e+lambda1)/SCALE
        dqe0 = [ np.sign(i)*self.options['MAXAD'] if abs(i)>self.options['MAXAD'] else i for i in dqe0 ]
        dqe0 = np.asarray(dqe0)
        
        # => Convert step back to DLC basis <= #
        dq = np.dot(v_temp,dqe0)
        #TODO THIS CAN BE MADE MORE EFFICIENT
        dq = [ np.sign(i)*self.options['MAXAD'] if abs(i)>self.options['MAXAD'] else i for i in dq ]
        dq = np.asarray(dq)

        print("check overlap")
        dq = np.reshape(dq,(-1,1))
        dq = dq - np.dot(molecule.constraints.T,dq)*molecule.constraints
        print(np.dot(dq.T,molecule.constraints))
        if self.options['print_level']>1:
            print(" dq ",dq.T)
        return np.reshape(dq,(-1,1))

    # need to modify this only for the DLC region
    def TS_eigenvector_step(self,molecule,g,ictan):
        '''
        Takes an eigenvector step using the Bofill updated Hessian ~1 negative eigenvalue in the
        direction of the reaction path.

        '''
        SCALE =self.options['SCALEQN']
        if molecule.newHess>0: SCALE = self.options['SCALEQN']*molecule.newHess
        if self.options['SCALEQN']>10.0: SCALE=10.0

        # constraint vector
        norm = np.linalg.norm(ictan)
        C = ictan/norm
        Vecs = molecule.coord_basis
        Cn = np.linalg.multi_dot([Vecs,Vecs.T,C])
        norm = np.linalg.norm(Cn)
        Cn = Cn/norm

        # => get eigensolution of Hessian <= 
        eigen,tmph = np.linalg.eigh(molecule.Hessian) #nicd,nicd 
        tmph = tmph.T

        #TODO nneg should be self and checked
        self.nneg = sum(1 for e in eigen if e<-0.01)

        #=> Overlap metric <= #
        overlap = np.dot(np.dot(tmph,Vecs.T),Cn) 

        print("overlap", overlap[:4].T)
        # Max overlap metrics
        path_overlap,maxoln = self.maxol_w_Hess(overlap[0:4])
        print(" t/ol %i: %3.2f" % (maxoln,path_overlap))

        # => set lamda1 scale factor <=#
        lambda1 = self.set_lambda1('TS',eigen,maxoln)

        maxol_good=True
        if path_overlap < self.options['HESS_TANG_TOL_TS']:
            maxol_good = False

        if maxol_good:
            # => grad in eigenvector basis <= #
            gqe = np.dot(tmph,g)
            path_overlap_e_g = gqe[maxoln]
            print(' gtse: {:1.4f} '.format(path_overlap_e_g[0]))
            # => calculate eigenvector step <=#
            dqe0 = np.zeros((molecule.num_coordinates,1))
            for i in range(molecule.num_coordinates):
                if i != maxoln:
                    dqe0[i] = -gqe[i] / (abs(eigen[i])+lambda1) / SCALE
            lambda0 = 0.0025
            dqe0[maxoln] = gqe[maxoln] / (abs(eigen[maxoln]) + lambda0)/SCALE

            # => Convert step back to DLC basis <= #
            dq = np.dot(tmph.T,dqe0)  # should it be transposed?
            dq = [ np.sign(i)*self.options['MAXAD'] if abs(i)>self.options['MAXAD'] else i for i in dq ]
        else:
            # => if overlap is small use Cn as Constraint <= #
            molecule.update_coordinate_basis(ictan)
            molecule.form_Hessian_in_basis()
            dq = self.eigenvector_step(molecule,g,1)  #hard coded!

        return dq,maxol_good

    def maxol_w_Hess(self,overlap):
        # Max overlap metrics
        absoverlap = np.abs(overlap)
        path_overlap = np.max(absoverlap)
        path_overlap_n = np.argmax(absoverlap)
        return path_overlap,path_overlap_n

    def update_Hessian(self,molecule,mode='BFGS'):
        '''
        mode 1 is BFGS, mode 2 is BOFILL
        '''
        assert mode=='BFGS' or mode=='BOFILL', "no update implemented with that mode"
        # do this even if mode==BOFILL
        change = self.update_bfgs(molecule)

        if molecule.coord_obj.__class__.__name__=='DelocalizedInternalCoordinates':
            molecule.update_Primitive_Hessian(change=change)
            if self.options['print_level']>1:
                print("change")
                print(change)
                print(" updated primitive internals Hessian")
                print(molecule.Primitive_Hessian)
            if mode=='BFGS':
                molecule.form_Hessian_in_basis()
            if mode=='BOFILL':
                change=self.update_bofill(molecule)
                molecule.update_Hessian(change)
        #else:
        #    self.Hessian += change
        molecule.newHess-=1

        return change

    def update_bfgs(self,molecule):
        if not molecule.coord_obj.__class__.__name__=='CartesianCoordinates':
            return self.update_bfgsp(molecule)
        else:
            raise NotImplementedError

    def update_bfgsp(self,molecule):
        if self.options['print_level']>1:
            print("In update bfgsp")
            print('dx_prim ',self.dx_prim.T)
            print('dg_prim ',self.dg_prim.T)

        Hdx = np.dot(molecule.Primitive_Hessian, self.dx_prim)
        dxHdx = np.dot(np.transpose(self.dx_prim),Hdx)
        dgdg = np.outer(self.dg_prim,self.dg_prim)
        dgtdx = np.dot(np.transpose(self.dg_prim),self.dx_prim)
        change = np.zeros_like(molecule.Primitive_Hessian)

        if self.options['print_level']>1:
            print("Hdx")
            print(Hdx.T)
            print("dgtdx: %1.8f dxHdx: %1.8f dgdg" % (dgtdx,dxHdx))
            print("dgdg")
            print(dgdg)

        if dgtdx>0.:
            if dgtdx<0.001: dgtdx=0.001
            change += dgdg/dgtdx
        if dxHdx>0.:
            if dxHdx<0.001: dxHdx=0.001
            change -= np.outer(Hdx,Hdx)/dxHdx

        return change

    def update_bofill(self,molecule):
        print("in update bofill")

        G = np.copy(molecule.Hessian) #nicd,nicd
        Gdx = np.dot(G,self.dx) #(nicd,nicd)(nicd,1) = (nicd,1)
        dgmGdx = self.dg - Gdx # (nicd,1)

        # MS
        dgmGdxtdx = np.dot(dgmGdx.T,self.dx) #(1,nicd)(nicd,1)
        Gms = np.outer(dgmGdx,dgmGdx)/dgmGdxtdx

        #PSB
        dxdx = np.outer(self.dx,self.dx)
        dxtdx = np.dot(self.dx.T,self.dx)
        dxtdg = np.dot(self.dx.T,self.dg)
        dxtGdx = np.dot(self.dx.T,Gdx)
        dxtdx2 = dxtdx*dxtdx
        dxtdgmdxtGdx = dxtdg - dxtGdx 
        Gpsb = np.outer(dgmGdx,self.dx)/dxtdx + np.outer(self.dx,dgmGdx)/dxtdx - dxtdgmdxtGdx*dxdx/dxtdx

        # Bofill mixing 
        dxtE = np.dot(self.dx.T,dgmGdx) #(1,nicd)(nicd,1)
        EtE = np.dot(dgmGdx.T,dgmGdx)  #E is dgmGdx
        phi = 1. - dxtE*dxtE/(dxtdx*EtE)

        change = (1.-phi)*Gms + phi*Gpsb
        return change

    def update_constraint_bfgsp(self):
        return change
