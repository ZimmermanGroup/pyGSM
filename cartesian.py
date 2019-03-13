
from pes import PES
import numpy as np
#from lbfgs import *
from conjugate_gradient import *
from prim_internals import PrimitiveInternalCoordinates

class CartesianCoordinates(PrimitiveInternalCoordinates):
    """
    Cartesian coordinate system, written as a kind of internal coordinate class.  
    This one does not support constraints, because that requires adding some 
    primitive internal coordinates.
    """
    def __init__(self, molecule, **kwargs):
        super(CartesianCoordinates, self).__init__(molecule)
        self.Internals = []
        self.cPrims = []
        self.cVals = []
        self.elem = molecule.elem
        for i in range(molecule.natoms):
            self.add(CartesianX(i, w=1.0))
            self.add(CartesianY(i, w=1.0))
            self.add(CartesianZ(i, w=1.0))
        if 'constraints' in kwargs and kwargs['constraints'] is not None:
            raise RuntimeError('Do not use constraints with Cartesian coordinates')

        self.Vecs = np.eye(3*self.natoms)

    def guess_hessian(self, xyz):
        return 0.5*np.eye(len(xyz.flatten()))

    def calcGrad(self,xyz,gradx):
        return gradx

    def newCartesian(self,xyz,dq,verbose=True):
        return xyz+dq

#class Cartesian:
#    def __init__(self,q,geom,pes):
#        self.q = q  # the value in the current basis
#        self.xs=[]
#        self.g=[]
#        self.fx=[]
#        self.xnorm=[]
#        self.gnorm=[]
#        self.step=[]
#        self.geom=geom
#        self.coords=manage_xyz.xyz_to_np(self.geom)
#        self.geoms=[]
#        self.natoms=len(geom)
#        self.PES = pes
#
#    def append_data(self,x,g,fx,xnorm,gnorm,step):
#        self.q = x
#        self.xs.append(x)
#        self.g.append(g)
#        self.coords = np.reshape(x,(self.natoms,3))
#        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords)
#        self.geoms.append(self.geom)
#        self.fx.append(fx)
#        self.xnorm.append(xnorm)
#        self.gnorm.append(gnorm)
#        self.step.append(step)
#        return
#
#
#    def proc_evaluate(self,x):
#        self.coords = np.reshape(x,(self.natoms,3))
#        self.geom = manage_xyz.np_to_xyz(self.geom,self.coords)
#        fx =self.PES.get_energy(self.geom)
#        g = np.ndarray.flatten(self.PES.get_gradient(self.geom)*KCAL_MOL_PER_AU)
#        self.PES.lot.hasRanForCurrentCoords= False
#        return fx,g
#
#if __name__ =='__main__':
#    from qchem import *
#    import pybel as pb
#    basis="sto-3g"
#    nproc=4
#
#    filepath="examples/tests/bent_benzene.xyz"
#    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
#    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)
#    geom=manage_xyz.read_xyz(filepath,scale=1)   
#
#
#    if False:
#        # cartesian optimization
#        param = lbfgs_parameters(cart.proc_evaluate,cart,min_step=0.0005)
#        x=np.ndarray.flatten(cart.coords)
#        fx=cart.PES.get_energy(cart.geom)
#        lb = lbfgs(len(x), x, fx, param,opt_steps=5)
#        ret = lb.do_lbfgs(opt_steps=4)
#        print cart.fx
#        manage_xyz.write_xyzs('prc.xyz',cart.geoms,scale=1.0)
#        #ret = lb.do_codys_lbfgs(opt_steps=4)
#        #print cart.fx
#        #manage_xyz.write_xyzs('prc2.xyz',cart.geoms,scale=1.0)
#    else:
#        # cartesian optimization
#
#        from conjugate_gradient import conjugate_gradient
#        from _linesearch import backtrack,parameters
#        # => Cartesian constructor <= #
#        coords = manage_xyz.xyz_to_np(geom)
#        q=coords.flatten()
#        print "initial q"
#        print q
#        cart = Cartesian(q,geom,pes)
#
#        #param = parameters(min_step=0.00001)
#        param = parameters.from_options(opt_type='UNCONSTRAINED',OPTTHRESH=1e-10)
#        cg = conjugate_gradient()
#
#        cg.optimize(cart,param,3)
#
#        print cart.fx
#        manage_xyz.write_xyzs('prc.xyz',cart.geoms,scale=1.0)
#
