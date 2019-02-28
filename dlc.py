from base_dlc import Base_DLC
from sklearn import preprocessing
import numpy as np


class DLC(Base_DLC):

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return DLC(DLC.default_options().set_values(kwargs))

    def get_nxyzics(self):
        return 0 

    def set_nicd(self):
        self.nicd=(self.natoms*3)-6
        self.nicd_DLC=self.nicd
  
    @staticmethod
    def create_DLC(icoordA,bondA,angleA,torsionA,mol,PES):
        return DLC(icoordA.options.copy().set_values({
            "bonds":bondA,
            "angles":angleA,
            "torsions":torsionA,
            'mol':mol,
            'PES':PES,
            }))

    def update_constraint_bfgsp(self):
        return change

    # should this be put in base_optimizer?
    def update_bfgsp(self):
        #print("In update bfgsp")
        dx = self.dqprim
        dg = self.gradqprim - self.pgradqprim
        Hdx = np.dot(self.Hintp,dx)
        if self.print_level>1:
            print "dg:", dg.T
            print "dx:", dx.T
            print "Hdx"
            print Hdx.T
        dxHdx = np.dot(np.transpose(dx),Hdx)
        dgdg = np.outer(dg,dg)
        dgtdx = np.dot(np.transpose(dg),dx)
        change = np.zeros_like(self.Hintp)

        if self.print_level>2:
            print "dgtdx: %1.3f dxHdx: %1.3f dgdg" % (dgtdx,dxHdx)
            print dgdg

        if dgtdx>0.:
            if dgtdx<0.001: dgtdx=0.001
            change += dgdg/dgtdx
        if dxHdx>0.:
            if dxHdx<0.001: dxHdx=0.001
            change -= np.outer(Hdx,Hdx)/dxHdx

        return change

    def update_bofill(self):

        #print "in update bofill"
        #unit test
        if False:
            self.Hint = np.loadtxt("hint.txt")
            self.Ut = np.loadtxt("Ut.txt")
            dx = np.loadtxt("dx.txt")
            dg = np.loadtxt("dg.txt")
            dx = np.reshape(dx,(self.nicd,1))
            dg = np.reshape(dg,(self.nicd,1))
        else:
            dx = np.copy(self.dq) #nicd,1
            dg = self.gradq - self.pgradq #nicd,1

        G = np.copy(self.Hint) #nicd,nicd
        Gdx = np.dot(G,dx) #(nicd,nicd)(nicd,1) = (nicd,1)
        dgmGdx = dg - Gdx # (nicd,1)

        # MS
        dgmGdxtdx = np.dot(dgmGdx.T,dx) #(1,nicd)(nicd,1)
        Gms = np.outer(dgmGdx,dgmGdx)/dgmGdxtdx

        #PSB
        dxdx = np.outer(dx,dx)
        dxtdx = np.dot(dx.T,dx)
        dxtdg = np.dot(dx.T,dg)
        dxtGdx = np.dot(dx.T,Gdx)
        dxtdx2 = dxtdx*dxtdx
        dxtdgmdxtGdx = dxtdg - dxtGdx 
        Gpsb = np.outer(dgmGdx,dx)/dxtdx + np.outer(dx,dgmGdx)/dxtdx - dxtdgmdxtGdx*dxdx/dxtdx

        # Bofill mixing 
        dxtE = np.dot(dx.T,dgmGdx) #(1,nicd)(nicd,1)
        EtE = np.dot(dgmGdx.T,dgmGdx)  #E is dgmGdx
        phi = 1. - dxtE*dxtE/(dxtdx*EtE)

        change = (1.-phi)*Gms + phi*Gpsb
        return change

    def opt_constraint(self,C):
        """
        This function takes a matrix of vectors wrtiten in the basis of ICs
        same as U vectors, and returns a new normalized Ut with those vectors as 
        basis vectors.
        """
        # normalize all constraints
        Cn = preprocessing.normalize(C.T,norm='l2')

        # orthogonalize
        Cn = self.orthogonalize(Cn) 

        # write Cn in terms of C_U
        dots = np.matmul(self.Ut,Cn.T)
        C_U = np.matmul(self.Ut.T,dots)

        # normalize C_U
        try:
            C_U = preprocessing.normalize(C_U.T,norm='l2')
            C_U = self.orthogonalize(C_U) 
            dots = np.matmul(C_U,np.transpose(C_U))
        except:
            print C
            exit(-1)
        #print C_U
        #print "shape of overlaps is %s, shape of Ut is %s, shape of C_U is %s" %(np.shape(dots),np.shape(self.Ut),np.shape(C_U))

        basis=np.zeros((self.nicd,self.num_ics),dtype=float)
        for n,row in enumerate(C_U):
            basis[self.nicd-len(C_U)+n,:] =row 
        count=0
        for v in self.Ut:
            w = v - np.sum( np.dot(v,b)*b  for b in basis )
            tmp = w/np.linalg.norm(w)
            if (abs(w) > 1e-4).any():  
                basis[count,:] =tmp
                count +=1
        self.Ut = np.array(basis)
        if self.print_level>1:
            print "printing Ut"
            print self.Ut
        #print "Check if Ut is orthonormal"
        #print dots
        dots = np.matmul(self.Ut,np.transpose(self.Ut))
        assert (np.allclose(dots,np.eye(dots.shape[0],dtype=float))),"error in orthonormality"

if __name__ =='__main__':
    from qchem import *
    import pybel as pb
    from dlbfgs import *
    from pes import PES
    basis="sto-3g"
    nproc=1

    filepath="examples/tests/bent_benzene.xyz"
    #filepath="examples/tests/cyclohexene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    lot=QChem.from_options(states=[(1,0)],charge=0,basis=basis,functional='HF',nproc=nproc)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=1)

    # => DLC constructor <= #
    ic1=DLC.from_options(mol=mol,PES=pes,print_level=1)
    ic1.form_unconstrained_DLC()


    if False:
        # => geometry data <= #
        data = geometry_data(ic1.geom)

        # lbfgs optimization
        param = lbfgs_parameters(m=10,min_step=0.00001)
        lb = dlbfgs(ic1,data,param)
        ret = lb.do_lbfgs(opt_steps=50)

        print data.fx
        manage_xyz.write_xyzs('prc.xyz',data.geoms,scale=1.0)
    else:
        from conjugate_gradient import *
        param = conjugate_gradient_parameters(min_step=0.00001)
        cg = conjugate_gradient(ic1,param)
        cg.do_cg(2)
        print ic1.fx
        manage_xyz.write_xyzs('prc.xyz',ic1.geoms,scale=1.0)



