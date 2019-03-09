from internalcoordinates import InternalCoordinates
from prim_internals import PrimitiveInternalCoordinates
from collections import OrderedDict, defaultdict
import itertools
from copy import deepcopy
import networkx as nx
from _math_utils import *
import options
from slots import *
from molecule import Molecule 
from nifty import click
from sklearn import preprocessing

    
class DelocalizedInternalCoordinates(InternalCoordinates):

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return DelocalizedInternalCoordinates(DelocalizedInternalCoordinates.default_options().set_values(kwargs))


    def __init__(self,
            options
            ):

        super(DelocalizedInternalCoordinates, self).__init__(options)

        # Cache  some useful attributes
        self.options = options
        constraints = options['constraints']
        cvals = options['cVals']
        molecule = options['molecule']
        self.na = molecule.natoms

        # The DLC contains an instance of primitive internal coordinates.
        #self.Prims = PrimitiveInternalCoordinates(molecule, connect=connect, addcart=addcart, constraints=constraints, cvals=cvals)
        self.Prims = PrimitiveInternalCoordinates(options.copy())
        xyz = molecule.xyz.flatten()

        # TODO
        print self

        self.build_dlc(xyz)


    def clearCache(self):
        super(DelocalizedInternalCoordinates, self).clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()
            
    def update(self, other):
        return self.Prims.update(other.Prims)
        
    def join(self, other):
        return self.Prims.join(other.Prims)
        
    def addConstraint(self, cPrim, cVal, xyz):
        self.Prims.addConstraint(cPrim, cVal, xyz)

    def getConstraints_from(self, other):
        self.Prims.getConstraints_from(other.Prims)
        
    def haveConstraints(self):
        return len(self.Prims.cPrims) > 0

    def getConstraintViolation(self, xyz):
        return self.Prims.getConstraintViolation(xyz)

    def printConstraints(self, xyz, thre=1e-5):
        self.Prims.printConstraints(xyz, thre=thre)

    def getConstraintTargetVals(self):
        return self.Prims.getConstraintTargetVals()

    def augmentGH(self, xyz, G, H):
        #CRA This fxn not used in current eigenvector follow regime
        """
        Add extra dimensions to the gradient and Hessian corresponding to the constrained degrees of freedom.
        The Hessian becomes:  H  c
                              cT 0
        where the elements of cT are the first derivatives of the constraint function 
        (typically a single primitive minus a constant) with respect to the DLCs. 
        
        Since we picked a DLC to represent the constraint (cProj), we only set one element 
        in each row of cT to be nonzero. Because cProj = a_i * Prim_i + a_j * Prim_j, we have
        d(Prim_c)/d(cProj) = 1.0/a_c where "c" is the index of the primitive being constrained.
        
        The extended elements of the Gradient are equal to the constraint violation.
        
        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        G : np.ndarray
            Flat array containing internal coordinate gradient
        H : np.ndarray
            Square array containing internal coordinate Hessian

        Returns
        -------
        GC : np.ndarray
            Flat array containing gradient extended by constraint violations
        HC : np.ndarray
            Square matrix extended by partial derivatives d(Prim)/d(cProj)
        """
        # Number of internals (elements of G)
        ni = len(G)
        # Number of constraints
        nc = len(self.Prims.cPrims)
        # Total dimension
        nt = ni+nc
        # Lower block of the augmented Hessian
        cT = np.zeros((nc, ni), dtype=float)
        c0 = np.zeros(nc, dtype=float)
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # The DLC corresponding to the constrained primitive (a.k.a. cProj) is self.Vecs[self.cDLC[ic]].
            # For a differential change in the DLC, the primitive that we are constraining changes by:
            cT[ic, self.cDLC[ic]] = 1.0/self.Vecs[iPrim, self.cDLC[ic]]
            # Calculate the further change needed in this constrained variable
            c0[ic] = self.Prims.cVals[ic] - c.value(xyz)
            if c.isPeriodic:
                Plus2Pi = c0[ic] + 2*np.pi
                Minus2Pi = c0[ic] - 2*np.pi
                if np.abs(c0[ic]) > np.abs(Plus2Pi):
                    c0[ic] = Plus2Pi
                if np.abs(c0[ic]) > np.abs(Minus2Pi):
                    c0[ic] = Minus2Pi
        # Construct augmented Hessian
        HC = np.zeros((nt, nt), dtype=float)
        HC[0:ni, 0:ni] = H[:,:]
        HC[ni:nt, 0:ni] = cT[:,:]
        HC[0:ni, ni:nt] = cT.T[:,:]
        # Construct augmented gradient
        GC = np.zeros(nt, dtype=float)
        GC[0:ni] = G[:]
        GC[ni:nt] = -c0[:]
        return GC, HC
    
    def applyConstraints(self, xyz):
        """
        Pass in Cartesian coordinates and return new coordinates that satisfy the constraints exactly.
        This is not used in the current constrained optimization code that uses Lagrange multipliers instead.
        """
        xyz1 = xyz.copy()
        niter = 0
        while True:
            dQ = np.zeros(len(self.Internals), dtype=float)
            for ic, c in enumerate(self.Prims.cPrims):
                # Look up the index of the primitive that is being constrained
                iPrim = self.Prims.Internals.index(c)
                # Look up the index of the DLC that corresponds to the constraint
                iDLC = self.cDLC[ic]
                # Calculate the further change needed in this constrained variable
                dQ[iDLC] = (self.Prims.cVals[ic] - c.value(xyz1))/self.Vecs[iPrim, iDLC]
                if c.isPeriodic:
                    Plus2Pi = dQ[iDLC] + 2*np.pi
                    Minus2Pi = dQ[iDLC] - 2*np.pi
                    if np.abs(dQ[iDLC]) > np.abs(Plus2Pi):
                        dQ[iDLC] = Plus2Pi
                    if np.abs(dQ[iDLC]) > np.abs(Minus2Pi):
                        dQ[iDLC] = Minus2Pi
            # print "applyConstraints calling newCartesian (%i), |dQ| = %.3e" % (niter, np.linalg.norm(dQ))
            xyz2 = self.newCartesian(xyz1, dQ, verbose=False)
            if np.linalg.norm(dQ) < 1e-6:
                return xyz2
            if niter > 1 and np.linalg.norm(dQ) > np.linalg.norm(dQ0):
                logger.warning("\x1b[1;93mWarning: Failed to apply Constraint\x1b[0m")
                return xyz1
            xyz1 = xyz2.copy()
            niter += 1
            dQ0 = dQ.copy()
            
    def newCartesian_withConstraint(self, xyz, dQ, thre=0.1, verbose=False):
        xyz2 = self.newCartesian(xyz, dQ, verbose)
        constraintSmall = len(self.Prims.cPrims) > 0
        for ic, c in enumerate(self.Prims.cPrims):
            w = c.w if type(c) in [RotationA, RotationB, RotationC] else 1.0
            current = c.value(xyz)/w
            reference = self.Prims.cVals[ic]/w
            diff = (current - reference)
            if np.abs(diff-2*np.pi) < np.abs(diff):
                diff -= 2*np.pi
            if np.abs(diff+2*np.pi) < np.abs(diff):
                diff += 2*np.pi
            if np.abs(diff) > thre:
                constraintSmall = False
        if constraintSmall:
            xyz2 = self.applyConstraints(xyz2)
        return xyz2
    
    def calcGradProj(self, xyz, gradx):
        """
        Project out the components of the internal coordinate gradient along the
        constrained degrees of freedom. This is used to calculate the convergence
        criteria for constrained optimizations.

        Parameters
        ----------
        xyz : np.ndarray
            Flat array containing Cartesian coordinates in atomic units
        gradx : np.ndarray
            Flat array containing gradient in Cartesian coordinates

        """
        if len(self.Prims.cPrims) == 0:
            return gradx
        q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx).T
        Gq = multi_dot([Ginv, Bmat, gradx.T])
        Gqc = np.array(Gq).flatten()
        # Remove the directions that are along the DLCs that we are constraining
        for i in self.cDLC:
            Gqc[i] = 0.0
        # Gxc = np.array(np.matrix(Bmat.T)*np.matrix(Gqc).T).flatten()
        Gxc = multi_dot([Bmat.T, Gqc.T]).flatten()
        return Gxc
    
    def build_dlc(self, xyz, cVec=None):
        """
        Build the delocalized internal coordinates (DLCs) which are linear 
        combinations of the primitive internal coordinates. Each DLC is stored
        as a column in self.Vecs.
        
        In short, each DLC is an eigenvector of the G-matrix, and the number of
        nonzero eigenvalues of G should be equal to 3*N. 
        
        After creating the DLCs, we construct special ones corresponding to primitive
        coordinates that are constrained (cProj).  These are placed in the front (i.e. left)
        of the list of DLCs, and then we perform a Gram-Schmidt orthogonalization.

        This function is called at the end of __init__ after the coordinate system is already
        specified (including which primitives are constraints).

        Parameters
        ----------
        xyz     : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units 
        cVec    : np.ndarray
                Float array containing difference in primitive coordinates
        """

        # Perform singular value decomposition
        click()
        G = self.Prims.GMatrix(xyz)
        time_G = click()

        L, Q = np.linalg.eigh(G)
        time_eig = click()
        # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)

        LargeVals = 0
        LargeIdx = []
        for ival, value in enumerate(L):
            # print ival, value
            if np.abs(value) > 1e-6:
                LargeVals += 1
                LargeIdx.append(ival)
        Expect = 3*self.na
        # print "%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L))
        # if LargeVals <= Expect:
        self.Vecs = Q[:, LargeIdx]
        self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            assert cVec is None,"can't have vector constraint and cprim."
            cVec=self.form_cVec_from_cPrims()
            print cVec.T

        if cVec is not None:
            click()
            # V contains the constraint vectors on the left, and the original DLCs on the right
            V = np.hstack((cVec, np.array(self.Vecs)))
            # Apply Gram-Schmidt to V, and produce U.
            self.Vecs = orthogonalize(V,cVec.shape[1])
            # print "Gram-Schmidt completed with thre=%.0e" % thre


    def form_cVec_from_cPrims(self):
        """ forms the constraint vector from self.cPrim -- not used in GSM"""
        #CRA 3/2019 warning:
        #I'm not sure how this works!!!
        #multiple constraints appears to be problematic!!!
        self.cDLC = [i for i in range(len(self.Prims.cPrims))]
        #print "Projecting out constraints...",
        #V=[]
        for ic, c in enumerate(self.Prims.cPrims):
            # Look up the index of the primitive that is being constrained
            iPrim = self.Prims.Internals.index(c)
            # Pick a row out of the eigenvector space. This is a linear combination of the DLCs.
            cVec = self.Vecs[iPrim, :]
            cVec = np.array(cVec)
            cVec /= np.linalg.norm(cVec)
            # This is a "new DLC" that corresponds to the primitive that we are constraining
            cProj = np.dot(self.Vecs,cVec.T)
            cProj /= np.linalg.norm(cProj)
            #V.append(np.array(cProj).flatten())
        V = cProj.reshape(-1,1)
        return V

    def form_cVecs_from_prim_Vecs(self,C):
        """
        This function takes a matrix of vectors wrtiten in the basis of primitives
        and returns new vectors written in the DLC basis. 
        
        Parameters
        ----------
        C   :   np.ndarray
                rectangular array containing column vectors of constraints. 
                The constraints are orthogonalized wrt the first column. 
        Returns
        -------
        cVecs:   np.ndarray
                rectangular array containing column vectors of orthogonal 
                constraints in DLC basis.
        """

        # normalize all constraints
        Cn = preprocessing.normalize(C,norm='l2')

        # orthogonalize
        Cn = orthogonalize(Cn) 

        # transform C into basis of DLC
        # CRA 3/2019 NOT SURE WHY THIS IS DONE
        # couldn't Cn just be used?
        cVecs = np.linalg.multi_dot([self.Vecs,self.Vecs.T,Cn])

        # normalize C_U
        try:
            cVecs = preprocessing.normalize(cVecs,norm='l2')
            cVecs = orthogonalize(cVecs) 
            dots = np.matmul(cVecs,np.transpose(cVecs))
        except:
            print cVecs
            print "error forming cVec"
            exit(-1)

        return cVecs

    def __eq__(self, other):
        return self.Prims == other.Prims

    def __ne__(self, other):
        return not self.__eq__(other)

    def largeRots(self):
        """ Determine whether a molecule has rotated by an amount larger than some threshold (hardcoded in Prims.largeRots()). """
        return self.Prims.largeRots()

    def calcDiff(self, coord1, coord2):
        """ Calculate difference in internal coordinates, accounting for changes in 2*pi of angles. """
        PMDiff = self.Prims.calcDiff(coord1, coord2)
        Answer = np.dot(PMDiff, self.Vecs)
        return np.array(Answer).flatten()

    def calculate(self, coords):
        """ Calculate the DLCs given the Cartesian coordinates. """
        PrimVals = self.Prims.calculate(coords)
        Answer = np.dot(PrimVals, self.Vecs)
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
        return np.array(Answer).flatten()

    def DLC_to_primitive(self,vecq):
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.

        return np.dot(self.Vecs,vecq)

    def derivatives(self, coords):
        """ Obtain the change of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.derivatives(coords)
        # The following code does the same as "tensordot"
        # print PrimDers.shape
        # print self.Vecs.shape
        # Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)
        # for i in range(self.Vecs.shape[1]):
        #     for j in range(self.Vecs.shape[0]):
        #         Answer[i, :, :] += self.Vecs[j, i] * PrimDers[j, :, :]
        # print Answer.shape
        Answer1 = np.tensordot(self.Vecs, PrimDers, axes=(0, 0))
        return np.array(Answer1)

    def GInverse(self, xyz):
        return self.GInverse_SVD(xyz)

    def repr_diff(self, other):
        return self.Prims.repr_diff(other.Prims)

    def guess_hessian(self, coords):
        """ Build the guess Hessian, consisting of a diagonal matrix 
        in the primitive space and changed to the basis of DLCs. """
        Hprim = self.Prims.guess_hessian(coords)
        return multi_dot([self.Vecs.T,Hprim,self.Vecs])

    def resetRotations(self, xyz):
        """ Reset the reference geometries for calculating the orientational variables. """
        self.Prims.resetRotations(xyz)


if __name__ =='__main__':
    M = Molecule('s1minima_with_h2o.pdb',fragments=True)
    objs = []
    objs.append(Distance(0,1))
    IC = PrimitiveInternalCoordinates.from_options(molecule=M,constraints=objs)
    print IC.cPrims
    print IC.cVals
    IC.printConstraints(M.xyz)
    DLC = DelocalizedInternalCoordinates(IC.options.copy())

    # dvec
    #dvec = self.PES.get_coupling(self.geom)
    #dgrad = self.PES.get_dgrad(self.geom)
    #dvecq = self.grad_to_q(dvec)
    #dgradq = self.grad_to_q(dgrad)

    #print(IC.Internals)
    ##print IC
    #print IC.derivatives(M.xyz)
    #print IC.Internals[0].a
    #print IC.Internals[0].b
    #print IC.Internals[0].value(M.xyz)

     
    #print IC.Internals[0].derivative(M.xyz)
    #print IC.wilsonB(M.xyz)

    #DLC = DelocalizedInternalCoordinates(M,build=True,remove_tr=True,connect=True)
    #q=DLC.calculate(M.xyz)
    #print q.shape

    #DLC.
