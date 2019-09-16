from __future__ import print_function

# standard library imports
from sys import exit
from time import time

# third party
import networkx as nx
from collections import OrderedDict, defaultdict
import itertools
from numpy.linalg import multi_dot
import numpy as np
np.set_printoptions(precision=4,suppress=True)

# local application imports
try:
    from .internal_coordinates import InternalCoordinates
    from .primitive_internals import PrimitiveInternalCoordinates
    from .topology import Topology
    from .slots import *
except:
    from internal_coordinates import InternalCoordinates
    from primitive_internals import PrimitiveInternalCoordinates
    from topology import Topology
    from slots import *

from utilities import *
   

class DelocalizedInternalCoordinates(InternalCoordinates):

    def __init__(self,
            options
            ):

        super(DelocalizedInternalCoordinates, self).__init__(options)

        # Cache  some useful attributes
        self.options = options
        constraints = options['constraints']
        cvals = options['cVals']
        self.atoms = options['atoms']
        self.natoms = len(self.atoms)

        # The DLC contains an instance of primitive internal coordinates.
        if self.options['primitives'] is None:
            print(" making primitives from options")
            t1 = time()
            self.Prims = PrimitiveInternalCoordinates(options.copy())
            dt = time() - t1
            print(" Time to make prims %.3f" % dt)
            self.options['primitives'] = self.Prims
        else:
            print(" setting primitives from options!")
            #print(" warning: not sure if a deep copy prims")
            #self.Prims=self.options['primitives']
            self.Prims = PrimitiveInternalCoordinates.copy(self.options['primitives'])
            self.Prims.clearCache()
        #print "in constructor",len(self.Prims.Internals)

        xyz = options['xyz']
        xyz = xyz.flatten()
        connect=options['connect']
        addcart=options['addcart']
        addtr=options['addtr']
        if addtr:
            if connect:
                raise RuntimeError(" Intermolecular displacements are defined by translation and rotations! \
                                    Don't add connect!")
        elif addcart:
            if connect:
                raise RuntimeError(" Intermolecular displacements are defined by cartesians! \
                                    Don't add connect!")
        else:
            pass

        self.build_dlc(xyz)
        #print "vecs after build"
        #print self.Vecs

    def clearCache(self):
        super(DelocalizedInternalCoordinates, self).clearCache()
        self.Prims.clearCache()

    def __repr__(self):
        return self.Prims.__repr__()
            
    def update(self, other):
        return self.Prims.update(other.Prims)
        
    def join(self, other):
        return self.Prims.join(other.Prims)

    def copy(self,xyz):
        return type(self)(self.options.copy().set_values({'xyz':xyz}))
        #return type(self)(self.options.copy().set_values({'primitives':self.Prims}))
        
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
    
    def wilsonB(self,xyz):
        Bp = self.Prims.wilsonB(xyz)
        #Vt = block_matrix.transpose(self.Vecs)
        #print(Vt.shape)
        #return block_matrix.dot(Vt,block_matrix.dot(Bp,self.Vecs))
        return block_matrix.dot(block_matrix.transpose(self.Vecs),Bp)

    def calcGrad(self, xyz, gradx):
        #q0 = self.calculate(xyz)
        Ginv = self.GInverse(xyz)
        Bmat = self.wilsonB(xyz)
        # Internal coordinate gradient
        # Gq = np.matrix(Ginv)*np.matrix(Bmat)*np.matrix(gradx)
        nifty.click()
        #Gq = multi_dot([Ginv, Bmat, gradx])
        Bg = block_matrix.dot(Bmat,gradx)
        Gq = block_matrix.dot( Ginv, Bg)
        #print("time to do block mult %.3f" % nifty.click())
        #Gq = np.dot(np.multiply(np.diag(Ginv)[:,None],Bmat),gradx)
        #print("time to do efficient mult %.3f" % nifty.click())
        return Gq
    
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
    
    def build_dlc(self, xyz, C=None):
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
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        nifty.click()
        print(" Beginning to build G Matrix")
        G = self.Prims.GMatrix(xyz)  # in primitive coords
        time_G = nifty.click()
        print(" Timings: Build G: %.3f " % (time_G))

        tmpvecs=[]
        for A in G.matlist:
            L,Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                #print("val=%.4f" %value,end=' ')
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            #print()
            #print("LargeVals %i" % LargeVals)
            tmpvecs.append(Q[:,LargeIdx])

        self.Vecs = block_matrix(tmpvecs)
        #print(" shape of DLC")
        #print(self.Vecs.shape)

        time_eig = nifty.click()
        print(" Timings: Build G: %.3f Eig: %.3f" % (time_G, time_eig))

        #self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]
        self.Internals = ["DLC %i" % (i+1) for i in range(self.Vecs.shape[0])]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            assert cVec is None,"can't have vector constraint and cprim."
            cVec=self.form_cVec_from_cPrims()

        #TODO use block diagonal
        if C is not None:
            # orthogonalize
            #C = C.copy()
            if (C[:]==0.).all():
                raise RuntimeError
            Cn = math_utils.orthogonalize(C)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?

            ## TMP TRYING TO USE Cn
            #cVecs = Cn
            cVecs = block_matrix.dot(block_matrix.dot(self.Vecs,block_matrix.transpose(self.Vecs)),Cn)

            # normalize C_U
            try:
                #print(cVecs.T)
                cVecs = math_utils.orthogonalize(cVecs) 
            except:
                print(cVecs)
                print("error forming cVec")
                exit(-1)

            
            # artificially increase component of distance,angle,dihedral 
            #typs=[Distance,Angle,LinearAngle,OutOfPlane,Dihedral]
            #for inum,p in enumerate(self.Prims.Internals):
            #    if type(p) in typs:
            #        pass
            #    else:
            #        cVecs[inum,:] = 0.


            #project constraints into vectors
            self.Vecs = block_matrix.project_constraint(self.Vecs,cVecs)
            #overDeterminedVecs = block_matrix.project_constraint(self.Vecs,cVecs)
            #print("shape overdetermined %s" %(overDeterminedVecs.shape,))
            #self.Vecs = block_matrix.gram_schmidt(overDeterminedVecs)

        return


    def build_dlc_conjugate(self, xyz, C=None):
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
        C       : np.ndarray
                Float array containing difference in primitive coordinates
        """

        print(" starting to build G prim")
        nifty.click()
        G = self.Prims.GMatrix(xyz)  # in primitive coords
        time_G = nifty.click()
        print(" Timings: Build G: %.3f " % (time_G))

        tmpvecs=[]
        for A in G.matlist:
            L,Q = np.linalg.eigh(A)
            LargeVals = 0
            LargeIdx = []
            for ival, value in enumerate(L):
                #print("val=%.4f" %value,end=' ')
                if np.abs(value) > 1e-6:
                    LargeVals += 1
                    LargeIdx.append(ival)
            #print("LargeVals %i" % LargeVals)
            tmpvecs.append(Q[:,LargeIdx])
        self.Vecs = block_matrix(tmpvecs)
        time_eig = nifty.click()
        #print(" Timings: Build G: %.3f Eig: %.3f" % (time_G, time_eig))

        self.Internals = ["DLC %i" % (i+1) for i in range(len(LargeIdx))]

        # Vecs has number of rows equal to the number of primitives, and
        # number of columns equal to the number of delocalized internal coordinates.
        if self.haveConstraints():
            assert cVec is None,"can't have vector constraint and cprim."
            cVec=self.form_cVec_from_cPrims()

        #TODO use block diagonal
        if C is not None:
            # orthogonalize
            if (C[:]==0.).all():
                raise RuntimeError
            G =  block_matrix.full_matrix(self.Prims.GMatrix(xyz))
            Cn = math_utils.conjugate_orthogonalize(C,G)

            # transform C into basis of DLC
            # CRA 3/2019 NOT SURE WHY THIS IS DONE
            # couldn't Cn just be used?
            cVecs = block_matrix.dot(block_matrix.dot(self.Vecs,block_matrix.transpose(self.Vecs)),Cn)

            # normalize C_U
            try:
                #print(cVecs.T)
                #cVecs = math_utils.orthogonalize(cVecs) 
                cVecs = math_utils.conjugate_orthogonalize(cVecs,G)
            except:
                print(cVecs)
                print("error forming cVec")
                exit(-1)

            #project constraints into vectors
            self.Vecs = block_matrix.project_conjugate_constraint(self.Vecs,cVecs,G)

        return

        def build_dlc_0(self, xyz):
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
              xyz : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
              """
              # Perform singular value decomposition
              click()
              G = self.Prims.GMatrix(xyz)
              # Manipulate G-Matrix to increase weight of constrained coordinates
              if self.haveConstraints():
                  for ic, c in enumerate(self.Prims.cPrims):
                      iPrim = self.Prims.Internals.index(c)
                      G[:, iPrim] *= 1.0
                      G[iPrim, :] *= 1.0
              ncon = len(self.Prims.cPrims)
              # Water Dimer: 100.0, no check -> -151.1892668451
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
        
              # Vecs has number of rows equal to the number of primitives, and
              # number of columns equal to the number of delocalized internal coordinates.
              if self.haveConstraints():
                  click()
                  # print "Projecting out constraints...",
                  V = []
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
                      V.append(np.array(cProj).flatten())
                      # print c, cProj[iPrim]
                  # V contains the constraint vectors on the left, and the original DLCs on the right
                  V = np.hstack((np.array(V).T, np.array(self.Vecs)))
                  # Apply Gram-Schmidt to V, and produce U.
                  # The Gram-Schmidt process should produce a number of orthogonal DLCs equal to the original number
                  thre = 1e-6
                  while True:
                      U = []
                      for iv in range(V.shape[1]):
                          v = V[:, iv].flatten()
                          U.append(v.copy())
                          for ui in U[:-1]:
                              U[-1] -= ui * np.dot(ui, v)
                          if np.linalg.norm(U[-1]) < thre:
                              U = U[:-1]
                              continue
                          U[-1] /= np.linalg.norm(U[-1])
                      if len(U) > self.Vecs.shape[1]:
                          thre *= 10
                      elif len(U) == self.Vecs.shape[1]:
                          break
                      elif len(U) < self.Vecs.shape[1]:
                          raise RuntimeError('Gram-Schmidt orthogonalization has failed (expect %i length %i)' % (self.Vecs.shape[1], len(U)))
                  # print "Gram-Schmidt completed with thre=%.0e" % thre
                  self.Vecs = np.array(U).T
                  # Constrained DLCs are on the left of self.Vecs.
                  self.cDLC = [i for i in range(len(self.Prims.cPrims))]
              # Now self.Internals is no longer a list of InternalCoordinate objects but only a list of strings.
              # We do not create objects for individual DLCs but 
              self.Internals = ["Constraint-DLC" if i < ncon else "DLC" + " %i" % (i+1) for i in range(self.Vecs.shape[1])]
        
        def build_dlc_1(self, xyz):
              """
              Build the delocalized internal coordinates (DLCs) which are linear 
              combinations of the primitive internal coordinates. Each DLC is stored
              as a column in self.Vecs.
              
              After some thought, build_dlc_0 did not implement constraint satisfaction
              in the most correct way. Constraint satisfaction was rather slow and
              the --enforce 0.1 may be passed to improve performance. Rethinking how the
              G matrix is constructed provides a more fundamental solution.
        
              In the new approach implemented here, constrained primitive ICs (PICs) are 
              first set aside from the rest of the PICs. Next, a G-matrix is constructed
              from the rest of the PICs and diagonalized to form DLCs, called "residual" DLCs. 
              The union of the cPICs and rDLCs forms a generalized set of DLCs, but the
              cPICs are not orthogonal to each other or to the rDLCs.
              
              A set of orthogonal DLCs is constructed by carrying out Gram-Schmidt
              on the generalized set. Orthogonalization is carried out on the cPICs in order.
              Next, orthogonalization is carried out on the rDLCs using a greedy algorithm
              that carries out projection for each cPIC, then keeps the one with the largest
              remaining norm. This way we avoid keeping rDLCs that are almost redundant with
              the cPICs. The longest projected rDLC is added to the set and continued until
              the expected number is reached.
        
              One special note in orthogonalization is that the "overlap" between internal
              coordinates corresponds to the G matrix element. Thus, for DLCs that's a linear
              combination of PICs, then the overlap is given by:
        
              v_i * B * B^T * v_j = v_i * G * v_j
        
              Notes on usage: 1) When this is activated, constraints tend to be satisfied very 
              rapidly even if the current coordinates are very far from the constraint values,
              leading to possible blowing up of the energies. In augment_GH, maximum steps in
              constrained degrees of freedom are restricted to 0.1 a.u./radian for this reason.
              
              2) Although the performance of this method is generally superior to the old method,
              the old method with --enforce 0.1 is more extensively tested and recommended.
              Thus, this method isn't enabled by default but provided as an optional feature.
        
              Parameters
              ----------
              xyz : np.ndarray
                  Flat array containing Cartesian coordinates in atomic units
              """
              click()
              G = self.Prims.GMatrix(xyz)
              nprim = len(self.Prims.Internals)
              cPrimIdx = []
              if self.haveConstraints():
                  for ic, c in enumerate(self.Prims.cPrims):
                      iPrim = self.Prims.Internals.index(c)
                      cPrimIdx.append(iPrim)
              ncon = len(self.Prims.cPrims)
              if cPrimIdx != list(range(ncon)):
                  raise RuntimeError("The constraint primitives should be at the start of the list")
              # Form a sub-G-matrix that doesn't include the constrained primitives and diagonalize it to form DLCs.
              Gsub = G[ncon:, ncon:]
              time_G = click()
              L, Q = np.linalg.eigh(Gsub)
              # Sort eigenvalues and eigenvectors in descending order (for cleanliness)
              L = L[::-1]
              Q = Q[:, ::-1]
              time_eig = click()
              # print "Build G: %.3f Eig: %.3f" % (time_G, time_eig)
              # Figure out which eigenvectors from the G submatrix to include
              LargeVals = 0
              LargeIdx = []
              GEigThre = 1e-6
              for ival, value in enumerate(L):
                  if np.abs(value) > GEigThre:
                      LargeVals += 1
                      LargeIdx.append(ival)
              # This is the number of nonredundant DLCs that we expect to have at the end
              Expect = np.sum(np.linalg.eigh(G)[0] > 1e-6)
        
              if (ncon + len(LargeIdx)) < Expect:
                  raise RuntimeError("Expected at least %i delocalized coordinates, but got only %i" % (Expect, ncon + len(LargeIdx)))
              # print("%i atoms (expect %i coordinates); %i/%i singular values are > 1e-6" % (self.na, Expect, LargeVals, len(L)))
              
              # Create "generalized" DLCs where the first six columns are the constrained primitive ICs
              # and the other columns are the DLCs formed from the rest
              self.Vecs = np.zeros((nprim, ncon+LargeVals), dtype=float)
              for i in range(ncon):
                  self.Vecs[i, i] = 1.0
              self.Vecs[ncon:, ncon:ncon+LargeVals] = Q[:, LargeIdx]
        
              # Perform Gram-Schmidt orthogonalization
              def ov(vi, vj):
                  return multi_dot([vi, G, vj])
              if self.haveConstraints():
                  click()
                  V = self.Vecs.copy()
                  nv = V.shape[1]
                  Vnorms = np.array([np.sqrt(ov(V[:,ic], V[:, ic])) for ic in range(nv)])
                  # U holds the Gram-Schmidt orthogonalized DLCs
                  U = np.zeros((V.shape[0], Expect), dtype=float)
                  Unorms = np.zeros(Expect, dtype=float)
                  
                  for ic in range(ncon):
                      # At the top of the loop, V columns are orthogonal to U columns up to ic.
                      # Copy V column corresponding to the next constraint to U.
                      U[:, ic] = V[:, ic].copy()
                      ui = U[:, ic]
                      Unorms[ic] = np.sqrt(ov(ui, ui))
                      if Unorms[ic]/Vnorms[ic] < 0.1:
                          logger.warning("Constraint %i is almost redundant; after projection norm is %.3f of original\n" % (ic, Unorms[ic]/Vnorms[ic]))
                      V0 = V.copy()
                      # Project out newest U column from all remaining V columns.
                      for jc in range(ic+1, nv):
                          vj = V[:, jc]
                          vj -= ui * ov(ui, vj)/Unorms[ic]**2
                      
                  for ic in range(ncon, Expect):
                      # Pick out the V column with the largest norm
                      norms = np.array([np.sqrt(ov(V[:, jc], V[:, jc])) for jc in range(ncon, nv)])
                      imax = ncon+np.argmax(norms)
                      # Add this column to U
                      U[:, ic] = V[:, imax].copy()
                      ui = U[:, ic]
                      Unorms[ic] = np.sqrt(ov(ui, ui))
                      # Project out the newest U column from all V columns
                      for jc in range(ncon, nv):
                          V[:, jc] -= ui * ov(ui, V[:, jc])/Unorms[ic]**2
                      
                  # self.Vecs contains the linear combination coefficients that are our new DLCs
                  self.Vecs = U.copy()
                  # Constrained DLCs are on the left of self.Vecs.
                  self.cDLC = [i for i in range(len(self.Prims.cPrims))]
        
              self.Internals = ["Constraint" if i < ncon else "DLC" + " %i" % (i+1) for i in range(self.Vecs.shape[1])]
              # # LPW: Coefficients of DLC's are in each column and DLCs corresponding to constraints should basically be like (0 1 0 0 0 ..)
              # pmat2d(self.Vecs, format='f', precision=2)
              # B = self.Prims.wilsonB(xyz)
              # Bdlc = np.einsum('ji,jk->ik', self.Vecs, B)
              # Gdlc = np.dot(Bdlc, Bdlc.T)
              # # Expect to see a diagonal matrix here
              # print("Gdlc")
              # pmat2d(Gdlc, format='e', precision=2)
              # # Expect to see "large" eigenvalues here (no less than 0.1 ideally)
              # print("L, Q")
              # L, Q = np.linalg.eigh(Gdlc)
              # print(L)

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
        #Answer = np.dot(PMDiff, self.Vecs)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs),PMDiff)
        return np.array(Answer).flatten()

    def calculate(self, coords):
        """ Calculate the DLCs given the Cartesian coordinates. """
        PrimVals = self.Prims.calculate(coords)
        #Answer = np.dot(PrimVals, self.Vecs)
        Answer = block_matrix.dot(block_matrix.transpose(self.Vecs),PrimVals)
        # print np.dot(np.array(self.Vecs[0,:]).flatten(), np.array(Answer).flatten())
        # print PrimVals[0]
        # raw_input()
        return np.array(Answer).flatten()

    def calcPrim(self,vecq):
        # To obtain the primitive coordinates from the delocalized internal coordinates,
        # simply multiply self.Vecs*Answer.T where Answer.T is the column vector of delocalized
        # internal coordinates. That means the "c's" in Equation 23 of Schlegel's review paper
        # are simply the rows of the Vecs matrix.
        #return self.Vecs.dot(vecq)
        return block_matrix.dot(Vecs,vecq)

    # overwritting the parent internalcoordinates GMatrix 
    # which is an elegant way to use the derivatives
    # but there is a more efficient way to compute G
    # using the block diagonal properties of G and V
    def GMatrix(self,xyz):
        tmpvecs=[]
        s3a=0
        sp=0
        Gp = self.Prims.GMatrix(xyz)
        Vt = block_matrix.transpose(self.Vecs)
        for vt,G,v in zip(Vt.matlist,Gp.matlist,self.Vecs.matlist):
            tmpvecs.append( np.dot(np.dot(vt,G),v))
        return block_matrix(tmpvecs)

    def MW_GMatrix(self,xyz,mass):
        tmpvecs=[]
        s3a=0
        sp=0
        Bp = self.Prims.wilsonB(xyz)
        Vt = block_matrix.transpose(self.Vecs)

        s3a = 0
        for vt,b,v in zip(Vt.matlist,Bp.matlist,self.Vecs.matlist):
            e3a = s3a + b.shape[1]
            tmpvecs.append( np.linalg.multi_dot([vt,b/mass[s3a:e3a],b.T,v]))
            s3a = e3a
        return block_matrix(tmpvecs)

    def derivatives(self, coords):
        """ Obtain the change of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.derivatives(coords)
        Answer = np.zeros((self.Vecs.shape[1], PrimDers.shape[1], PrimDers.shape[2]), dtype=float)

        # block matrix tensor dot
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :] += block[j, i] * PrimDers[j, :, :]
                count+=1
        
        #print(" block matrix way")
        #print(Answer)
        #tmp = block_matrix.full_matrix(self.Vecs)
        #Answer1 = np.tensordot(tmp, PrimDers, axes=(0, 0))
        #print(" np way")
        #print(Answer1)
        #return np.array(Answer1)

        return Answer

    def second_derivatives(self, coords):
        """ Obtain the second derivatives of the DLCs with respect to the Cartesian coordinates. """
        PrimDers = self.Prims.second_derivatives(coords)

        # block matrix tensor dot
        Answer = np.zeros((self.Vecs.shape[1],coords.shape[0],3,coords.shape[0],3))
        count = 0
        for block in self.Vecs.matlist:
            for i in range(block.shape[1]):
                for j in range(block.shape[0]):
                    Answer[count, :, :, :, :] += block[j, i] * PrimDers[j, :, :, :, :]
                count+=1

        #print(" block matrix way.")
        #print(Answer[0])
    
        #tmp = block_matrix.full_matrix(self.Vecs)
        #Answer2 = np.tensordot(tmp, PrimDers, axes=(0, 0))
        #print(" np tensor dot with full mat")
        #print(Answer2[0])
        #return np.array(Answer2)

        return Answer

    def MW_GInverse(self,xyz,mass):
        xyz = xyz.reshape(-1,3)
        #nifty.click()
        G = self.MW_GMatrix(xyz,mass)
        #time_G = nifty.click()
        tmpGi = [ np.linalg.inv(g) for g in G.matlist ]
        #time_inv = nifty.click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return block_matrix(tmpGi)

    def GInverse(self, xyz):
        #return self.GInverse_diag(xyz)
        return self.GInverse_EIG(xyz)

    #TODO this needs to be fixed
    def GInverse_diag(self,xyz):
        t0 = time()
        G = self.GMatrix(xyz)
        #print(G)
        dt = time() - t0
        #print(" total time to get GMatrix %.3f" % dt)
        d = np.diagonal(G)
        #print(d)
        return np.diag(1./d)

    def GInverse_EIG(self, xyz):
        xyz = xyz.reshape(-1,3)
        nifty.click()
        G = self.GMatrix(xyz)
        time_G = nifty.click()
        #Gi = np.linalg.inv(G)
        tmpGi = [ np.linalg.inv(g) for g in G.matlist ]
        time_inv = nifty.click()
        # print "G-time: %.3f Inv-time: %.3f" % (time_G, time_inv)
        return block_matrix(tmpGi)

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

    def create2dxyzgrid(self,xyz,xvec,yvec,nx,ny,mag):
        '''
        xvec and yvec are some delocalized coordinate basis vector (or some linear combination of them)
        nx and ny are the number of grid points
        mag is the step along the delocalized coordinate basis. Don't recommend using greater than 0.5
        
        returns an xyz grid to calculate energies on (see potential_energy_surface modules).

        '''

        x=np.linspace(-mag,mag,nx)
        y=np.linspace(-mag,mag,ny)
        xv,yv = np.meshgrid(x,y)
        xyz1 = xyz.flatten()
        xyzgrid = np.zeros((xv.shape[0],xv.shape[1],xyz1.shape[0]))
        print(self.Vecs.shape)
        print(xvec.shape)

        # find what linear combination of DLC basis xvec and yvec is
        proj_xvec = block_matrix.dot(block_matrix.transpose(self.Vecs),xvec)
        proj_yvec = block_matrix.dot(block_matrix.transpose(self.Vecs),yvec)

        #proj_xvec = block_matrix.dot(self.Vecs,xvec)
        #proj_yvec = block_matrix.dot(self.Vecs,yvec)

        print(proj_xvec.T)
        print(proj_yvec.T)
        print(proj_xvec.shape)

        rc=0
        for xrow,yrow in zip(xv,yv):
            cc=0
            for xx,yy in zip(xrow,yrow):

                # first form the vector in the grid as the linear comb of the projected vectors
                dq = xx* proj_xvec + yy*proj_yvec
                print(dq.T)
                print(dq.shape)

                # convert to xyz and save to xyzgrid
                xyzgrid[rc,cc,:] = self.newCartesian(xyz,dq).flatten()
                cc+=1
            rc+=1
        
        return xyzgrid

if __name__ =='__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    #filepath='../../data/butadiene_ethene.xyz'
    #filepath='crystal.xyz'
    filepath='multi1.xyz'

    geom1 = manage_xyz.read_xyz(filepath)
    atom_symbols  = manage_xyz.get_atoms(geom1)

    xyz1 = manage_xyz.xyz_to_np(geom1)

    ELEMENT_TABLE = elements.ElementData()
    atoms = [ELEMENT_TABLE.from_symbol(atom) for atom in atom_symbols]

    hybrid_indices = list(range(0,5)) + list(range(21,26))
    #hybrid_indices = list(range(0,74)) + list(range(3348, 3358))
    #hybrid_indices = None
    #print(hybrid_indices)
    #with open('frozen.txt') as f:
    #    hybrid_indices = f.read().splitlines()
    #hybrid_indices = [int(x) for x in hybrid_indices]
    #print(hybrid_indices)

    print(" Making topology")
    G1 = Topology.build_topology(xyz1,atoms,hybrid_indices=hybrid_indices)

    print(xyz1.shape) 
    print(" Making prim")

    G1 = Topology.build_topology(xyz1,atoms,hybrid_indices=hybrid_indices)

    p = DelocalizedInternalCoordinates.from_options(
            xyz=xyz1,
            atoms=atoms,
            addtr = True,
            topology=G1,
            ) 


    #print(" Len p.prims")
    #print(len(p.Prims.Internals))
    #
    #print(" Len prim vals")
    #prim_vals = p.Prims.calculate(xyz)
    #print(prim_vals)
    #print(len(prim_vals))

    #print(" Len dlc")
    q = p.calculate(xyz1)
    #print(len(q))

    dQ = np.zeros(q.shape)
    dQ[0] = 0.1

    new_xyz = p.newCartesian(xyz1,dQ,verbose=True)

    new_geom = manage_xyz.np_to_xyz(geom1,new_xyz)

    both = [geom1,new_geom]
    manage_xyz.write_xyzs('check.xyz',both,scale=1.)

    #print(p.Internals)


