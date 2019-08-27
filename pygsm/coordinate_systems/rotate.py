from __future__ import division

# standard library imports
import sys

# third party
import numpy as np
from numpy.linalg import multi_dot

# local application imports
from utilities import nifty,manage_xyz


"""
References
----------
1. E. A. Coutsias, C. Seok, K. A. Dill. "Using Quaternions to Calculate RMSD.". J. Comput. Chem 2004.
"""

# def invert_svd(X,thresh=1e-12):
    
#     """ 

#     Invert a matrix using singular value decomposition. 
#     @param[in] X The matrix to be inverted
#     @param[in] thresh The SVD threshold; eigenvalues below this are not inverted but set to zero
#     @return Xt The inverted matrix

#     """

#     u,s,vh = np.linalg.svd(X, full_matrices=0)
#     uh     = np.matrix(np.transpose(u))
#     v      = np.matrix(np.transpose(vh))
#     si     = s.copy()
#     for i in range(s.shape[0]):
#         # reg = s[i]**2 / (s[i]**2 + thresh**2)
#         si[i] = s[i] / (s[i]**2 + thresh**2)
#         # if abs(s[i]) > thresh:
#         #     si[i] = 1./s[i]
#         # else:
#         #     si[i] = 0.0
#     si     = np.matrix(np.diag(si))
#     Xt     = v*si*uh
#     return Xt

def build_correlation(x, y):
    """
    Build the 3x3 correlation matrix given by the sum over all atoms k:
    xk_i * yk_j
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        3x3 correlation matrix

    """
    if not (x.ndim == 2 and y.ndim == 2 and x.shape[1] == 3 and y.shape[1] == 3 and x.shape[0] == y.shape[0]):
        raise ValueError("Input dimensions not (any_same_value, 3): x ({}), y ({})".format(x.shape, y.shape))
    xmat = x.T
    ymat = y.T
    return np.dot(xmat, ymat.T)

def build_F(x, y):
    """
    Build the 4x4 F-matrix used in constructing the rotation quaternion
    given by Equation 10 of Reference 1
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates
    """
    R = build_correlation(x, y)
    F = np.zeros((4,4),dtype=float)
    R11 = R[0,0]
    R12 = R[0,1]
    R13 = R[0,2]
    R21 = R[1,0]
    R22 = R[1,1]
    R23 = R[1,2]
    R31 = R[2,0]
    R32 = R[2,1]
    R33 = R[2,2]
    F[0,0] = R11 + R22 + R33
    F[0,1] = R23 - R32
    F[0,2] = R31 - R13
    F[0,3] = R12 - R21
    F[1,0] = R23 - R32
    F[1,1] = R11 - R22 - R33
    F[1,2] = R12 + R21
    F[1,3] = R13 + R31
    F[2,0] = R31 - R13
    F[2,1] = R12 + R21
    F[2,2] = R22 - R33 - R11
    F[2,3] = R23 + R32
    F[3,0] = R12 - R21
    F[3,1] = R13 + R31
    F[3,2] = R23 + R32
    F[3,3] = R33 - R22 - R11
    return F

def al(p):
    """
    Given a quaternion p, return the 4x4 matrix A_L(p)
    which when multiplied with a column vector q gives
    the quaternion product pq.
    
    Parameters
    ----------
    p : numpy.ndarray
        4 elements, represents quaternion
    
    Returns
    -------
    numpy.ndarray
        4x4 matrix describing action of quaternion multiplication

    """
    # Given a quaternion p, return the 4x4 matrix A_L(p)
    # which when multiplied with a column vector q gives
    # the quaternion product pq.
    return np.array([[ p[0], -p[1], -p[2], -p[3]],
                     [ p[1],  p[0], -p[3],  p[2]],
                     [ p[2],  p[3],  p[0], -p[1]],
                     [ p[3], -p[2],  p[1],  p[0]]])
     
def ar(p):
    """
    Given a quaternion p, return the 4x4 matrix A_R(p)
    which when multiplied with a column vector q gives
    the quaternion product qp.
    
    Parameters
    ----------
    p : numpy.ndarray
        4 elements, represents quaternion
    
    Returns
    -------
    numpy.ndarray
        4x4 matrix describing action of quaternion multiplication

    """
    return np.array([[ p[0], -p[1], -p[2], -p[3]],
                     [ p[1],  p[0],  p[3], -p[2]],
                     [ p[2], -p[3],  p[0],  p[1]],
                     [ p[3],  p[2], -p[1],  p[0]]])

def conj(q):
    """
    Given a quaternion p, return its conjugate, simply the second
    through fourth elements changed in sign.
    
    Parameters
    ----------
    q : numpy.ndarray
        4 elements, represents quaternion
    
    Returns
    -------
    numpy.ndarray
        New array representing conjugate of q
    """
    assert q.ndim == 1
    assert q.shape[0] == 4
    qc = np.zeros_like(q)
    qc[0] =  q[0]
    qc[1] = -q[1]
    qc[2] = -q[2]
    qc[3] = -q[3]
    return qc

def form_rot(q):
    """
    Given a quaternion p, form a rotation matrix from it.
    
    Parameters
    ----------
    q : numpy.ndarray
        4 elements, represents quaternion
    
    Returns
    -------
    numpy.array
        3x3 rotation matrix
    """
    qc = conj(q)
    R4 = np.dot(al(q),ar(qc))
    return R4[1:, 1:]

def sorted_eigh(mat, b=None, asc=False):
    """ Return eigenvalues of a symmetric matrix in descending order and associated eigenvectors """
    if b is not None:
        L, Q = np.linalg.eigh(mat, b)
    else:
        L, Q = np.linalg.eigh(mat)
    if asc:
        idx = L.argsort()
    else:
        idx = L.argsort()[::-1]   
    L = L[idx]
    Q = Q[:,idx]
    return L, Q

def calc_rmsd(x, y):
    """
    Calculate the minimal RMSD between two structures x and y following
    the algorithm in Reference 1.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        3x3 correlation matrix
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    N = x.shape[0]
    L, Q = sorted_eigh(build_F(x, y))
    idx = L.argsort()[::-1]   
    L = L[idx]
    Q = Q[:,idx]

    lmax = np.max(L)
    rmsd = np.sqrt((np.sum(x**2) + np.sum(y**2) - 2*lmax)/N)
    return rmsd

def is_linear(x, y):
    """
    Returns True if molecule is linear 
    (largest eigenvalue almost equivalent to second largest)
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    N = x.shape[0]
    L, Q = sorted_eigh(build_F(x, y))
    if L[0]/L[1] < 1.01 and L[0]/L[1] > 0.0:
        return True
    else:
        return False

def get_quat(x, y, eig=False):
    """
    Calculate the quaternion that rotates x into maximal coincidence with y
    to minimize the RMSD, following the algorithm in Reference 1.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        4-element array representing quaternion
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    N = x.shape[0]
    L, Q = sorted_eigh(build_F(x, y))
    q = Q[:,0]
    # Standardize the orientation somewhat
    if q[0] < 0:
        q *= -1
    if eig:
        return q, L[0]
    else:
        return q

def get_rot(x, y):
    """
    Calculate the rotation matrix that brings x into maximal coincidence with y
    to minimize the RMSD, following the algorithm in Reference 1.  Mainly
    used to check the correctness of the quaternion.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.array
        3x3 rotation matrix

    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    N = x.shape[0]
    q = get_quat(x, y)
    U = form_rot(q)
    # x = np.matrix(x)
    # xr = np.array((U*x.T).T)
    xr = np.dot(U,x.T).T
    rmsd = np.sqrt(np.sum((xr-y)**2)/N)
    return U


def get_R_der(x, y):
    """
    Calculate the derivatives of the correlation matrix with respect
    to the Cartesian coordinates.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        u, w, i, j : 
        First two dimensions are (n_atoms, 3), the variables being differentiated
        Second two dimensions are (3, 3), the elements of the R-matrix derivatives with respect to atom u, dimension w
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    # 3 x 3 x N_atoms x 3
    ADiffR = np.zeros((x.shape[0], 3, 3, 3), dtype=float)
    for u in range(x.shape[0]):
        for w in range(3):
            for i in range(3):
                for j in range(3):
                    if i == w:
                        ADiffR[u, w, i, j] = y[u, j]
    fdcheck = False
    if fdcheck:
        h = 1e-4
        R0 = build_correlation(x, y)
        for u in range(x.shape[0]):
            for w in range(3):
                x[u, w] += h
                RPlus = build_correlation(x, y)
                x[u, w] -= 2*h
                RMinus = build_correlation(x, y)
                x[u, w] += h
                FDiffR = (RPlus-RMinus)/(2*h)
                nifty.logger.info(u, w, np.max(np.abs(ADiffR[u, w]-FDiffR)))
    return ADiffR

def get_F_der(x, y):
    """
    Calculate the derivatives of the F-matrix with respect
    to the Cartesian coordinates.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        u, w, i, j : 
        First two dimensions are (n_atoms, 3), the variables being differentiated
        Second two dimensions are (4, 4), the elements of the R-matrix derivatives with respect to atom u, dimension w
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    dR = get_R_der(x, y)
    dF = np.zeros((x.shape[0], 3, 4, 4),dtype=float)
    for u in range(x.shape[0]):
        for w in range(3):
            dR11 = dR[u,w,0,0]
            dR12 = dR[u,w,0,1]
            dR13 = dR[u,w,0,2]
            dR21 = dR[u,w,1,0]
            dR22 = dR[u,w,1,1]
            dR23 = dR[u,w,1,2]
            dR31 = dR[u,w,2,0]
            dR32 = dR[u,w,2,1]
            dR33 = dR[u,w,2,2]
            dF[u,w,0,0] = dR11 + dR22 + dR33
            dF[u,w,0,1] = dR23 - dR32
            dF[u,w,0,2] = dR31 - dR13
            dF[u,w,0,3] = dR12 - dR21
            dF[u,w,1,0] = dR23 - dR32
            dF[u,w,1,1] = dR11 - dR22 - dR33
            dF[u,w,1,2] = dR12 + dR21
            dF[u,w,1,3] = dR13 + dR31
            dF[u,w,2,0] = dR31 - dR13
            dF[u,w,2,1] = dR12 + dR21
            dF[u,w,2,2] = dR22 - dR33 - dR11
            dF[u,w,2,3] = dR23 + dR32
            dF[u,w,3,0] = dR12 - dR21
            dF[u,w,3,1] = dR13 + dR31
            dF[u,w,3,2] = dR23 + dR32
            dF[u,w,3,3] = dR33 - dR22 - dR11
    fdcheck = False
    if fdcheck:
        h = 1e-4
        F0 = build_F(x, y)
        for u in range(x.shape[0]):
            for w in range(3):
                x[u, w] += h
                FPlus = build_F(x, y)
                x[u, w] -= 2*h
                FMinus = build_F(x, y)
                x[u, w] += h
                FDiffF = (FPlus-FMinus)/(2*h)
                nifty.logger.info(u, w, np.max(np.abs(dF[u, w]-FDiffF)))
    return dF

def get_q_der(x, y):
    """
    Calculate the derivatives of the quaternion with respect
    to the Cartesian coordinates.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        u, w, i: 
        First two dimensions are (n_atoms, 3), the variables being differentiated
        Third dimension is 4, the elements of the quaternion derivatives with respect to atom u, dimension w
    """
    x = x - np.mean(x,axis=0)
    y = y - np.mean(y,axis=0)
    q, l = get_quat(x, y, eig=True)
    F = build_F(x, y)
    dF = get_F_der(x, y)
    mat = np.eye(4)*l - F
    # pinv = np.matrix(np.linalg.pinv(np.eye(4)*l - F))
    pinv = nifty.invert_svd(np.eye(4)*l - F, thresh=1e-6)
    dq = np.zeros((x.shape[0], 3, 4), dtype=float)
    for u in range(x.shape[0]):
        for w in range(3):
            # dquw = pinv*np.matrix(dF[u, w])*np.matrix(q).T
            dquw = multi_dot([pinv,dF[u, w],q.T])
            dq[u, w] = np.array(dquw).flatten()
    fdcheck = False
    if fdcheck:
        h = 1e-6
        for u in range(x.shape[0]):
            for w in range(3):
                x[u, w] += h
                QPlus = get_quat(x, y)
                x[u, w] -= 2*h
                QMinus = get_quat(x, y)
                x[u, w] += h
                FDiffQ = (QPlus-QMinus)/(2*h)
                nifty.logger.info(QPlus, QMinus)
                nifty.logger.info(dq[u, w], FDiffQ)
                nifty.logger.info(u, w, np.dot(QPlus, QMinus), np.max(np.abs(dq[u, w]-FDiffQ)))
    return dq

def calc_fac_dfac(q0):
    """
    Calculate the prefactor mapping the quaternion to the exponential map
    and also its derivative. Takes the first element of the quaternion only
    """
    # Ill-defined around q0=1.0
    qm1 = q0-1.0
    # if np.abs(q0) == 1.0:
    #     fac = 2
    #     dfac = -2/3
    if np.abs(qm1) < 1e-8:
        fac = 2 - 2*qm1/3
        dfac = -2/3
    else:
        fac = 2*np.arccos(q0)/np.sqrt(1-q0**2)
        dfac = -2/(1-q0**2)
        dfac += 2*q0*np.arccos(q0)/(1-q0**2)**1.5
    return fac, dfac

def get_expmap(x, y):
    """
    Calculate the exponential map that rotates x into maximal coincidence with y
    to minimize the RMSD.
    
    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        3-element array representing exponential map
    """
    q = get_quat(x, y)
    # print q
    fac, _ = calc_fac_dfac(q[0])
    v = fac*q[1:]
    return v

def get_expmap_der(x,y):
    """
    Given trial coordinates x and target coordinates y, 
    return the derivatives of the exponential map that brings
    x into maximal coincidence (minimum RMSD) with y, with
    respect to the coordinates of x.

    Parameters
    ----------
    x : numpy.ndarray
        Trial coordinates, dimensionality (number of atoms) x 3
    y : numpy.ndarray
        Target coordinates, dimensionalty must match trial coordinates

    Returns
    -------
    numpy.ndarray
        u, w, i: 
        First two dimensions are (n_atoms, 3), the variables being differentiated
        Third dimension is 3, the elements of the exponential map derivatives with respect to atom u, dimension w
    """
    q = get_quat(x,y)
    v = get_expmap(x,y)
    fac, dfac = calc_fac_dfac(q[0])
    dvdq = np.zeros((4, 3), dtype=float)
    dvdq[0, :] = dfac*q[1:]
    for i in range(3):
        dvdq[i+1, i] = fac
    fdcheck = False
    if fdcheck:
        h = 1e-6
        fac, _ = calc_fac_dfac(q[0])
        VZero = fac*q[1:]
        for i in range(4):
            # Do backwards difference only, because arccos of q[0] > 1 is undefined
            q[i] -= h
            fac, _ = calc_fac_dfac(q[0])
            VMinus = fac*q[1:]
            q[i] += h
            FDiffV = (VZero-VMinus)/h
            nifty.logger.info(i, dvdq[i], FDiffV, np.max(np.abs(dvdq[i]-FDiffV)))
    # Dimensionality: Number of atoms, number of dimensions (3), number of elements in q (4)
    dqdx = get_q_der(x, y)
    # Dimensionality: Number of atoms, number of dimensions (3), number of elements in v (3)
    dvdx = np.zeros((x.shape[0], 3, 3), dtype=float)
    for u in range(x.shape[0]):
        for w in range(3):
            dqdx_uw = dqdx[u, w]
            for i in range(4):
                dvdx[u, w, :] += dvdq[i, :] * dqdx[u, w, i]
    if fdcheck:
        h = 1e-3
        for u in range(x.shape[0]):
            for w in range(3):
                x[u, w] += h
                VPlus = get_expmap(x, y)
                x[u, w] -= 2*h
                VMinus = get_expmap(x, y)
                x[u, w] += h
                FDiffV = (VPlus-VMinus)/(2*h)
                nifty.logger.info(u, w, np.max(np.abs(dvdx[u, w]-FDiffV)))
    return dvdx

def eckart_frame(
    geom,
    masses,
    ):

    """ Moves the molecule to the Eckart frame

    Params:
        geom ((natoms,4) np.ndarray) - Contains atom symbol and xyz coordinates
        masses ((natoms) np.ndarray) - Atom masses

    Returns:
        COM ((3), np.ndarray) - Molecule center of mess
        L ((3), np.ndarray) - Principal moments
        O ((3,3), np.ndarray)- Principle axes of inertial tensor
        geom2 ((natoms,4 np.ndarray) - Contains new geometry (atom symbol and xyz coordinates)

    """

    # Center of mass
    COM = np.sum(manage_xyz.xyz_to_np(geom) * np.outer(masses, [1.0]*3), 0) / np.sum(masses)
    # Inertial tensor
    I = np.zeros((3,3))
    for atom, mass in zip(geom, masses): 
        I[0,0] += mass * (atom[1] - COM[0]) * (atom[1] - COM[0])
        I[0,1] += mass * (atom[1] - COM[0]) * (atom[2] - COM[1])
        I[0,2] += mass * (atom[1] - COM[0]) * (atom[3] - COM[2])
        I[1,0] += mass * (atom[2] - COM[1]) * (atom[1] - COM[0])
        I[1,1] += mass * (atom[2] - COM[1]) * (atom[2] - COM[1])
        I[1,2] += mass * (atom[2] - COM[1]) * (atom[3] - COM[2])
        I[2,0] += mass * (atom[3] - COM[2]) * (atom[1] - COM[0])
        I[2,1] += mass * (atom[3] - COM[2]) * (atom[2] - COM[1])
        I[2,2] += mass * (atom[3] - COM[2]) * (atom[3] - COM[2])
    I /= np.sum(masses)
    # Principal moments/Principle axes of inertial tensor
    L, O = np.linalg.eigh(I)
    
    # Eckart geometry
    geom2 = manage_xyz.np_to_xyz(geom, np.dot((manage_xyz.xyz_to_np(geom) - np.outer(np.ones((len(masses),)), COM)), O))

    return COM, L, O, geom2

def eckart_align(geom1,geom2,masses,rfrac,max_iter=200):

    COMreact = np.sum(manage_xyz.xyz_to_np(geom1) * np.outer(masses, [1.0]*3), 0) / np.sum(masses)

    COMproduct = np.sum(manage_xyz.xyz_to_np(geom2) * np.outer(masses, [1.0]*3), 0) / np.sum(masses)

    xyzreact = manage_xyz.xyz_to_np(geom1)
    xyzproduct = manage_xyz.xyz_to_np(geom2)

    # Convert to MW coordinates
    mwcreact = xyzreact*units.ANGSTROM_TO_AU*np.outer(np.sqrt(masses),[1.0]*3)
    mwcproduct = xyzproduct*units.ANGSTROM_TO_AU*np.outer(np.sqrt(masses),[1.0]*3)
    thetas = np.zeros(3)
    total_thetas = np.zeros(3)
    rot_mat = np.zeros((3,3))

    if rfrac < 1.:
        max_iter=1

    for i in range(maxiter):

        # get gradmag
        grad = np.zeros(3)
        for atom1,atom2 in zip(mwcreact,mwcproduct):
            grad[0] += 2*(atom1[1]*atom2[2]-atom1[2]*atom2[1])
            grad[1] += 2*(atom1[2]*atom2[0]-atom1[0]*atom2[2])
            grad[2] += 2*(atom1[0]*atom2[1]-atom1[1]*atom2[0])
        gradmag = np.linalg(grad)
        print(" the gradient of the Eckart distance is %5.4f" % gradmag)

        # get hessian
        dot = np.dot(mwcreact.flatten(),mwcproduct.flatten())
        xd = np.zeros((3,3))
        for atom1,atom2 in zip(mwcreact,mwcproduct):
            xd[0,0] += atom1[0]*atom2[0]
            xd[1,1] += atom1[1]*atom2[1]
            xd[2,2] += atom1[2]*atom2[2]
            xd[1,0] += atom1[1]*atom2[0]
            xd[2,0] += atom1[2]*atom2[0]
            xd[2,1] += atom1[2]*atom2[1]
        xd[0,1] = xd[1,0]
        xd[0,2] = xd[2,0]
        xd[1,2] = xd[2,1]
        hess = np.zeros((3,3))
        hess[0,0] = 2*(dot-xd[0,0])
        hess[1,1] = 2*(dot-xd[1,1])
        hess[2,2] = 2*(dot-xd[2,2])
        hess[1,0] = -2*xd[1,0]
        hess[2,0] = -2*xd[2,0]
        hess[2,1] = -2*xd[2,1]
        hess[0,1] = hess[1,0]
        hess[0,2] = hess[2,0]
        hess[1,2] = hess[2,1]

        hess_evals,hess_evecs = np.linalg.eigh(hess)

        mag_thetas = np.linalg.norm(thetas)

        if gradmag<tol and all(hess_evals>0.):
            break
        
        temp=0.
        vec_index=0
        for j in range(3):
            if hess_evals[j]<temp and abs(hess_evals[j])>0.01:
                temp = hess_evals[j]
                vec_index=j


        if vec_index!=0:
            # Rotate around structure
            RotMat = np.zeros((3,3))
            vec = hess_evecs[vec_index]
            RotMat[0,0] = 2*vec[0]*vec[0] -1
            RotMat[1,1] = 2*vec[1]*vec[1] -1
            RotMat[2,2] = 2*vec[2]*vec[2] -1
            RotMat[1,0] = 2*vec[0]*vec[1]
            RotMat[2,0] = 2*vec[2]*vec[0]
            RotMat[2,1] = 2*vec[2]*vec[1]
            RotMat[0,1] = RotMat[1,0]
            RotMat[0,2] = RotMat[2,0]
            RotMat[1,2] = RotMat[2,1]

            # rotate product
            mwcprod = np.dot(mwcprod,RotMat)

        hess_inverse = np.linalg.inv(hess)

        thetas = np.dot(hess_inverse,grad)
        thetas -= np.ones(3)*rfrac
        total_thetas += thetas

        x = thetas[0]
        y= thetas[1]
        z = thetas[2]
        rot_mat[0,0] = np.cos(y)*np.cos(z)
        rot_mat[0,1] = -np.cos(y)*np.sin(z)
        rot_mat[0,2] = np.sin(y)
        rot_mat[1,0] = np.sin(x)*np.sin(y)*np.cos(z) + np.cos(x)*np.sin(z)
        rot_mat[1,1] = -np.sin(x)*np.sin(y)*np.sin(z) + np.cos(x)*np.cos(z)
        rot_mat[1,2] = -np.sin(x)*np.cos(y)
        rot_mat[2,0] = -np.cos(x)*np.sin(y)*np.cos(z) + np.sin(x)*np.sin(z)
        rot_mat[2,1] = np.cos(x)*np.sin(y)*np.sin(z) + np.sin(x)*np.cos(z)
        rot_mat[2,2] = np.cos(x)*np.cos(y)



def vibrational_basis(
    geom,
    masses,
    ):

    """ Compute the vibrational basis in mass-weighted Cartesian coordinates.
    This is the null-space of the translations and rotations in the Eckart frame.
    
    Params: 
        geom (geometry struct) - minimimum geometry structure
        masses (list of float) - masses for the geometry

    Returns:
        B ((3*natom, 3*natom-6) np.ndarray) - orthonormal basis for vibrations. Mass-weighted cartesians in rows, mass-weighted vibrations in columns. 

    """

    # Compute Eckart frame geometry
    COM, L, O, geom2 = eckart_frame(geom, masses)
    G = manage_xyz.xyz_to_np(geom2)

    # Known basis functions for translations
    TR = np.zeros((3*len(geom),6))
    # Translations
    TR[0::3,0] = np.sqrt(masses) # +X
    TR[1::3,1] = np.sqrt(masses) # +Y
    TR[2::3,2] = np.sqrt(masses) # +Z
    # Rotations in the Eckart frame
    for A, mass in enumerate(masses):
        mass_12 = np.sqrt(mass)
        for j in range(3):
            TR[3*A+j,3] = + mass_12 * (G[A,1] * O[j,2] - G[A,2] * O[j,1]) # + Gy Oz - Gz Oy 
            TR[3*A+j,4] = - mass_12 * (G[A,0] * O[j,2] - G[A,2] * O[j,0]) # - Gx Oz + Gz Ox 
            TR[3*A+j,5] = + mass_12 * (G[A,0] * O[j,1] - G[A,1] * O[j,0]) # + Gx Oy - Gy Ox 

    # Single Value Decomposition (review)        
    U, s, V = np.linalg.svd(TR, full_matrices=True)

    # The null-space of TR
    B = U[:,6:]

    return B

def main():
    M = Molecule(sys.argv[1])
    # The target structure
    Y = M.xyzs[0]
    # The structure being rotated
    X = M.xyzs[1]
    # Copy the structure being rotated
    Z = X.copy()
    get_expmap_der(X, Y)

if __name__ == "__main__":
    main()
