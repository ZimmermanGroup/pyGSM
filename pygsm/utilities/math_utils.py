import numpy as np

## Some vector calculus functions
def unit_vector(a):
    """
    Vector function: Given a vector a, return the unit vector
    """
    return a / np.linalg.norm(a)

def d_unit_vector(a, ndim=3):
    term1 = np.eye(ndim)/np.linalg.norm(a)
    term2 = np.outer(a, a)/(np.linalg.norm(a)**3)
    answer = term1-term2
    return answer

def d_cross(a, b):
    """
    Given two vectors a and b, return the gradient of the cross product axb w/r.t. a.
    (Note that the answer is independent of a.)
    Derivative is on the first axis.
    """
    d_cross = np.zeros((3, 3), dtype=float)
    for i in range(3):
        ei = np.zeros(3, dtype=float)
        ei[i] = 1.0
        d_cross[i] = np.cross(ei, b)
    return d_cross

def d_cross_ab(a, b, da, db):
    """
    Given two vectors a, b and their derivatives w/r.t. a parameter, return the derivative
    of the cross product
    """
    answer = np.zeros((da.shape[0], 3), dtype=float)
    for i in range(da.shape[0]):
        answer[i] = np.cross(a, db[i]) + np.cross(da[i], b)
    return answer

def ncross(a, b):
    """
    Scalar function: Given vectors a and b, return the norm of the cross product
    """
    cross = np.cross(a, b)
    return np.linalg.norm(cross)

def d_ncross(a, b):
    """
    Return the gradient of the norm of the cross product w/r.t. a
    """
    ncross = np.linalg.norm(np.cross(a, b))
    term1 = a * np.dot(b, b)
    term2 = -b * np.dot(a, b)
    answer = (term1+term2)/ncross
    return answer

def nudot(a, b):
    r"""
    Given two vectors a and b, return the dot product (\hat{a}).b.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(ev, b)
    
def d_nudot(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the dot product (\hat{a}).b w/r.t. a.
    """
    return np.dot(d_unit_vector(a), b)

def ucross(a, b):
    r"""
    Given two vectors a and b, return the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.cross(ev, b)
    
def d_ucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_cross(ev, b))

def nucross(a, b):
    r"""
    Given two vectors a and b, return the norm of the cross product (\hat{a})xb.
    """
    ev = a / np.linalg.norm(a)
    return np.linalg.norm(np.cross(ev, b))
    
def d_nucross(a, b):
    r"""
    Given two vectors a and b, return the gradient of 
    the norm of the cross product (\hat{a})xb w/r.t. a.
    """
    ev = a / np.linalg.norm(a)
    return np.dot(d_unit_vector(a), d_ncross(ev, b))
## End vector calculus functions


#8/10/2019
# LPW wrote a way to perform conjugate orthogonalization
# here is my own implementation which is similar to the orthogonalize  
# GS algorithm. Also different is that LPW algorithm does not
# normalize the basis vectors -- they are simply the 
# orthogonalized projection on the G surface.
# Here I think the DLC vectors are orthonormalized
# on the G surface.

def conjugate_orthogonalize(vecs,G,numCvecs=0):
    """
    vecs contains some set of vectors 
    we want to orthogonalize them on the surface G
    numCvecs is the number of vectors that should be linearly dependent 
    after the gram-schmidt process is applied
    """
    rows=vecs.shape[0]
    cols=vecs.shape[1]
    Expect = cols - numCvecs

    # basis holds the Gram-schmidt orthogonalized DLCs
    basis=np.zeros((rows,Expect))
    norms = np.zeros(Expect,dtype=float)

    def ov(vi, vj):
        return np.linalg.multi_dot([vi, G, vj])

    # first orthogonalize the Cvecs
    for ic in range(numCvecs):  # orthogonalize with respect to these
        basis[:,ic]= vecs[:,ic].copy()
        ui = basis[:,ic]
        norms[ic] = np.sqrt(ov(ui,ui))

        # Project out newest basis column from all remaining vecs columns.
        for jc in range(ic+1, cols):
            vj = vecs[:, jc]
            vj -= ui * ov(ui, vj)/norms[ic]**2

    count=numCvecs
    for v in vecs[:,numCvecs:].T:
        #w = v - np.sum( np.dot(v,b)*b  for b in basis.T)
        w = v - np.sum( ov(v,b)*b/ov(b,b)  for b in basis[:,:count].T)
        #A =np.linalg.multi_dot([w[:,np.newaxis].T,G,w[:,np.newaxis]])
        #wnorm = np.sqrt(ov(w[:,np.newaxis],w[:,np.newaxis]))
        wnorm = np.sqrt(ov(w,w))
        if wnorm > 1e-5 and (abs(w) > 1e-6).any():
            try:
                basis[:,count]=w/wnorm
                count+=1
            except:
                print("this vector should be vanishing, exiting")
                print("norm=",wnorm)
                print(w)
                exit(1)
    dots = np.linalg.multi_dot([basis.T,G,basis])
    if not (np.allclose(dots,np.eye(dots.shape[0],dtype=float))):
        print("np.dot(b.T,b)")
        print(dots)
        raise RuntimeError("error in orthonormality")
    return basis


#TODO cVecs can be orthonormalized first to make it less confusing 
# since they are being added to basis before being technically orthonormal
def orthogonalize(vecs,numCvecs=0):
    """
    """
    rows=vecs.shape[0]
    cols=vecs.shape[1]
    basis=np.zeros((rows,cols-numCvecs))

    for i in range(numCvecs):  # orthogonalize with respect to these
        basis[:,i]= vecs[:,i]

    count=numCvecs
    for v in vecs.T:
        w = v - np.sum( np.dot(v,b)*b  for b in basis.T)
        wnorm = np.linalg.norm(w)
        if wnorm > 1e-5 and (abs(w) > 1e-6).any():
            try:
                basis[:,count]=w/wnorm
                count+=1
            except:
                print("this vector should be vanishing, exiting")
                print("norm=",wnorm)
                print(w)
                exit(1)
    dots = np.matmul(basis.T,basis)
    if not (np.allclose(dots,np.eye(dots.shape[0],dtype=float))):
        print("np.dot(b.T,b)")
        print(dots)
        raise RuntimeError("error in orthonormality")
    return basis

