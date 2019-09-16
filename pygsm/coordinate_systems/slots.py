
# standard library imports
import sys
import os
from os import path

# third party
import numpy as np

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import nifty,math_utils

try:
    from .rotate import get_expmap, get_expmap_der, is_linear
except:
    from rotate import get_expmap, get_expmap_der, is_linear

class CartesianX(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        return "Cartesian-X %i" % (self.a+1)

    @property
    def atoms(self):
        return [self.a]
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            nifty.logger.warning("Warning: CartesianX same atoms, different weights (%.4f %.4f)" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][0]*self.w
        
    def derivative(self, xyz, start_idx=0):
        '''
        start idx is used for fragments, in that case pass in only the xyz of the fragment 
        Expecting shape of xyz to be (N,3)
        '''
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = self.a - start_idx
        derivatives[relative_a][0] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class CartesianY(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Y %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Y %i" % (self.a+1)

    @property
    def atoms(self):
        return [self.a]
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            nifty.logger.warning("Warning: CartesianY same atoms, different weights (%.4f %.4f)" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][1]*self.w
        
    def derivative(self, xyz, start_idx=0):
        '''
        start idx is used for fragments, in that case pass in only the xyz of the fragment 
        Expecting shape of xyz to be (N,3)
        '''
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = self.a - start_idx
        derivatives[relative_a][1] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class CartesianZ(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w=1.0):
        self.a = a
        self.w = w
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Cartesian-Z %i : Weight %.3f" % (self.a+1, self.w)
        return "Cartesian-Z %i" % (self.a+1)

    @property
    def atoms(self):
        return [self.a]
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = self.a == other.a
        if eq and self.w != other.w:
            nifty.logger.warning("Warning: CartesianZ same atoms, different weights (%.4f %.4f)" % (self.w, other.w))
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        return xyz[a][2]*self.w
        
    def derivative(self, xyz, start_idx=0):
        '''
        start idx is used for fragments, in that case pass in only the xyz of the fragment 
        Expecting shape of xyz to be (N,3)
        '''
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = self.a - start_idx
        derivatives[relative_a][2] = self.w
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationX(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-X %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-X %s" % (nifty.commadash(self.a))
        
    @property
    def atoms(self):
        return list(self.a)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            nifty.logger.warning("Warning: TranslationX same atoms, different weights")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,0]*self.w)
        
    def derivative(self, xyz, start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = [ a-start_idx for a in self.a]
        for i, a in enumerate(relative_a):
            derivatives[a][0] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationY(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Y %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Y %s" % (nifty.commadash(self.a))

    @property
    def atoms(self):
        return list(self.a)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            nifty.logger.warning("Warning: TranslationY same atoms, different weights")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,1]*self.w)
        
    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = [ a-start_idx for a in self.a]
        for i, a in enumerate(relative_a):
            derivatives[a][1] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class TranslationZ(object):
    __slots__ = ['a','w','isAngular','isPeriodic']
    def __init__(self, a, w):
        self.a = a
        self.w = w
        assert len(a) == len(w)
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        # return "Translation-Z %s : Weights %s" % (' '.join([str(i+1) for i in self.a]), ' '.join(['%.2e' % i for i in self.w]))
        return "Translation-Z %s" % (nifty.commadash(self.a))

    @property
    def atoms(self):
        return list(self.a)
        
    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.w-other.w)**2) > 1e-6:
            nifty.logger.warning("Warning: TranslationZ same atoms, different weights")
            eq = False
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        return np.sum(xyz[a,2]*self.w)
        
    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        relative_a = [ a-start_idx for a in self.a]
        for i, a in enumerate(relative_a):
            derivatives[a][2] = self.w[i]
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        return deriv2

class Rotator(object):
    __slots__=['a','x0','stored_value','stored_valxyz','stored_deriv','stored_derxyz','stored_deriv2','stored_deriv2xyz','stored_norm','e0','stored_dot2','linear']
    def __init__(self, a, x0):
        self.a = list(tuple(sorted(a)))
        x0 = x0.reshape(-1, 3)
        self.x0 = x0.copy()
        self.stored_valxyz = np.zeros_like(x0)
        self.stored_value = None
        self.stored_derxyz = np.zeros_like(x0)
        self.stored_deriv = None
        self.stored_deriv2xyz = np.zeros_like(x0)
        self.stored_deriv2 = None
        self.stored_norm = 0.0
        # Extra variables to account for the case of linear molecules
        # The reference axis used for computing dummy atom position
        self.e0 = None
        # Dot-squared measures alignment of molecule long axis with reference axis.
        # If molecule becomes parallel with reference axis, coordinates must be reset.
        self.stored_dot2 = 0.0
        # Flag that records linearity of molecule
        self.linear = False

    def reset(self, x0):
        x0 = x0.reshape(-1, 3)
        self.x0 = x0.copy()
        self.stored_valxyz = np.zeros_like(x0)
        self.stored_value = None
        self.stored_derxyz = np.zeros_like(x0)
        self.stored_deriv = None
        self.stored_norm = 0.0
        self.e0 = None
        self.stored_dot2 = 0.0
        self.linear = False

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
            nifty.logger.warning("Warning: Rotator same atoms, different reference positions")
        return eq

    def __repr__(self):
        return "Rotator %s" % nifty.commadash(self.a)

    def __ne__(self, other):
        return not self.__eq__(other)

    def calc_e0(self):
        """
        Compute the reference axis for adding dummy atoms. 
        Only used in the case of linear molecules.

        We first find the Cartesian axis that is "most perpendicular" to the molecular axis.
        Next we take the cross product with the molecular axis to create a perpendicular vector.
        Finally, this perpendicular vector is normalized to make a unit vector.
        """
        ysel = self.x0[self.a, :]
        vy = ysel[-1]-ysel[0]
        ev = vy / np.linalg.norm(vy)
        # Cartesian axes.
        ex = np.array([1.0,0.0,0.0])
        ey = np.array([0.0,1.0,0.0])
        ez = np.array([0.0,0.0,1.0])
        self.e0 = np.cross(vy, [ex, ey, ez][np.argmin([np.dot(i, ev)**2 for i in [ex, ey, ez]])])
        self.e0 /= np.linalg.norm(self.e0)

    def value(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_valxyz)) < 1e-12:
            return self.stored_value
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            xmean = np.mean(xsel,axis=0)
            ymean = np.mean(ysel,axis=0)
            if not self.linear and is_linear(xsel, ysel):
                # print "Setting linear flag for", self
                self.linear = True
            if self.linear:
                # Handle linear molecules.
                vx = xsel[-1]-xsel[0]
                vy = ysel[-1]-ysel[0]
                # Calculate reference axis (if needed)
                if self.e0 is None: self.calc_e0()
                #log.debug(vx)
                ev = vx / np.linalg.norm(vx)
                # Measure alignment of molecular axis with reference axis
                self.stored_dot2 = np.dot(ev, self.e0)**2
                # Dummy atom is located one Bohr from the molecular center, direction
                # given by cross-product of the molecular axis with the reference axis
                xdum = np.cross(vx, self.e0)
                ydum = np.cross(vy, self.e0)
                exdum = xdum / np.linalg.norm(xdum)
                eydum = ydum / np.linalg.norm(ydum)
                xsel = np.vstack((xsel, exdum+xmean))
                ysel = np.vstack((ysel, eydum+ymean))
            answer = get_expmap(xsel, ysel)
            self.stored_norm = np.linalg.norm(answer)
            self.stored_valxyz = xyz.copy()
            self.stored_value = answer
            return answer

    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1, 3)
        relative_a = [ a-start_idx for a in self.a]
        if np.max(np.abs(xyz-self.stored_derxyz[relative_a])) < 1e-12:
            return self.stored_deriv[relative_a]
        else:
            xsel = xyz[relative_a, :]
            ysel = self.x0[relative_a, :]
            xmean = np.mean(xsel,axis=0)
            ymean = np.mean(ysel,axis=0)
            if not self.linear and is_linear(xsel, ysel):
                # print "Setting linear flag for", self
                self.linear = True
            if self.linear:
                vx = xsel[-1]-xsel[0]
                vy = ysel[-1]-ysel[0]
                if self.e0 is None: self.calc_e0()
                xdum = np.cross(vx, self.e0)
                ydum = np.cross(vy, self.e0)
                exdum = xdum / np.linalg.norm(xdum)
                eydum = ydum / np.linalg.norm(ydum)
                xsel = np.vstack((xsel, exdum+xmean))
                ysel = np.vstack((ysel, eydum+ymean))
            deriv_raw = get_expmap_der(xsel, ysel)
            if self.linear:
                # Chain rule is applied to get terms from
                # dummy atom derivatives
                nxdum = np.linalg.norm(xdum)
                dxdum = math_utils.d_cross(vx, self.e0)
                dnxdum = math_utils.d_ncross(vx, self.e0)
                # Derivative of dummy atom position w/r.t. molecular axis vector
                dexdum = (dxdum*nxdum - np.outer(dnxdum,xdum))/nxdum**2
                # Here we may compute finite difference derivatives to check
                # h = 1e-6
                # fdxdum = np.zeros((3, 3), dtype=float)
                # for i in range(3):
                #     vx[i] += h
                #     dPlus = np.cross(vx, self.e0)
                #     dPlus /= np.linalg.norm(dPlus)
                #     vx[i] -= 2*h
                #     dMinus = np.cross(vx, self.e0)
                #     dMinus /= np.linalg.norm(dMinus)
                #     vx[i] += h
                #     fdxdum[i] = (dPlus-dMinus)/(2*h)
                # if np.linalg.norm(dexdum - fdxdum) > 1e-6:
                #     print dexdum - fdxdum
                #     raise Exception()
                # Apply terms from chain rule
                deriv_raw[0]  -= np.dot(dexdum, deriv_raw[-1])
                for i in range(len(self.a)):
                    deriv_raw[i]  += np.dot(np.eye(3), deriv_raw[-1])/len(self.a)
                deriv_raw[-2] += np.dot(dexdum, deriv_raw[-1])
                deriv_raw = deriv_raw[:-1]
            derivatives = np.zeros((xyz.shape[0], 3, 3), dtype=float)
            #for i, a in enumerate(self.a):
            for i, a in enumerate(relative_a):
                derivatives[a, :, :] = deriv_raw[i, :, :]
            self.stored_derxyz = xyz.copy()
            self.stored_deriv = derivatives
            return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1, 3)
        if np.max(np.abs(xyz-self.stored_deriv2xyz)) < 1e-12:
            return self.stored_deriv2
        else:
            xsel = xyz[self.a, :]
            ysel = self.x0[self.a, :]
            xmean = np.mean(xsel,axis=0)
            ymean = np.mean(ysel,axis=0)
            if not self.linear and is_linear(xsel, ysel):
                # print "Setting linear flag for", self
                self.linear = True
            if self.linear:
                vx = xsel[-1]-xsel[0]
                vy = ysel[-1]-ysel[0]
                if self.e0 is None: self.calc_e0()
                xdum = np.cross(vx, self.e0)
                ydum = np.cross(vy, self.e0)
                exdum = xdum / np.linalg.norm(xdum)
                eydum = ydum / np.linalg.norm(ydum)
                xsel = np.vstack((xsel, exdum+xmean))
                ysel = np.vstack((ysel, eydum+ymean))

            deriv_raw, deriv2_raw = get_expmap_der(xsel, ysel, second=True)
            if self.linear:
                # Chain rule is applied to get terms from dummy atom derivatives
                def dexdum_(vx_):
                    xdum_ = np.cross(vx_, self.e0)
                    nxdum_ = np.linalg.norm(xdum_)
                    dxdum_ = math_utils.d_cross(vx_, self.e0)
                    dnxdum_ = math_utils.d_ncross(vx_, self.e0)
                    dexdum_ = (dxdum_*nxdum_ - np.outer(dnxdum_,xdum_))/nxdum_**2
                    return dexdum_.copy()

                # First indices: elements of vx that are being differentiated w/r.t.
                # Last index: elements of exdum itself
                dexdum = dexdum_(vx)
                dexdum2 = np.zeros((3, 3, 3), dtype=float)
                h = 1.0e-3
                for i in range(3):
                    vx[i] += h
                    dPlus = dexdum_(vx)
                    vx[i] -= 2*h
                    dMinus = dexdum_(vx)
                    vx[i] += h
                    dexdum2[i] = (dPlus-dMinus)/(2*h)
                # Build arrays that contain derivative of dummy atom position
                # w/r.t. real atom positions
                ddum1 = np.zeros((len(self.a), 3, 3), dtype=float)
                ddum1[0] = -dexdum
                ddum1[-1] = dexdum
                for i in range(len(self.a)):
                    ddum1[i] += np.eye(3)/len(self.a)
                ddum2 = np.zeros((len(self.a), 3, len(self.a), 3, 3), dtype=float)
                ddum2[ 0, : , 0, :] =  dexdum2
                ddum2[-1, : , 0, :] = -dexdum2
                ddum2[ 0, :, -1, :] = -dexdum2
                ddum2[-1, :, -1, :] =  dexdum2
                # =====

                # Do not delete - reference codes using loops for chain rule terms
                # for j in range(len(self.a)): # Loop over atom 1
                #     for m in range(3):       # Loop over xyz of atom 1
                #         for k in range(len(self.a)): # Loop over atom 2
                #             for n in range(3):       # Loop over xyz of atom 2
                #                 for i in range(3):   # Loop over elements of exponential map
                #                     for p in range(3): # Loop over xyz of dummy atom
                #                         deriv2_raw[j, m, k, n, i] += deriv2_raw[j, m, -1, p, i] * ddum1[k, n, p]
                #                         deriv2_raw[j, m, k, n, i] += deriv2_raw[-1, p, k, n, i] * ddum1[j, m, p]
                #                         deriv2_raw[j, m, k, n, i] += deriv_raw[-1, p, i] * ddum2[j, m, k, n, p]
                #                         for q in range(3):
                #                             deriv2_raw[j, m, k, n, i] += deriv2_raw[-1, p, -1, q, i] * ddum1[j, m, p] * ddum1[k, n, q]
                # =====

                deriv2_raw[:-1, :, :-1, :] += np.einsum('jmpi,knp->jmkni', deriv2_raw[:-1, :, -1, :, :], ddum1, optimize=True)
                deriv2_raw[:-1, :, :-1, :] += np.einsum('pkni,jmp->jmkni', deriv2_raw[-1, :, :-1, :, :], ddum1, optimize=True)
                deriv2_raw[:-1, :, :-1, :] += np.einsum('pi,jmknp->jmkni', deriv_raw[-1, :, :], ddum2, optimize=True)
                deriv2_raw[:-1, :, :-1, :] += np.einsum('pqi,jmp,knq->jmkni', deriv2_raw[-1, :, -1, :, :], ddum1, ddum1, optimize=True)
                deriv2_raw = deriv2_raw[:-1, :, :-1, :, :]
            second_derivatives = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3, 3), dtype=float)

            for i, a in enumerate(self.a):
                for j, b in enumerate(self.a):
                    second_derivatives[a, :, b, :, :] = deriv2_raw[i, :, j, :, :]
            return second_derivatives

class RotationA(object):
    __slots__=['a','x0','w','Rotator','isAngular','isPeriodic']
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

    def __repr__(self):
        # return "Rotation-A %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-A %s" % (nifty.commadash(self.a))

    @property
    def atoms(self):
        return list(self.a)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationA same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationA same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[0]*self.w
        
    def derivative(self, xyz,start_idx=0):
        der_all = self.Rotator.derivative(xyz,start_idx)
        derivatives = der_all[:, :, 0]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 0]*self.w
        return second_derivatives

class RotationB(object):
    __slots__=['a','x0','w','Rotator','isAngular','isPeriodic']
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

    def __repr__(self):
        # return "Rotation-B %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-B %s" % (nifty.commadash(self.a))

    @property
    def atoms(self):
        return list(self.a)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationB same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationB same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[1]*self.w
        
    def derivative(self, xyz,start_idx=0):
        der_all = self.Rotator.derivative(xyz,start_idx)
        derivatives = der_all[:, :, 1]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 1]*self.w
        return second_derivatives

class RotationC(object):
    __slots__=['a','x0','w','Rotator','isAngular','isPeriodic']
    def __init__(self, a, x0, Rotators, w=1.0):
        self.a = tuple(sorted(a))
        self.x0 = x0
        self.w = w
        if self.a not in Rotators:
            Rotators[self.a] = Rotator(self.a, x0)
        self.Rotator = Rotators[self.a]
        self.isAngular = True
        self.isPeriodic = False

    def __repr__(self):
        # return "Rotation-C %s : Weight %.3f" % (' '.join([str(i+1) for i in self.a]), self.w)
        return "Rotation-C %s" % (nifty.commadash(self.a))

    @property
    def atoms(self):
        return list(self.a)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        eq = set(self.a) == set(other.a)
        # if eq and np.sum((self.w-other.w)**2) > 1e-6:
        #     print "Warning: RotationC same atoms, different weights"
        # if eq and np.sum((self.x0-other.x0)**2) > 1e-6:
        #     print "Warning: RotationC same atoms, different reference positions"
        return eq

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        return self.Rotator.value(xyz)[2]*self.w
        
    def derivative(self, xyz,start_idx=0):
        der_all = self.Rotator.derivative(xyz,start_idx)
        derivatives = der_all[:, :, 2]*self.w
        return derivatives

    def second_derivative(self, xyz):
        deriv2_all = self.Rotator.second_derivative(xyz)
        second_derivatives = deriv2_all[:, :, :, :, 2]*self.w
        return second_derivatives

class Distance(object):
    __slots__=['a','b','isAngular','isPeriodic']
    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a == b:
            raise RuntimeError('a and b must be different')
        self.isAngular = False
        self.isPeriodic = False

    def __repr__(self):
        return "Distance %i-%i" % (self.a+1, self.b+1)

    @property
    def atoms(self):
        return [self.a,self.b]

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                return True
        if self.a == other.b:
            if self.b == other.a:
                return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        return np.sqrt(np.sum((xyz[a]-xyz[b])**2))
        
    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a-start_idx
        n = self.b-start_idx
        u = (xyz[m] - xyz[n]) / np.linalg.norm(xyz[m] - xyz[n])
        derivatives[m, :] = u
        derivatives[n, :] = -u
        return derivatives

    def second_derivative(self, xyz):
          xyz = xyz.reshape(-1,3)
          deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
          m = self.a
          n = self.b
          l = np.linalg.norm(xyz[m] - xyz[n])
          u = (xyz[m] - xyz[n]) / l
          mtx = (np.outer(u, u) - np.eye(3))/l
          deriv2[m, :, m, :] = -mtx
          deriv2[n, :, n, :] = -mtx
          deriv2[m, :, n, :] = mtx
          deriv2[n, :, m, :] = mtx
          return deriv2

class Angle(object):
    __slots__=['a','b','c','isAngular','isPeriodic']
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        return "Angle %i-%i-%i" % (self.a+1, self.b+1, self.c+1)

    @property
    def atoms(self):
        return [self.a,self.b,self.c]

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if self.a == other.a:
                if self.c == other.c:
                    return True
            if self.a == other.c:
                if self.c == other.a:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        if dot / (norm1 * norm2) >= 1.0:
            if (np.abs(dot / (norm1 * norm2)) - 1.0) > 1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return 0.0
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # vector from first atom to central atom
        vector1 = xyz[a] - xyz[b]
        # vector from last atom to central atom
        vector2 = xyz[c] - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a-start_idx
        o = self.b-start_idx
        n = self.c-start_idx
        # Unit displacement vectors
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
        VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)
        w = w_prime / np.linalg.norm(w_prime)
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm
        derivatives[m, :] = term1
        derivatives[n, :] = term2
        derivatives[o, :] = -(term1 + term2)
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        o = self.b
        n = self.c
        # Unit displacement vectors
        u_prime = (xyz[m] - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyz[n] - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        # Deriv2 derivatives are set to zero in the case of parallel or antiparallel vectors
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            return deriv2
        # cosine and sine of the bond angle
        cq = np.dot(u, v)
        sq = np.sqrt(1-cq**2)
        uu = np.outer(u, u)
        uv = np.outer(u, v)
        vv = np.outer(v, v)
        de = np.eye(3)
        term1 = (uv + uv.T - (3*uu - de)*cq)/(u_norm**2*sq)
        term2 = (uv + uv.T - (3*vv - de)*cq)/(v_norm**2*sq)
        term3 = (uu + vv - uv*cq   - de)/(u_norm*v_norm*sq)
        term4 = (uu + vv - uv.T*cq - de)/(u_norm*v_norm*sq)
        der1 = self.derivative(xyz)
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        for a in [m, n, o]:
            for b in [m, n, o]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1
                                      + zeta(a, n, o)*zeta(b, n, o)*term2
                                      + zeta(a, m, o)*zeta(b, n, o)*term3
                                      + zeta(a, n, o)*zeta(b, m, o)*term4
                                      - (cq/sq) * np.outer(der1[a], der1[b]))
        return deriv2


class LinearAngle(object):
    __slots__=['a','b','c','axis','e0','stored_dot2','isAngular','isPeriodic']
    def __init__(self, a, b, c, axis):
        self.a = a
        self.b = b
        self.c = c
        self.axis = axis
        self.isAngular = False
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')
        self.e0 = None
        self.stored_dot2 = 0.0

    @property
    def atoms(self):
        return [self.a,self.b,self.c]

    def reset(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        # Cartesian axes.
        ex = np.array([1.0,0.0,0.0])
        ey = np.array([0.0,1.0,0.0])
        ez = np.array([0.0,0.0,1.0])
        self.e0 = [ex, ey, ez][np.argmin([np.dot(i, ev)**2 for i in [ex, ey, ez]])]
        self.stored_dot2 = 0.0

    def __repr__(self):
        return "LinearAngle%s %i-%i-%i" % (["X","Y"][self.axis], self.a+1, self.b+1, self.c+1)

    def __eq__(self, other):
        if not hasattr(other, 'axis'): return False
        if self.axis is not other.axis: return False
        if type(self) is not type(other): return False
        if self.b == other.b:
            if self.a == other.a:
                if self.c == other.c:
                    return True
            if self.a == other.c:
                if self.c == other.a:
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def value(self, xyz):
        """
        This function measures the displacement of the BA and BC unit
        vectors in the linear angle "ABC". The displacements are measured
        along two axes that are perpendicular to the AC unit vector.
        """
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        self.stored_dot2 = np.dot(ev, e0)**2
        # Now make two unit vectors that are perpendicular to this one.
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        if self.axis == 0:
            answer = np.dot(eba, e1) + np.dot(ebc, e1)
        else:
            answer = np.dot(eba, e2) + np.dot(ebc, e2)
        return answer

    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        a = self.a-start_idx
        b = self.b-start_idx
        c = self.c-start_idx
        derivatives = np.zeros_like(xyz)
        ## Finite difference derivatives
        ## fderivatives = np.zeros_like(xyz)
        ## h = 1e-6
        ## for u in range(xyz.shape[0]):
        ##     for v in range(3):
        ##         xyz[u, v] += h
        ##         vPlus = self.value(xyz)
        ##         xyz[u, v] -= 2*h
        ##         vMinus = self.value(xyz)
        ##         xyz[u, v] += h
        ##         fderivatives[u, v] = (vPlus-vMinus)/(2*h)
        # Unit vector pointing from a to c.
        v = xyz[c] - xyz[a]
        ev = v / np.linalg.norm(v)
        if self.e0 is None: self.reset(xyz)
        e0 = self.e0
        c1 = np.cross(ev, e0)
        e1 = c1 / np.linalg.norm(c1)
        c2 = np.cross(ev, e1)
        e2 = c2 / np.linalg.norm(c2)
        # BA and BC unit vectors in ABC angle
        vba = xyz[a]-xyz[b]
        eba = vba / np.linalg.norm(vba)
        vbc = xyz[c]-xyz[b]
        ebc = vbc / np.linalg.norm(vbc)
        # Derivative terms
        de0 = np.zeros((3, 3), dtype=float)
        dev = math_utils.d_unit_vector(v)
        dc1 = math_utils.d_cross_ab(ev, e0, dev, de0)
        de1 = np.dot(dc1, math_utils.d_unit_vector(c1))
        dc2 = math_utils.d_cross_ab(ev, e1, dev, de1)
        de2 = np.dot(dc2, math_utils.d_unit_vector(c2))
        deba = math_utils.d_unit_vector(vba)
        debc = math_utils.d_unit_vector(vbc)
        if self.axis == 0:
            derivatives[a, :] = np.dot(deba, e1) + np.dot(-de1, eba) + np.dot(-de1, ebc)
            derivatives[b, :] = np.dot(-deba, e1) + np.dot(-debc, e1)
            derivatives[c, :] = np.dot(de1, eba) + np.dot(de1, ebc) + np.dot(debc, e1)
        else:
            derivatives[a, :] = np.dot(deba, e2) + np.dot(-de2, eba) + np.dot(-de2, ebc)
            derivatives[b, :] = np.dot(-deba, e2) + np.dot(-debc, e2)
            derivatives[c, :] = np.dot(de2, eba) + np.dot(de2, ebc) + np.dot(debc, e2)
        ## Finite difference derivatives
        ## if np.linalg.norm(derivatives - fderivatives) > 1e-6:
        ##     print np.linalg.norm(derivatives - fderivatives)
        ##     raise Exception()
        return derivatives

    def second_derivative(self, xyz):
         xyz = xyz.reshape(-1,3)
         a = self.a
         b = self.b
         c = self.c
         deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
         h = 1.0e-3
         for i in range(3):
             for j in range(3):
                 ii = [a, b, c][i]
                 xyz[ii, j] += h
                 FPlus = self.derivative(xyz)
                 xyz[ii, j] -= 2*h
                 FMinus = self.derivative(xyz)
                 xyz[ii, j] += h
                 fderiv = (FPlus-FMinus)/(2*h)
                 deriv2[ii, j, :, :] = fderiv
         return deriv2
    
class MultiAngle(object):
    __slots__=['a','b','c','isAngular','isPeriodic']
    def __init__(self, a, b, c):
        if type(a) is int:
            a = (a,)
        if type(c) is int:
            c = (c,)
        self.a = tuple(a)
        self.b = b
        self.c = tuple(c)
        self.isAngular = True
        self.isPeriodic = False
        if len({a, b, c}) != 3:
            raise RuntimeError('a, b, and c must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strc = ("("+','.join(["%i" % (i+1) for i in self.c])+")") if len(self.c) > 1 else "%i" % (self.c[0]+1)
        return "%sAngle %s-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.c) > 1) else "", stra, self.b+1, strc)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.b == other.b:
            if set(self.a) == set(other.a):
                if set(self.c) == set(other.c):
                    return True
            if set(self.a) == set(other.c):
                if set(self.c) == set(other.a):
                    return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        dot = np.dot(vector1, vector2)
        # Catch the edge case that very rarely this number is -1.
        if dot / (norm1 * norm2) <= -1.0:
            if (np.abs(dot / (norm1 * norm2)) + 1.0) < -1e-6:
                raise RuntimeError('Encountered invalid value in angle')
            return np.pi
        return np.arccos(dot / (norm1 * norm2))

    def normal_vector(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = np.array(self.c)
        xyza = np.mean(xyz[a], axis=0)
        xyzc = np.mean(xyz[c], axis=0)
        # vector from first atom to central atom
        vector1 = xyza - xyz[b]
        # vector from last atom to central atom
        vector2 = xyzc - xyz[b]
        # norm of the two vectors
        norm1 = np.sqrt(np.sum(vector1**2))
        norm2 = np.sqrt(np.sum(vector2**2))
        crs = np.cross(vector1, vector2)
        crs /= np.linalg.norm(crs)
        return crs
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = np.array(self.a)
        o = self.b
        n = np.array(self.c)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        # Unit displacement vectors
        u_prime = (xyzm - xyz[o])
        u_norm = np.linalg.norm(u_prime)
        v_prime = (xyzn - xyz[o])
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        v = v_prime / v_norm
        VECTOR1 = np.array([1, -1, 1]) / np.sqrt(3)
        VECTOR2 = np.array([-1, 1, 1]) / np.sqrt(3)
        if np.linalg.norm(u + v) < 1e-10 or np.linalg.norm(u - v) < 1e-10:
            # if they're parallel
            if ((np.linalg.norm(u + VECTOR1) < 1e-10) or
                    (np.linalg.norm(u - VECTOR2) < 1e-10)):
                # and they're parallel o [1, -1, 1]
                w_prime = np.cross(u, VECTOR2)
            else:
                w_prime = np.cross(u, VECTOR1)
        else:
            w_prime = np.cross(u, v)
        w = w_prime / np.linalg.norm(w_prime)
        term1 = np.cross(u, w) / u_norm
        term2 = np.cross(w, v) / v_norm
        for i in m:
            derivatives[i, :] = term1/len(m)
        for i in n:
            derivatives[i, :] = term2/len(n)
        derivatives[o, :] = -(term1 + term2)
        return derivatives

    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)
    
class Dihedral(object):
    __slots__=['a','b','c','d','isAngular','isPeriodic']
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Dihedral %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    @property
    def atoms(self):
        return [self.a,self.b,self.c,self.d]

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if self.b == other.b:
                if self.c == other.c:
                    if self.d == other.d:
                        return True
        if self.a == other.d:
            if self.b == other.c:
                if self.c == other.b:
                    if self.d == other.a:
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz, start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a-start_idx
        o = self.b-start_idx
        p = self.c-start_idx
        n = self.d-start_idx
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        deriv2 = np.zeros((xyz.shape[0], xyz.shape[1], xyz.shape[0], xyz.shape[1]))
        m = self.a
        o = self.b
        p = self.c
        n = self.d
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        lu = np.linalg.norm(u_prime)
        lw = np.linalg.norm(w_prime)
        lv = np.linalg.norm(v_prime)
        u = u_prime / lu
        w = w_prime / lw
        v = v_prime / lv
        cu = np.dot(u, w)
        su = (1 - np.dot(u, w)**2)**0.5
        su4 = su**4
        cv = np.dot(v, w)
        sv = (1 - np.dot(v, w)**2)**0.5
        sv4 = sv**4
        if su < 1e-6 or sv < 1e-6 : return deriv2
        
        uxw = np.cross(u, w)
        vxw = np.cross(v, w)

        term1 = np.outer(uxw, w*cu - u)/(lu**2*su4)
        term2 = np.outer(vxw, -w*cv + v)/(lv**2*sv4)
        term3 = np.outer(uxw, w - 2*u*cu + w*cu**2)/(2*lu*lw*su4)
        term4 = np.outer(vxw, w - 2*v*cv + w*cv**2)/(2*lv*lw*sv4)
        term5 = np.outer(uxw, u + u*cu**2 - 3*w*cu + w*cu**3)/(2*lw**2*su4)
        term6 = np.outer(vxw,-v - v*cv**2 + 3*w*cv - w*cv**3)/(2*lw**2*sv4)
        term1 += term1.T
        term2 += term2.T
        term3 += term3.T
        term4 += term4.T
        term5 += term5.T
        term6 += term6.T
        def mk_amat(vec):
            amat = np.zeros((3,3))
            for i in range(3):
                for j in range(3):
                    if i == j: continue
                    k = 3 - i - j
                    amat[i, j] = vec[k] * (j-i) * ((-0.5)**np.abs(j-i))
            return amat
        term7 = mk_amat((-w*cu + u)/(lu*lw*su**2))
        term8 = mk_amat(( w*cv - v)/(lv*lw*sv**2))
        def zeta(a_, m_, n_):
            return (int(a_==m_) - int(a_==n_))
        # deriv2_terms = [np.zeros_like(deriv2) for i in range(9)]
        # Accumulate the second derivative
        for a in [m, n, o, p]:
            for b in [m, n, o, p]:
                deriv2[a, :, b, :] = (zeta(a, m, o)*zeta(b, m, o)*term1 +
                                      zeta(a, n, p)*zeta(b, n, p)*term2 +
                                      (zeta(a, m, o)*zeta(b, o, p) + zeta(a, p, o)*zeta(b, o, m))*term3 +
                                      (zeta(a, n, p)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, n, p))*term4 +
                                      zeta(a, o, p)*zeta(b, p, o)*term5 +
                                      zeta(a, p, o)*zeta(b, o, p)*term6)
                if a != b:
                    deriv2[a, :, b, :] += ((zeta(a, m, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, m))*term7 +
                                           (zeta(a, n, o)*zeta(b, p, o) + zeta(a, p, o)*zeta(b, o, n))*term8)
        return deriv2

class MultiDihedral(object):
    __slots__=['a','b','c','d','isAngular','isPeriodic']
    def __init__(self, a, b, c, d):
        if type(a) is int:
            a = (a, )
        if type(d) is int:
            d = (d, )
        self.a = tuple(a)
        self.b = b
        self.c = c
        self.d = tuple(d)
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        stra = ("("+','.join(["%i" % (i+1) for i in self.a])+")") if len(self.a) > 1 else "%i" % (self.a[0]+1)
        strd = ("("+','.join(["%i" % (i+1) for i in self.d])+")") if len(self.d) > 1 else "%i" % (self.d[0]+1)
        return "%sDihedral %s-%i-%i-%s" % ("Multi" if (len(self.a) > 1 or len(self.d) > 1) else "", stra, self.b+1, self.c+1, strd)

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if set(self.a) == set(other.a):
            if self.b == other.b:
                if self.c == other.c:
                    if set(self.d) == set(other.d):
                        return True
        if set(self.a) == set(other.d):
            if self.b == other.c:
                if self.c == other.b:
                    if set(self.d) == set(other.a):
                        return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = np.array(self.a)
        b = self.b
        c = self.c
        d = np.array(self.d)
        xyza = np.mean(xyz[a], axis=0)
        xyzd = np.mean(xyz[d], axis=0)
        
        vec1 = xyz[b] - xyza
        vec2 = xyz[c] - xyz[b]
        vec3 = xyzd - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = np.array(self.a)
        o = self.b
        p = self.c
        n = np.array(self.d)
        xyzm = np.mean(xyz[m], axis=0)
        xyzn = np.mean(xyz[n], axis=0)
        
        u_prime = (xyzm - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyzn - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        for i in self.a:
            derivatives[i, :] = term1/len(self.a)
        for i in self.d:
            derivatives[i, :] = -term2/len(self.d)
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

    def second_derivative(self, xyz):
        raise NotImplementedError("Second derivatives have not been implemented for IC type %s" % self.__name__)
    
class OutOfPlane(object):
    __slots__=['a','b','c','d','isAngular','isPeriodic']
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.isAngular = True
        self.isPeriodic = True
        if len({a, b, c, d}) != 4:
            raise RuntimeError('a, b, c and d must be different')

    def __repr__(self):
        return "Out-of-Plane %i-%i-%i-%i" % (self.a+1, self.b+1, self.c+1, self.d+1)

    @property
    def atoms(self):
        return [self.a,self.b,self.c,self.d]

    def __eq__(self, other):
        if type(self) is not type(other): return False
        if self.a == other.a:
            if {self.b, self.c, self.d} == {other.b, other.c, other.d}:
                if [self.b, self.c, self.d] != [other.b, other.c, other.d]:
                    nifty.logger.warning("Warning: OutOfPlane atoms are the same, ordering is different")
                return True
        #     if self.b == other.b:
        #         if self.c == other.c:
        #             if self.d == other.d:
        #                 return True
        # if self.a == other.d:
        #     if self.b == other.c:
        #         if self.c == other.b:
        #             if self.d == other.a:
        #                 return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)
        
    def value(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        a = self.a-start_idx
        b = self.b-start_idx
        c = self.c-start_idx
        d = self.d-start_idx
        vec1 = xyz[b] - xyz[a]
        vec2 = xyz[c] - xyz[b]
        vec3 = xyz[d] - xyz[c]
        cross1 = np.cross(vec2, vec3)
        cross2 = np.cross(vec1, vec2)
        arg1 = np.sum(np.multiply(vec1, cross1)) * \
               np.sqrt(np.sum(vec2**2))
        arg2 = np.sum(np.multiply(cross1, cross2))
        answer = np.arctan2(arg1, arg2)
        return answer
        
    def derivative(self, xyz,start_idx=0):
        xyz = xyz.reshape(-1,3)
        derivatives = np.zeros_like(xyz)
        m = self.a-start_idx
        o = self.b-start_idx
        p = self.c-start_idx
        n = self.d-start_idx
        u_prime = (xyz[m] - xyz[o])
        w_prime = (xyz[p] - xyz[o])
        v_prime = (xyz[n] - xyz[p])
        u_norm = np.linalg.norm(u_prime)
        w_norm = np.linalg.norm(w_prime)
        v_norm = np.linalg.norm(v_prime)
        u = u_prime / u_norm
        w = w_prime / w_norm
        v = v_prime / v_norm
        if (1 - np.dot(u, w)**2) < 1e-6:
            term1 = np.cross(u, w) * 0
            term3 = np.cross(u, w) * 0
        else:
            term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
            term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        if (1 - np.dot(v, w)**2) < 1e-6:
            term2 = np.cross(v, w) * 0
            term4 = np.cross(v, w) * 0
        else:
            term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
            term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        # term1 = np.cross(u, w) / (u_norm * (1 - np.dot(u, w)**2))
        # term2 = np.cross(v, w) / (v_norm * (1 - np.dot(v, w)**2))
        # term3 = np.cross(u, w) * np.dot(u, w) / (w_norm * (1 - np.dot(u, w)**2))
        # term4 = np.cross(v, w) * np.dot(v, w) / (w_norm * (1 - np.dot(v, w)**2))
        derivatives[m, :] = term1
        derivatives[n, :] = -term2
        derivatives[o, :] = -term1 + term3 - term4
        derivatives[p, :] = term2 - term3 + term4
        return derivatives

    def second_derivative(self, xyz):
        xyz = xyz.reshape(-1,3)
        a = self.a
        b = self.b
        c = self.c
        d = self.d
        deriv2 = np.zeros((xyz.shape[0], 3, xyz.shape[0], 3), dtype=float)
        h = 1.0e-3
        for i in range(4):
            for j in range(3):
                ii = [a, b, c, d][i]
                xyz[ii, j] += h
                FPlus = self.derivative(xyz)
                xyz[ii, j] -= 2*h
                FMinus = self.derivative(xyz)
                xyz[ii, j] += h
                fderiv = (FPlus-FMinus)/(2*h)
                deriv2[ii, j, :, :] = fderiv
        return deriv2

def logArray(mat, precision=3, fmt="f"):
    fmt="%% .%i%s" % (precision, fmt)
    if len(mat.shape) == 1:
        for i in range(mat.shape[0]):
            nifty.logger.info(fmt % mat[i]),
        print()
    elif len(mat.shape) == 2:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                nifty.logger.info(fmt % mat[i,j]),
            print()
    else:
        raise RuntimeError("One or two dimensional arrays only")

