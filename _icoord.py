import numpy as np
import openbabel as ob
import pybel as pb
from units import *

class Mixin:

    def bond_exists(self,bond):
        if bond in self.bonds:
            return True
        else:
            return False

    def print_xyz(self):
        for a in ob.OBMolAtomIter(self.mol.OBMol):
            print(" %1.4f %1.4f %1.4f" %(a.GetX(), a.GetY(), a.GetZ()) )

    def distance(self,i,j):
        """ for some reason openbabel has this one based """
        a1=self.mol.OBMol.GetAtom(i)
        a2=self.mol.OBMol.GetAtom(j)
        return a1.GetDistance(a2)
    def get_angle(self,i,j,k):
        a=self.mol.OBMol.GetAtom(i)
        b=self.mol.OBMol.GetAtom(j)
        c=self.mol.OBMol.GetAtom(k)
        return self.mol.OBMol.GetAngle(b,a,c) #a is the vertex #in degrees

    def get_torsion(self,i,j,k,l):
        a=self.mol.OBMol.GetAtom(i)
        b=self.mol.OBMol.GetAtom(j)
        c=self.mol.OBMol.GetAtom(k)
        d=self.mol.OBMol.GetAtom(l)
        tval=self.mol.OBMol.GetTorsion(a,b,c,d)*np.pi/180.
        #if tval >3.14159:
        if tval>=np.pi:
            tval-=2.*np.pi
        #if tval <-3.14159:
        if tval<=-np.pi:
            tval+=2.*np.pi
        return tval*180./np.pi


    def getIndex(self,i):
        return self.mol.OBMol.GetAtom(i+1).GetIndex()

    def getCoords(self,i):
        a= self.mol.OBMol.GetAtom(i+1)
        return [a.GetX(),a.GetY(),a.GetZ()]

    def getAtomicNum(self,i):
        return self.mol.OBMol.GetAtom(i).GetAtomicNum()

    def isTM(self,i):
        anum= self.getIndex(i)
        if anum>20:
            if anum<31:
                return True
            elif anum >38 and anum < 49:
                return True
            elif anum >71 and anum <81:
                return True

    def bmatp_dqbdx(self,i,j):
        u = np.zeros(3,dtype=float)
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        u=np.subtract(coora,coorb)
        norm= np.linalg.norm(u)
        u = u/norm
        dqbdx = np.zeros(6,dtype=float)
        dqbdx[0] = u[0]
        dqbdx[1] = u[1]
        dqbdx[2] = u[2]
        dqbdx[3] = -u[0]
        dqbdx[4] = -u[1]
        dqbdx[5] = -u[2]
        return dqbdx

    def bmatp_dqadx(self,i,j,k):
        u = np.zeros(3,dtype=float)
        v = np.zeros(3,dtype=float)
        w = np.zeros(3,dtype=float)
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1) #vertex
        c=self.mol.OBMol.GetAtom(k+1)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        coorc=np.array([c.GetX(),c.GetY(),c.GetZ()])
        u=np.subtract(coora,coorb)
        v=np.subtract(coorc,coorb)
        n1=self.distance(i+1,j+1)
        n2=self.distance(j+1,k+1)
        u=u/n1
        v=v/n2


        w=np.cross(u,v)
        nw = np.linalg.norm(w)
        if nw < 1e-3:
            print(" linear angle detected")
            vn = np.zeros(3,dtype=float)
            vn[2]=1.
            w=np.cross(u,vn)
            nw = np.linalg.norm(w)
            if nw < 1e-3:
                vn[2]=0.
                vn[1]=1.
                w=np.cross(u,vn)

        n3=np.linalg.norm(w)
        w=w/n3
        uw=np.cross(u,w)
        wv=np.cross(w,v)
        dqadx = np.zeros(9,dtype=float)
        dqadx[0] = uw[0]/n1
        dqadx[1] = uw[1]/n1
        dqadx[2] = uw[2]/n1
        dqadx[3] = -uw[0]/n1 + -wv[0]/n2
        dqadx[4] = -uw[1]/n1 + -wv[1]/n2
        dqadx[5] = -uw[2]/n1 + -wv[2]/n2
        dqadx[6] = wv[0]/n2
        dqadx[7] = wv[1]/n2
        dqadx[8] = wv[2]/n2


        return dqadx

    def bmatp_dqtdx(self,i,j,k,l):
        a=self.mol.OBMol.GetAtom(i+1)
        b=self.mol.OBMol.GetAtom(j+1) 
        c=self.mol.OBMol.GetAtom(k+1)
        d=self.mol.OBMol.GetAtom(l+1)
        dqtdx = np.zeros(12,dtype=float)

        angle1=self.mol.OBMol.GetAngle(a,b,c)*np.pi/180.
        angle2=self.mol.OBMol.GetAngle(b,c,d)*np.pi/180.
        if angle1>3. or angle2>3.:
            print(" near-linear angle")
            return dqtdx
        u = np.zeros(3,dtype=float)
        v = np.zeros(3,dtype=float)
        w = np.zeros(3,dtype=float)
        coora=np.array([a.GetX(),a.GetY(),a.GetZ()])
        coorb=np.array([b.GetX(),b.GetY(),b.GetZ()])
        coorc=np.array([c.GetX(),c.GetY(),c.GetZ()])
        coord=np.array([d.GetX(),d.GetY(),d.GetZ()])
        u=np.subtract(coora,coorb)
        w=np.subtract(coorc,coorb)
        v=np.subtract(coord,coorc)
        
        n1=self.distance(i+1,j+1)
        n2=self.distance(j+1,k+1)
        n3=self.distance(k+1,l+1)

        u=u/n1
        v=v/n3
        w=w/n2

        uw=np.cross(u,w)
        vw=np.cross(v,w)

        cosphiu = np.dot(u,w)
        cosphiv = -1*np.dot(v,w)
        sin2phiu = 1.-cosphiu*cosphiu
        sin2phiv = 1.-cosphiv*cosphiv
        #print(" cos's: %1.4f %1.4f vs %1.4f %1.4f " % (cosphiu,cosphiv,np.cos(angle1),np.cos(angle2))) 
        #print(" sin2's: %1.4f %1.4f vs %1.4f %1.4f " % (sin2phiu,sin2phiv,np.sin(angle1)*np.sin(angle1),np.sin(angle2)*np.sin(angle2)))

        #TODO why does this cause problems
        if sin2phiu < 1e-3 or sin2phiv <1e-3:
            #print "sin2phiu"
            return dqtdx

        #CPMZ possible error in uw calc
        dqtdx[0]  = uw[0]/(n1*sin2phiu);
        dqtdx[1]  = uw[1]/(n1*sin2phiu);
        dqtdx[2]  = uw[2]/(n1*sin2phiu);
        dqtdx[3]   = -uw[0]/(n1*sin2phiu) + ( uw[0]*cosphiu/(n2*sin2phiu) + vw[0]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[4]   = -uw[1]/(n1*sin2phiu) + ( uw[1]*cosphiu/(n2*sin2phiu) + vw[1]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[5]   = -uw[2]/(n1*sin2phiu) + ( uw[2]*cosphiu/(n2*sin2phiu) + vw[2]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[6]   =  vw[0]/(n3*sin2phiv) - ( uw[0]*cosphiu/(n2*sin2phiu) + vw[0]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[7]   =  vw[1]/(n3*sin2phiv) - ( uw[1]*cosphiu/(n2*sin2phiu) + vw[1]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[8]   =  vw[2]/(n3*sin2phiv) - ( uw[2]*cosphiu/(n2*sin2phiu) + vw[2]*cosphiv/(n2*sin2phiv) )                  
        dqtdx[9]   = -vw[0]/(n3*sin2phiv)                                                                                  
        dqtdx[10]  = -vw[1]/(n3*sin2phiv)                                                                                  
        dqtdx[11]  = -vw[2]/(n3*sin2phiv)

        if np.isnan(dqtdx).any():
            print "Error!"
        return dqtdx


    def close_bond(self,bond):
        A = 0.2
        d = self.distance(bond[0],bond[1])
        #dr = (vdw_radii.radii[self.getAtomicNum(bond[0])] + vdw_radii.radii[self.getAtomicNum(bond[1])] )/2
        a=self.getAtomicNum(bond[0])
        b=self.getAtomicNum(bond[1])
        dr = (self.Elements.from_atomic_number(a).vdw_radius + self.Elements.from_atomic_number(b).vdw_radius )/2.
        val = np.exp(-A*(d-dr))
        if val>1: val=1
        return val

    def update_ic_eigen(self,gradq):
        SCALE =self.SCALEQN
        if self.newHess>0: SCALE = self.SCALEQN*self.newHess
        if self.SCALEQN>10.0: SCALE=10.0
        lambda1 = 0.0

        e,v_temp = np.linalg.eigh(self.Hint)
        v = np.transpose(v_temp)
        e = np.reshape( e,(len(e),1))
        leig = e[0]

        if leig < 0:
            lambda1 = -leig+0.015
        else:
            lambda1 = 0.005
        if abs(lambda1)<0.005: lambda1 = 0.005

        # => grad in eigenvector basis <= #
        gqe = np.matmul(v,gradq)

        dqe0 = np.divide(-gqe,e+lambda1)/SCALE
        for i in dqe0:
            if abs(i)>self.MAXAD:
                i=np.sign(i)*self.MAXAD

        dq0 = np.matmul(v_temp,dqe0)

        for i in dq0:
            if abs(i)>self.MAXAD:
                i=np.sign(i)*self.MAXAD

        return dq0

    def compute_predE(self,dq0):
        # compute predicted change in energy 
        assert np.shape(dq0)==(self.nicd,1), "dq0 not (nicd,1) "
        assert np.shape(self.gradq)==(self.nicd,1), "gradq not (nicd,1) "
        assert np.shape(self.Hint)==(self.nicd,self.nicd), "Hint not (nicd,nicd) "
        dEtemp = np.matmul(self.Hint,dq0)
        dEpre = np.dot(np.transpose(dq0),self.gradq) + 0.5*np.dot(np.transpose(dEtemp),dq0)
        dEpre *=KCAL_MOL_PER_AU
        if abs(dEpre)<0.05: dEpre = np.sign(dEpre)*0.05
        print( "predE: %1.4f " % dEpre) 
        return dEpre

    def grad_to_q(self,grad):
        #np.set_printoptions(precision=4)
        #np.set_printoptions(suppress=True)
        gradq = np.matmul(self.bmatti,grad)
        return gradq
