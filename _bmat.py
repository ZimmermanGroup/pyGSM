import numpy as np
import openbabel as ob
import pybel as pb
from units import *

""" This class has functions for creating the Bmatrix"""

class Bmat:

    def diagonalize_G(self,G):
        SVD=False
        if SVD:
            v_temp,e,vh  = np.linalg.svd(G)
        else:
            e,v_temp = np.linalg.eigh(G)
            idx = e.argsort()[::-1]
            e = e[idx]
            v_temp = v_temp[:,idx]
        v = np.transpose(v_temp)
        return e,v

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

    def bmatp_create(self):
        N3 = 3*self.natoms
        #print "Number of internal coordinates is %i " % self.num_ics
        bmatp=np.zeros((self.num_ics,N3),dtype=float)
        i=0
        for bond in self.BObj.bonds:
            a1=bond[0]-1
            a2=bond[1]-1
            dqbdx = self.bmatp_dqbdx(a1,a2)
            bmatp[i,3*a1+0] = dqbdx[0]
            bmatp[i,3*a1+1] = dqbdx[1]
            bmatp[i,3*a1+2] = dqbdx[2]
            bmatp[i,3*a2+0] = dqbdx[3]
            bmatp[i,3*a2+1] = dqbdx[4]
            bmatp[i,3*a2+2] = dqbdx[5]
            i+=1
            #print "%s" % ((a1,a2),)

        for angle in self.AObj.angles:
            a1=angle[0]-1
            a2=angle[1]-1 #vertex
            a3=angle[2]-1
            dqadx = self.bmatp_dqadx(a1,a2,a3)
            bmatp[i,3*a1+0] = dqadx[0]
            bmatp[i,3*a1+1] = dqadx[1]
            bmatp[i,3*a1+2] = dqadx[2]
            bmatp[i,3*a2+0] = dqadx[3]
            bmatp[i,3*a2+1] = dqadx[4]
            bmatp[i,3*a2+2] = dqadx[5]
            bmatp[i,3*a3+0] = dqadx[6]
            bmatp[i,3*a3+1] = dqadx[7]
            bmatp[i,3*a3+2] = dqadx[8]
            i+=1
            #print "%s" % ((a1,a2,a3),)

        for torsion in self.TObj.torsions:
            a1=torsion[0]-1
            a2=torsion[1]-1
            a3=torsion[2]-1
            a4=torsion[3]-1
            #print "%s" % ((a1,a2,a3,a4),)
            dqtdx = self.bmatp_dqtdx(a1,a2,a3,a4)
            bmatp[i,3*a1+0] = dqtdx[0]
            bmatp[i,3*a1+1] = dqtdx[1]
            bmatp[i,3*a1+2] = dqtdx[2]
            bmatp[i,3*a2+0] = dqtdx[3]
            bmatp[i,3*a2+1] = dqtdx[4]
            bmatp[i,3*a2+2] = dqtdx[5]
            bmatp[i,3*a3+0] = dqtdx[6]
            bmatp[i,3*a3+1] = dqtdx[7]
            bmatp[i,3*a3+2] = dqtdx[8]
            bmatp[i,3*a4+0] = dqtdx[9]
            bmatp[i,3*a4+1] = dqtdx[10]
            bmatp[i,3*a4+2] = dqtdx[11]
            i+=1
        if self.print_level>2:
            print "printing bmatp"
            print bmatp
            print "\n"
        #print "shape of bmatp is %s" %(np.shape(bmatp),)
        return bmatp

    def q_create(self):  
        """Determines the scalars in delocalized internal coordinates"""

        #print(" Determining q in ICs")
        N3=3*self.natoms
        q = np.zeros((self.nicd,1),dtype=float)

        dists=[self.distance(bond[0],bond[1]) for bond in self.BObj.bonds ]
        angles=[self.get_angle(angle[0],angle[1],angle[2])*np.pi/180. for angle in self.AObj.angles ]
        tmp =[self.get_torsion(torsion[0],torsion[1],torsion[2],torsion[3]) for torsion in self.TObj.torsions]
        torsions=[]
        for i,j in zip(self.torv0,tmp):
            tordiff = i-j
            if tordiff>180.:
                torfix=360.
            elif tordiff<-180.:
                torfix=-360.
            else:
                torfix=0.
            torsions.append((j+torfix)*np.pi/180.)

        for i in range(self.nicd):
            q[i] = np.dot(self.Ut[i,0:self.BObj.nbonds],dists) + \
                    np.dot(self.Ut[i,self.BObj.nbonds:self.AObj.nangles+self.BObj.nbonds],angles) \
                    + np.dot(self.Ut[i,self.BObj.nbonds+self.AObj.nangles:self.num_ics_p],torsions)

        #print("Printing q")
        #print np.transpose(q)
        return q

    def Hintp_to_Hint(self):
        tmp = np.dot(self.Ut,self.Hintp) #(nicd,numic)(num_ic,num_ic)
        return np.matmul(tmp,np.transpose(self.Ut)) #(nicd,numic)(numic,numic)

    def fromDLC_to_ICbasis(self,vecq):
        """
        This function takes a matrix of vectors wrtiten in the basis of U.
        The components in this basis are called q.
        """
        vec_U = np.zeros((self.num_ics,1),dtype=float)
        assert np.shape(vecq) == (self.nicd,1), "vecq is not nicd long"
        vec_U = np.dot(self.Ut.T,vecq)
        return vec_U/np.linalg.norm(vec_U)
