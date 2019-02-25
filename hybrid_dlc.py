from base_dlc import Base_DLC
from time import time
import openbabel as ob
import numpy as np

class Hybrid_DLC(Base_DLC): 
    """
    Hybrid DLC for systems containing a large amount of atoms, the coordinates are partitioned 
    into a QM-region which is simulated with ICs, and a MM-region which is modeled with Cartesians. 
    """
    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Hybrid_DLC(Hybrid_DLC.default_options().set_values(kwargs))

    def create_DLC(icoordA,bondA,angleA,torsionA,mol,PES):
        return Hybrid_DLC(icoordA.options.copy().set_values({
            "bonds":bondA,
            "angles":angleA,
            "torsions":torsionA,
            'mol':mol,
            'PES':PES,
            }))

    def get_nxyzics(self):
        '''
        This function gets the number of xyz ics,
        a list of booleans that describes whether the atom is xyz ic or not,
        and finally it saves the xyzics_coords
        '''
        startidx=0
        self.nxyzatoms=0
        for i,res in enumerate(ob.OBResidueIter(self.mol.OBMol)):
            if res.GetName() not in self.IC_region:
                self.nxyzatoms+=res.GetNumAtoms()
                for j,a in enumerate(ob.OBResidueAtomIter(res)):
                    self.xyzatom_bool[startidx+j]=True
            startidx+= res.GetNumAtoms()
        return self.nxyzatoms

    def set_nicd(self):
        self.nicd=3*(self.natoms-self.nxyzatoms)-6+3*self.nxyzatoms
        self.nicd_DLC=self.nicd

    def bmatp_create(self):
        only_ics= self.BObj.nbonds + self.AObj.nangles + self.TObj.ntor
        bmatp=super(Hybrid_DLC,self).bmatp_create()
        bmatp[only_ics:,(self.natoms-self.nxyzatoms)*3:] = np.eye(self.nxyzatoms*3)
        return bmatp

    def q_create(self):
        q=super(Hybrid_DLC,self).q_create()
        xyzic_atom_coords = self.get_xyz_atom_coords() 
        n3m6 = (self.natoms-self.nxyzatoms)*3-6
        for i in range(self.nicd):
            q[i]+=np.dot(self.Ut[i,self.num_ics_p:],xyzic_atom_coords)
        return q

    def primitive_internal_values(self):
        return np.concatenate((self.BObj.bondd,self.AObj.anglev,self.TObj.torv,self.get_xyz_atom_coords()))

    def primitive_internal_difference(self,qprim1,qprim2):
        dqprim_internals = super(Hybrid_DLC,self).primitive_internal_difference(qprim1,qprim2)
        dqprim_xyzatoms = qprim1[self.num_ics_p:] - qprim2[self.num_ics_p:]
        dqprim_xyzatoms = np.reshape(dqprim_xyzatoms,(3*self.nxyzatoms,1))
        dqprim = np.concatenate((dqprim_internals,dqprim_xyzatoms))
        return dqprim

    def get_xyz_atom_coords(self):
        xyzic_atom_coords=np.zeros((self.nxyzatoms,3))
        count=0
        for i,a in enumerate(self.coords):
            if self.xyzatom_bool[i]==True:
                xyzic_atom_coords[count]=a
                count+=1
        return xyzic_atom_coords.flatten()

    def bmat_create(self):
        self.q = self.q_create()
        if self.print_level>2:
            print "printing q"
            print self.q.T

        #t = time()
        #bmat = np.matmul(self.Ut,self.bmatp)
        #bbt = np.matmul(bmat,np.transpose(bmat))
        #t = time()
        #bbti = np.linalg.inv(bbt)
        #delta = time() - t
        #self.bmatti= np.matmul(bbti,bmat)
        ##return
        #print "time for full inverse ",delta

        #t=time()
        n3m6 = (self.natoms-self.nxyzatoms)*3-6
        UtDLC = self.Ut[:n3m6-self.lowev,:self.num_ics_p] # shoud it be + lowev?
        UtC = self.Ut[n3m6:,self.num_ics_p:]

        Bp=self.bmatp[:self.num_ics_p,:n3m6+6]
        Bc=self.bmatp[self.num_ics_p:,n3m6+6:]

        UtDLC_Bp = np.matmul(UtDLC,Bp)
        UtC_Bc = np.eye(3*self.nxyzatoms)

        bmat = np.block([
                    [ UtDLC_Bp, np.zeros((UtDLC_Bp.shape[0],UtC_Bc.shape[1]))],
                    [ np.zeros((UtC_Bc.shape[0],UtDLC_Bp.shape[1])),UtC_Bc ]
                    ])

        UtDLC_Bp2 = np.matmul(UtDLC_Bp,np.transpose(UtDLC_Bp))
        UtDLC_Bp2i = np.linalg.inv(UtDLC_Bp2)
        UtC_Bc2i = np.eye(3*self.nxyzatoms)

        bbti = np.block([
                [ UtDLC_Bp2i, np.zeros((UtDLC_Bp2i.shape[0],UtC_Bc2i.shape[1]))],
                [ np.zeros((UtC_Bc2i.shape[0],UtDLC_Bp2i.shape[1])),UtC_Bc2i ]
            ])
        #self.bmatti= np.matmul(bbti,bmat)
        bmatti_DLC = np.matmul(UtDLC_Bp2i,UtDLC_Bp)
        self.bmatti = np.block([ 
                [bmatti_DLC, np.zeros((bmatti_DLC.shape[0],3*self.nxyzatoms))],
                [ np.zeros((3*self.nxyzatoms,bmatti_DLC.shape[1])),np.eye(3*self.nxyzatoms)]
                    ])


        #delta= time()-t
        #print "time for partial inverse ",delta

        if self.print_level>2:
            print "bmatti"
            print self.bmatti

    
    def diagonalize_G(self,G):
        #print "using special diagonalize"
        #t = time()
        #total_eig,total_v = super(DLC,self).diagonalize_G(G)
        ##return total_eig,total_v
        #delta = time() - t
        #print "time for full matrix diagonalization ",delta

        # => initialize <= #
        #t = time()
        total_eig = np.zeros(self.nicd)
        total_v = np.zeros((self.nicd,self.num_ics))

        # =>  diagonalize sublock <= #
        eig1,v1 = super(Hybrid_DLC,self).diagonalize_G(G[:self.num_ics_p,:self.num_ics_p])

        # => take only 3N-6 sublock of eig1 <= #
        n3m6 = (self.natoms-self.nxyzatoms)*3-6
        self.lowev=0
        for eig in eig1[:n3m6]:
            if eig<0.001:
                self.lowev+=1
        #print "lowev=",self.lowev
        eig1 = eig1[:n3m6-self.lowev]
        v1 = v1[:n3m6-self.lowev]
        #print eig1

        # => append to initialzed matrices <= #
        print "Hybrid coordinates:"
        print ' [ Ut_DLC 0 ] ' 
        print ' [   0  Ut_C] '

        total_v[:v1.shape[0],:v1.shape[1]] = v1
        total_eig[:eig1.shape[0]] = eig1

        # => xyz sublock <= #
        d = G.diagonal()[self.num_ics_p:]
        np.fill_diagonal(total_v[v1.shape[0]:,v1.shape[1]:],d)
        total_eig[eig1.shape[0]:] = d

        #delta = time() - t
        #print "time for partial matrix diagonalization ",delta

        return total_eig,total_v
        

if __name__ =='__main__':
    #filepath="r001.ttt_meh_oh_ff-solvated-dftb3-dyn1_000.pdb"
    #filepath="solvated.pdb"
    from qchem import *
    from pes import *
    import pybel as pb
    #lot1=QChem.from_options(states=[(1,0)],basis='6-31gs')
    #pes = PES.from_options(lot=lot1,ad_idx=0,multiplicity=1)
    #mol1=pb.readfile("pdb",filepath).next()
    ##ic1=Hybrid_DLC.from_options(mol=mol1,PES=pes,IC_region=['TTT'])
    #ic1=Hybrid_DLC.from_options(mol=mol1,PES=pes,IC_region=['LIG'])

    filepath="firstnode.pdb"
    mol=pb.readfile("pdb",filepath).next()

    lot = QChem.from_options(states=[(2,0)],lot_inp_file='qstart',nproc=1)
    pes = PES.from_options(lot=lot,ad_idx=0,multiplicity=2)
    ic = Hybrid_DLC.from_options(mol=mol,PES=pes,IC_region=["UNL"],print_level=1)

    #dq = np.zeros((ic.nicd,1))
    #dq[0]= 0.2
    #print "dq = "
    #print dq.T
    #ic.ic_to_xyz_opt(dq)

    
