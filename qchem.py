from base import *
import numpy as np


class QChem(Base):
    
    def getEnergy(self):
        energy =0.
        average_over =0
        for i in self.calc_states:
            energy += self.compute_energy(S=i[0],index=i[1])
            average_over+=1
        return energy/average_over

    def getGrad(self):
        average_over=0
        grad = np.zeros((self.molecule.natom,3))
        for i in self.calc_states:
            tmp = self.lot.compute_gradient(S=i[0],index=i[1])
            print(np.shape(tmp[...]))
            grad += tmp[...] 
            average_over+=1
        final_grad = grad/average_over

        return np.reshape(final_grad,(3*self.molecule.natom,1))
    
    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return QChem(QChem.default_options().set_values(kwargs))
    
if __name__ == '__main__':

    from obutils import *
    import ico as ic
    import pybel as pb    

    filepath="tests/fluoroethene.xyz"
    mol=pb.readfile("xyz",filepath).next()
    ic1=ic.ICoord.from_options(mol=mol)
    ic1.ic_create()
    ic1.bmatp_create()
    ic1.bmatp_to_U()
    ic1.bmat_create()
     
    lot=QChem.from_options(calc_states=[(0,0)],nstates=1,filepath=filepath,basis='6-31gs')
    #lot.cas_from_geom()
    lot.getEnergy()
