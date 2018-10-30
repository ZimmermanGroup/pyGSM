import options
from base_lot import * 

class Penalty_PES(Base):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Penalty_PES(Penalty_PES.default_options().set_values(kwargs))

    def getGrad(self):
        avg_grad = self.PES.getGrad() 
        dgrad = self.PES.grada[1] - self.PES.grada[0]
        dgrad = dgrad.reshape((3*len(self.coords),1))

        prefactor = (self.dE**2. + 2.*self.alpha*self.dE)/((self.dE + self.alpha)**2.)
        #print prefactor
        #print self.PES.grada[0].flatten()
        #print self.PES.grada[1].flatten()
        #print avg_grad.T
        tmp = dgrad*prefactor*self.sigma
        #print prefactor
        print tmp.T
        print avg_grad.T
        grad = avg_grad + tmp
        print grad.T

        return grad

    def getEnergy(self):
        self.PES.coords = self.coords
        E= self.PES.getEnergy()
        self.E  = self.PES.E
        # wstates can only be len=2
        self.dE= self.E[self.wstates[1]] - self.E[self.wstates[0]]
        G = (self.dE**2.)/(self.dE + self.alpha)
        totalE=E+self.sigma*G
        #print "avg E is %1.4f, G is %1.4f,totalE is %1.4f, dE is %1.4f" % (E,G,totalE,self.dE)
        print "dE = %1.4f" % self.dE,
        return totalE


if __name__ == '__main__':
    if 1:
        from pytc import *
        import icoord as ic
        import pybel as pb
        from units import  *
        #filepath1="tests/ethylene.xyz"
        filepath2="tests/twisted_ethene.xyz"
        nocc=7
        nactive=2
        lot=PyTC.from_options(E_states=[(0,0),(0,1)],filepath=filepath2,nocc=nocc,nactive=nactive,basis='6-31gs')
        lot.cas_from_file(filepath2)

        #lot.casci_from_file_from_template(filepath1,filepath2,nocc,nocc)
        p = Penalty_PES(lot.options.copy().set_values({
            "PES":lot,
            }))
        #p.getEnergy()
        #p.getGrad()
        print "alpha is %1.4f kcal/mol" % p.alpha
        mol=pb.readfile("xyz",filepath2).next()
        mol.OBMol.AddBond(6,1,1)
        print "ic1"
        ic1=ic.ICoord.from_options(mol=mol,lot=p,resetopt=False)

        for i in range(1):
            ic1.optimize(25)
            if ic1.gradrms<ic1.OPTTHRESH:
                break
            if  ic1.lot.dE>0.001*KCAL_MOL_PER_AU:
                ic1.lot.sigma *=2.
                print "increasing sigma %1.2f" % ic1.lot.sigma
        print "Finished"

