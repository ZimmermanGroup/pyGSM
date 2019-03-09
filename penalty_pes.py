import options
from pes import * 
import sys

class Penalty_PES(PES):
    """ penalty potential energy surface calculators """

    def __init__(self,
            PES1,
            PES2,
            lot,
            sigma=1.0,
            alpha=0.02*KCAL_MOL_PER_AU,
            ):
        #self.PES1 = PES1
        #self.PES2 = PES2
        self.PES1 = PES(PES1.options.copy().set_values({
            "lot": lot,
            }))
        self.PES2 = PES(PES2.options.copy().set_values({
            "lot": lot,
            }))
        self.lot = lot
        self.alpha = alpha
        self.dE = 1000.
        self.sigma = sigma
        print ' PES1 multiplicity: {} PES2 multiplicity: {}'.format(self.PES1.multiplicity,self.PES2.multiplicity)


    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Penalty_PES(Penalty_PES.default_options().set_values(kwargs))

    @classmethod
    def create_pes_from(cls,PES,lot):
        return cls(PES.PES1,PES.PES2,lot)

    def get_energy(self,geom):
        E1 = self.PES1.get_energy(geom)
        E2 = self.PES2.get_energy(geom)
        #avgE = 0.5*(self.PES1.get_energy(geom) + self.PES2.get_energy(geom))
        avgE = 0.5*(E1+E2)
        #self.dE = self.PES2.get_energy(geom) - self.PES1.get_energy(geom)
        self.dE = E2-E1
        #print "E1: %1.4f E2: %1.4f"%(E1,E2),
        #print "delta E = %1.4f" %self.dE,
        #TODO what to do if PES2 is or goes lower than PES1?
        G = (self.dE*self.dE)/(abs(self.dE) + self.alpha)
        #if self.dE < 0:
        #    G*=-1
        #print "G = %1.4f" % G
        #print "alpha: %1.4f sigma: %1.4f"%(self.alpha,self.sigma),
        #print "F: %1.4f"%(avgE+self.sigma*G)
        sys.stdout.flush()
        return avgE+self.sigma*G

    def get_gradient(self,geom):
        self.grad1 = self.PES1.get_gradient(geom)
        self.grad2 = self.PES2.get_gradient(geom)
        avg_grad = 0.5*(self.grad1 + self.grad2)
        dgrad = self.grad2 - self.grad1
        if self.dE < 0:
            dgrad *= -1
        factor = self.sigma*((self.dE*self.dE) + 2.*self.alpha*abs(self.dE))/((abs(self.dE) + self.alpha)**2)
        #print "factor is %1.4f" % factor
        return avg_grad + factor*dgrad


