import options
from pes import * 

class Avg_PES(PES):
    """ Avg potential energy surface calculators """

    #def __init__(self,
    #        PES1,
    #        PES2):
    #    self.PES1 = PES1
    #    self.PES2 = PES2
    #    self.alpha = 0.02*KCAL_MOL_PER_AU
    #    self.sigma = 3.5


    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Avg_PES(Avg_PES.default_options().set_values(kwargs))

    def get_energy(self,geom):
        return 0.5*(self.PES1.get_energy(geom) + self.PES2.get_energy(geom))

    def get_gradient(self,geom):
        return 0.5*(self.PES1.get_gradient(geom) + self.PES2.get_gradient(geom))


