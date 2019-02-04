import options
from pes import * 

class Avg_PES(PES):
    """ Avg potential energy surface calculators """

    #TODO can fix this up so it automatically initializes PES1 and PES2?
    def __init__(self,
            PES1,
            PES2,
            lot,
            ):
        self.options = PES1.options
        #problem!!!! initialize PES1 and PES2
        #self.PES1 = PES1
        #self.PES2 = PES2
        self.PES1 = PES(PES1.options.copy().set_values({
            "lot": lot,
            }))
        self.PES2 = PES(PES2.options.copy().set_values({
            "lot": lot,
            }))
        self.dE = 1000.
        self.lot = lot

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Avg_PES(Avg_PES.default_options().set_values(kwargs))

    def get_energy(self,geom):
        if self.PES1.multiplicity==self.PES2.multiplicity:
            assert self.PES2.ad_idx>self.PES1.ad_idx,"dgrad wrong direction"
        self.dE = self.PES2.get_energy(geom) - self.PES1.get_energy(geom)
        return 0.5*(self.PES1.get_energy(geom) + self.PES2.get_energy(geom))

    def get_gradient(self,geom):
        return 0.5*(self.PES1.get_gradient(geom) + self.PES2.get_gradient(geom))

    def get_coupling(self,geom):
        assert self.PES1.multiplicity==self.PES2.multiplicity,"coupling is 0"
        assert self.PES1.ad_idx!=self.PES2.ad_idx,"coupling is 0"
        return self.lot.get_coupling(geom,self.PES1.multiplicity,self.PES1.ad_idx,self.PES2.ad_idx)

    def get_dgrad(self,geom):
        if self.PES1.multiplicity==self.PES2.multiplicity:
            assert self.PES2.ad_idx>self.PES1.ad_idx,"dgrad wrong direction"
        return (self.PES2.get_gradient(geom) - self.PES1.get_gradient(geom))

