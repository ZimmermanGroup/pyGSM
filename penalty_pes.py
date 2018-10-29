import options
from base_lot import * 

class Penalty_PES(object):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Penalty_PES, '_default_options'): return Penalty_PES._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='PES',
            required=True,
            doc='')

        opt.add_option(
                key='sigma',
                required=False,
                value=3.5,
                doc='')

        opt.add_option(
                key='alpha',
                required=False,
                value=0.02*KCAL_MOL_TO_AU,
                doc='')


        Base._default_options = opt
        return Base._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Penalty_PES(Penalty_PES.default_options().set_values(kwargs))

    def __init__(self,
            options,
            ):
        """ Constructor """
        # Cache some useful atributes
        self.options = options
        self.PES = self.options['PES'] 
        self.sigma = self.options['sigma'] 
        self.alpha = self.options['alpha'] 

    def getGrad(self):
        print "hello"
        avg_grad = self.PES.getGrad() 
        avg_grad = avg_grad.reshape((np.shape(self.PES.coords)))
        dgrad = self.PES.grada[1] - self.PES.grada[0]

        # wstates can only be len=2
        dE= self.PES.E[self.PES.wstate[1]] - self.PES.E[self.PES.wstate[0]]

        prefactor = (dE**2. + 2.*self.alpha*dE)/(2*dE + self.alpha)**2.
        grad = avg_grad + self.sigma*prefactor*dgrad

        print grad

    def getEnergy(self):
        self.PES.getEnergy()


if __name__ == '__main__':
    if 1:
        from pytc import *
        filepath="tests/stretched_fluoroethene.xyz"
        nocc=11
        nactive=2
        lot=PyTC.from_options(E_states=[(0,0),(0,1)],filepath=filepath,nocc=nocc,nactive=nactive,basis='6-31gs')
        lot.cas_from_file(filepath)
        p = Penalty_PES.from_options(PES=lot)
        p.getEnergy()
        p.getGrad()
        print p.PES.E
        print p.PES.grada

