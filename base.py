import options

class Base(object):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Base, '_default_options'): return Base._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='calc_states',
            value=(0,0),
            required=True,
            allowed_types=[list],
            doc='')

        opt.add_option(
                key='functional',
                required=False,
                allowed_types=[str],
                doc='density functional')

        opt.add_option(
                key='nocc',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of occupied orbitals (for CAS)')

        opt.add_option(
                key='nactive',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of active orbitals (for CAS)')

        opt.add_option(
                key='nstates',
                value=1,
                required=False,
                allowed_types=[int],
                doc='Number of states')

        opt.add_option(
                key='basis',
                value=0,
                required=False,
                allowed_types=[str],
                doc='Basis set')

        opt.add_option(
                key='geom',
                required=True,
                allowed_types=[list],
                doc='geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)')

        opt.add_option(
                key='charge',
                value=0,
                required=False,
                allowed_types=[int],
                doc='charge of molecule')
                

        Base._default_options = opt
        return Base._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """

        self.options = options
        # Cache some useful atributes
        self.calc_states=self.options['calc_states']
        self.nstates=self.options['nstates']
        self.geom = self.options['geom']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']
        self.functional=self.options['functional']

    def getEnergy(self):
        raise NotImplementedError()

    def getGrad(self):
        raise NotImplementedError()

    def finite_difference(self):
        self.getEnergy() 
        print("Not yet implemented")
        return 0
