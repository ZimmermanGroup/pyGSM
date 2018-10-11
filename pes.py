import options

class Base(object):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Base, '_default_options'): return Base._default_options.copy()
        opt = options.Options() 
        opt.add_option(
            key='wstate',
            value=0,
            required=False,
            allowed_types=[int],
            doc='what state')

        opt.add_option(
            key='wspin',
            value=0,
            required=False,
            allowed_types=[int],
            doc='what spin')
            
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
                key='basis',
                value=0,
                required=False,
                allowed_types=[str],
                doc='Basis set')

        opt.add_option(
                key='filepath',
                value='initial0000.xyz',
                required=False,
                allowed_types=[str],
                doc='path to xyz file')


        Base._default_options = opt
        return Base._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """

        self.options = options
        # Cache some useful atributes
        self.wstate=self.options['wstate']
        self.wspin = self.options['wspin']
        self.filepath = self.options['filepath']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']

    def getEnergy(self):
        raise NotImplementedError()

    def finite_difference(self):
        self.getEnergy() 
        print("Not yet implemented")
        return 0
