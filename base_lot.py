import options
import manage_xyz
import numpy as np

class Base(object):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Base, '_default_options'): return Base._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='E_states',
            value=(0,0),
            required=True,
            allowed_types=[list],
            doc='')

        opt.add_option(
            key='G_states',
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
                value=None,
                required=False,
                allowed_types=[list,None],
                doc='geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)')

        opt.add_option(
                key='filepath',
                required=False,
                allowed_types=[str],
                doc='filepath')

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
        self.E_states=self.options['E_states']
        self.G_states=self.options['G_states']
        self.nstates=self.options['nstates']
        self.geom = self.options['geom']
        self.filepath = self.options['filepath']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']
        self.functional=self.options['functional']

        if self.filepath is not None:
            print "reading geom from %s" % self.filepath
            self.geom=manage_xyz.read_xyz(self.filepath,scale=1)
        if self.geom is not None:
            print "setting coords from geom"
            self.coords = manage_xyz.xyz_to_np(self.geom)

        if self.G_states is None:
            print "assuming G_states same as E_states"
            self.G_states=self.E_states

        # used for taking difference of states
        self.wstates=[]
        for n,i in enumerate(self.E_states):
            for j in self.G_states:
                if i==j:
                    self.wstates.append(n)
        print self.wstates

    def getEnergy(self):
        tmpE = []
        energy =0.
        average_over =0
        # calculate E_states
        for i in self.E_states:
            c_E = self.compute_energy(S=i[0],index=i[1])
            tmpE.append(c_E)
            # average E of G_states:
            # should only only used for crossing optimizations
            if i in self.G_states:
                energy += c_E
                average_over+=1

        # sort and save E states adiabatically
        self.sort_index = np.argsort(tmpE)
        self.E=[]
        for i in self.sort_index:
            self.E.append(tmpE[i])
        
        return energy/average_over

    def compute_energy(self,spin,index):
        raise NotImplementedError()

    def getGrad(self):
        grad = np.zeros((np.shape(self.coords)))

        average_over=0
        tmpGrad = []
        for i in self.G_states:
            tmp = self.compute_gradient(S=i[0],index=i[1])
            grad += tmp
            tmpGrad.append(tmp)
            average_over+=1

        final_grad = grad/average_over

        # sort and save grads adiabatically
        self.grada = []
        for i in self.sort_index:
            self.grada.append(tmpGrad[i])

        return np.reshape(final_grad,(3*len(self.coords),1))

    def compute_gradient(self,S,index):
        raise NotImplementedError()

    def finite_difference(self):
        self.getEnergy() 
        print("Not yet implemented")
        return 0
