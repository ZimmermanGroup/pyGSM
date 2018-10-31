import options
import manage_xyz
import numpy as np
from units import *
import elements 

ELEMENT_TABLE = elements.ElementData()

class Base(object):
    """ Base object for potential energy surface calculators """

    @staticmethod
    def default_options():
        """ Base default options. """

        if hasattr(Base, '_default_options'): return Base._default_options.copy()
        opt = options.Options() 

        opt.add_option(
            key='multiplicity',
            value=None,
            required=False,
            doc="List of Spin Mulitplicities (2S+1)  (e.g., [1, 3, 5] for singlet, triplet, quintet, will be assigned a value of 1 or 2 if None")

        opt.add_option(
            key='states',
            allowed_types=[list],
            doc='list of states 0-indexed')

        opt.add_option(
            key='G_states',
            value=None,
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
                key='coords',
                value=None,
                required=False,
                doc='coords ((natoms,3) np.ndarray) - system geometry (x,y,z)')

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

        opt.add_option(
            key='PES',
            required=False,
            doc='')

        opt.add_option(
                key='nproc',
                required=False,
                value=1,
                )

        opt.add_option(
                key='sigma',
                required=False,
                value=3.5,
                doc='')

        opt.add_option(
                key='alpha',
                required=False,
                value=0.02*KCAL_MOL_PER_AU,
                doc='')

        Base._default_options = opt
        return Base._default_options.copy()

    def __init__(self,
            options,
            ):
        """ Constructor """

        self.options = options
        # Cache some useful atributes
        self.states =self.options['states']
        self.G_states=self.options['G_states']
        self.nstates=self.options['nstates']
        self.geom = self.options['geom']
        self.filepath = self.options['filepath']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']
        self.functional=self.options['functional']
        self.PES = self.options['PES'] 
        self.sigma = self.options['sigma'] 
        self.alpha = self.options['alpha'] 
        self.nproc=self.options['nproc']
        self.coords = self.options['coords']
        self.charge = self.options['charge']
        self.multiplicity=[]

        if self.filepath is not None:
            print "reading geom from %s" % self.filepath
            self.geom=manage_xyz.read_xyz(self.filepath,scale=1)
        if self.geom is not None:
            print "setting coords from geom"
            self.coords = manage_xyz.xyz_to_np(self.geom)
            self.options['coords'] = self.coords

        atoms = manage_xyz.get_atoms(self.geom)
        print atoms

        elements = [ELEMENT_TABLE.from_symbol(atom) for atom in atoms]
        atomic_num = [ele.atomic_num for ele in elements]

        self.n_electrons = sum(atomic_num) - self.charge                                       
        if self.n_electrons < 0:
            raise ValueError("Molecule has fewer than 0 electrons!!!")                         

        multiplicity = self.options['multiplicity']

        for i in multiplicity:
            if multiplicity is None:                                                               
                self.multiplicity.append(1)
                infer_multiplicity = True                                                          
            else:
                self.multiplicity.append(i)
                infer_multiplicity = False   


        if not isinstance(self.charge, int) or not isinstance(multiplicity,list):                   
            raise TypeError("Charge and spin must be integers")                                
     
        # set multiplicty to 2 if needed
        for n,i in enumerate(self.multiplicity):
            if i < 1:
                raise ValueError("Spin multiplicity must be at least 1")       
            if (self.n_electrons + i + 1) % 2:    #true if odd
                if infer_multiplicity:
                    self.multiplicity[n] = 2                                                          
                    print(" self.multiplicity to 2")
                else:
                    raise ValueError("Inconsistent charge and multiplicity.")

        for i in self.multiplicity:
            if i > self.n_electrons + 1:
                raise ValueError("Spin multiplicity too high.")                                    


        if self.G_states is None:
            self.G_states=[]
            print "creating G states [(mult,state)]"
            for m,s in zip(self.multiplicity,self.states):
                self.G_states.append((m,s))

        print self.G_states
        print self.multiplicity
        print self.states

        # used for taking difference of states
        self.wstates=[]
        for n,i in enumerate(self.states):
            for j in self.G_states:
                if i==j[1]:
                    self.wstates.append(n)
        #print "in init"
        #print self.wstates
        self.dE = 1000.


    def getEnergy(self):
        tmpE = []
        energy =0.
        average_over =0
        # calculate E_states
        #for i in self.E_states:
        for m,s in zip(self.multiplicity,self.states):
            c_E = self.compute_energy(m,self.charge,s)
            tmpE.append(c_E)
            # average E of G_states:
            # should only only used for crossing optimizations
            if (m,s) in self.G_states:
                energy += c_E
                average_over+=1

        # sort and save E states adiabatically
        self.sort_index = np.argsort(tmpE)
        self.E=[]
        for i in self.sort_index:
            self.E.append(tmpE[i])
        
        return energy/float(average_over)

    def compute_energy(self,mulitplicity,charge,state):
        raise NotImplementedError()

    def getGrad(self):
        grad = np.zeros((np.shape(self.coords)))
        average_over=0
        tmpGrad = []
        for i in self.G_states:
            tmp = self.compute_gradient(i[0],self.charge,i[1])
            grad += tmp
            tmpGrad.append(tmp)
            average_over+=1

        final_grad = grad/float(average_over)

        # sort and save grads adiabatically
        if len(self.wstates)>1:
            self.grada = []
            for i in self.sort_index:
                self.grada.append(tmpGrad[i])

        return np.reshape(final_grad,(3*len(self.coords),1))

    def compute_gradient(self,multiplicity,charge,state):
        raise NotImplementedError()

    def finite_difference(self):
        self.getEnergy() 
        print("Not yet implemented")
        return 0
