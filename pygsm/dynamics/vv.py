#import lightspeed as ls
import numpy as np

class VV(object):

    """ class VV implements Velocity Verlet.

    VV is used to implement dynamics in a model where the potential/force
    evaluation (referred to herein as 'compute_potential') and the time propagation
    are on the same level, and are called in an external loop.

    An example usage of VV dynamics is:

        # => Initialization <= #
    
        tmax # Pre-determined total simulation time
        dt   # Pre-determined timestep
        M    # Pre-determined masses
        X0   # Pre-determined initial positions
        P0   # Pre-determined initial momenta
        V0, F0 = compute_potential(X0)  # User-provided compute_potential
        prop = VV(dt,M,X0,P0,F0,V0)  # VV object initialization

        # => Dynamics Loop <= #
    
        while prop.t < tmax

            # Compute potential/force at prop.Xnew
            V, F = prop.compute_potential(prop.Xnew)
            # Advance VV time
            prop.step(V,F)

    """

    def __init__(
        self,
        dt, # timestep 
        M,  # masses
        X0, # initial positions
        P0, # initial momenta
        F0, # initial forces
        V0, # initial potential
        ):

        """ Initialize the VV object.

        Params:
            dt (float) - timestep
            M (Tensor) - masses
            X0 (Tensor) - initial positions 
            P0 (Tensor) - initial momenta
            F0 (Tensor) - initial forces 
            V0 (float) - initial potential energy

        Object Attributes:
            dt (float) - timestep
            M (Tensor) - masses
            I (int) - current step
            X (Tensor) - current position
            P (Tensor) - current momenta
            F (Tensor) - current force
            V (float) - current potential energy
            T (float) - current kinetic energy
            E (float) - current total energy
            t (float) - current time
            Xnew (Tensor) - proposed new position - force and potential should
                be computed here and passed to step to move forward in time.

        The Tensor objects in VV can be of arbitrary shape (e.g., arbitrary
            problem dimensionality), but must all be the same shape.

        """

        # Global parameters
        self.dt = dt # Timestep
        self.M = M   # Mass
        
        # Current frame
        self.I = 0 # Frame counter
        self.X = X0
        self.P = P0
        self.F = F0
        self.V = V0

        # Positions for next force
        #self.Xnew = ls.Tensor.array(self.X[...] + (self.P[...] / self.M[...]) * self.dt + 0.5 * (self.F[...] / self.M[...]) * self.dt**2)
        self.Xnew = self.X + self.P / self.M * self.dt + 0.5 * self.F / self.M * self.dt**2 
     
    @property
    def T(self):
        """ The kinetic energy """
        return 0.5 * np.sum(self.P[...]**2 / self.M[...])

    @property
    def E(self):
        """ The total energy """
        return self.T + self.V
    
    @property
    def t(self):
        """ The elapsed simulation time, I * dt """
        return self.I * self.dt

    def step(
        self,
        F, # Force computed at self.Xnew
        V, # Potential energy computed at self.Xnew
        ):

        """ Step forward in time using the VV propagator.

        Params:
            F (Tensor) - the force computed at self.Xnew
            V (float) - the potential energy computed at self.Xnew
        
        The function changes the internal state of the VV object to advance one frame in time.

        """
        
        # Update current frame
        self.X = self.Xnew
        self.P = self.P + 0.5 * (F + self.F) *self.dt
        #self.P = ls.Tensor.array(self.P[...] + 0.5 * (F[...] + self.F[...]) * self.dt)  # VV
        #self.P = ls.Tensor.array(self.P[...] + self.F[...] * self.dt)  # 1st-Order Euler
        self.F = F
        self.V = V
        self.I += 1

        # Positions for next force
        self.Xnew = self.X + (self.P/self.M) * self.dt + 0.5 * (self.F/self.M)*self.dt**2
        #self.Xnew = ls.Tensor.array(self.X[...] + (self.P[...] / self.M[...]) * self.dt + 0.5 * (self.F[...] / self.M[...]) * self.dt**2)

# => Test Utility <= #

def test_ho(
    ):

    """ Example use of VV to propagate a 1D harmonic oscillator """

    # Simulation conditions
    M = ls.Tensor.array([1.0])
    k = 1.0
    dt = 0.1
    tmax = 100.0
    P0 = ls.Tensor.array([1.0])
    X0 = ls.Tensor.array([0.0])

    # Initial potential/force
    V0 = 0.5 * k * np.sum(X0[...]**2)
    F0 = ls.Tensor.array(-k * X0[...])

    # VV propagator
    prop = VV(dt,M,X0,P0,F0,V0)

    # Energy statistics over time
    stats = {
        't' : [],
        'T' : [],
        'V' : [],
        'E' : [],
    }

    print 'NVE: %5s %14s %24s %24s %24s' % (
        "I",
        "t", 
        "T",
        "V",
        "E",
        )

    # => Simulation Time Loop <= #

    while prop.t < tmax:

        # Iteration trace
        print 'NVE: %5d %14.6f %24.16E %24.16E %24.16E' % (
            prop.I,
            prop.t,
            prop.T,
            prop.V,
            prop.E,
            )
        # Statistics save
        stats['t'].append(prop.t)
        stats['T'].append(prop.T)
        stats['V'].append(prop.V)
        stats['E'].append(prop.E)
        
        # Potential/Force evaluation at prop.Xnew
        V = 0.5 * k * np.sum(prop.Xnew[...]**2)
        F = ls.Tensor.array(-k * prop.Xnew[...])

        # VV propagation
        prop.step(F,V)

    # => Energy Conservation Plot <= #

    stats = { key : np.array(val) for key, val in stats.iteritems() }

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.clf()
    plt.plot(stats['t'],stats['T']-stats['T'][0],label='T')
    plt.plot(stats['t'],stats['V']-stats['V'][0],label='V')
    plt.plot(stats['t'],stats['E']-stats['E'][0],label='E')
    plt.xlabel('t')
    plt.ylabel('E')
    plt.legend()
    plt.savefig('ho.pdf')

if __name__ == '__main__':

    test_ho()
