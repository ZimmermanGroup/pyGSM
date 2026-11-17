from __future__ import print_function

# third party
import numpy as np

# local application imports
from .conjugate_gradient import conjugate_gradient


class beales_cg(conjugate_gradient):
    """Beale's restarted conjugate gradient.

    Like conjugate_gradient, plus Beale's restart term (h.g) s0 in every search direction, where
    s0_prim is the restart direction and gp_prim the gradient there (primitive basis), and
    h = (g - gp)/(s0.(g - gp)) is fixed at the first step. Without s0_prim, or with constraints,
    there is no restart term and this is plain conjugate_gradient.
    """

    def optimize(self, molecule, s0_prim=None, gp_prim=None, **kwargs):
        constrained = self.get_nconstraints(kwargs.get('opt_type', 'UNCONSTRAINED')) > 0
        self.s0_prim = None if constrained else s0_prim
        self.restart_gp_prim = gp_prim
        self.h = None
        return super(beales_cg, self).optimize(molecule, **kwargs)

    def restart_term(self, g_prim):
        if self.s0_prim is None:
            return 0.
        if self.h is None:
            dg = g_prim - self.restart_gp_prim
            self.h = dg/np.dot(self.s0_prim.T, dg).item()
        return np.dot(self.h.T, g_prim).item()*self.s0_prim
