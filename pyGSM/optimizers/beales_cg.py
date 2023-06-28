from __future__ import print_function

# standard library imports
import sys
try:
    from io import StringIO
except:
    from StringIO import StringIO

# third party
import numpy as np

# local application imports
from .base_optimizer import base_optimizer
from utilities import manage_xyz, block_matrix


class beales_cg(base_optimizer):

    def optimize(
            self,
            molecule,
            s0_prim,
            gp_prim,
            refE=0.,
            opt_type="BEALES_CG",
            opt_steps=3,
            ictan=None,
            xyzframerate=4,
            verbose=False,
    ):

        print(" initial E %5.4f" % (molecule.energy - refE))
        sys.stdout.flush()

        # stash/initialize some useful attributes
        geoms = []
        energies = []
        geoms.append(molecule.geometry)
        energies.append(molecule.energy-refE)
        self.initial_step = True
        self.disp = 1000.
        self.Ediff = 1000.
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)
        self.buf = StringIO()

        # for cartesian these are the same
        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)

        # number of coordinates
        n = molecule.num_coordinates

        # Evaluate the function value and its gradient.
        fx = molecule.energy
        g = molecule.gradient

        # if np.all(s0==0.):
        # #if np.all(s0_prim==0.) or s0_prim==None:
        #    print(" already at TS")
        #    nconstraints=1
        #    s0_prim = 0.*gp_prim
        #    g = g - np.dot(g.T,molecule.constraints)*molecule.constraints
        # else:
        #    s0_prim = block_matrix.dot(molecule.coord_basis,s0)

        molecule.gradrms = np.sqrt(np.dot(g.T, g)/n)
        g_prim = block_matrix.dot(molecule.coord_basis, g)

        update_hess = False

        for ostep in range(opt_steps):
            print(" On opt step {} ".format(ostep+1))

            if update_hess and self.options['update_hess_in_bg']:
                if opt_type != 'TS':
                    self.update_Hessian(molecule, 'BFGS')
                else:
                    raise NotImplementedError
            update_hess = True

            if ostep == 0:

                # IF USE BEALES CG
                if nconstraints < 1:
                    dg = g_prim - gp_prim
                    h = dg/np.dot(s0_prim.T, dg)
                else:  # DONT USE BG
                    h = 0.*g_prim
                d_prim = -g_prim + np.dot(h.T, g_prim)*s0_prim   # the search direction
            else:
                dnew = g_prim
                deltanew = np.dot(dnew.T, dnew)
                deltaold = np.dot(gp_prim.T, gp_prim)
                beta = deltanew/deltaold
                d_prim = -g_prim + np.dot(h.T, g_prim)*s0_prim + beta*d_prim

            # form in DLC basis (does nothing if cartesian)
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis), d_prim)

            # normalize the direction
            stepsize = np.linalg.norm(d)
            print(" stepsize = %1.2f" % stepsize)
            d = d/stepsize  # normalize
            if stepsize > self.DMAX:
                stepsize = self.DMAX
                print(" reducing step, new step = %1.2f" % stepsize)

            # store
            xp = x.copy()
            gp = g.copy()
            gp_prim = block_matrix.dot(molecule.coord_basis, gp)
            xyzp = xyz.copy()
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            # line search
            print(" Linesearch")
            sys.stdout.flush()
            ls = self.Linesearch(n, x, fx, g, d, stepsize, xp, constraint_steps, self.linesearch_parameters, molecule, verbose)
            print(" Done linesearch")

            # save new values from linesearch
            molecule = ls['molecule']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']
            step = ls['step']
            if nconstraints > 0:
                g = g - np.dot(g.T, molecule.constraints)*molecule.constraints

            if ls['step'] > self.options['DMAX']:
                if ls['step'] <= self.options['abs_max_step']:     # absolute max
                    print(" Increasing DMAX to {}".format(ls['step']))
                    self.options['DMAX'] = ls['step']
                else:
                    self.options['DMAX'] = self.options['abs_max_step']
            elif ls['step'] < self.options['DMAX']:
                if ls['step'] >= self.DMIN:     # absolute min
                    print(" Decreasing DMAX to {}".format(ls['step']))
                    self.options['DMAX'] = ls['step']
                elif ls['step'] <= self.DMIN:
                    self.options['DMAX'] = self.DMIN
                    print(" Decreasing DMAX to {}".format(self.DMIN))

            # dE
            dEstep = fx - fxp
            print(" dEstep=%5.4f" % dEstep)

            # revert to the previous point
            if ls['status'] < 0 or dEstep > 0.:
                x = xp.copy()
                molecule.xyz = xyzp
                g = gp.copy()
                fx = fxp
                print('[ERROR] the point return to the previous point')
                if nconstraints == 1:
                    print(" opt-summary")
                    print(self.buf.getvalue())
                    return geoms, energies
                else:
                    nconstraints = 1
                    constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    x = np.copy(molecule.coordinates)
                    g = molecule.gradient.copy()
                    # project out the constraint
                    for c in molecule.constraints.T:
                        g -= np.dot(g.T, c[:, np.newaxis])*c[:, np.newaxis]
                    step = 0.
                    g_prim = block_matrix.dot(molecule.coord_basis, g)
                    h = 0.*g_prim
            else:
                # update molecule xyz
                xyz = molecule.update_xyz(x-xp)
                g_prim = block_matrix.dot(molecule.coord_basis, g)

            if ostep % xyzframerate == 0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy-refE)
                manage_xyz.write_xyzs_w_comments('opt_{}.xyz'.format(molecule.node_id), geoms, energies, scale=1.)

            # save variables for update Hessian!
            if not molecule.coord_obj.__class__.__name__ == 'CartesianCoordinates' or self.options['update_hess_in_bg']:
                self.dx_prim = molecule.coord_obj.Prims.calcDiff(xyz, xyzp)
                self.dx_prim = np.reshape(self.dx_prim, (-1, 1))
                self.dg_prim = g_prim - gp_prim
            else:
                raise NotImplementedError(" ef not implemented for CART")

            if self.options['print_level'] > 0:
                print(" Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (ostep+1, fx-refE, molecule.gradrms, step, self.options['DMAX']))
            self.buf.write(u' Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (ostep+1, fx-refE, molecule.gradrms, step, self.options['DMAX']))

            # gmax = np.max(g)/ANGSTROM_TO_AU/units.KCAL_MOL_PER_AU
            # print "current gradrms= %r au" % gradrms
            # gmax = np.max(g)/units.ANGSTROM_TO_AU
            # self.disp = np.max(x - xp)/units.ANGSTROM_TO_AU
            # self.Ediff = fx -fxp / units.KCAL_MOL_PER_AU
            # print(" maximum displacement component %1.2f (au)" % self.disp)
            # print(" maximum gradient component %1.2f (au)" % gmax)

            # check for convergence TODO
            molecule.gradrms = np.sqrt(np.dot(g.T, g)/n)
            gmax = float(np.max(g))
            disp = float(np.linalg.norm((xyz-xyzp).flatten()))
            print(" gmax %5.4f disp %5.4f Ediff %5.4f gradrms %5.4f\n" % (gmax, disp, dEstep, molecule.gradrms))

            self.converged = False
            if self.opt_cross and abs(dE) < self.conv_dE and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp and ls['status'] == 0:
                self.converged = True
            elif not self.opt_cross and molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp:
                self.converged = True

            if self.converged:
                print(" converged")
                if ostep % xyzframerate != 0:
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy-refE)
                    manage_xyz.write_xyzs_w_comments('opt_{}.xyz'.format(molecule.node_id), geoms, energies, scale=1.)
                break

            #update DLC  --> this changes q, g, Hint
            if not molecule.coord_obj.__class__.__name__ == 'CartesianCoordinates':
                if opt_type == 'SEAM' or opt_type == "MECI":
                    print(" updating DLC")
                    sys.stdout.flush()
                    #if opt_type=="ICTAN":
                    constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                    molecule.update_coordinate_basis(constraints=constraints)
                    #else:
                    #    molecule.update_coordinate_basis()
                    x = np.copy(molecule.coordinates)
                    fx = molecule.energy
                    dE = molecule.difference_energy
                    if dE != 1000.:
                        print(" difference energy is %5.4f" % dE)
                    g = molecule.gradient.copy()
                    if nconstraints > 0:
                        g = g - np.dot(g.T, molecule.constraints)*molecule.constraints
                    g_prim = block_matrix.dot(molecule.coord_basis, g)
                    print(" Done update")
            sys.stdout.flush()

        print(" opt-summary")
        print(self.buf.getvalue())
        return geoms, energies
