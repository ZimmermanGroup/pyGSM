from __future__ import print_function

# standard library imports
import os
import sys
from io import StringIO

# third party
import numpy as np

# local application imports
from .base_optimizer import base_optimizer
from utilities import manage_xyz, block_matrix


class conjugate_gradient(base_optimizer):
    """Nonlinear conjugate gradient (Fletcher-Reeves) in delocalized internal coordinates.

    Gradient-only: it minimizes (UNCONSTRAINED, ICTAN, SEAM, MECI) and climbs (CLIMB) but cannot
    follow an eigenvector (TS); GSM switches the TS node to eigenvector_follow for the exact TS search.
    Same driver interface and convergence criteria as lbfgs.
    """

    @staticmethod
    def project_constraints(g, molecule):
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T, c[:, np.newaxis])*c[:, np.newaxis]
        return gc

    def restart_term(self, g_prim):
        """Extra term added to every search direction (none for plain CG, see beales_cg)."""
        return 0.

    def optimize(
            self,
            molecule,
            refE=0.,
            opt_type='UNCONSTRAINED',
            opt_steps=3,
            ictan=None,
            xyzframerate=4,
            verbose=False,
            path=None,
    ):
        path = os.getcwd() if path is None else path
        if opt_type in ['TS', 'TS-SEAM']:
            raise NotImplementedError("{} is gradient-only and cannot do {}; use eigenvector_follow".format(type(self).__name__, opt_type))

        print(" initial E %5.4f" % (molecule.energy - refE))
        geoms = [molecule.geometry]
        energies = [molecule.energy-refE]
        self.check_inputs(molecule, opt_type, ictan)
        nconstraints = self.get_nconstraints(opt_type)
        self.buf = StringIO()
        if opt_type == 'SEAM' or opt_type == 'MECI':
            self.opt_cross = True

        # form initial coord basis
        constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
        molecule.update_coordinate_basis(constraints=constraints)

        x = np.copy(molecule.coordinates)
        xyz = np.copy(molecule.xyz)
        num_coords = molecule.num_coordinates - nconstraints - molecule.num_frozen_atoms*3

        fx = molecule.energy
        g = molecule.gradient.copy()
        gc = self.project_constraints(g, molecule)
        g_prim = block_matrix.dot(molecule.coord_basis, gc)
        molecule.gradrms = np.sqrt(np.dot(gc.T, gc)/num_coords).item()

        d_prim = None  # None = (re)start with steepest descent
        gp_prim = None

        for ostep in range(opt_steps):
            print(" On opt step {} ".format(ostep+1))

            # search direction: Fletcher-Reeves, restarted after a rejected step
            d_new = -g_prim + self.restart_term(g_prim)
            if d_prim is None:
                d_prim = d_new
            else:
                beta = (np.dot(g_prim.T, g_prim)/np.dot(gp_prim.T, gp_prim)).item()
                print(" beta = %1.3f" % beta)
                d_prim = d_new + beta*d_prim

            # form in DLC basis (does nothing if cartesian) and normalize
            d = block_matrix.dot(block_matrix.transpose(molecule.coord_basis), d_prim)
            actual_step = np.linalg.norm(d)
            print(" actual_step= %1.5f" % actual_step)
            d = d/actual_step
            step = min(actual_step, self.DMAX)

            # store
            xp = x.copy()
            xyzp = xyz.copy()
            gp = g.copy()
            gp_prim = g_prim.copy()
            fxp = fx

            # => calculate constraint step <= #
            constraint_steps = self.get_constraint_steps(molecule, opt_type, g)

            # line search
            ls = self.Linesearch(nconstraints, x, fx, gc, d, step, xp, constraint_steps, self.linesearch_parameters, molecule, verbose)
            molecule = ls['molecule']
            step = ls['step']
            x = ls['x']
            fx = ls['fx']
            g = ls['g']
            dEstep = fx - fxp
            print(" dEstep=%5.4f" % dEstep)

            if ls['status'] < 0 or (dEstep > 0. and opt_type != 'CLIMB'):
                # back to the previous point, restart CG with a smaller step
                print('[ERROR] the point return to the previous point')
                x = xp.copy()
                xyz = xyzp.copy()
                molecule.xyz = xyzp.copy()
                g = gp.copy()
                fx = fxp
                dEstep = 0.
                d_prim = None
                if self.DMAX <= self.DMIN:
                    print(" Reached minimum step,exiting")
                    break
                self.DMAX = max(step/2., self.DMIN)
            else:
                xyz = molecule.update_xyz(x-xp)
                if step >= self.DMAX:
                    self.DMAX = min(self.DMAX*1.2, self.options['abs_max_step'])

            gc = self.project_constraints(g, molecule)
            g_prim = block_matrix.dot(molecule.coord_basis, gc)
            molecule.gradrms = np.sqrt(np.dot(gc.T, gc)/num_coords).item()
            dE = molecule.difference_energy

            if ostep % xyzframerate == 0:
                geoms.append(molecule.geometry)
                energies.append(molecule.energy-refE)
                manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path, molecule.node_id), geoms, energies, scale=1.)

            if self.options['print_level'] > 0:
                print(" Node: %d Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f" % (molecule.node_id, ostep+1, fx-refE, molecule.gradrms, step, self.DMAX))
            self.buf.write(u' Node: %d Opt step: %d E: %5.4f gradrms: %1.5f ss: %1.3f DMAX: %1.3f\n' % (molecule.node_id, ostep+1, fx-refE, molecule.gradrms, step, self.DMAX))

            # convergence, same criteria as lbfgs
            gmax = float(np.max(np.absolute(gc)))
            disp = float(np.linalg.norm((xyz-xyzp).flatten()))
            print(" gmax %5.4f disp %5.4f dEstep %5.4f gradrms %5.4f\n" % (gmax, disp, dEstep, molecule.gradrms))
            self.converged = (
                molecule.gradrms < self.conv_grms and abs(gmax) < self.conv_gmax
                and abs(dEstep) < self.conv_Ediff and abs(disp) < self.conv_disp
            )
            if self.opt_cross:
                self.converged = self.converged and abs(dE) < self.conv_dE and ls['status'] == 0
            if self.converged and opt_type == 'CLIMB':
                self.converged = abs(np.dot(g.T, molecule.constraints[:, 0]).item()) < self.conv_grms*5

            if self.converged:
                print(" converged")
                if ostep % xyzframerate != 0:
                    geoms.append(molecule.geometry)
                    energies.append(molecule.energy-refE)
                    manage_xyz.write_xyzs_w_comments('{}/opt_{}.xyz'.format(path, molecule.node_id), geoms, energies, scale=1.)
                break

            # update DLC  --> this changes q, g
            if molecule.coord_obj.__class__.__name__ != 'CartesianCoordinates' and opt_type in ['SEAM', 'MECI']:
                constraints = self.get_constraint_vectors(molecule, opt_type, ictan)
                molecule.update_coordinate_basis(constraints=constraints)
                x = np.copy(molecule.coordinates)
                g = molecule.gradient.copy()
                gc = self.project_constraints(g, molecule)
                g_prim = block_matrix.dot(molecule.coord_basis, gc)
                d_prim = None  # new basis: restart
            sys.stdout.flush()

        print(" opt-summary")
        print(self.buf.getvalue())
        return geoms, energies
