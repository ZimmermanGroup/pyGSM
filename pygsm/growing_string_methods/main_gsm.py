from __future__ import print_function
import numpy as np
import sys
import os
from os import path

# local application imports
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

try:
    from .gsm import GSM
except:
    from gsm import GSM

from wrappers.molecule import Molecule
from utilities.nifty import printcool
from utilities.manage_xyz import xyz_to_np
from utilities import block_matrix
from coordinate_systems import rotate
from optimizers import eigenvector_follow
import multiprocessing as mp
from itertools import chain
from copy import deepcopy

def worker(arg):
    obj, methname = arg[:2]
    return getattr(obj, methname)(*arg[2:])


#######################################################################################
############### This class contains the main GSM functions  ###########################
#######################################################################################


class MainGSM(GSM):
    def grow_string(self, max_iters=30, max_opt_steps=3, nconstraints=1):
        '''
        Grow the string

        Parameters
        ----------
        max_iter : int
             Maximum number of GSM iterations
        nconstraints : int
        optsteps : int
            Maximum number of optimization steps per node of string
        '''
        printcool("In growth_iters")

        ncurrent, nlist = self.make_difference_node_list()
        self.ictan, self.dqmaga = self.get_tangents_growing()
        self.refresh_coordinates()
        self.set_active(self.nR-1, self.nnodes-self.nP)

        isGrown = False
        iteration = 0
        while not isGrown:
            if iteration > max_iters:
                print(" Ran out of iterations")
                return
                # raise Exception(" Ran out of iterations")
            printcool("Starting growth iteration %i" % iteration)
            self.optimize_iteration(max_opt_steps)
            totalgrad, gradrms, sum_gradrms = self.calc_optimization_metrics(self.nodes)
            self.xyz_writer('scratch/growth_iters_{:03}_{:03}.xyz'.format(self.ID, iteration), self.geometries, self.energies, self.gradrmss, self.dEs)
            print(" gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n".format(iteration, float(totalgrad), float(gradrms), float(self.emax)))

            try:
                self.grow_nodes()
            except Exception as error:
                print("can't add anymore nodes, bdist too small")

                if self.__class__.__name__ == "SE_GSM":  # or self.__class__.__name__=="SE_Cross":
                    # Don't do SE_cross because that already does optimization later
                    if self.nodes[self.nR-1].PES.lot.do_coupling:
                        opt_type = 'MECI'
                    else:
                       opt_type = 'UNCONSTRAINED'
                    print(" optimizing last node")
                    self.optimizer[self.nR-1].conv_grms = self.CONV_TOL
                    print(self.optimizer[self.nR-1].conv_grms)
                    path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.nR-1))
                    self.optimizer[self.nR-1].optimize(
                        molecule=self.nodes[self.nR-1],
                        refE=self.nodes[0].V0,
                        opt_steps=50,
                        opt_type=opt_type,
                        path=path,
                    )
                elif self.__class__.__name__ == "SE_Cross":
                    print(" Will do extra optimization of this node in SE-Cross")
                else:
                    raise RuntimeError
                break

            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.ic_reparam_g()
            self.ictan, self.dqmaga = self.get_tangents_growing()
            self.refresh_coordinates()

            iteration += 1
            isGrown = self.check_if_grown()

        # create newic object
        print(" creating newic molecule--used for ic_reparam")
        self.newic = Molecule.copy_from_options(self.nodes[0])

        # TODO should something be done for growthdirection 2?
        if self.growth_direction == 1:
            print("Setting LOT of last node")
            self.nodes[-1] = Molecule.copy_from_options(
                MoleculeA=self.nodes[-2],
                xyz=self.nodes[-1].xyz,
                new_node_id=self.nnodes-1
            )
        return

    def optimize_string(self, max_iter=30, nconstraints=1, opt_steps=1, rtype=2):
        '''
        Optimize the grown string until convergence

        Parameters
        ----------
        max_iter : int
             Maximum number of GSM iterations 
        nconstraints : int
        optsteps : int
            Maximum number of optimization steps per node of string
        rtype : int
            An option to change how GSM optimizes  
            TODO change this s***
            0 is no-climb
            1 is climber
            2 is finder
        '''
        printcool("In opt_iters")

        self.nclimb = 0
        self.nhessreset = 10  # are these used??? TODO
        self.hessrcount = 0   # are these used?!  TODO
        self.newclimbscale = 2.
        self.set_finder(rtype)

        self.isConverged = False
        oi = 0

        # enter loop
        while not self.isConverged:
            printcool("Starting opt iter %i" % oi)
            if self.climb and not self.find:
                print(" CLIMBING")
            elif self.find:
                print(" TS SEARCHING")

            # stash previous TSnode
            self.pTSnode = self.TSnode
            self.emaxp = self.emax

            # store reparam energies
            print(" V_profile (beginning of iteration): ", end=' ')
            self.print_energies()

            # => Get all tangents 3-way <= #
            self.get_tangents_opting()
            self.refresh_coordinates()

            # => do opt steps <= #
            self.set_node_convergence()
            self.optimize_iteration(opt_steps)

            print(" V_profile: ", end=' ')
            self.print_energies()

            # TODO resetting
            # TODO special SSM criteria if first opt'd node is too high?
            if self.TSnode == self.nnodes-2 and (self.climb or self.find):
                printcool("WARNING\n: TS node shouldn't be second to last node for tangent reasons")
                self.add_node_after_TS()
                added = True
            elif self.TSnode == 1 and (self.climb or self.find):
                printcool("WARNING\n: TS node shouldn't be first  node for tangent reasons")
                self.add_node_before_TS()
                added = True
            else:
                added = False

            # => find peaks <= #
            fp = self.find_peaks('opting')

            ts_cgradq = 0.
            if not self.find:
                ts_cgradq = np.linalg.norm(np.dot(self.nodes[self.TSnode].gradient.T, self.nodes[self.TSnode].constraints[:, 0])*self.nodes[self.TSnode].constraints[:, 0])
                print(" ts_cgradq %5.4f" % ts_cgradq)

            ts_gradrms = self.nodes[self.TSnode].gradrms
            self.dE_iter = abs(self.emax-self.emaxp)
            print(" dE_iter ={:2.2f}".format(self.dE_iter))

            # => calculate totalgrad <= #
            totalgrad, gradrms, sum_gradrms = self.calc_optimization_metrics(self.nodes)

            # Check if allup or alldown
            energies = np.array(self.energies)
            if (np.all(energies[1:]+0.5 >= energies[:-1]) or np.all(energies[1:]-0.5 <= energies[:-1])) and (self.climber or self.finder):
                printcool(" There is no TS, turning off TS search")
                rtype = 0
                self.climber = self.finder = self.find = self.climb = False
                self.CONV_TOL = self.options['CONV_TOL']*5

            # if self.has_intermediate(5) and rtype>0 and (self.climb or self.find):
            #    printcool(" THERE IS AN INTERMEDIATE, OPTIMIZE THE INTERMEDIATE AND TRY AGAIN")
            #    self.endearly=True
            #    isConverged=True
            #    self.tscontinue=False

            # => Check Convergence <= #
            self.isConverged = self.is_converged(totalgrad, fp, rtype, ts_cgradq)

            # => set stage <= #
            stage_changed = self.set_stage(totalgrad, sum_gradrms, ts_cgradq, ts_gradrms, fp)

            if not stage_changed:
                # Decrement stuff that controls stage
                if self.climb:
                    self.nclimb -= 1
                self.nhessreset -= 1
                if self.nopt_intermediate > 0:
                    self.nopt_intermediate -= 1

                if self.pTSnode != self.TSnode and self.climb:
                    print("TS node changed after opting")
                    self.climb = False
                    #self.slow_down_climb()
                    self.pTSnode = self.TSnode

                # opt decided Hess is not good because of overlap
                if self.find and (not self.optimizer[self.TSnode].maxol_good or added):
                    self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                    self.modify_TS_Hess()
                elif self.find and (self.optimizer[self.TSnode].nneg > 3 or self.optimizer[self.TSnode].nneg == 0 or self.hess_counter > 10 or np.abs(self.TS_E_0 - self.emax) > 10.) and not self.optimizer[self.TSnode].converged:

                    # Reform the guess primitive Hessian
                    self.nodes[self.TSnode].form_Primitive_Hessian()
                    if self.hessrcount < 1 and self.pTSnode == self.TSnode:
                        print(" resetting TS node coords Ut (and Hessian)")
                        self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                        self.modify_TS_Hess()
                        self.nhessreset = 10
                        self.hessrcount = 1
                    else:
                        print(" Hessian consistently bad, going back to climb (for 3 iterations)")
                        self.find = False
                        self.nclimb = 2
                elif self.find and self.optimizer[self.TSnode].nneg <= 3:
                    self.hessrcount -= 1
                    self.hess_counter += 1

            # => write Convergence to file <= #
            filename = 'scratch/opt_iters_{:03}_{:03}.xyz'.format(self.ID, oi)
            self.xyz_writer(filename, self.geometries, self.energies, self.gradrmss, self.dEs)

            print(" End early counter {}".format(self.endearly_counter))

            # TODO prints tgrads and jobGradCount
            print("opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}\n".format(oi, float(totalgrad), float(gradrms), self.TSnode, float(self.emax)))
            oi += 1

            # => Reparam the String <= #
            if oi < max_iter and not self.isConverged:
                self.reparameterize(nconstraints=nconstraints)
                self.get_tangents_opting()
                self.refresh_coordinates()
                if self.pTSnode != self.TSnode and self.climb:
                    print("TS node changed after reparameterizing")
                    self.slow_down_climb()
            elif oi >= max_iter and not self.isConverged:
                self.ran_out = True
                print(" Ran out of iterations")
                return
                # raise Exception(" Ran out of iterations")

        # TODO Optimize TS node to a finer convergence
        # if rtype==2:
        return

    def refresh_coordinates(self, update_TS=False):
        '''
        Refresh the DLC coordinates for the string
        '''

        if not self.done_growing:
            # TODO

            if self.mp_cores == 1:
                for n in range(1, self.nnodes-1):
                    if self.nodes[n] is not None:
                        Vecs = self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n])
                        self.nodes[n].coord_basis = Vecs

            else:
                pool = mp.Pool(self.mp_cores)
                Vecs = pool.map(worker, ((self.newic.coord_obj, "build_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.nnodes-1) if self.nodes[n] is not None))
                pool.close()
                pool.join()

                i = 0
                for n in range(1, self.nnodes-1):
                    if self.nodes[n] is not None:
                        self.nodes[n].coord_basis = Vecs[i]
                        i += 1
        else:
            if self.find or self.climb:
                TSnode = self.TSnode
                if self.mp_cores == 1:
                    for n in range(1, self.nnodes-1):
                        # don't update tsnode coord basis
                        if n != TSnode or (n == TSnode and update_TS):
                            Vecs = self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n])
                            self.nodes[n].coord_basis = Vecs
                else:
                    pool = mp.Pool(self.mp_cores)
                    Vecs = pool.map(worker, ((self.newic.coord_obj, "build_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.nnodes-1) if n != TSnode))
                    pool.close()
                    pool.join()
                    for i, n in enumerate(chain(range(1, TSnode), range(TSnode+1, self.nnodes-1))):
                        self.nodes[n].coord_basis = Vecs[i]

                    if update_TS:
                        Vec = self.newic.coord_obj.build_dlc(self.nodes[TSnode].xyz, self.ictan[TSnode])
                        self.nodes[TSnode].coord_basis = Vec

            else:
                if self.mp_cores == 1:
                    Vecs = []
                    for n in range(1, self.nnodes-1):
                        Vecs.append(self.newic.coord_obj.build_dlc(self.nodes[n].xyz, self.ictan[n]))
                elif self.mp_cores > 1:
                    pool = mp.Pool(self.mp_cores)
                    Vecs = pool.map(worker, ((self.newic.coord_obj, "build_dlc", self.nodes[n].xyz, self.ictan[n]) for n in range(1, self.nnodes-1)))
                    pool.close()
                    pool.join()
                for n, node in enumerate(self.nodes[1:self.nnodes-1]):
                    node.coord_basis = Vecs[n]

    def optimize_iteration(self, opt_steps):
        '''
        Optimize string iteration
        '''

        refE = self.nodes[0].energy

        for n in range(self.nnodes):
            if self.nodes[n] and self.active[n]:
                print()
                path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, n))
                printcool("Optimizing node {}".format(n))
                opt_type = self.set_opt_type(n)
                osteps = self.mult_steps(n, opt_steps)
                self.optimizer[n].optimize(
                    molecule=self.nodes[n],
                    refE=refE,
                    opt_type=opt_type,
                    opt_steps=osteps,
                    ictan=self.ictan[n],
                    xyzframerate=1,
                    path=path,
                )

        if self.__class__.__name__ == "SE-GSM" and self.done_growing:
            fp = self.find_peaks('opting')
            if self.energies[self.nnodes-1] > self.energies[self.nnodes-2] and fp > 0 and self.nodes[self.nnodes-1].gradrms > self.CONV_TOL:
                printcool('Last node is not a minimum, Might need to verify that the last node is a minimum')
                path = os.path.join(os.getcwd(), 'scratch/{:03d}/{}'.format(self.ID, self.nnodes-1))
                self.optimizer[self.nnodes-1].optimize(
                    molecule=self.nodes[self.nnodes-1],
                    refE=refE,
                    opt_type='UNCONSTRAINED',
                    opt_steps=osteps,
                    ictan=None,
                    path=path
                )

    def get_tangents_opting(self, print_level=1):
        if self.climb or self.find:
            self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
        else:
            self.ictan, self.dqmaga = self.get_tangents(self.nodes)

    def get_tangents_growing(self, print_level=1):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase.
        Also updates coordinates
        Not a static method beause no one should ever call this outside of GSM
        """

        ncurrent, nlist = self.make_difference_node_list()
        dqmaga = [0.]*self.nnodes
        ictan = [[]]*self.nnodes

        if self.print_level > 1:
            print("ncurrent, nlist")
            print(ncurrent)
            print(nlist)

        for n in range(ncurrent):
            # ictan0,_ = self.get_tangent(
            #        node1=self.nodes[nlist[2*n]],
            #        node2=self.nodes[nlist[2*n+1]],
            #        driving_coords=self.driving_coords,
            #        )

            if self.__class__.__name__ == "DE_GSM":  # or self.__class__.__name__=="SE_Cross":
                print(" getting tangent [%i ]from between %i %i pointing towards %i" % (nlist[2*n], nlist[2*n], nlist[2*n+1], nlist[2*n]))
                ictan0 = self.get_tangent_xyz(self.nodes[nlist[2*n]].xyz,
                                              self.nodes[nlist[2*n+1]].xyz,
                                              self.nodes[0].primitive_internal_coordinates)
            else:
                ictan0, _ = self.get_tangent(
                    node1=self.nodes[nlist[2*n]],
                    node2=self.nodes[nlist[2*n+1]],
                    driving_coords=self.driving_coords,
                )

            if self.print_level > 1:
                print("forming space for", nlist[2*n+1])
            if self.print_level > 1:
                print("forming tangent for ", nlist[2*n])

            if (ictan0[:] == 0.).all():
                print(" ICTAN IS ZERO!")
                print(nlist[2*n])
                print(nlist[2*n+1])
                raise RuntimeError

            # normalize ictan
            norm = np.linalg.norm(ictan0)
            ictan[nlist[2*n]] = ictan0/norm

            # NOTE regular GSM does something weird here
            # Vecs = self.nodes[nlist[2*n]].update_coordinate_basis(constraints=self.ictan[nlist[2*n]])
            # constraint = self.nodes[nlist[2*n]].constraints
            # prim_constraint = block_matrix.dot(Vecs,constraint)
            # but this is not followed here anymore 7/1/2020
            # dqmaga[nlist[2*n]] = np.dot(prim_constraint.T,ictan0)
            # dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))
            # tmp_dqmaga = np.dot(prim_constraint.T,ictan0)
            # tmp_dqmaga = np.sqrt(tmp_dqmaga)

            dqmaga[nlist[2*n]] = norm

        if print_level > 0:
            print('------------printing dqmaga---------------')
            for n in range(self.nnodes):
                print(" {:5.3}".format(dqmaga[n]), end=' ')
                if (n+1) % 5 == 0:
                    print()
            print()

        if print_level > 1:
            for n in range(ncurrent):
                print("dqmag[%i] =%1.2f" % (nlist[2*n], self.dqmaga[nlist[2*n]]))
                print("printing ictan[%i]" % nlist[2*n])
                print(self.ictan[nlist[2*n]].T)
        for i, tan in enumerate(ictan):
            if np.all(tan == 0.0):
                print("tan %i of the tangents is 0" % i)
                raise RuntimeError

        return ictan, dqmaga

    # Refactor this code!
    # TODO remove return form_TS hess  3/2021
    def set_stage(self, totalgrad, sumgradrms, ts_cgradq, ts_gradrms, fp):

        # checking sum gradrms is not good because if one node is converged a lot while others a re not this is bad
        all_converged = all([self.nodes[n].gradrms < self.optimizer[n].conv_grms*1.1 for n in range(1, self.nnodes-1)])
        all_converged_climb = all([self.nodes[n].gradrms < self.optimizer[n].conv_grms*2.5 for n in range(1, self.nnodes-1)])
        stage_changed = False

        # TODO totalgrad is not a good criteria for large systems
        # if fp>0 and (((totalgrad < 0.3 or ts_cgradq < 0.01) and self.dE_iter < 2.) or all_converged) and self.nopt_intermediate<1: # extra criterion in og-gsm for added

        if fp > 0 and all_converged_climb and self.dE_iter < 2.:  # and self.nopt_intermediate<1:
            if not self.climb and self.climber:
                print(" ** starting climb **")
                self.climb = True
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" % (totalgrad, ts_gradrms, ts_cgradq))
                # overwrite this here just in case TSnode changed wont cause slow down climb
                self.pTSnode = self.TSnode
                stage_changed = True

            # TODO deserves to be rethought 3/2021
            elif (self.climb and not self.find and self.finder and self.nclimb < 1 and
                    ((totalgrad < 0.2 and ts_gradrms < self.CONV_TOL*10. and ts_cgradq < 0.01) or  # I hate totalgrad
                     (totalgrad < 0.1 and ts_gradrms < self.CONV_TOL*10. and ts_cgradq < 0.02) or  #
                     (all_converged) or
                     (ts_gradrms < self.CONV_TOL*2.5 and ts_cgradq < 0.01)  # used to be 5
                     )) and self.dE_iter < 1.:
                print(" ** starting exact climb **")
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" % (totalgrad, ts_gradrms, ts_cgradq))
                self.find = True

                # Modify TS Hessian
                self.ictan, self.dqmaga = self.get_three_way_tangents(self.nodes, self.energies)
                self.modify_TS_Hess()

                if self.optimizer[self.TSnode].options['DMAX'] > 0.1:
                    self.optimizer[self.TSnode].options['DMAX'] = 0.1
                self.optimizer[self.TSnode] = eigenvector_follow(self.optimizer[self.TSnode].options.copy())
                self.optimizer[self.TSnode].options['SCALEQN'] = 1.
                self.nhessreset = 10  # are these used??? TODO
                self.hessrcount = 0   # are these used?!  TODO
                stage_changed = True

        return stage_changed

    def add_GSM_nodeR(self, newnodes=1):
        '''
        Add a node between endpoints on the reactant side, should only be called inside GSM
        '''
        printcool("Adding reactant node")

        if self.current_nnodes+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            iR = self.nR-1
            iP = self.nnodes-self.nP
            iN = self.nR
            print(" adding node: %i between %i %i from %i" % (iN, iR, iP, iR))
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1./float(self.nnodes-self.current_nnodes+1)
            else:
                stepsize = 0.5

            self.nodes[self.nR] = GSM.add_node(
                self.nodes[iR],
                self.nodes[iP],
                stepsize,
                iN,
                DQMAG_MAX=self.DQMAG_MAX,
                DQMAG_MIN=self.DQMAG_MIN,
                driving_coords=self.driving_coords,
            )

            if self.nodes[self.nR] is None:
                raise Exception('Ran out of space')

            if self.__class__.__name__ != "DE_GSM":
                ictan, bdist = self.get_tangent(
                    self.nodes[self.nR],
                    None,
                    driving_coords=self.driving_coords,
                )
                self.nodes[self.nR].bdist = bdist

            self.optimizer[self.nR].DMAX = self.optimizer[self.nR-1].DMAX
            self.current_nnodes += 1
            self.nR += 1
            print(" nn=%i,nR=%i" % (self.current_nnodes, self.nR))
            self.active[self.nR-1] = True

            # align center of mass  and rotation
            # print("%i %i %i" %(iR,iP,iN))

            # print(" Aligning")
            # self.nodes[self.nR-1].xyz = self.com_rotate_move(iR,iP,iN)

    def add_GSM_nodeP(self, newnodes=1):
        '''
        Add a node between endpoints on the product side, should only be called inside GSM
        '''
        printcool("Adding product node")
        if self.current_nnodes+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")

        for i in range(newnodes):
            # self.nodes[-self.nP-1] = BaseClass.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nnodes-self.nP)
            n1 = self.nnodes-self.nP
            n2 = self.nnodes-self.nP-1
            n3 = self.nR-1
            print(" adding node: %i between %i %i from %i" % (n2, n1, n3, n1))
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1./float(self.nnodes-self.current_nnodes+1)
            else:
                stepsize = 0.5

            self.nodes[-self.nP-1] = GSM.add_node(
                self.nodes[n1],
                self.nodes[n3],
                stepsize,
                n2
            )
            if self.nodes[-self.nP-1] is None:
                raise Exception('Ran out of space')

            self.optimizer[n2].DMAX = self.optimizer[n1].DMAX
            self.current_nnodes += 1
            self.nP += 1
            print(" nn=%i,nP=%i" % (self.current_nnodes, self.nP))
            self.active[-self.nP] = True

            # align center of mass  and rotation
            # print("%i %i %i" %(n1,n3,n2))
            # print(" Aligning")
            # self.nodes[-self.nP].xyz = self.com_rotate_move(n1,n3,n2)
            # print(" getting energy for node %d: %5.4f" %(self.nnodes-self.nP,self.nodes[-self.nP].energy - self.nodes[0].V0))
        return

    def reparameterize(self, ic_reparam_steps=8, n0=0, nconstraints=1):
        '''
        Reparameterize the string
        '''
        if self.interp_method == 'DLC':
            # print('reparameterizing')
            self.ic_reparam(nodes=self.nodes, energies=self.energies, climbing=(self.climb or self.find), ic_reparam_steps=ic_reparam_steps, NUM_CORE=self.mp_cores)
        return

    def ic_reparam_g(self, ic_reparam_steps=4, n0=0, reparam_interior=True):  # see line 3863 of gstring.cpp
        """
        Reparameterize during growth phase
        """

        printcool("Reparamerizing string nodes")
        # close_dist_fix(0) #done here in GString line 3427.
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)
        disprms = 0.0

        if self.current_nnodes == self.nnodes:
            return

        for i in range(ic_reparam_steps):
            self.ictan, self.dqmaga = self.get_tangents_growing()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
            if self.print_level > 0:
                if i == 0:
                    print(" totaldqmag (without inner): {:1.2}\n".format(totaldqmag))
                print(" printing spacings dqmaga: ")
                for n in range(self.nnodes):
                    print(" {:2.3}".format(self.dqmaga[n]), end=' ')
                    if (n+1) % 5 == 0:
                        print()
                print()

            if i == 0:
                if self.current_nnodes != self.nnodes:
                    rpart = np.zeros(self.nnodes)
                    for n in range(n0+1, self.nR):
                        rpart[n] = 1.0/(self.current_nnodes-2)
                    for n in range(self.nnodes-self.nP, self.nnodes-1):
                        rpart[n] = 1.0/(self.current_nnodes-2)
                else:
                    for n in range(n0+1, self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
                if self.print_level > 0:
                    if i == 0:
                        print(" rpart: ")
                        for n in range(1, self.nnodes-1):
                            print(" {:1.2}".format(rpart[n]), end=' ')
                            if (n) % 5 == 0:
                                print()
                        print()
            nR0 = self.nR
            nP0 = self.nP

            # TODO CRA 3/2019 why is this here?
            if not reparam_interior:
                if self.nnodes-self.current_nnodes > 2:
                    nR0 -= 1
                    nP0 -= 1

            deltadq = 0.0
            for n in range(n0+1, nR0):
                deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes-nP0, self.nnodes-1):
                deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(n0+1, self.nnodes-1):
                if abs(rpmove[n]) > MAXRE:
                    rpmove[n] = float(np.sign(rpmove[n])*MAXRE)

            disprms = float(np.linalg.norm(rpmove[n0+1:self.nnodes-1]))
            if self.print_level > 0:
                for n in range(n0+1, self.nnodes-1):
                    print(" disp[{}]: {:1.2f}".format(n, rpmove[n]), end=' ')
                    if (n) % 5 == 0:
                        print()
                print()
                print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 1e-2:
                break

            move_list = self.make_move_list()
            tan_list = self.make_tan_list()

            if self.mp_cores > 1:
                pool = mp.Pool(self.mp_cores)
                Vecs = pool.map(worker, ((self.nodes[0].coord_obj, "build_dlc", self.nodes[n].xyz, self.ictan[ntan]) for n, ntan in zip(move_list, tan_list) if rpmove[n] < 0))
                pool.close()
                pool.join()

                i = 0
                for n in move_list:
                    if rpmove[n] < 0:
                        self.nodes[n].coord_basis = Vecs[i]
                        i += 1

                # move the positions
                pool = mp.Pool(self.mp_cores)
                newXyzs = pool.map(worker, ((self.nodes[n].coord_obj, "newCartesian", self.nodes[n].xyz, rpmove[n]*self.nodes[n].constraints[:, 0]) for n in move_list if rpmove[n] < 0))
                pool.close()
                pool.join()
                i = 0
                for n in move_list:
                    if rpmove[n] < 0:
                        self.nodes[n].xyz = newXyzs[i]
                        i += 1
            else:
                for nmove, ntan in zip(move_list, tan_list):
                    if rpmove[nmove] < 0:
                        print('Moving {} along ictan[{}]'.format(nmove, ntan))
                        self.nodes[nmove].update_coordinate_basis(constraints=self.ictan[ntan])
                        constraint = self.nodes[nmove].constraints[:, 0]
                        dq0 = rpmove[nmove]*constraint
                        self.nodes[nmove].update_xyz(dq0, verbose=True)

        print(" spacings (end ic_reparam, steps: {}/{}):".format(i+1, ic_reparam_steps), end=' ')
        for n in range(self.nnodes):
            print(" {:1.2}".format(self.dqmaga[n]), end=' ')
        print("  disprms: {:1.3}".format(disprms))

        # TODO old GSM does this here
        # Failed = check_array(self.nnodes,self.dqmaga)
        # If failed, do exit 1

    def modify_TS_Hess(self):
        ''' Modifies Hessian using RP direction'''
        print("modifying %i Hessian with RP" % self.TSnode)

        TSnode = self.TSnode
        # a variable to determine how many time since last modify
        self.hess_counter = 0
        self.TS_E_0 = self.energies[TSnode]

        E0 = self.energies[TSnode]/GSM.units.KCAL_MOL_PER_AU
        Em1 = self.energies[TSnode-1]/GSM.units.KCAL_MOL_PER_AU
        if self.TSnode+1 < self.nnodes:
            Ep1 = self.energies[TSnode+1]/GSM.units.KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        # Update TS node coord basis
        Vecs = self.nodes[TSnode].update_coordinate_basis(constraints=None)

        # get constrained coord basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        const_vec = self.newic.update_coordinate_basis(constraints=self.ictan[TSnode])
        q0 = self.newic.coordinates[0]
        constraint = self.newic.constraints[:, 0]

        # this should just give back ictan[TSnode]?
        tan0 = block_matrix.dot(const_vec, constraint)

        # get qm1 (don't update basis)
        self.newic.xyz = self.nodes[TSnode-1].xyz.copy()
        qm1 = self.newic.coordinates[0]

        if TSnode+1 < self.nnodes:
            # get qp1 (don't update basis)
            self.newic.xyz = self.nodes[TSnode+1].xyz.copy()
            qp1 = self.newic.coordinates[0]
        else:
            qp1 = qm1

        print(" TS Hess init'd w/ existing Hintp")

        # Go to non-constrained basis
        self.newic.xyz = self.nodes[TSnode].xyz.copy()
        self.newic.coord_basis = Vecs
        self.newic.Primitive_Hessian = self.nodes[TSnode].Primitive_Hessian.copy()
        self.newic.form_Hessian_in_basis()

        tan = block_matrix.dot(block_matrix.transpose(Vecs), tan0)   # (nicd,1
        Ht = np.dot(self.newic.Hessian, tan)                         # (nicd,nicd)(nicd,1) = nicd,1
        tHt = np.dot(tan.T, Ht)

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        print(" tHt %1.3f a: %1.1f b: %1.1f c: %1.3f" % (tHt, a[0], b[0], c[0]))

        ttt = np.outer(tan, tan)

        # Hint before
        # with np.printoptions(threshold=np.inf):
        #    print self.newic.Hessian
        # eig,tmph = np.linalg.eigh(self.newic.Hessian)
        # print "initial eigenvalues"
        # print eig

        # Finalize Hessian
        self.newic.Hessian += (c-tHt)*ttt
        self.nodes[TSnode].Hessian = self.newic.Hessian.copy()

        # Hint after
        # with np.printoptions(threshold=np.inf):
        #    print self.nodes[TSnode].Hessian
        # print "shape of Hessian is %s" % (np.shape(self.nodes[TSnode].Hessian),)

        self.nodes[TSnode].newHess = 5

        if False:
            print("newHess of node %i %i" % (TSnode, self.nodes[TSnode].newHess))
            eigen, tmph = np.linalg.eigh(self.nodes[TSnode].Hessian)  # nicd,nicd
            print("eigenvalues of new Hess")
            print(eigen)

        # reset pgradrms ?

    def mult_steps(self, n, opt_steps):
        exsteps = 1
        tsnode = int(self.TSnode)

        if (self.find or self.climb) and self.energies[n] > self.energies[self.TSnode]*0.9 and n != tsnode:  #
            exsteps = 2
            print(" multiplying steps for node %i by %i" % (n, exsteps))
        elif self.find and n == tsnode and self.energies[tsnode] > self.energies[tsnode-1]*1.1 and self.energies[tsnode] > self.energies[tsnode+1]*1.1:  # Can also try self.climb but i hate climbing image
            exsteps = 2
            print(" multiplying steps for node %i by %i" % (n, exsteps))
        # elif not self.find and not self.climb and n==tsnode  and self.energies[tsnode]>self.energies[tsnode-1]*1.5 and self.energies[tsnode]>self.energies[tsnode+1]*1.5 and self.climber:
        #    exsteps=2
        #    print(" multiplying steps for node %i by %i" % (n,exsteps))

        # elif not (self.find and self.climb) and self.energies[tsnode] > 1.75*self.energies[tsnode-1] and self.energies[tsnode] > 1.75*self.energies[tsnode+1] and self.done_growing and n==tsnode:  #or self.climb
        #    exsteps=2
        #    print(" multiplying steps for node %i by %i" % (n,exsteps))
        return exsteps*opt_steps

    def set_opt_type(self, n, quiet=False):
        # TODO error for seam climb
        opt_type = 'ICTAN'
        if self.climb and n == self.TSnode and not self.find and self.nodes[n].PES.__class__.__name__ != "Avg_PES":
            opt_type = 'CLIMB'
        elif self.find and n == self.TSnode:
            opt_type = 'TS'
        elif self.nodes[n].PES.__class__.__name__ == "Avg_PES":
            opt_type = 'SEAM'
            if self.climb and n == self.TSnode:
                opt_type = 'TS-SEAM'
        if not quiet:
            print((" setting node %i opt_type to %s" % (n, opt_type)))

        # if isinstance(self.optimizer[n],beales_cg) and opt_type!="BEALES_CG":
        #    raise RuntimeError("This shouldn't happen")

        return opt_type

    #TODO Remove me does not deserve to be a function
    def set_finder(self, rtype):
        assert rtype in [0, 1, 2], "rtype not defined"
        print('')
        print("*********************************************************************")
        if rtype == 2:
            print("****************** set climber and finder to True *******************")
            self.climber = True
            self.finder = True
        elif rtype == 1:
            print("***************** setting climber to True*************************")
            self.climber = True
        else:
            print("******** Turning off climbing image and exact TS search **********")
        print("*********************************************************************")

    def com_rotate_move(self, iR, iP, iN):
        print(" aligning com and to Eckart Condition")

        mfrac = 0.5
        if self.nnodes - self.current_nnodes+1 != 1:
            mfrac = 1./(self.nnodes - self.current_nnodes+1)

        # if self.__class__.__name__ != "DE_GSM":
        #    # no "product" structure exists, use initial structure
        #    iP = 0

        xyz0 = self.nodes[iR].xyz.copy()
        xyz1 = self.nodes[iN].xyz.copy()
        com0 = self.nodes[iR].center_of_mass
        com1 = self.nodes[iN].center_of_mass
        masses = self.nodes[iR].mass_amu

        # From the old GSM code doesn't work
        # com1 = mfrac*(com2-com0)
        # print("com1")
        # print(com1)
        # # align centers of mass
        # xyz1 += com1
        # Eckart_align(xyz1,xyz2,masses,mfrac)

        # rotate to be in maximal coincidence with 0
        # assumes iP i.e. 2 is also in maximal coincidence
        U = rotate.get_rot(xyz0, xyz1)
        xyz1 = np.dot(xyz1, U)

        # # align
        # if self.nodes[iP] != None:
        #    xyz2 = self.nodes[iP].xyz.copy()
        #    com2 = self.nodes[iP].center_of_mass

        #    if abs(iN-iR) > abs(iN-iP):
        #        avg_com = mfrac*com2 + (1.-mfrac)*com0
        #    else:
        #        avg_com = mfrac*com0 + (1.-mfrac)*com2
        #    dist = avg_com - com1  #final minus initial
        # else:
        #    dist = com0 - com1  #final minus initial

        # print("aligning to com")
        # print(dist)
        # xyz1 += dist

        return xyz1

    def find_peaks(self, rtype='opting'):
        '''
        This doesnt actually calculate peaks, it calculates some other thing
        '''
        # rtype 1: growing
        # rtype 2: opting
        # rtype 3: intermediate check
        if rtype not in ['growing', 'opting', 'intermediate']:
            raise RuntimeError

        # if rtype==1:
        if rtype == "growing":
            nnodes = self.nR
        elif rtype == "opting" or rtype == "intermediate":
            nnodes = self.nnodes
        else:
            raise ValueError("find peaks bad input")
        # if rtype==1 or rtype==2:
        #    print "Energy"
        alluptol = 0.1
        alluptol2 = 0.5
        allup = True
        diss = False
        energies = self.energies
        for n in range(1, len(energies[:nnodes])):
            if energies[n]+alluptol < energies[n-1]:
                allup = False
                break

        if energies[nnodes-1] > 15.0:
            if nnodes-3 > 0:
                if ((energies[nnodes-1]-energies[nnodes-2]) < alluptol2 and
                    (energies[nnodes-2]-energies[nnodes-3]) < alluptol2 and
                        (energies[nnodes-3]-energies[nnodes-4]) < alluptol2):
                    print(" possible dissociative profile")
                    diss = True

        print(" nnodes ", nnodes)
        print(" all uphill? ", allup)
        print(" dissociative? ", diss)
        npeaks1 = 0
        npeaks2 = 0
        minnodes = []
        maxnodes = []
        if energies[1] > energies[0]:
            minnodes.append(0)
        if energies[nnodes-1] < energies[nnodes-2]:
            minnodes.append(nnodes-1)
        for n in range(self.n0, nnodes-1):
            if energies[n+1] > energies[n]:
                if energies[n] < energies[n-1]:
                    minnodes.append(n)
            if energies[n+1] < energies[n]:
                if energies[n] > energies[n-1]:
                    maxnodes.append(n)

        print(" min nodes ", minnodes)
        print(" max nodes ", maxnodes)
        npeaks1 = len(maxnodes)
        # print "number of peaks is ",npeaks1
        ediff = 0.5
        PEAK4_EDIFF = 2.0
        if rtype == "growing":
            ediff = 1.
        if rtype == "intermediate":
            ediff = PEAK4_EDIFF

        if rtype == "growing":
            nmax = np.argmax(energies[:self.nR])
            emax = float(max(energies[:self.nR]))
        else:
            emax = float(max(energies))
            nmax = np.argmax(energies)

        print(" emax and nmax in find peaks %3.4f,%i " % (emax, nmax))

        #check if any node after peak is less than 2 kcal below
        for n in maxnodes:
            diffs = (energies[n]-e > ediff for e in energies[n:nnodes])
            if any(diffs):
                found = n
                npeaks2 += 1
        npeaks = npeaks2
        print(" found %i significant peak(s) TOL %3.2f" % (npeaks, ediff))

        # handle dissociative case
        if rtype == "intermediate" and npeaks == 1:
            nextmin = 0
            for n in range(found, nnodes-1):
                if n in minnodes:
                    nextmin = n
                    break
            if nextmin > 0:
                npeaks = 2

        # if rtype==3:
        #    return nmax
        if allup is True and npeaks == 0:
            return -1
        if diss is True and npeaks == 0:
            return -2

        return npeaks

    def is_converged(self, totalgrad, fp, rtype, ts_cgradq):
        '''
        Check if optimization is converged
        '''

        # Important the factor 5 here corresponds to the same convergence criteria in the TS optimizer
        TS_conv = self.CONV_TOL*5
        # => Check if intermediate exists
        # ALEX REMOVED CLIMB REQUIREMENT
        if self.has_intermediate(self.noise):
            print("New pot min: {}".format(self.get_intermediate(self.noise)))
            print("Old pot min: {}".format(self.pot_min))
            if self.get_intermediate(self.noise) == self.pot_min:
                self.endearly_counter += 1
            else:
                self.pot_min = self.get_intermediate(self.noise)
                self.endearly_counter = 1
            if self.endearly_counter >= 3:
                self.end_early = True
                self.tscontinue = False
                printcool(" THERE IS AN INTERMEDIATE, OPTIMIZE THE INTERMEDIATE AND TRY AGAIN")
                return True

        elif not self.has_intermediate(self.noise):
            self.endearly_counter = 0
            self.pot_min = self.get_intermediate(self.noise)

        # print(" Number of imaginary frequencies %i" % self.optimizer[self.TSnode].nneg)

        # or (totalgrad<0.1 and self.nodes[self.TSnode].gradrms<2.5*TS_conv and self.dE_iter<0.02 and self.optimizer[self.TSnode].nneg <2)  #TODO extra crit here
        if (self.finder and self.find):
            return (self.nodes[self.TSnode].gradrms < self.CONV_TOL and abs(ts_cgradq) < TS_conv and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff*3 and self.optimizer[self.TSnode].nneg < 2)
        elif self.climber and self.climb:
            return (self.nodes[self.TSnode].gradrms < self.CONV_TOL and abs(ts_cgradq) < TS_conv and self.dE_iter < self.optimizer[self.TSnode].conv_Ediff*3)
        elif not self.climber and not self.finder:
            print(" CONV_TOL=%.4f" % self.CONV_TOL)
            return all([self.optimizer[n].converged for n in range(1, self.nnodes-1)])

        return False

    def print_energies(self):
        for n in range(len(self.energies)):
            print(" {:7.3f}".format(float(self.energies[n])), end=' ')
        print()

    def get_intermediate(self, noise):
        '''
        Check string for intermediates
        noise is a leeway factor for determining intermediate
        '''

        energies = self.energies
        potential_min = []
        for i in range(1, (len(energies) - 1)):
            rnoise = 0
            pnoise = 0
            a = 1
            b = 1
            while (energies[i-a] >= energies[i]):
                if (energies[i-a] - energies[i]) > rnoise:
                    rnoise = energies[i-a] - energies[i]
                if rnoise > noise:
                    break
                if (i-a) == 0:
                    break
                a += 1

            while (energies[i+b] >= energies[i]):
                if (energies[i+b] - energies[i]) > pnoise:
                    pnoise = energies[i+b] - energies[i]
                if pnoise > noise:
                    break
                if (i+b) == len(energies) - 1:
                    break
                b += 1
            if ((rnoise > noise) and (pnoise > noise)):
                print('Potential minimum at image %s' % i)
                potential_min.append(i)

        return potential_min

    def has_intermediate(self, noise):
        pot_min = self.get_intermediate(noise)
        return len(pot_min) > 0

    def setup_from_geometries(self, input_geoms, reparametrize=True, restart_energies=True, start_climb_immediately=False):
        '''
        Restart
        input_geoms list of geometries
        reparameterize (boolean) : reparameterize the initial string to make the nodes equidistant
        restart_energies (boolean) : generate the initial energies
        start_climb_immediately (boolean) : set climb to True or False
        '''

        printcool("Restarting GSM from geometries")
        self.growth_direction = 0
        nstructs = len(input_geoms)

        if nstructs != self.nnodes:
            print('need to interpolate')
            # if self.interp_method=="DLC": TODO
            raise NotImplementedError
        else:
            geoms = input_geoms

        self.gradrms = [0.]*nstructs
        self.dE = [1000.]*nstructs

        self.isRestarted = True
        self.done_growing = True

        # set coordinates from geoms
        self.nodes[0].xyz = xyz_to_np(geoms[0])
        self.nodes[nstructs-1].xyz = xyz_to_np(geoms[-1])
        for struct in range(1, nstructs-1):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[struct-1],
                                                            xyz_to_np(geoms[struct]),
                                                            new_node_id=struct,
                                                            copy_wavefunction=False)
            self.nodes[struct].newHess = 5
            # Turning this off
            # self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
            # self.nodes[struct].gradrms=grmss[struct]
            # self.nodes[struct].PES.dE = dE[struct]
        self.nnodes = self.nR = nstructs

        if start_climb_immediately:
            # should check that this is a climber...
            self.climb = True
        #ALEX CHANGE - rearranged reparameterize and restart_energies 'if' blocks
        if restart_energies:
            # self.interpolate_orbitals()
            print(" V_profile: ", end=' ')
            energies = self.energies
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(energies[n])), end=' ')
            print()
        if reparametrize:
            printcool("Reparametrizing")
            self.reparameterize(ic_reparam_steps=8)
            self.xyz_writer('grown_string_{:03}.xyz'.format(self.ID), self.geometries, self.energies, self.gradrmss, self.dEs)


        self.ictan, self.dqmaga = self.get_tangents(self.nodes)
        self.refresh_coordinates()
        print(" setting all interior nodes to active")
        for n in range(1, self.nnodes-1):
            self.active[n] = True
            self.optimizer[n].conv_grms = self.CONV_TOL*2.5
            self.optimizer[n].options['DMAX'] = 0.05

        return

    def add_node_before_TS(self):
        '''
        '''

        # New node is TS node
        new_node = GSM.add_node(
            self.nodes[self.TSnode-1],
            self.nodes[self.TSnode],
            stepsize=0.5,
            node_id=self.TSnode,
        )
        new_node_list = [None]*(self.nnodes+1)
        new_optimizers = [None]*(self.nnodes+1)

        for n in range(0, self.TSnode):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]

        new_node_list[self.TSnode] = new_node
        new_optimizers[self.TSnode] = self.optimizer[0].__class__(self.optimizer[0].options.copy())

        for n in range(self.TSnode+1, self.nnodes+1):
            new_node_list[n] = Molecule.copy_from_options(MoleculeA=self.nodes[n-1], new_node_id=n)
            new_optimizers[n] = self.optimizer[n-1]

        self.nodes = new_node_list
        self.optimizer = new_optimizers
        self.nnodes = len(self.nodes)
        print(' New number of nodes %d' % self.nnodes)
        self.active = [True] * self.nnodes
        self.active[0] = False
        self.active[self.nnodes-1] = False
        print("0")
        print(self.nodes[0].xyz)
        print("1")
        print(self.nodes[1].xyz)
        print("-1")
        print(self.nodes[-1].xyz)

    def add_node_after_TS(self):
        '''
        '''
        new_node = GSM.add_node(
            self.nodes[self.TSnode],
            self.nodes[self.TSnode+1],
            stepsize=0.5,
            node_id=self.TSnode+1,
        )
        new_node_list = [None]*(self.nnodes+1)
        new_optimizers = [None]*(self.nnodes+1)
        for n in range(0, self.TSnode+1):
            new_node_list[n] = self.nodes[n]
            new_optimizers[n] = self.optimizer[n]
        new_node_list[self.TSnode+1] = new_node
        new_optimizers[self.TSnode+1] = self.optimizer[0].__class__(self.optimizer[0].options.copy())

        for n in range(self.TSnode+2, self.nnodes+1):
            new_node_list[n] = Molecule.copy_from_options(MoleculeA=self.nodes[n-1], new_node_id=n)
            new_optimizers[n] = self.optimizer[n-1]
        self.nodes = new_node_list
        self.optimizer = new_optimizers
        self.nnodes = len(self.nodes)
        print(' New number of nodes %d' % self.nnodes)
        self.active = [True] * self.nnodes
        self.active[0] = False
        self.active[self.nnodes-1] = False

    def set_node_convergence(self):
        ''' set convergence for nodes
        '''

        factor = 5. if (self.climber or self.finder) else 1.
        TSnode = self.TSnode
        for n in range(1, self.nnodes-1):
            if self.nodes[n] is not None:
                self.optimizer[n].conv_grms = self.CONV_TOL*factor
                self.optimizer[n].conv_gmax = self.options['CONV_gmax']*factor
                self.optimizer[n].conv_Ediff = self.options['CONV_Ediff']*factor
                if self.optimizer[n].converged and n != TSnode:
                    self.optimizer[n].check_only_grad_converged=True
                if (self.climb or self.find) and self.energies[n]>self.energies[TSnode]*0.75 and n!=TSnode:
                    self.optimizer[n].conv_grms = self.CONV_TOL     
                    self.optimizer[n].conv_gmax = self.options['CONV_gmax']
                    self.optimizer[n].conv_Ediff = self.options['CONV_Ediff']
                    self.optimizer[n].check_only_grad_converged = False
                if n == self.TSnode and (self.climb or self.find):
                    self.optimizer[n].conv_grms = self.CONV_TOL
                    self.optimizer[n].conv_gmax = self.options['CONV_gmax']
                    self.optimizer[n].conv_Ediff = self.options['CONV_Ediff']
                    self.optimizer[n].check_only_grad_converged = False

    def slow_down_climb(self):
        if self.climb and not self.find:
            print(" slowing down climb optimization")
            self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale
            self.optimizer[self.TSnode].options['SCALEQN'] = 2.
            if self.optimizer[self.TSnode].SCALE_CLIMB < 5.:
                self.optimizer[self.TSnode].SCALE_CLIMB += 1.
            self.optimizer[self.pTSnode].options['SCALEQN'] = 1.
            self.ts_exsteps = 1
            if self.newclimbscale < 5.0:
                self.newclimbscale += 1.
        elif self.find:
            self.find = False
            self.climb = True
            self.nclimb = 1
            print(" Find bad, going back to climb")

    def interpolate_orbitals(self):
        '''
        Interpolate orbitals
        '''
        print("Interpolating orbitals")
        nnodes = len(self.nodes)
        #nn = nnodes//2 + 1
        nn = - (nnodes // -2)
        couples = [(i, nnodes-i-1) for i in range(nn)]
        first = True
        options = {}
        for i, j in couples:
            if first:
                # Calculate the energy of the i, j
                print("Calculating initial energy for node: {}".format(i))
                self.nodes[i].energy
                print("Calculating initial energy for node: {}".format(j))
                self.nodes[j].energy
                first = False
            elif j - i <= 1:
                #even nnodes case
                if i == j - 1:  
                    self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                        self.nodes[i-1].PES.lot, copy_wavefunction=True)
                    self.nodes[i].PES.lot.node_id = i
                    self.nodes[i].energy
                    i += 1
                if i == j:
                    print("Checking if energies match for wavefunction guesses from either direction for node: {}".format(i))

                    options={'node_id':self.nodes[i].node_id}
                    self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                        self.nodes[i-1].PES.lot, options, copy_wavefunction=True)
                    print("Getting forward energy")
                    self.nodes[i].PES.lot.node_id = i
                    energy_forward = self.nodes[i].energy
                    print("Getting backward energy")
                    options={'node_id':self.nodes[i+1].node_id}
                    self.nodes[i].PES.lot = type(self.nodes[i+1].PES.lot).copy(
                        self.nodes[i+1].PES.lot, options, copy_wavefunction=True)
                    self.nodes[i].PES.lot.node_id = i
                    self.nodes[i].PES.lot.hasRanForCurrentCoords = False
                    energy_backward = self.nodes[i].energy
                    print("Forward direction energy: {}".format(energy_forward))
                    print("Backward direction energy: {}".format(energy_backward))
                    if abs(energy_forward - energy_backward) < 0.1:
                        print("Energies match")
                    else:
                        print("Energies do not match")
                        if energy_backward < energy_forward:
                            for k in range(i):
                                options={'node_id':self.nodes[i+k-1].node_id}
                                self.nodes[i-k-1].PES.lot = type(self.nodes[i-k].PES.lot).copy(
                                    self.nodes[i-k].PES.lot, options, copy_wavefunction=True)
                                self.nodes[i-k-1].PES.lot.node_id = i-k-1
                                self.nodes[i-k -1].PES.lot.hasRanForCurrentCoords = False
                                print("node_id {}".format(self.nodes[i+k].node_id))
                                print("Calculating new initial energy for node: {}".format(i-k-1))
                                print("New energy: {}".format(self.nodes[i-k-1].energy))
                        else:
                            for k in range(i+1):
                                #lower energy is in forward direction, so do node i using i-1's wavefunction
                                options={'node_id':self.nodes[i+k].node_id}
                                self.nodes[i+k].PES.lot = type(self.nodes[i+k-1].PES.lot).copy(
                                    self.nodes[i+k-1].PES.lot, options, copy_wavefunction=True)
                                self.nodes[i+k].PES.lot.node_id = i+k
                                self.nodes[i+k].PES.lot.hasRanForCurrentCoords = False
                                print("node_id {}".format(self.nodes[i+k].node_id))
                                print("Calculating new initial energy for node: {}".format(i+k))
                                print("New energy: {}".format(self.nodes[i+k].energy))

                     
                    
            else:
                # Copy the orbital of i-1 to i
                self.nodes[i].PES.lot = type(self.nodes[i-1].PES.lot).copy(
                    self.nodes[i-1].PES.lot, options, copy_wavefunction=True)
                self.nodes[i].PES.lot.node_id = i
                print("Calculating initial energy for node: {}".format(i))
                self.nodes[i].energy
                # Copy the orbital of j+1 to j
                self.nodes[j].PES.lot = type(self.nodes[j+1].PES.lot).copy(
                    self.nodes[j+1].PES.lot, options, copy_wavefunction=True)
                self.nodes[j].PES.lot.node_id = j
                print("Calculating initial energy for node: {}".format(j))
                self.nodes[j].energy
        return
