from __future__ import print_function
import numpy as np

try:
    from .gsm import GSM
except:
    from .gsm import GSM


#######################################################################################
############### This class contains the main GSM functions  ###########################
#######################################################################################

class MainGSM(GSM):
    
    def grow_string(self,max_iters=30,max_opt_steps=3,nconstraints=1):
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
        nifty.printcool("In growth_iters")

        #TODO

        ncurrent,nlist = self.make_nlist()
        self.ictan,self.dqmaga = self.get_tangents_growing()
        self.set_active(self.nR-1, self.nnodes-self.nP)

        isGrown=False
        iteration=0
        while not isGrown:
            if iteration>max_iters:
                raise RuntimeError
            nifty.printcool("Starting growth iteration %i" % iteration)
            self.optimize_iteration(max_opt_steps)
            totalgrad,gradrms,sum_gradrms = self.calc_grad()
            self.write_xyz_files('scratch/growth_iters_{:03}_{:03}.xyz'.format(self.ID,iteration))
            print(" gopt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E: {:5.4}\n".format(iteration,float(totalgrad),float(gradrms),float(self.emax)))
                
            try:
                self.grow_nodes()
            except Exception as error:
                print("can't add anymore nodes, bdist too small")

                if self.__class__.__name__=="SE_GSM": # or self.__class__.__name__=="SE_Cross":
                    # Don't do SE_cross because that already does optimization later
                    if self.nodes[self.nR-1].PES.lot.do_coupling:
                        opt_type='MECI'
                    else:
                       opt_type='UNCONSTRAINED'

                    print(" optimizing last node")
                    self.optimizer[self.nR-1].conv_grms = self.CONV_TOL
                    print(self.optimizer[self.nR-1].conv_grms)
                    self.optimizer[self.nR-1].optimize(
                            molecule=self.nodes[self.nR-1],
                            refE=self.nodes[0].V0,
                            opt_steps=50,
                            opt_type=opt_type,
                            )
                elif self.__class__.__name__=="SE_Cross":
                    print(" Will do extra optimization of this node in SE-Cross")
                else:
                    raise RuntimeError

            self.set_active(self.nR-1, self.nnodes-self.nP)
            self.ic_reparam_g()
            #TODO
            self.ictan,self.dqmaga = self.get_tangents_growing()

            iteration+=1
            isGrown = self.check_if_grown()

        # create newic object
        print(" creating newic molecule--used for ic_reparam")
        self.newic  = Molecule.copy_from_options(self.nodes[0])

        # TODO should something be done for growthdirection 2?
        if self.growth_direction==1:
            print("Setting LOT of last node")
            self.nodes[-1] = Molecule.copy_from_options(
                    MoleculeA = self.nodes[-2],
                    xyz = self.nodes[-1].xyz,
                    new_node_id = self.nnodes-1
                    )
        return 


    def optimize_string(self,max_iter=30,nconstraints=1,opt_steps=1,rtype=2):
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
            
        '''
        nifty.printcool("In opt_iters")

        self.nclimb=0
        self.nhessreset=10  # are these used??? TODO 
        self.hessrcount=0   # are these used?!  TODO
        self.newclimbscale=2.
        self.set_finder(rtype)

        # set convergence for nodes
        if (self.climber or self.finder):
            factor = 2.5
        else: 
            factor = 1.
        for i in range(self.nnodes):
            if self.nodes[i] !=None:
                self.optimizer[i].conv_grms = self.CONV_TOL*factor
                self.optimizer[i].conv_gmax = self.options['CONV_gmax']*factor
                self.optimizer[i].conv_Ediff = self.options['CONV_Ediff']*factor

        # enter loop
        for oi in range(max_iter):

            nifty.printcool("Starting opt iter %i" % oi)
            if self.climb and not self.find: print(" CLIMBING")
            elif self.find: print(" TS SEARCHING")

            sys.stdout.flush()

            # stash previous TSnode  
            self.pTSnode = self.TSnode
            self.emaxp = self.emax

            # => Get all tangents 3-way <= #
            self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
           
            # => do opt steps <= #
            self.optimize_iteration(optsteps)
            #self.store_energies()

            print(" V_profile: ", end=' ')
            energies = self.energies
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(energies[n])), end=' ')
            print()

            #TODO resetting
            #TODO special SSM criteria if TSNode is second to last node
            #TODO special SSM criteria if first opt'd node is too high?
            if self.TSnode == self.nnodes-2 and (self.climb or self.find):
                nifty.printcool("WARNING\n: TS node shouldn't be second to last node for tangent reasons")

            # => find peaks <= #
            fp = self.find_peaks(2)

            # => get TS node <=
            self.emax= self.energies[self.TSnode]
            print(" max E({}) {:5.4}".format(self.TSnode,self.emax))

            ts_cgradq = 0.
            if not self.find:
                ts_cgradq = np.linalg.norm(np.dot(self.nodes[self.TSnode].gradient.T,self.nodes[self.TSnode].constraints[:,0])*self.nodes[self.TSnode].constraints[:,0])
                print(" ts_cgradq %5.4f" % ts_cgradq)

            ts_gradrms=self.nodes[self.TSnode].gradrms
            self.dE_iter=abs(self.emax-self.emaxp)
            print(" dE_iter ={:2.2f}".format(self.dE_iter))

            # => calculate totalgrad <= #
            totalgrad,gradrms,sum_gradrms = self.calc_grad()

            # => Check Convergence <= #
            isDone = self.check_opt(totalgrad,fp,rtype,ts_cgradq)
            if isDone:
                self.found_ts=True
                break

            # => Check if intermediate exists 
            if check_for_intermediate():
                self.exit_early
                return 

            sum_conv_tol = (self.nnodes-2)*self.CONV_TOL 
            if not self.climber and not self.finder:
                print(" CONV_TOL=%.4f" %self.CONV_TOL)
                print(" convergence criteria is %.5f, current convergence %.5f" % (sum_conv_tol,sum_gradrms))
                all_conv=True
                for  n in range(1,self.nnodes-1):
                    if not self.optimizer[n].converged:
                        all_conv=False
                        break

                if sum_gradrms<sum_conv_tol and all_conv: #Break even if not climb/find
                    break

            # => set stage <= #
            form_TS_hess = self.set_stage(totalgrad,sum_gradrms,ts_cgradq,ts_gradrms,fp)

            # => Reparam the String <= #
            if oi!=max_iter-1:
                self.reparameterize(nconstraints=nconstraints)

            # store reparam energies
            #self.store_energies()
            energies = self.energies
            self.emax= energies[self.TSnode]
            print(" V_profile (after reparam): ", end=' ')
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(energies[n])), end=' ')
            print()

            ## Modify TS Hess if necessary ##
            # from set stage
            if form_TS_hess:
                #TODO
                self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
                self.get_eigenv_finite(self.TSnode)
                if self.optimizer[self.TSnode].options['DMAX']>0.1:
                    self.optimizer[self.TSnode].options['DMAX']=0.1

            # opt decided Hess is not good because of overlap
            elif self.find and not self.optimizer[n].maxol_good:
                #TODO
                self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
                self.get_eigenv_finite(self.TSnode)

            # 
            elif self.find and (self.optimizer[self.TSnode].nneg > 3 or self.optimizer[self.TSnode].nneg==0 or self.hess_counter > 3 or (self.TS_E_0 - self.emax) > 10.) and ts_gradrms >self.CONV_TOL:
                if self.hessrcount<1 and self.pTSnode == self.TSnode:
                    print(" resetting TS node coords Ut (and Hessian)")
                    self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
                    self.get_eigenv_finite(self.TSnode)
                    self.nhessreset=10
                    self.hessrcount=1
                else:
                    print(" Hessian consistently bad, going back to climb (for 3 iterations)")
                    self.find=False
                    #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                    self.nclimb=2

            #elif self.find and self.optimizer[self.TSnode].nneg > 1 and ts_gradrms < self.CONV_TOL:
            #     print(" nneg > 1 and close to converging -- reforming Hessian")                
            #         self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
            #     self.get_eigenv_finite(self.TSnode)                    

            elif self.find and self.optimizer[self.TSnode].nneg <= 3:
                self.hessrcount-=1
                self.hess_counter += 1

            if self.pTSnode!=self.TSnode and self.climb:
                #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy())
                #self.optimizer[self.pTSnode] = self.optimizer[0].__class__(self.optimizer[self.TSnode].options.copy())

                if self.climb and not self.find:
                    print(" slowing down climb optimization")
                    self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale
                    self.optimizer[self.TSnode].options['SCALEQN'] = 2.
                    if self.optimizer[self.TSnode].SCALE_CLIMB <5.:
                        self.optimizer[self.TSnode].SCALE_CLIMB +=1.
                    self.optimizer[self.pTSnode].options['SCALEQN'] = 1.
                    self.ts_exsteps=1
                    if self.newclimbscale<5.0:
                        self.newclimbscale +=1.
                elif self.find:
                    self.find = False
                    self.nclimb=1
                    print(" Find bad, going back to climb")
                    #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.pTSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                    #self.optimizer[self.pTSnode] = self.optimizer[0].__class__(self.optimizer[self.TSnode].options.copy())
                    #self.self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
                    #self.get_eigenv_finite(self.TSnode)

            # => write Convergence to file <= #
            self.write_xyz_files('scratch/opt_iters_{:03}_{:03}.xyz'.format(self.ID,oi))

            #TODO prints tgrads and jobGradCount
            print("opt_iter: {:2} totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(oi,float(totalgrad),float(gradrms),self.TSnode,float(self.emax)))
            print('\n')

        ## Optimize TS node to a finer convergence
        #if rtype==2:

        #    # loop 10 times, 5 optimization steps each
        #    for i in range(10):
        #        nifty.printcool('cycle {}'.format(i))
        #        geoms,energies = self.nodes[self.TSnode].optimizer.optimize(
        #                molecule=self.nodes[gsm.TSnode],
        #                refE=self.nodes[0].V0,
        #                opt_steps=5,
        #                opt_type="TS",
        #                ictan=self.ictan[gsm.TSnode],
        #                verbose=True,
        #                )
        #        self.self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
        #        tmp_geoms+=geoms
        #        tmp_E+=energies
        #        manage_xyz.write_xyzs_w_comments(
        #                'optimization.xyz',
        #                tmp_geoms,
        #                tmp_E,
        #                )
        #        if self.nodes[self.TSnode].optimizer.converged:
        #            break

        filename="opt_converged_{:03d}.xyz".format(self.ID)
        print(" Printing string to " + filename)
        self.write_xyz_files(filename)
        sys.stdout.flush()
        return


    def refresh_coordinates(self,update_TS=True):
        '''
        Refresh the DLC coordinates for the string
        '''
        energies = self.energies
        TSnode = self.TSnode

        if not self.done_growing:
            self.ictan,self.dqmaga = self.get_tangents_growing(nodes,node_list)
             #TODO
        else:
            self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
            for n in range(1,self.nnodes-1):
                # don't update tsnode coord basis 
                if n!=TSnode or (n==TSnode and update_TS): 
                    # update newic coordinate basis
                    self.newic.xyz = self.nodes[n].xyz
                    Vecs = self.newic.update_coordinate_basis(self.ictan[n])
                    self.nodes[n].coord_basis = Vecs
        

    def optimize_iteration(self,opt_steps):
        '''
        Optimize string iteration
        '''

        refE=self.nodes[0].energy

        if self.use_multiprocessing:
            cpus = mp.cpu_count()/self.nodes[0].PES.lot.nproc                                  
            print(" Parallelizing over {} processes".format(cpus))                             
            out_queue = mp.Queue()

            workers = [ mp.Process(target=mod, args=(self.nodes[n],self.optimizer[n],self.ictan[n],self.mult_steps(n,opt_steps),self.set_opt_type(n),refE,n,s,gp_prim, out_queue) ) for n in range(self.nnodes) if self.nodes[n] and self.active[n] ]
            
            for work in workers: work.start()                                                  
            
            for work in workers: work.join()                                                   
            
            res_lst = []
            for j in range(len(workers)):
                res_lst.append(out_queue.get())                                                
                                             
        else:
            for n in range(self.nnodes):
                if self.nodes[n] and self.active[n]:
                    print()
                    path=os.path.join(os.getcwd(),'scratch/{:03d}/{}'.format(self.ID,n))
                    nifty.printcool("Optimizing node {}".format(n))
                    opt_type = self.set_opt_type(n)
                    osteps = self.mult_steps(n,opt_steps)
                    self.optimizer[n].optimize(
                            molecule=self.nodes[n],
                            refE=refE,
                            opt_type=opt_type,
                            opt_steps=osteps,
                            ictan=self.ictan[n],
                            xyzframerate=1,
                            path=path,
                            )

        if self.__class__.__name__=="SE-GSM" and self.done_growing:
            fp = self.find_peaks(2)
            if self.energies[self.nnodes-1]>self.energies[self.nnodes-2] and fp>0 and self.nodes[self.nnodes-1].gradrms>self.CONV_TOL:
                self.optimizer[self.nnodes-1].optimize(
                        molecule=self.nodes[self.nnodes-1],
                        refE=refE,
                        opt_type='UNCONSTRAINED',
                        opt_steps=osteps,
                        ictan=None,
                        )

    def get_tangents_growing(self):
        """
        Finds the tangents during the growth phase. 
        Tangents referenced to left or right during growing phase.
        Also updates coordinates
        Not a static method beause no one should ever call this outside of GSM
        """

        ncurrent,nlist = self.make_nlist()
        dqmaga = [0.]*self.nnodes
        ictan = [[]]*nnodes
    
        if self.print_level>1:
            print("ncurrent, nlist")
            print(ncurrent)
            print(nlist)
    
        for n in range(ncurrent):
            print(" ictan[{}]".format(nlist[2*n]))
            ictan0,_ = get_tangent(
                    node1=self.nodes[nlist[2*n]],
                    node2=self.nodes[nlist[2*n+1]],
                    driving_coords=self.driving_coords,
                    )
    
            if self.print_level>1:
                print("forming space for", nlist[2*n+1])
            if self.print_level>1:
                print("forming tangent for ",nlist[2*n])
    
            if (ictan0[:]==0.).all():
                print(" ICTAN IS ZERO!")
                print(nlist[2*n])
                print(nlist[2*n+1])
                raise RuntimeError
    
            #normalize ictan
            norm = np.linalg.norm(ictan0)  
            ictan[nlist[2*n]] = ictan0/norm
           
            Vecs = self.nodes[nlist[2*n]].update_coordinate_basis(constraints=self.ictan[nlist[2*n]])
            constraint = self.nodes[nlist[2*n]].constraints
            prim_constraint = block_matrix.dot(Vecs,constraint)
    
            # NOTE regular GSM does something weird here 
            # but this is not followed here anymore 7/1/2020
            #dqmaga[nlist[2*n]] = np.dot(prim_constraint.T,ictan0) 
            #dqmaga[nlist[2*n]] = float(np.sqrt(abs(dqmaga[nlist[2*n]])))
            tmp_dqmaga = np.dot(prim_constraint.T,ictan0)
            tmp_dqmaga = np.sqrt(tmp_dqmaga)
            dqmaga[nlist[2*n]] = norm
    
    
        if self.print_level>0:
            print('------------printing dqmaga---------------')
            for n in range(self.nnodes):
                print(" {:5.3}".format(dqmaga[n]), end=' ')
                if (n+1)%5==0:
                    print()
            print() 
       
        if print_level>1:
            for n in range(ncurrent):
                print("dqmag[%i] =%1.2f" %(nlist[2*n],self.dqmaga[nlist[2*n]]))
                print("printing ictan[%i]" %nlist[2*n])       
                print(self.ictan[nlist[2*n]].T)
        for i,tan in enumerate(ictan):
            if np.all(tan==0.0):
                print("tan %i of the tangents is 0" %i)
                raise RuntimeError
   
        return ictan,dqmaga


    # TODO remove return form_TS hess  3/2021
    def set_stage(self,totalgrad,sumgradrms, ts_cgradq,ts_gradrms,fp):
        form_TS_hess=False

        sum_conv_tol = np.sum([self.optimizer[n].conv_grms for n in range(1,self.nnodes-1)])+ 0.0005

        #TODO totalgrad is not a good criteria for large systems
        if (totalgrad < 0.3 or sumgradrms<sum_conv_tol or ts_cgradq < 0.01)  and fp>0: # extra criterion in og-gsm for added
            if not self.climb and self.climber:
                print(" ** starting climb **")
                self.climb=True
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq))
       
                # set to beales
                #self.optimizer[self.TSnode] = beales_cg(self.optimizer[self.TSnode].options.copy().set_values({"Linesearch":"backtrack"}))
                #self.optimizer[self.TSnode].options['DMAX'] /= self.newclimbscale

                # overwrite this here just in case TSnode changed wont cause slow down climb  
                self.pTSnode = self.TSnode

            elif (self.climb and not self.find and self.finder and self.nclimb<1 and
                    ((totalgrad<0.2 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.01) or #
                    (totalgrad<0.1 and ts_gradrms<self.options['CONV_TOL']*10. and ts_cgradq<0.02) or  #
                    (sumgradrms< sum_conv_tol) or
                    (ts_gradrms<self.options['CONV_TOL']*5.)  #  used to be 5
                    )) and self.dE_iter<1. :
                print(" ** starting exact climb **")
                print(" totalgrad %5.4f gradrms: %5.4f gts: %5.4f" %(totalgrad,ts_gradrms,ts_cgradq))
                self.find=True
                form_TS_hess=True
                self.optimizer[self.TSnode] = eigenvector_follow(self.optimizer[self.TSnode].options.copy())
                print(type(self.optimizer[self.TSnode]))
                self.optimizer[self.TSnode].options['SCALEQN'] = 1.
                self.nhessreset=10  # are these used??? TODO 
                self.hessrcount=0   # are these used?!  TODO
            if self.climb: 
                self.nclimb-=1

            #for n in range(1,self.nnodes-1):
            #    self.active[n]=True
            #    self.optimizer[n].options['OPTTHRESH']=self.options['CONV_TOL']*2
            self.nhessreset-=1

        return form_TS_hess


    def add_GSM_nodeR(self,newnodes=1):
        nifty.printcool("Adding reactant node")

        if self.current_nnodes+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")
        for i in range(newnodes):
            iR = self.nR-1
            iP = self.nnodes-self.nP
            iN = self.nR
            print(" adding node: %i between %i %i from %i" %(iN,iR,iP,iR))
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1./float(self.nnodes-self.current_nnodes+1)
            else:
                stepsize = 0.5

            self.nodes[self.nR] = add_node(
                    self.nodes[iR],
                    self.nodes[iP],
                    stepsize,
                    iN,
                    DQMAG_MAX = self.DQMAG_MAX,
                    DQMAG_MIN = self.DQMAG_MIN,
                    driving_coords = self.driving_coords,
                    )

            if self.nodes[self.nR]==None:
                raise Exception('Ran out of space')

            if self.__class__.__name__!="DE_GSM":
                ictan,bdist =  self.get_tangent(
                        self.nodes[self.nR],
                        None,
                        driving_coords=self.driving_coords,
                        )
                self.nodes[self.nR].bdist = bdist

            self.optimizer[self.nR].DMAX = self.optimizer[self.nR-1].DMAX
            self.current_nnodes+=1
            self.nR+=1
            print(" nn=%i,nR=%i" %(self.current_nnodes,self.nR))
            self.active[self.nR-1] = True

            # align center of mass  and rotation
            #print("%i %i %i" %(iR,iP,iN))

            #print(" Aligning")
            self.nodes[self.nR-1].xyz = self.com_rotate_move(iR,iP,iN)


    def add_GSM_nodeP(self,newnodes=1):
        nifty.printcool("Adding product node")
        if self.current_nnodes+newnodes > self.nnodes:
            raise ValueError("Adding too many nodes, cannot interpolate")

        for i in range(newnodes):
            #self.nodes[-self.nP-1] = BaseClass.add_node(self.nnodes-self.nP,self.nnodes-self.nP-1,self.nnodes-self.nP)
            n1=self.nnodes-self.nP
            n2=self.nnodes-self.nP-1
            n3=self.nR-1
            print(" adding node: %i between %i %i from %i" %(n2,n1,n3,n1))
            if self.nnodes - self.current_nnodes > 1:
                stepsize = 1./float(self.nnodes-self.current_nnodes+1)
            else:
                stepsize = 0.5

            self.nodes[-self.nP-1] = add_node(
                    self.nodes[n1],
                    self.nodes[n3],
                    stepsize,
                    n2
                    )
            if self.nodes[-self.nP-1]==None:
                raise Exception('Ran out of space')

            self.optimizer[n2].DMAX = self.optimizer[n1].DMAX
            self.current_nnodes+=1
            self.nP+=1
            print(" nn=%i,nP=%i" %(self.current_nnodes,self.nP))
            self.active[-self.nP] = True

            # align center of mass  and rotation
            #print("%i %i %i" %(n1,n3,n2))
            #print(" Aligning")
            self.nodes[-self.nP].xyz = self.com_rotate_move(n1,n3,n2)
            #print(" getting energy for node %d: %5.4f" %(self.nnodes-self.nP,self.nodes[-self.nP].energy - self.nodes[0].V0))
        return


    def reparameterize(self,ic_reparam_steps=8,n0=0,nconstraints=1,rtype=0):
        print(self.interp_method)
        if self.interp_method == 'DLC':
            print('reparameterizing')
            self.ic_reparam(nconstraints=nconstraints)
        elif self.interp_method == 'Geodesic':
             self.geodesic_reparam()
        return

    def geodesic_reparam(self):
        TSnode = self.TSnode
        print(TSnode)
        if self.climb or self.find:
            a  = geodesic_reparam( self.nodes[0:self.TSnode] )
            b = geodesic_reparam( self.nodes[self.TSnode:] )
            new_xyzs = np.vstack((a,b))
        else:
            new_xyzs =  geodesic_reparam(self.nodes) 

        print(new_xyzs)

        for i,xyz in enumerate(new_xyzs):
            self.nodes[i].xyz  = xyz


    def ic_reparam(self,ic_reparam_steps=8,n0=0,nconstraints=1,rtype=0):
        nifty.printcool("reparametrizing string nodes")
        ictalloc = self.nnodes+1
        rpmove = np.zeros(ictalloc)
        rpart = np.zeros(ictalloc)
        totaldqmag = 0.0
        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(ictalloc)
        edist = np.zeros(ictalloc)
    
        # align first and last nodes
        # assuming they are aligned
        #self.nodes[self.nnodes-1].xyz = self.com_rotate_move(0,self.nnodes-1,self.nnodes-2)

        #for n in range(1,self.nnodes-1):
        #    self.nodes[n].xyz = self.com_rotate_move(n-1,n+1,n)

        # stash energies
        energies = np.copy(self.energies)
        # do this or else it will recalculate energies every step!
        TSnode = self.TSnode

        for i in range(ic_reparam_steps):

            self.ictan,self.dqmaga = self.get_tangents(self.nodes)

            # copies of original ictan
            ictan0 = np.copy(self.ictan)
            ictan = np.copy(self.ictan)

            if self.print_level>0:
                print(" printing spacings dqmaga:")
                for n in range(1,self.nnodes):
                    print(" dq[%d] %1.4f" % (n,self.dqmaga[n]), end=' ') 
                print() 

            totaldqmag = 0.
            totaldqmag = np.sum(self.dqmaga[n0+1:self.nnodes])
            print(" totaldqmag = %1.3f" %totaldqmag)
            dqavg = totaldqmag/(self.nnodes-1)

            #if climb:
            if self.climb or rtype==2:
                h1dqmag = np.sum(self.dqmaga[1:TSnode+1])
                h2dqmag = np.sum(self.dqmaga[TSnode+1:self.nnodes])
                if self.print_level>0:
                    print(" h1dqmag, h2dqmag: %3.2f %3.2f" % (h1dqmag,h2dqmag))
           
            # => Using average <= #
            if i==0 and rtype==0:
                print(" using average")
                if not self.climb:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
                else:
                    for n in range(n0+1,TSnode):
                        rpart[n] = 1./(TSnode-n0)
                    for n in range(TSnode+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-TSnode-1)
                    rpart[TSnode]=0.

            if rtype==1 and i==0:
                dEmax = 0.
                for n in range(n0+1,self.nnodes):
                    dE[n] = abs(energies[n]-energies[n-1])
                dEmax = max(dE)
                for n in range(n0+1,self.nnodes):
                    edist[n] = dE[n]*self.dqmaga[n]

                print(" edist: ", end=' ')
                for n in range(n0+1,self.nnodes):
                    print(" {:1.1}".format(edist[n]), end=' ')
                print() 
                
                totaledq = np.sum(edist[n0+1:self.nnodes])
                edqavg = totaledq/(self.nnodes-1)

            if i==0:
                print(" rpart: ", end=' ')
                for n in range(1,self.nnodes):
                    print(" {:1.2}".format(rpart[n]), end=' ')
                print()

            if not self.climb and rtype!=2:
                for n in range(n0+1,self.nnodes-1):
                    deltadq = self.dqmaga[n] - totaldqmag * rpart[n]
                    #if n==self.nnodes-2:
                    #    deltadq += (totaldqmag * rpart[n] - self.dqmaga[n+1]) # this shifts the last node backwards
                    #    deltadq /= 2.
                    #print(deltadq)
                    rpmove[n] = -deltadq
            else:
                deltadq = 0.
                rpmove[TSnode] = 0.
                for n in range(n0+1,TSnode):
                    deltadq = self.dqmaga[n] - h1dqmag * rpart[n]
                    #if n==TSnode-1:
                    #    deltadq += h1dqmag * rpart[n] - self.dqmaga[n+1]
                    #    deltadq /= 2.
                    rpmove[n] = -deltadq
                for n in range(TSnode+1,self.nnodes-1):
                    deltadq = self.dqmaga[n] - h2dqmag * rpart[n]
                    #if n==self.nnodes-2:
                    #    deltadq += h2dqmag * rpart[n] - self.dqmaga[n+1]
                    #    deltadq /= 2.
                    rpmove[n] = -deltadq

            MAXRE = 0.5
            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>MAXRE:
                    rpmove[n] = np.sign(rpmove[n])*MAXRE
            # There was a really weird rpmove code here from GSM but
            # removed 7/1/2020
            if self.climb or rtype==2:
                rpmove[TSnode] = 0.

            disprms = np.linalg.norm(rpmove[n0+1:self.nnodes-1])/np.sqrt(len(rpmove[n0+1:self.nnodes-1]))
            lastdispr = disprms

            if self.print_level>0:
                for n in range(n0+1,self.nnodes-1):
                    print(" disp[{}]: {:1.2}".format(n,rpmove[n]), end=' ')
                print()
                print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 0.02:
                break

            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n])>0.:
                    #print "moving node %i %1.3f" % (n,rpmove[n])
                    self.newic.xyz = self.nodes[n].xyz.copy()

                    if rpmove[n] < 0.:
                        ictan[n] = np.copy(ictan0[n]) 
                    else:
                        ictan[n] = np.copy(ictan0[n+1]) 

                    # TODO
                    # it would speed things up a lot if we don't reform  coordinate basis
                    # perhaps check if similarity between old ictan and new ictan is close?
                    self.newic.update_coordinate_basis(ictan[n])

                    constraint = self.newic.constraints[:,0]

                    if n==TSnode and (self.climb or rtype==2):
                        pass
                    else:
                        dq = rpmove[n]*constraint
                        self.newic.update_xyz(dq,verbose=True)
                        self.nodes[n].xyz = self.newic.xyz.copy()

                    # new 6/7/2019
                    if self.nodes[n].newHess==0:
                        if not (n==TSnode and (self.climb or self.find)):
                            self.nodes[n].newHess=2

                #TODO might need to recalculate energy here for seam? 

        #for n in range(1,self.nnodes-1):
        #    self.nodes[n].xyz = self.com_rotate_move(n-1,n+1,n)

        print(' spacings (end ic_reparam, steps: {}/{}):'.format(i+1,ic_reparam_steps))
        for n in range(1,self.nnodes):
            print(" {:1.2}".format(self.dqmaga[n]), end=' ')
        print()
        print("  disprms: {:1.3}\n".format(disprms))

    def ic_reparam_g(self,ic_reparam_steps=4,n0=0,reparam_interior=True):  #see line 3863 of gstring.cpp
        """
        
        """
        nifty.printcool("Reparamerizing string nodes")
        #close_dist_fix(0) #done here in GString line 3427.
        rpmove = np.zeros(self.nnodes)
        rpart = np.zeros(self.nnodes)
        dqavg = 0.0
        disprms = 0.0
        h1dqmag = 0.0
        h2dqmag = 0.0
        dE = np.zeros(self.nnodes)
        edist = np.zeros(self.nnodes)
        emax = -1000 # And this?

        if self.current_nnodes==self.nnodes:
            self.ic_reparam(4)
            return

        for i in range(ic_reparam_steps):
            self.ictan,self.dqmaga = self.get_tangents_growing()
            totaldqmag = np.sum(self.dqmaga[n0:self.nR-1])+np.sum(self.dqmaga[self.nnodes-self.nP+1:self.nnodes])
            if self.print_level>0:
                if i==0:
                    print(" totaldqmag (without inner): {:1.2}\n".format(totaldqmag))
                print(" printing spacings dqmaga: ")
                for n in range(self.nnodes):
                    print(" {:2.3}".format(self.dqmaga[n]), end=' ')
                    if (n+1)%5==0:
                        print()
                print() 
            
            if i == 0:
                if self.current_nnodes!=self.nnodes:
                    rpart = np.zeros(self.nnodes)
                    for n in range(n0+1,self.nR):
                        rpart[n] = 1.0/(self.current_nnodes-2)
                    for n in range(self.nnodes-self.nP,self.nnodes-1):
                        rpart[n] = 1.0/(self.current_nnodes-2)
                else:
                    for n in range(n0+1,self.nnodes):
                        rpart[n] = 1./(self.nnodes-1)
                if self.print_level>0:
                    if i==0:
                        print(" rpart: ")
                        for n in range(1,self.nnodes-1):
                            print(" {:1.2}".format(rpart[n]), end=' ')
                            if (n)%5==0:
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
            for n in range(n0+1,nR0):
                deltadq = self.dqmaga[n-1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq
            for n in range(self.nnodes-nP0,self.nnodes-1):
                deltadq = self.dqmaga[n+1] - totaldqmag*rpart[n]
                rpmove[n] = -deltadq

            MAXRE = 1.1

            for n in range(n0+1,self.nnodes-1):
                if abs(rpmove[n]) > MAXRE:
                    rpmove[n] = float(np.sign(rpmove[n])*MAXRE)

            disprms = float(np.linalg.norm(rpmove[n0+1:self.nnodes-1]))
            lastdispr = disprms
            if self.print_level>0:
                for n in range(n0+1,self.nnodes-1):
                    print(" disp[{}]: {:1.2f}".format(n,rpmove[n]), end=' ')
                    if (n)%5==0:
                        print()
                print()
                print(" disprms: {:1.3}\n".format(disprms))

            if disprms < 1e-2:
                break

            ncurrent,nlist = self.make_nlist()
            param_list=[]
            for n in range(ncurrent):
                if nlist[2*n+1] not in param_list:
                    if rpmove[nlist[2*n+1]]>0:
                        # Using tangent pointing inner?
                        print('Moving {} along ictan[{}]'.format(nlist[2*n+1],nlist[2*n+1]))
                        self.nodes[nlist[2*n+1]].update_coordinate_basis(constraints=self.ictan[nlist[2*n+1]])
                        constraint = self.nodes[nlist[2*n+1]].constraints[:,0]
                        dq0 = rpmove[nlist[2*n+1]]*constraint
                        self.nodes[nlist[2*n+1]].update_xyz(dq0,verbose=True)
                        param_list.append(nlist[2*n+1])
                    else:
                        # Using tangent point outer
                        print('Moving {} along ictan[{}]'.format(nlist[2*n+1],nlist[2*n]))
                        self.nodes[nlist[2*n+1]].update_coordinate_basis(constraints=self.ictan[nlist[2*n]])
                        constraint = self.nodes[nlist[2*n+1]].constraints[:,0]
                        dq0 = rpmove[nlist[2*n+1]]*constraint
                        self.nodes[nlist[2*n+1]].update_xyz(dq0,verbose=True)
                        param_list.append(nlist[2*n+1])
        print(" spacings (end ic_reparam, steps: {}/{}):".format(i+1,ic_reparam_steps), end=' ')
        for n in range(self.nnodes):
            print(" {:1.2}".format(self.dqmaga[n]), end=' ')
        print("  disprms: {:1.3}".format(disprms))

        #TODO old GSM does this here
        #Failed = check_array(self.nnodes,self.dqmaga)
        #If failed, do exit 1


    # TODO Move this to a util 
    def get_eigenv_finite(self,en):
        ''' Modifies Hessian using RP direction'''
        print("modifying %i Hessian with RP" % en)
    
        # a variable to determine how many time since last modify
        self.hess_counter = 0
        self.TS_E_0 = self.energies[self.TSnode]

        E0 = self.energies[en]/units.KCAL_MOL_PER_AU
        Em1 = self.energies[en-1]/units.KCAL_MOL_PER_AU
        if en+1<self.nnodes:
            Ep1 = self.energies[en+1]/units.KCAL_MOL_PER_AU
        else:
            Ep1 = Em1

        # Update TS node coord basis
        Vecs = self.nodes[en].update_coordinate_basis(constraints=None)

        # get constrained coord basis
        self.newic.xyz = self.nodes[en].xyz.copy()
        const_vec = self.newic.update_coordinate_basis(constraints=self.ictan[en])
        q0 = self.newic.coordinates[0]
        constraint = self.newic.constraints[:,0]

        # this should just give back ictan[en]? 
        tan0 = block_matrix.dot(const_vec,constraint)

        # get qm1 (don't update basis)
        self.newic.xyz = self.nodes[en-1].xyz.copy()
        qm1 = self.newic.coordinates[0]

        if en+1<self.nnodes:
            # get qp1 (don't update basis)
            self.newic.xyz = self.nodes[en+1].xyz.copy()
            qp1 = self.newic.coordinates[0]
        else:
            qp1 = qm1

        if en == self.TSnode:
            print(" TS Hess init'd w/ existing Hintp")

        # Go to non-constrained basis
        self.newic.xyz = self.nodes[en].xyz.copy()
        self.newic.coord_basis = Vecs
        self.newic.Primitive_Hessian = self.nodes[en].Primitive_Hessian.copy()
        self.newic.form_Hessian_in_basis()

        tan = block_matrix.dot(block_matrix.transpose(Vecs),tan0)   # (nicd,1
        Ht = np.dot(self.newic.Hessian,tan)                         # (nicd,nicd)(nicd,1) = nicd,1
        tHt = np.dot(tan.T,Ht) 

        a = abs(q0-qm1)
        b = abs(qp1-q0)
        c = 2*(Em1/a/(a+b) - E0/a/b + Ep1/b/(a+b))
        print(" tHt %1.3f a: %1.1f b: %1.1f c: %1.3f" % (tHt,a[0],b[0],c[0]))

        ttt = np.outer(tan,tan)

        # Hint before
        #with np.printoptions(threshold=np.inf):
        #    print self.newic.Hessian
        #eig,tmph = np.linalg.eigh(self.newic.Hessian)
        #print "initial eigenvalues"
        #print eig
      
        # Finalize Hessian
        self.newic.Hessian += (c-tHt)*ttt
        self.nodes[en].Hessian = self.newic.Hessian.copy()

        # Hint after
        #with np.printoptions(threshold=np.inf):
        #    print self.nodes[en].Hessian
        #print "shape of Hessian is %s" % (np.shape(self.nodes[en].Hessian),)

        self.nodes[en].newHess = 5

        if False:
            print("newHess of node %i %i" % (en,self.nodes[en].newHess))
            eigen,tmph = np.linalg.eigh(self.nodes[en].Hessian) #nicd,nicd
            print("eigenvalues of new Hess")
            print(eigen)

        # reset pgradrms ? 


    def set_V0(self):
        raise NotImplementedError 


    def mult_steps(self,n,opt_steps):
        exsteps=1
        tsnode = int(self.TSnode)

        if (self.find or self.climb) and self.energies[n] > self.energies[self.TSnode]*0.9 and n!=tsnode:  #
            exsteps=2
            print(" multiplying steps for node %i by %i" % (n,exsteps))
            self.optimizer[n].conv_grms = self.options['CONV_TOL']      # TODO this is not perfect here
            self.optimizer[n].conv_gmax = self.options['CONV_gmax']
            self.optimizer[n].conv_Ediff = self.options['CONV_Ediff']
        if (self.find or (self.climb and self.energies[tsnode]>self.energies[tsnode-1]+5 and self.energies[tsnode]>self.energies[tsnode+1]+5.)) and n==tsnode: #or self.climb
            exsteps=2
            print(" multiplying steps for node %i by %i" % (n,exsteps))

        #elif not (self.find and self.climb) and self.energies[tsnode] > 1.75*self.energies[tsnode-1] and self.energies[tsnode] > 1.75*self.energies[tsnode+1] and self.done_growing and n==tsnode:  #or self.climb
        #    exsteps=2
        #    print(" multiplying steps for node %i by %i" % (n,exsteps))
        return exsteps*opt_steps


    def set_opt_type(self,n,quiet=False):
        #TODO error for seam climb
        opt_type='ICTAN' 
        if self.climb and n==self.TSnode and not self.find and self.nodes[n].PES.__class__.__name__!="Avg_PES":
            opt_type='CLIMB'
            #opt_type='BEALES_CG'
        elif self.find and n==self.TSnode:
            opt_type='TS'
        elif self.nodes[n].PES.__class__.__name__=="Avg_PES":
            opt_type='SEAM'
            if self.climb and n==self.TSnode:
                opt_type='TS-SEAM'
        if not quiet:
            print((" setting node %i opt_type to %s" %(n,opt_type)))

        if isinstance(self.optimizer[n],beales_cg) and opt_type!="BEALES_CG":
            raise RuntimeError("This shouldn't happen")

        return opt_type


    def set_finder(self,rtype):
        assert rtype in [0,1,2], "rtype not defined"
        print('')
        print("*********************************************************************")
        if rtype==2:
            print("****************** set climber and finder to True *******************")
            self.climber=True
            self.finder=True
        elif rtype==1:
            print("***************** setting climber to True*************************")
            self.climber=True
        else:
            print("******** Turning off climbing image and exact TS search **********")
        print("*********************************************************************")
 

    def restart_from_geoms(self,input_geoms,reparametrize=False,restart_energies=True):
        '''
        '''

        nifty.printcool("Restarting GSM from geometries")
        self.growth_direction=0
        nstructs=len(input_geoms)

        if nstructs != self.nnodes:
            print('need to interpolate')
            #if self.interp_method=="DLC":
            #    # determine how many times to upsample, then upsample that many times
            #    #self.upsample()
            #    raise NotImplementedError
            #elif self.interp_method=="Geodesic":
            old_xyzs = [ manage_xyz.xyz_to_np(geom) for geom in input_geoms ]
            symbols = manage_xyz.get_atoms(input_geoms[0])
            xyzs = redistribute(symbols,old_xyzs,self.nnodes,tol=2e-3*5)
            geoms = [ manage_xyz.np_to_xyz(input_geoms[0],xyz) for xyz in xyzs ]
            nstructs = len(geoms)
        else:
            geoms = input_geoms

        self.gradrms = [0.]*nstructs
        self.dE = [1000.]*nstructs

        self.isRestarted=True
        self.done_growing=True

        # set coordinates from geoms
        self.nodes[0].xyz = manage_xyz.xyz_to_np(geoms[0])
        self.nodes[nstructs-1].xyz = manage_xyz.xyz_to_np(geoms[-1])
        for struct in range(1,nstructs-1):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[struct-1],
                    manage_xyz.xyz_to_np(geoms[struct]),
                    new_node_id=struct,
                    copy_wavefunction=False)
            self.nodes[struct].newHess=5
            # Turning this off
            #self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
            #self.nodes[struct].gradrms=grmss[struct]
            #self.nodes[struct].PES.dE = dE[struct]
        self.nnodes=self.nR=nstructs

        if reparametrize:
            nifty.printcool("Reparametrizing")
            self.reparameterize(ic_reparam_steps=8)
            self.write_xyz_files('grown_string1_{:03}.xyz'.format(self.ID))

        if restart_energies:
            # initial energy
            self.nodes[0].V0 = self.nodes[0].energy 
            self.energies[0] = 0.
            print(" initial energy is %3.4f" % self.nodes[0].energy)

            for struct in range(1,nstructs-1):
                print(" energy of node %i is %5.4f" % (struct,self.nodes[struct].energy))
                self.energies[struct] = self.nodes[struct].energy - self.nodes[0].V0
                print(" Relative energy of node %i is %5.4f" % (struct,self.energies[struct]))

            print(" V_profile: ", end=' ')
            energies= self.energies
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(energies[n])), end=' ')
            print()

        print(" setting all interior nodes to active")
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.optimizer[n].conv_grms=self.options['CONV_TOL']*2.5
            self.optimizer[n].options['DMAX'] = 0.05


        return

    def restart_string(self,xyzfile='restart.xyz',rtype=2,reparametrize=False,restart_energies=True):
        nifty.printcool("Restarting string from file")
        self.growth_direction=0
        with open(xyzfile) as f:
            nlines = sum(1 for _ in f)
        #print "number of lines is ", nlines
        with open(xyzfile) as f:
            natoms = int(f.readlines()[2])

        #print "number of atoms is ",natoms
        nstructs = (nlines-6)/ (natoms+5) #this is for three blocks after GEOCON
        nstructs = int(nstructs)
        
        #print "number of structures in restart file is %i" % nstructs
        coords=[]
        grmss = []
        atomic_symbols=[]
        dE = []
        with open(xyzfile) as f:
            f.readline()
            f.readline() #header lines
            # get coords
            for struct in range(nstructs):
                tmpcoords=np.zeros((natoms,3))
                f.readline() #natoms
                f.readline() #space
                for a in range(natoms):
                    line=f.readline()
                    tmp = line.split()
                    tmpcoords[a,:] = [float(i) for i in tmp[1:]]
                    if struct==0:
                        atomic_symbols.append(tmp[0])
                coords.append(tmpcoords)
            ## Get energies
            #f.readline() # line
            #f.readline() #energy
            #for struct in range(nstructs):
            #    self.energies[struct] = float(f.readline())
            ## Get grms
            #f.readline() # max-force
            #for struct in range(nstructs):
            #    grmss.append(float(f.readline()))
            ## Get dE
            #f.readline()
            #for struct in range(nstructs):
            #    dE.append(float(f.readline()))


        # initialize lists
        self.gradrms = [0.]*nstructs
        self.dE = [1000.]*nstructs

        self.isRestarted=True
        self.done_growing=True
        # TODO

        # set coordinates from restart file
        self.nodes[0].xyz = coords[0].copy()
        self.nodes[nstructs-1].xyz = coords[nstructs-1].copy()
        for struct in range(1,nstructs):
            self.nodes[struct] = Molecule.copy_from_options(self.nodes[struct-1],coords[struct],new_node_id=struct,copy_wavefunction=False)
            self.nodes[struct].newHess=5
            # Turning this off
            #self.nodes[struct].gradrms = np.sqrt(np.dot(self.nodes[struct].gradient,self.nodes
            #self.nodes[struct].gradrms=grmss[struct]
            #self.nodes[struct].PES.dE = dE[struct]
        self.nnodes=self.nR=nstructs

        if reparametrize:
            nifty.printcool("Reparametrizing")
            self.reparameterize(ic_reparam_steps=8)
            self.write_xyz_files('grown_string1_{:03}.xyz'.format(self.ID))

        if restart_energies:
            # initial energy
            self.nodes[0].V0 = self.nodes[0].energy 
            self.energies[0] = 0.
            print(" initial energy is %3.4f" % self.nodes[0].energy)

            for struct in range(1,nstructs-1):
                print(" energy of node %i is %5.4f" % (struct,self.nodes[struct].energy))
                self.energies[struct] = self.nodes[struct].energy - self.nodes[0].V0
                print(" Relative energy of node %i is %5.4f" % (struct,self.energies[struct]))

            print(" V_profile: ", end=' ')
            energies= self.energies
            for n in range(self.nnodes):
                print(" {:7.3f}".format(float(energies[n])), end=' ')
            print()

            #print(" grms_profile: ", end=' ')
            #for n in range(self.nnodes):
            #    print(" {:7.3f}".format(float(self.nodes[n].gradrms)), end=' ')
            #print()
            #print(" dE_profile: ", end=' ')
            #for n in range(self.nnodes):
            #    print(" {:7.3f}".format(float(self.nodes[n].difference_energy)), end=' ')
            #print()

        print(" setting all interior nodes to active")
        for n in range(1,self.nnodes-1):
            self.active[n]=True
            self.optimizer[n].conv_grms=self.options['CONV_TOL']*2.5
            self.optimizer[n].options['DMAX'] = 0.05


        if self.__class__.__name__!="SE_Cross":
            self.set_finder(rtype)


            if restart_energies:
                self.ictan,self.dqmaga = self.get_three_way_tangents(self.nodes,self.energies)
                num_coords =  self.nodes[0].num_coordinates - 1

                # project out the constraint
                for n in range(0,self.nnodes):
                    gc=self.nodes[n].gradient.copy()
                    for c in self.nodes[n].constraints.T:
                        gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
                    self.nodes[n].gradrms = np.sqrt(np.dot(gc.T,gc)/num_coords)

                fp = self.find_peaks(2)
                totalgrad,gradrms,sumgradrms = self.calc_grad()
                print(" totalgrad: {:4.3} gradrms: {:5.4} max E({}) {:5.4}".format(float(totalgrad),float(gradrms),self.TSnode,float(self.emax)))

                ts_cgradq = np.linalg.norm(np.dot(self.nodes[self.TSnode].gradient.T,self.nodes[self.TSnode].constraints[:,0])*self.nodes[self.TSnode].constraints[:,0])
                print(" ts_cgradq %5.4f" % ts_cgradq)
                ts_gradrms=self.nodes[self.TSnode].gradrms

                self.set_stage(totalgrad,sumgradrms,ts_cgradq,ts_gradrms,fp)
            else:
                return



    def com_rotate_move(self,iR,iP,iN):
        print(" aligning com and to Eckart Condition")

        mfrac = 0.5
        if self.nnodes - self.current_nnodes+1  != 1:
            mfrac = 1./(self.nnodes - self.current_nnodes+1)

        #if self.__class__.__name__ != "DE_GSM":
        #    # no "product" structure exists, use initial structure
        #    iP = 0

        xyz0 = self.nodes[iR].xyz.copy()
        xyz1 = self.nodes[iN].xyz.copy()
        com0 = self.nodes[iR].center_of_mass
        com1 = self.nodes[iN].center_of_mass
        masses = self.nodes[iR].mass_amu

        # From the old GSM code doesn't work
        #com1 = mfrac*(com2-com0)
        #print("com1")
        #print(com1)
        ## align centers of mass
        #xyz1 += com1
        #Eckart_align(xyz1,xyz2,masses,mfrac)

        # rotate to be in maximal coincidence with 0
        # assumes iP i.e. 2 is also in maximal coincidence
        U = rotate.get_rot(xyz0,xyz1)
        xyz1 = np.dot(xyz1,U)

        ## align 
        #if self.nodes[iP] != None:
        #    xyz2 = self.nodes[iP].xyz.copy()
        #    com2 = self.nodes[iP].center_of_mass

        #    if abs(iN-iR) > abs(iN-iP):
        #        avg_com = mfrac*com2 + (1.-mfrac)*com0
        #    else:
        #        avg_com = mfrac*com0 + (1.-mfrac)*com2
        #    dist = avg_com - com1  #final minus initial
        #else:
        #    dist = com0 - com1  #final minus initial

        #print("aligning to com")
        #print(dist)
        #xyz1 += dist




        return xyz1

    # TODO move to string utils or delete altogether
    def get_current_rotation(self,frag,a1,a2):
        '''
        calculate current rotation for single-ended nodes
        '''
    
        # Get the information on fragment to rotate
        sa,ea,sp,ep = self.nodes[0].coord_obj.Prims.prim_only_block_info[frag]
    
        theta = 0.
        # Haven't added any nodes yet
        if self.nR==1:
            return theta
   
        for n in range(1,self.nR):
            xyz_frag = self.nodes[n].xyz[sa:ea].copy()
            axis = self.nodes[n].xyz[a2] - self.nodes[n].xyz[a1]
            axis /= np.linalg.norm(axis)
    
            # only want the fragment of interest
            reference_xyz = self.nodes[n-1].xyz.copy()

            # Turn off
            ref_axis = reference_xyz[a2] - reference_xyz[a1]
            ref_axis /= np.linalg.norm(ref_axis)
   
            # ALIGN previous and current node to get rotation around axis of rotation
            #print(' Rotating reference axis to current axis')
            I = np.eye(3)
            v = np.cross(ref_axis,axis)
            if v.all()==0.:
                print('Rotation is identity')
                R=I
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                c = np.dot(ref_axis,axis)
                s = np.linalg.norm(v)
                R = I + vx + np.dot(vx,vx) * (1. - c)/(s**2)
            new_ref_axis = np.dot(ref_axis,R.T)
            #print(' overlap of ref-axis and axis (should be 1.) %1.2f' % np.dot(new_ref_axis,axis))
            new_ref_xyz = np.dot(reference_xyz,R.T)

            
            # Calculate dtheta 
            ca = self.nodes[n].primitive_internal_coordinates[sp+3]
            cb = self.nodes[n].primitive_internal_coordinates[sp+4]
            cc = self.nodes[n].primitive_internal_coordinates[sp+5]
            dv12_a = ca.calcDiff(self.nodes[n].xyz,new_ref_xyz)
            dv12_b = cb.calcDiff(self.nodes[n].xyz,new_ref_xyz)
            dv12_c = cc.calcDiff(self.nodes[n].xyz,new_ref_xyz)
            dv12 = np.array([dv12_a,dv12_b,dv12_c])
            #print(dv12)
            dtheta = np.linalg.norm(dv12)  #?
        
            dtheta = dtheta + np.pi % (2*np.pi) - np.pi
            theta += dtheta
   
        theta = theta/ca.w
        angle = theta * 180./np.pi
        print(angle) 

        return theta

    #TODO Move to manage_xyz
    def write_xyz_files(self,filename):
        #xyzfile = os.getcwd()+'/scratch/'+base+'_{:03}_{:03}.xyz'.format(self.ID,iters)
        geoms = []
        for ico in self.nodes:
            if ico != None:
                geoms.append(ico.geometry)

        with open(filename,'w') as f:
            f.write("[Molden Format]\n[Geometries] (XYZ)\n")
            for geom in geoms:
                f.write('%d\n\n' % len(geom))
                for atom in geom:
                    f.write('%-2s %14.6f %14.6f %14.6f\n' % (
                        atom[0],
                        atom[1],
                        atom[2],
                        atom[3],
                        ))
            f.write("[GEOCONV]\n")
            f.write('energy\n')
            V0=self.nodes[0].energy
            for n,ico in enumerate(self.nodes):
                if ico!=None:
                    f.write('{}\n'.format(ico.energy-V0))
                    #f.write('{}\n'.format(self.energies[n]))
            f.write("max-force\n")
            for ico in self.nodes:
                if ico != None:
                    f.write('{}\n'.format(float(ico.gradrms)))
            #print(" WARNING: Printing dE as max-step in molden output ")
            f.write("max-step\n")
            for ico,act in zip(self.nodes,self.active):
                if ico!=None:
                    f.write('{}\n'.format(float(ico.difference_energy)))
        f.close()

    #TODO move to string_utils
    def find_peaks(self,rtype):
        #rtype 1: growing
        #rtype 2: opting
        #rtype 3: intermediate check
        if rtype==1:
            nnodes=self.nR
        elif rtype==2 or rtype==3:
            nnodes=self.nnodes
        else:
            raise ValueError("find peaks bad input")
        #if rtype==1 or rtype==2:
        #    print "Energy"
        alluptol=0.1
        alluptol2=0.5
        allup=True
        diss=False
        energies = self.energies
        for n in range(1,len(energies[:nnodes])):
            if energies[n]+alluptol<energies[n-1]:
                allup=False
                break

        if energies[nnodes-1]>15.0:
            if nnodes-3>0:
                if ((energies[nnodes-1]-energies[nnodes-2])<alluptol2 and 
                (energies[nnodes-2]-energies[nnodes-3])<alluptol2 and
                (energies[nnodes-3]-energies[nnodes-4])<alluptol2):
                    print(" possible dissociative profile")
                    diss=True

        print(" nnodes ",nnodes)  
        print(" all uphill? ",allup)
        print(" dissociative? ",diss)
        npeaks1=0
        npeaks2=0
        minnodes=[]
        maxnodes=[]
        if energies[1]>energies[0]:
            minnodes.append(0)
        if energies[nnodes-1]<energies[nnodes-2]:
            minnodes.append(nnodes-1)
        for n in range(self.n0,nnodes-1):
            if energies[n+1]>energies[n]:
                if energies[n]<energies[n-1]:
                    minnodes.append(n)
            if energies[n+1]<energies[n]:
                if energies[n]>energies[n-1]:
                    maxnodes.append(n)

        print(" min nodes ",minnodes)
        print(" max nodes ", maxnodes)
        npeaks1 = len(maxnodes)
        #print "number of peaks is ",npeaks1
        ediff=0.5
        PEAK4_EDIFF = 2.0
        if rtype==1:
            ediff=1.
        if rtype==3:
            ediff=PEAK4_EDIFF

        if rtype==1:
            nmax = np.argmax(energies[:self.nR])
            emax = float(max(energies[:self.nR]))
        else:
            emax = float(max(energies))
            nmax = np.argmax(energies)

        print(" emax and nmax in find peaks %3.4f,%i " % (emax,nmax))

        #check if any node after peak is less than 2 kcal below
        for n in maxnodes:
            diffs=( energies[n]-e>ediff for e in energies[n:nnodes])
            if any(diffs):
                found=n
                npeaks2+=1
        npeaks = npeaks2
        print(" found %i significant peak(s) TOL %3.2f" %(npeaks,ediff))

        #handle dissociative case
        if rtype==3 and npeaks==1:
            nextmin=0
            for n in range(found,nnodes-1):
                if n in minnodes:
                    nextmin=n
                    break
            if nextmin>0:
                npeaks=2

        #if rtype==3:
        #    return nmax
        if allup==True and npeaks==0:
            return -1
        if diss==True and npeaks==0:
            return -2

        return npeaks


    #TODO move to 
    def check_for_reaction_g(self,rtype):

        c = Counter(elem[0] for elem in self.driving_coords)
        nadds = c['ADD']
        nbreaks = c['BREAK']
        isrxn=False

        if (nadds+nbreaks) <1:
            return False
        nadded=0
        nbroken=0 
        nnR = self.nR-1
        xyz = self.nodes[nnR].xyz
        atoms = self.nodes[nnR].atoms

        for i in self.driving_coords:
            if "ADD" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0],index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d<d0:
                    nadded+=1
            if "BREAK" in i:
                index = [i[1]-1, i[2]-1]
                bond = Distance(index[0],index[1])
                d = bond.value(xyz)
                d0 = (atoms[index[0]].vdw_radius + atoms[index[1]].vdw_radius)/2
                if d>d0:
                    nbroken+=1
        if rtype==1:
            if (nadded+nbroken)>=(nadds+nbreaks): 
                isrxn=True
                #isrxn=nadded+nbroken
        else:
            isrxn=True
            #isrxn=nadded+nbroken
        print(" check_for_reaction_g isrxn: %r nadd+nbrk: %i" %(isrxn,nadds+nbreaks))
        return isrxn

    def check_for_reaction(self):
        isrxn = self.check_for_reaction_g(1)
        minnodes=[]
        maxnodes=[]
        wint=0
        energies = self.energies
        if energies[1]>energies[0]:
            minnodes.append(0)
        if energies[self.nnodes-1]<energies[self.nnodes-2]:
            minnodes.append(self.nnodes-1)
        for n in range(self.n0,self.nnodes-1):
            if energies[n+1]>energies[n]:
                if energies[n]<energies[n-1]:
                    minnodes.append(n)
            if energies[n+1]<energies[n]:
                if energies[n]>energies[n-1]:
                    maxnodes.append(n)
        if len(minnodes)>2 and len(maxnodes)>1:
            wint=minnodes[1] # the real reaction ends at first minimum
            print(" wint ", wint)

        return isrxn,wint


    def calc_grad(self):
        totalgrad = 0.0
        gradrms = 0.0
        sum_gradrms = 0.0
        for i,ico in zip(list(range(1,self.nnodes-1)),self.nodes[1:self.nnodes-1]):
            if ico!=None:
                print(" node: {:02d} gradrms: {:.6f}".format(i,float(ico.gradrms)),end='')
                if i%5 == 0:
                    print()
                totalgrad += ico.gradrms*self.rn3m6
                gradrms += ico.gradrms*ico.gradrms
                sum_gradrms += ico.gradrms
        print('')
        #TODO wrong for growth
        gradrms = np.sqrt(gradrms/(self.nnodes-2))
        return totalgrad,gradrms,sum_gradrms



