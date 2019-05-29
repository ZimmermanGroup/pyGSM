import numpy as np 

def NoLineSearch(n, x, fx, g, d, step, xp, gp,constraint_step, parameters,molecule):
   
    x = x + d * step  + constraint_step  # 
    xyz = molecule.coord_obj.newCartesian(molecule.xyz, x-xp,verbose=False)

    # use these so molecule xyz doesn't change
    fx = molecule.PES.get_energy(xyz)
    gx = molecule.PES.get_gradient(xyz)
    g = molecule.coord_obj.calcGrad(xyz,gx)

    #print(" [INFO]end line evaluate fx = %5.4f step = %1.2f." %(fx, step))
    result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x}
    return result

def backtrack(nconstraints, x, fx, g, d, step, xp, gp,constraint_step, parameters,molecule):

    # n is the non-constrained
    count = 0
    dec = 0.5
    inc = 2.1

    # project out the constraint
    gc = g - np.dot(g.T,molecule.constraints)*molecule.constraints

    result = {'status':0,'fx':fx,'step':step,'x':x, 'g':gc}

    # Compute the initial gradient in the search direction.
    dginit = np.dot(gc.T, d)

    # Make sure that s points to a descent direction.
    if 0 < dginit:
    	print('[ERROR] not descent direction')
    	result['status'] = -2
    	return result

    # The initial value of the objective function. 
    finit = fx
    dgtest = parameters['ftol'] * dginit
    
    while True:
        x = xp
        x = x + d * step  + constraint_step 
        xyz = molecule.coord_obj.newCartesian(molecule.xyz, x-xp,verbose=False)

        # Evaluate the function and gradient values. 
        # use these so molecule xyz doesn't change
        fx = molecule.PES.get_energy(xyz)
        gx = molecule.PES.get_gradient(xyz)
        g = molecule.coord_obj.calcGrad(xyz,gx)

        # project out the constraint
        gc = g - np.dot(g.T,molecule.constraints)*molecule.constraints
        #print(" [INFO]end line evaluate fx = %5.4f step = %1.2f." %(fx, step))

        count = count + 1
        # check the sufficient decrease condition (Armijo condition).
        if fx > finit + (step * dgtest) and np.all(constraint_step==0):  #+ np.dot(g.T,constraint_step): # doesn't work with constraint :(
            print(" [INFO] not satisfy sufficient decrease condition.")
            width = dec
            print(" step %1.2f" % (step*width))
        else:
            # check the wolfe condition
            # now g is the gradient of f(xk + step * d)
            dg = np.dot(gc.T, d)
            if dg < parameters['wolfe'] * dginit:
                print(" [INFO] dg = %r < parameters.wolfe * dginit = %r" %(dg, parameters['wolfe'] * dginit))
                print(" [INFO] not satisfy wolf condition.")
                width = inc
            else:
                # check the strong wolfe condition
                if dg > -parameters['wolfe'] * dginit:
                	print(" [INFO] not satisfy strong wolf condition.")
                	width = dec
                else:
                    result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}
                    return result
        if step < parameters['min_step']:
            result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}
            print(' [INFO] the linesearch step is too small')
            return result
        if step > parameters['max_step']:
            result = {'status':-1,'fx':fx,'step':step,'x':x, 'g':g}
            print(' [INFO] the linesearch step is too large')
            return result
        if parameters['max_linesearch'] <= count:
            print(' [INFO] the iteration of linesearch is many')
            result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}
            return result	

        # update the step		
        step = step * width


