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


def golden_section(n, x, fx, g, d, step, xp, gp,constraint_step, parameters,molecule):

    z = (1 + np.sqrt(5))/2
    x1 = x.copy()
    x4 = x + d*step
    x2 = x4 - (x4-x1)/z
    x3 = x1 + (x4-x1)/z

    f1 = molecule.PES.get_energy(x1)
    f2 = molecule.PES.get_energy(x2)
    f3 = molecule.PES.get_energy(x3)
    f4 = molecule.PES.get_energy(x4)

    #check that a min exists
    min_exists=False
    if f2>f1 and f2>f4 and f3>f1 and f3>f4:
        min_exists=True
    
    if not min_exists:
        print(" no minimum exists")
        return {'status':1,'fx':f1,'step':step*0.,'x':x1, 'g':g}

    accuracy = 1.0e-3
    count=0
    while x4-x1>accuracy:
        if f2<f3:
            x4,f4 = x3,f3
            x3,f3 = x2,f2
            x2 = x4-(x4-x1)/z
            f2 = molecule.PES.get_energy(x2)
        else:
            x1,f1 = x2,f2
            x2,f2 = x3,f3
            x3 = x1 + (x4-x1)/z
            f3 = molecule.PES.get_energy(x3)

        count+=1
        if count>10:
            break

    x = 0.5*(x1+x4)
    fx = molecule.PES.get_energy(x)
    g = molecule.PES.get_gradient(x)
    step = x - x1
    result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}
    
    return

def secant_method(nconstraints, x, fx, gc, d, step, xp, gp,constraint_step, parameters,molecule):
    raise NotImplementedError

def steepest_ascent(nconstraints, x, fx, g, d, step, xp, gp,constraint_step, parameters,molecule):
    '''
    along the direction d
    x_new = x + step*d

    find the step that maximizes f(x)
    such that f'(x_new) normal to d is zero.
    np.dot(d.T,g_new) = 0


    x_new = x + gamma*g(x)

    gamma = abs( np.dot(dx.T,df))/abs(
    '''

    while True:

        # the gradient orthogonal to d
        gc = g - np.dot(g.T,d)*d
        step = np.linalg.norm(gc)

        # store 
        xp = x.copy()
        gp = g.copy()
        fxp = fx
       
        ls = backtrack(0, x, fx, gc, d, step, xp, gp,constraint_steps,parameters,molecule)

        # get values from linesearch
        p_step = step
        step = ls['step']
        x = ls['x']
        fx = ls['fx']
        g  = ls['g']

        #print(" [INFO]end line evaluate fx = %5.4f step = %1.2f." %(fx, step))
        x = x + d * step
        #xyz = molecule.coord_obj.newCartesian(molecule.xyz, x-xp,verbose=False)
        xyz = moleule.update_xyz(x-xp)

        # check for convergence TODO
        gradrms = np.sqrt(np.dot(g.T,g)/n)
        if gradrms < self.conv_grms:
            break

    result = {'status':0,'fx':fx,'step':step,'x':x, 'g':gc}



