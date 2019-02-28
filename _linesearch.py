import numpy as np 

def backtrack(n, x, fx, g, d, step, xp, gp,constraint_step, parameters,proc_evaluate):

    # n is the non-constrained
    # need to do something if n!= len(x) e.g. because of constraints

    count = 0
    dec = 0.5
    inc = 2.1
    #result = {'status':0,'fx':fx,'step':step,'x':x, 'g':g}
    result = {'status':0, 'proc_results': None, 'step':step, 'x':x}

    # Compute the initial gradient in the search direction.
    dginit = np.dot(g[:n].T, d[:n])
    # Make sure that s points to a descent direction.
    if 0 < dginit:
    	print '[ERROR] not descent direction'
    	result['status'] = -1
    	return result
    # The initial value of the objective function. 
    finit = fx
    dgtest = parameters.ftol * dginit
    
    while True:
        x = xp
        x = x + d * step  + constraint_step # n goes up to constraint (hopefully)

        # Evaluate the function and gradient values. 
        proc_results = proc_evaluate(x,n)
        fx=proc_results['fx']
        g = proc_results['g']

        #print " [INFO] end line evaluate fx = %r step = %r." %(fx, step)
        count = count + 1
        # check the sufficient decrease condition (Armijo condition).
        if fx > finit + (step * dgtest) and np.all(constraint_step==0):  #+ np.dot(g.T,constraint_step): # doesn't work with constraint :(
            print " [INFO] not satisfy sufficient decrease condition."
            width = dec
        else:
            # check the wolfe condition
            # now g is the gradient of f(xk + step * d)
            dg = np.dot(g[:n].T, d[:n])
            if dg < parameters.wolfe * dginit:
                print " [INFO] dg = %r < parameters.wolfe * dginit = %r" %(dg, parameters.wolfe * dginit)
                print " [INFO] not satisfy wolf condition."
                width = inc
            else:
                # check the strong wolfe condition
                if dg > -parameters.wolfe * dginit:
                	print " [INFO] not satisfy strong wolf condition."
                	width = dec
                else:
                    #result = {'status':0, 'fx':fx, 'step':step, 'x':x,'g':g}
                    result = {'status':0, 'proc_results': proc_results, 'step':step, 'x':x}
                    return result
        if step < parameters.min_step:
            result['status'] = -1
            print ' [ERROR] the linesearch step is too small'
            return result
        if step > parameters.max_step:
            result['status'] = -1
            print ' [ERROR] the linesearch step is too large'
            return result
        if parameters.max_linesearch <= count:
            print ' [INFO] the iteration of linesearch is many'
            #result = {'status':0, 'fx':fx, 'step':step, 'x':x,'g':g}
            result = {'status':0, 'proc_results': proc_results, 'step':step, 'x':x}
            return result	

        # update the step		
        step = step * width


