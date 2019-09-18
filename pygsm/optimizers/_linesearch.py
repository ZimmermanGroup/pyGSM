import sys
import numpy as np 
from utilities import manage_xyz

#TODO remove unecessary arguments: nconstraints, xp, ,...

def NoLineSearch(n, x, fx, g, d, step, xp, constraint_step, parameters,molecule):

    x = x + d * step  + constraint_step  # 
    xyz = molecule.coord_obj.newCartesian(molecule.xyz, x-xp,verbose=False)

    # use these so molecule xyz doesn't change
    print(" evaluate fx in linesearch")
    fx = molecule.PES.get_energy(xyz)
    gx = molecule.PES.get_gradient(xyz)
    g = molecule.coord_obj.calcGrad(xyz,gx)

    #print(" [INFO]end line evaluate fx = %5.4f step = %1.2f." %(fx, step))
    result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}
    return result



# TODO might be wise to add to backtrack a condition that says if 
# the number of iterations was many and the energy increased
# just return the initial point
def backtrack(nconstraints, x, fx, g, d, step, xp,constraint_step, parameters,molecule):
    print(" In backtrack")

    # n is the non-constrained
    count = 0
    dec = 0.5
    inc = 2.1

    # project out the constraint
    gc = g.copy()
    for c in molecule.constraints.T:
        gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]

    result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}

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
        gc = g.copy()
        for c in molecule.constraints.T:
            gc -= np.dot(gc.T,c[:,np.newaxis])*c[:,np.newaxis]
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
                    result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}
                    return result
        if step < parameters['min_step']:
            result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}
            print(' [INFO] the linesearch step is too small')
            return result
        if step > parameters['max_step']:
            print(' [INFO] the linesearch step is too large, returning with step {}'.format(step))
            result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}
            return result
        if parameters['max_linesearch'] <= count:
            print(' [INFO] the iteration of linesearch is many')
            result = {'status':0, 'fx':fx, 'g':g, 'step':step, 'x':x,'molecule':molecule}
            return result	

        # update the step		
        step = step * width

def double_golden_section(x,xyz1,xyz7,f1,f7,molecule):
    print("in")
    # read in xyz1,xyz7 and molecule with xyz set to xyz4
    x1 = molecule.coord_obj.calculate(xyz1)
    x7 = molecule.coord_obj.calculate(xyz7)
    x4 = molecule.coordinates.copy()  # initially the same as x
    x4 = x4.flatten()
    x = x.flatten()
    xyz4 = molecule.xyz.copy()
    f4 = molecule.energy
   
    # stash coordinates for 4
    xyz = molecule.xyz.copy()
    
    z = (1+np.sqrt(5))/2.
    
    # form x2,x3,x5,x6
    x2 = x4 - (x4-x1)/z
    x3 = x1 + (x4-x1)/z
    x5 = x7 - (x7-x4)/z
    x6 = x4 + (x7-x4)/z
    
    xyz2 = xyz4  - (xyz4 - xyz1)/z
    xyz3 = xyz1  + (xyz4 - xyz1)/z
    xyz5 = xyz7 - (xyz7 - xyz4)/z
    xyz6 = xyz4 + (xyz7 - xyz4)/z

    xyzs = [xyz1,xyz2,xyz3,xyz4,xyz5,xyz6,xyz7]
    geoms = [manage_xyz.combine_atom_xyz(molecule.atom_symbols,xyz) for xyz in xyzs ]
    manage_xyz.write_xyzs('test.xyz',geoms,scale=1.)
    
    sys.stdout.flush()
    
    f2 = molecule.PES.get_energy(xyz2)
    f3 = molecule.PES.get_energy(xyz3)
    f5 = molecule.PES.get_energy(xyz5)
    f6 = molecule.PES.get_energy(xyz6)
    print(" %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f %5.4f" % (f1,f2,f3,f4,f5,f6,f7))
    l = [f1, f2, f3, f4, f5, f6, f7 ]
    sys.stdout.flush()
    
    def ismax(l1,val):
        m = max(l1)
        return (True if m==val else False)
    
    left=False
    right=False
    center=False
    if ismax(l,f3):
        print(" 3")
        #xyz4,f4 = xyz3,f3
        xyz1,f1 = xyz2,f2
        left=True
    elif ismax(l,f5):
        print(" 5")
        #xyz4,f4 = xyz5,f5
        xyz7,f7 = xyz6,f6
        right=True
    elif ismax(l,f2):
        print(" 2")
        #xyz1,f1 = xyz2,f2
        xyz4,f4 = xyz3,f3
        left=True
    elif ismax(l,f6):
        print(" 6")
        #xyz7,f7 = xyz6,f6
        xyz4,f4 = xyz5,f5
        right=True
    elif ismax(l,f4):
        print(" 4")
        print(" Center")
        center=True
        #result = {'status':False,'fx':f4,'step': 0.*x,'xyz':xyz}
        #return result
    else:
        #something is wrong with TSnode
        print(" End point is TSnode")
        raise RuntimeError
    
    
    # rearrange right to be in canonical order
    if right:
        xyz1,f1 = xyz4,f4
        xyz2,f2 = xyz5,f5
        xyz3,f3 = xyz6,f6
        xyz4,f4 = xyz7,f7
    if center:
        xyz1,f1 = xyz3,f3
        xyz4,f4 = xyz5,f5
        xyz2 = xyz4  - (xyz4 - xyz1)/z
        xyz3 = xyz1  + (xyz4 - xyz1)/z
        f2 = molecule.PES.get_energy(xyz2)
        f3 = molecule.PES.get_energy(xyz3)
        print(" f1: %5.4f f2: %5.4f f3: %5.4f f4: %5.4f " % (f1,f2,f3,f4))

    
    TOLF = 0.1 # kcal/mol
    TOLC = 1.e-3 #
    print('entering while loop')
    sys.stdout.flush()
    count=0
    dxyz = np.linalg.norm(xyz2.flatten()-xyz3.flatten())
    while abs(f2-f3)>TOLF and dxyz>TOLC:
        if f2>f3:
            #x4,f4 = x3,f3
            xyz4,f4 = xyz3,f3
        else:
            xyz1,f1 = xyz2,f2
        #x2 = x4 - (x4 - x1)/z
        #x3 = x1 + (x4 - x1)/z
        #xyz2 = molecule.coord_obj.newCartesian(xyz,x2-x)
        #xyz3 = molecule.coord_obj.newCartesian(xyz,x3-x)
        xyz2 = xyz4 - (xyz4-xyz1)/z
        xyz3 = xyz1 + (xyz4-xyz1)/z
        f2 = molecule.PES.get_energy(xyz2)
        f3 = molecule.PES.get_energy(xyz3)
        print("f2: %5.4f f3: %5.4f" %(f2,f3))
        print(abs(f2-f3))
        dxyz = np.linalg.norm(xyz2.flatten()-xyz3.flatten())
        print(dxyz)
        sys.stdout.flush()

        count+=1
        if count>3:
            break
        
    #xnew = 0.5*(x1+x4)
    #xyznew = molecule.coord_obj.newCartesian(xyz,xnew-x)
    xyznew = 0.5*(xyz1+xyz4)
    fnew = molecule.PES.get_energy(xyznew)
    xnew = molecule.coord_obj.calculate(xyznew)
    print(" GS: %5.4f" % fnew)
    
    step = xnew - x
    result = {'status':True,'fx':fnew,'step':step,'xyz':xyznew}

    return result


def golden_section(x, g, d, step, molecule,maximize=False):

    z = (1 + np.sqrt(5))/2
    x1 = x.copy()
    x4 = x + d*step
    x2 = x4 - (x4-x1)/z
    x3 = x1 + (x4-x1)/z

    xyz1 = molecule.coord_obj.newCartesian(molecule.xyz, x1-x,verbose=False)
    xyz2 = molecule.coord_obj.newCartesian(molecule.xyz, x2-x,verbose=False)
    xyz3 = molecule.coord_obj.newCartesian(molecule.xyz, x3-x,verbose=False)
    xyz4 = molecule.coord_obj.newCartesian(molecule.xyz, x4-x,verbose=False)

    sign=1.
    if maximize:
        sign=-1.

    f1 = sign*molecule.PES.get_energy(xyz1)
    f2 = sign*molecule.PES.get_energy(xyz2)
    f3 = sign*molecule.PES.get_energy(xyz3)
    f4 = sign*molecule.PES.get_energy(xyz4)

    #check that a min exists
    min_exists=False
    if f2>f1 and f2>f4 and f3>f1 and f3>f4:
        min_exists=True
    
    if not min_exists:
        print(" no minimum exists")
        if f1<f4:
            print(" returning initial point")
            return {'status':0,'fx':sign*f1,'step':step*0.,'x':x1, 'g':g,'xyznew':xyz1}
        else:
            print(" returning endpoint")
            return {'status':0,'fx':sign*f4,'step':x4-x1,'x':x4, 'g':g,'xyznew':xyz4}

    accuracy = 1.0e-3
    count=0
    while x4-x1>accuracy:
        if f2<f3:
            x4,f4 = x3,f3
            x3,f3 = x2,f2
            x2 = x4-(x4-x1)/z
            xyz2 = molecule.coord_obj.newCartesian(molecule.xyz, x2-x,verbose=False)
            f2 = sign*molecule.PES.get_energy(xyz2)
        else:
            x1,f1 = x2,f2
            x2,f2 = x3,f3
            x3 = x1 + (x4-x1)/z
            xyz3 = molecule.coord_obj.newCartesian(molecule.xyz, x3-x,verbose=False)
            f3 = sign*molecule.PES.get_energy(xyz3)

        count+=1
        if count>10:
            break

    xnew = 0.5*(x1+x4)
    xyznew = molecule.coord_obj.newCartesian(molecule.xyz, xnew-x,verbose=False)
    fx = molecule.PES.get_energy(xyxznew)
    g = molecule.PES.get_gradient(xyznew)
    step = x - x1
    result = {'status':0,'fx':fx,'step':step,'x':xnew, 'g':g, 'xyznew':xyznew}
    
    return result

def secant_method(nconstraints, x, fx, gc, d, step, xp,constraint_step, parameters,molecule):
    raise NotImplementedError

def steepest_ascent(nconstraints, x, fx, g, d, step, xp,constraint_step, parameters,molecule):
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



