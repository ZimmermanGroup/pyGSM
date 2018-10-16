import options
import numpy as np
import os
import pybel as pb
import icoords

class BaseGSM(object):
    
    @staticmethod
    def default_options():
        if hasattr(BaseGSM, '_default_options'): return BaseGSM._default_options.copy()

        opt = options.Options() 
        
        opt.add_option(
            key='ICoord1',
            required=True,
            allowed_types=[icoords.ICoord],
            doc='')

        opt.add_option(
            key='ICoord2',
            required=False,
            allowed_types=[icoords.ICoord],
            doc='')

        opt.add_option(
            key='nnodes',
            required=False,
            value=0,
            allowed_types=[int],
            doc='number of string nodes')
        
        opt.add_option(
            key='isSSM',
            required=False,
            value=False,
            allowed_types=[bool],
            doc='specify SSM or DSM')

        opt.add_option(
            key='isomers',
            required=False,
            allowed_types=[list],
            doc='Provide a list of tuples to select coordinates to modify')

        BaseGSM._default_options = opt
        return BaseGSM._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return BaseGSM(BaseGSM.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes

        #TODO What is optCG Ask Paul
        self.optCG = False
        self.isTSnode =False
        self.nnodes = self.options['nnodes']
        self.icoords = [0]*self.nnodes
        self.icoords[0] = self.options['ICoord1']
        self.icoords[-1] = self.options['ICoord2']
        self.nn = 2
        self.nR = 1
        self.nP = 1        
        self.isSSM = self.options['isSSM']
        self.active = [-1] * self.nnodes
        self.active[0] = -2
        self.active[-1] = -2


    @staticmethod
    def tangent_1(ICoord1,ICoord2):
        ictan = []
        for bond1,bond2 in zip(ICoord1.bondd,ICoord2.bondd):
            ictan.append(bond1 - bond2)
        for angle1,angle2 in zip(ICoord1.anglev,ICoord2.anglev):
            ictan.append(angle1-angle2)
        for torsion1,torsion2 in zip(ICoord1.torv,ICoord2.torv):
            temptorsion = (torsion1-torsion2)*np.pi/180.0
            if temptorsion > np.pi:
                ictan.append(-1*((2*np.pi) - temptorsion))
            elif temptorsion < -np.pi:
                ictan.append((2*np.pi)+temptorsion)
            else:
                ictan.append(temptorsion)
        return ictan
            
    @staticmethod
    def tangent_1b(ICoord1,ICooord2):
        ictan = []
        bdist = 0.


        pass

    def add_node(self,n1,n2,n3):
        
        if self.isSSM:
            raise Exception("Cannot use add_node with SSM. Use add_node_SSM")

        print "Adding Node {} between Nodes {} and {}".format(n2,n1,n3)
        if n1 == n2 or n1 == n3 or n2 == n3:
            print "Cannot add node {} between {} and {}".format(n2,n1,n3)
            raise ValueError("n1==n2, or n1==n3, or n2==n3")

        BDISTMIN = 0.01
        bdist = 0.
        

        newic = self.icoords[n1]
        intic = self.icoords[n3]
        dq0 = [0.] * newic.nicd
        ictan = BaseGSM.tangent_1(newic,intic)

        dqmag = 0.0
        
        newic.bmatp_create()
        newic.bmatp_to_U()
        ictan0 = ictan[:]
        
        if not self.isSSM:
            dqmag += np.dot(ictan0,newic.Ut[-1])

        print " dqmag: %1.3f"%dqmag

        newic.bmat_create()

        if not self.isSSM:
            if self.nnodes-self.nn != 1:
                dq0[-1] = -dqmag/float(self.nnodes-self.nn)
            else:
                dq0[-1] = -dqmag/2.0;
        
        elif self.isSSM:
            dq0[-1] = -dqmag
        
        print " dq0[-1]: %1.3f \n"%dq0[-1]
        
        newic.ic_to_xyz(dq0)

        newic.update_ics()
        
        self.icoords[n2] = newic
        
        #TODO com_rotate_move(n1,n3,n2,1.0)
        
        self.icoords[n2].bmatp_create()
        self.icoords[n2].bmatp_to_U()
        self.icoords[n2].bmat_create()

        if not self.isSSM:
            self.icoords[n2].make_Hint()
            self.icoords[n2].newHess = 5
        
        self.active[n2] = 1;

        success = True
        
        self.nn += 1

        return success


    def add_node_SSM(self,n1,n2):
        pass

