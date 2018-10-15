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

    def add_node(self,ic1,ic2):
        pass

