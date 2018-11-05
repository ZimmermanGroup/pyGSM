import numpy as np
import openbabel as ob
import pybel as pb
import options
import elements 
import os
from units import *
import itertools
from copy import deepcopy
import manage_xyz
from _icoord import ICoords
from _bmat import Bmat
from _obutils import Utils

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)


class Hybrid_DLC(Base_DLC,Utils): # write new mixins _Hyb_ICoords for hybrid water,_Hyb_Bmat,
    """
    Hybrid DLC for systems containing a large amount of atoms, the coordinates are partitioned 
    into a QM-region which is simulated with ICs, and a MM-region which is modeled with Cartesians. 
    """
    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Hybrid_DLC(Hybrid_DLC.default_options().set_values(kwargs))


    def setup(self):
        raise NotImplementedError()

    def ic_create(self):
        raise NotImplementedError()

    def update_ics(self):
        raise NotImplementedError()

    def linear_ties(self):
        raise NotImplementedError()

    def bond_frags(self):
        raise NotImplementedError()

    def update_residue(self):
        pass

