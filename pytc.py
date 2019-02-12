import lightspeed as ls
import psiw
from base_lot import * 
import numpy as np
import manage_xyz as mx
from units import *
from collections import Counter
from rhf_lot import *
from contextlib import contextmanager
import sys

@contextmanager
def custom_redirection(fileobj):
    old = sys.stdout
    sys.stdout = fileobj
    try:
        yield fileobj
    finally:
        sys.stdout = old


class PyTC(Lot):
    """
    Level of theory is a wrapper object to do DFT and CASCI calculations 
    Inherits from Lot. Requires a PSIW object
    """

    def get_energy(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getE(state,multiplicity)

    def getE(self,state,multiplicity):
        tmp = self.search_tuple(self.E,multiplicity)
        return tmp[state][1]*KCAL_MOL_PER_AU

    def run(self,geom):
        assert self.hasRanForCurrentCoords==False,"don't run"
        self.E=[]
        self.grada=[]
        #normal update
        coords = mx.xyz_to_np(geom)
        T = ls.Tensor.array(coords*ANGSTROM_TO_AU)

        with open('psiw_jobs.txt','a') as out:
            with custom_redirection(out):
                self.psiw = self.psiw.update_xyz(T)

                #filename="{}_rhf_update.molden".format(self.node_id)
                #self.psiw.casci.reference.save_molden_file(filename)

                for state in self.states:
                    multiplicity=state[0]
                    ad_idx=state[1]
                    S=multiplicity-1
                    if self.psiw.__class__.__name__=="CASCI_LOT":
                        self.E.append((multiplicity,self.psiw.compute_energy(S=S,index=ad_idx)))
                        tmp = self.psiw.compute_gradient(S=S,index=ad_idx)
                    elif self.psiw.__class__.__name__=="RHF_LOT": 
                        self.E.append((multiplicity,self.psiw.compute_energy()))
                        tmp = self.psiw.compute_gradient()
                    self.grada.append((multiplicity,tmp[...]))
                if self.do_coupling==True:
                    state1=self.states[0][1]
                    state2=self.states[1][1]
                    tmp = self.psiw.compute_coupling(S=S,indexA=state1,indexB=state2)
                    self.coup = tmp[...]

        self.hasRanForCurrentCoords=True
        print "setting has Ran to true"

        return

    def getgrad(self,state,multiplicity):
        tmp = self.search_tuple(self.grada,multiplicity)
        return np.asarray(tmp[state][1])*ANGSTROM_TO_AU

    def get_gradient(self,geom,multiplicity,state):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getgrad(state,multiplicity)

    def getcoup(self,state1,state2,multiplicity):
        #TODO this could be better
        return np.reshape(self.coup,(3*len(self.coup),1))*ANGSTROM_TO_AU

    def get_coupling(self,geom,multiplicity,state1,state2):
        if self.hasRanForCurrentCoords==False:
            self.run(geom)
        return self.getcoup(state1,state2,multiplicity)

    @staticmethod
    def copy(PyTCA,node_id):
        """ create a copy of this psiw object"""
        do_coupling = PyTCA.do_coupling
        obj = PyTC(PyTCA.options.copy().set_values({
            "node_id" :node_id,
            "do_coupling":do_coupling,
            }))

        if PyTCA.psiw.__class__.__name__=="CASCI_LOT":
            obj.psiw = psiw.CASCI_LOT(PyTCA.psiw.options.copy())
                    
        elif PyTCA.psiw.__class__.__name__=="RHF_LOT": 
            obj.psiw = RHF_LOT(PyTCA.psiw.options.copy())
                    
        return obj

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return PyTC(PyTC.default_options().set_values(kwargs))


if __name__ == '__main__':

    from pytc import *
    import manage_xyz

    if 1:
        nocc=11
        nactive=2

        psiw=PyTC.from_options(states=[(1,0),(1,1)],nocc=nocc,nactive=nactive,basis='6-31gs',do_coupling=True)
        x="tests/fluoroethene.xyz"
        #psiw.cas_from_file(x)
        #lot.casci_from_file_from_template(x,x,nocc,nocc) # hack to get it to work,need casci1

        geom=manage_xyz.read_xyz(x,scale=1)   
        e=lot.get_energy(geom,1,0)
        print e
        g=lot.get_gradient(geom,1,0)
        print g
        g=lot.get_gradient(geom,1,1)
        print g
        c=lot.get_coupling(geom,1,0,1)
        print c
