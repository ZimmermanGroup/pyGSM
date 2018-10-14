"""Class structures of important chemical concepts
"""
import copy
import numpy as np
#import fileio
import elements
import icoords as ic
import options
import pytc
import qchem
import manage_xyz


import pybel as pb


ELEMENT_TABLE = elements.ElementData()


class Molecule(object):
    """ wraps the lot and icoord object for optimizers"""

    @staticmethod
    def default_options():
        """ Molecule default options. """

        if hasattr(Molecule, '_default_options'): return Molecule._default_options.copy()
        opt = options.Options() 
        opt.add_option(
            key='filepath',
            value="initial0000.xyz",
            required=False,
            allowed_types=[str],
            doc='file containing geometry')

        opt.add_option(
                key='geom',
                required=False,
                allowed_types=[list],
                doc='geom ((natoms,4) np.ndarray) - system geometry (atom symbol, x,y,z)')

        opt.add_option(
                key='hess',
                required=False,
                allowed_types=[list],
                doc='hess:       2d array of Force constants')

        opt.add_option(
            key='calc_states',
            value=(0,0),
            required=True,
            allowed_types=[list],
            doc='')

        opt.add_option(
                key='functional',
                required=False,
                allowed_types=[str],
                doc='density functional')

        opt.add_option(
                key='nocc',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of occupied orbitals (for CAS)')

        opt.add_option(
                key='nactive',
                value=0,
                required=False,
                allowed_types=[int],
                doc='number of active orbitals (for CAS)')

        opt.add_option(
                key='basis',
                required=True,
                allowed_types=[str],
                doc='Basis set')

        opt.add_option(
                key='charge',
                value=0,
                required=False,
                allowed_types=[int],
                doc='charge of mol')

        opt.add_option(
                key='package',
                required=True,
                allowed_types=[str],
                doc='Electronic structure theory package')

        Molecule._default_options = opt
        return Molecule._default_options.copy()

    @staticmethod
    def from_options(**kwargs):
        """ Returns an instance of this class with default options updated from values in kwargs"""
        return Molecule(Molecule.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """Constructor """
        self.options = options

        # Cache some useful attributes
        self.filepath = self.options['filepath']
        self.geom=self.options['geom']
        self.calc_states=self.options['calc_states']
        self.geom = self.options['geom']
        self.nocc=self.options['nocc']
        self.nactive=self.options['nactive']
        self.basis=self.options['basis']
        self.functional=self.options['functional']
        self.charge=self.options['charge']
        self.package=self.options['package']
        self.functional=self.options['functional']

        #if self.geom is not None:
        self.geom=manage_xyz.read_xyz(self.filepath,scale=1)

        if self.package=="PyTC":
            self.lot=pytc.PyTC.from_options(calc_states=self.calc_states,geom=self.geom,nocc=self.nocc,nactive=self.nactive,basis=self.basis)
        elif self.package=="QChem":
            self.lot=qchem.QChem.from_options(calc_states=self.calc_states,geom=self.geom,basis=self.basis,functional=self.functional)
        else:
            raise NotImplementedError()

        mol=pb.readfile("xyz",self.filepath).next()
        self.ic=ic.ICoord.from_options(mol=mol)



if __name__ == '__main__':
    filepath="tests/fluoroethene.xyz"
    #geom=manage_xyz.read_xyz(filepath,scale=1)
    nocc=23
    nactive=2
    calc_states=[(0,0)]
    basis="6-31gs"
    molecule=Molecule.from_options(filepath=filepath,nocc=nocc,nactive=nactive,calc_states=calc_states,basis=basis,package='PyTC')
    molecule.lot.cas_from_geom()


