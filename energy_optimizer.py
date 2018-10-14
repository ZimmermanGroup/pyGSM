import numpy as np
import options
import os
import molecule
#import grad

class EOpt(object):

    @staticmethod
    def default_options():
        if hasattr(EOpt, '_default_options'): return EOpt._default_options.copy()

        opt = options.Options() 

        opt.add_option(
                key='Molecule',
                required=True,
                #allowed_types=[molecule.Molecule],
                doc='Molecule wrapper ')

        EOpt._default_options = opt
        return EOpt._default_options.copy()


    @staticmethod
    def from_options(**kwargs):
        return EOpt(EOpt.default_options().set_values(kwargs))

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        # Cache some useful attributes
        self.Molecule = self.options['Molecule']

        #TODO What is optCG Ask Paul
        self.optCG = False
        self.isTSnode =False


    def optimize(self,nsteps):
        xyzfile=os.getcwd()+"/xyzfile.xyz"
        output_format = 'xyz'
        obconversion = ob.OBConversion()
        obconversion.SetOutFormat(output_format)
        opt_molecules=[]
        opt_molecules.append(self.Molecule.ic.mol.OBMol)

        self.Molecule.ic.make_Hint()
        for step in range(nsteps):
            self.opt_step()

        #step controller 
        with open(xyzfile,'w') as f:
            for mol in opt_molecules:
                f.write(obconversion.WriteString(mol))

    def opt_step(self):

        self.Molecule.ic.make_Hint()
        energy=0.

        energy = self.Molecule.lot.getEnergy()
        print("energy is %1.4f" % energy)
        grad = self.Molecule.lot.getGrad()

        gradq=self.Molecule.ic.grad_to_q(grad)


        #update_ic_eigen
        #Molecule.update(grad)
            #gradq = self.Molecule.ic.grad_to_q(grad)
            #self.Molecule.ic.ic_to_xyz() #updates geom of ICoords
            # need to update coords in lot as well


if __name__ == '__main__':

    from obutils import *
    import manage_xyz
    
    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)
    nocc=23
    nactive=2
    calc_states=[(0,0)]
    basis='6-31gs'
    molecule=molecule.Molecule.from_options(filepath=filepath,nocc=nocc,nactive=nactive,calc_states=calc_states,basis=basis,package='PyTC')
    molecule.lot.cas_from_geom()

    opt = EOpt.from_options(Molecule=molecule) 
    opt.optimize(1)


