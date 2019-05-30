# standard library imports
import sys
import os
from os import path
import importlib

#third party
import argparse

# local application imports
sys.path.append(path.dirname( path.dirname( path.abspath(__file__))))
from utilities import *
from potential_energy_surfaces import PES
from potential_energy_surfaces import Avg_PES
from potential_energy_surfaces import Penalty_PES
from molecule import Molecule
from optimizers import *
from growing_string_methods import *

class GSM(object):
    """ """

    @classmethod
    def from_options(cls,**kwargs):
        nifty.printcool_dictionary(kwargs)
        return cls(kwargs)

    def __init__(
            self,
            options,
            ):
        """ Constructor """
        self.options = options

        #LOT
        nifty.printcool("Build the pyGSM level of theory (LOT) object")
        est_package=importlib.import_module("level_of_theories."+options['EST_Package'].lower())
        lot_class = getattr(est_package,options['EST_Package'])

        geoms = manage_xyz.read_xyzs(self.options['xyzfile'])
        lot = lot_class.from_options(
                lot_inp_file=self.options['lot_inp_file'],
                states=self.options['states'],
                job_data=self.options['job_data'],
                geom=geoms[0],
                )

        #PES
        nifty.printcool("Building the PES objects")
        pes_class = getattr(sys.modules[__name__], options['PES_type'])
        if options['PES_type']=='PES':
            pes = pes_class.from_options(
                    lot=lot,
                    ad_idx=self.options['adiabatic_index'],
                    multiplicity=self.options['multiplicity'],
                    FORCE=self.options['FORCE']
                    )
        else:
            pes1 = PES.from_options(
                    lot=lot,multiplicity=self.options['states'][0][0],
                    ad_idx=self.options['states'][0][1],
                    FORCE=self.options['FORCE']
                    )
            pes2 = PES.from_options(
                    lot=lot,
                    multiplicity=self.options['states'][1][0],
                    ad_idx=self.options['states'][1][1],
                    FORCE=self.options['FORCE']
                    )
            pes = pes_class.from_options(PES1=pes1,PES2=pes2,lot=lot)

        # Molecule
        nifty.printcool("Building the reactant object")
        Form_Hessian = True if self.options['optimizer']=='eigenvector_follow' else False
        reactant = Molecule.from_options(
                geom=geoms[0],
                PES=pes,
                coordinate_type=self.options['coordinate_type'],
                Form_Hessian=Form_Hessian
                )

        if self.options['gsm_type']=='DE_GSM':
            nifty.printcool("Building the product object")
            product = Molecule.from_options(
                    geom=geoms[1],
                    PES=pes,
                    coordinate_type=self.options['coordinate_type'],
                    Form_Hessian=Form_Hessian,
                    node_id=self.options['num_nodes']-1,
                    )
       
        # optimizer
        nifty.printcool("Building the Optimizer object")
        opt_class = getattr(sys.modules[__name__], options['optimizer'])
        optimizer = opt_class.from_options(print_level=self.options['opt_print_level'],Linesearch=self.options['linesearch'])

        # GSM
        nifty.printcool("Building the GSM object")
        gsm_class = getattr(sys.modules[__name__], options['gsm_type'])
        if options['gsm_type']=="DE_GSM":
            self.gsm = gsm_class.from_options(
                    reactant=reactant,
                    product=product,
                    nnodes=self.options['num_nodes'],
                    CONV_TOL=self.options['CONV_TOL'],
                    ADD_NODE_TOL=self.options['ADD_NODE_TOL'],
                    growth_direction=self.options['growth_direction'],
                    product_geom_fixed=self.options['product_geom_fixed'],
                    optimizer=optimizer,
                    ID=self.options['ID'],
                    #print_level=self.options['gsm_print_level'],
                    )
        else:
            driving_coordinates = self.read_isomers_file()
            self.gsm = gsm_class.from_options(
                    reactant=reactant,
                    nnodes=self.options['num_nodes'],
                    DQMAG_MAX=self.options['DQMAG_MAX'],
                    BDIST_RATIO=self.options['BDIST_RATIO'],
                    CONV_TOL=self.options['CONV_TOL'],
                    ADD_NODE_TOL=self.options['ADD_NODE_TOL'],
                    optimizer=optimizer,
                    print_level=self.options['gsm_print_level'],
                    driving_coords=driving_coordinates,
                    ID=self.options['ID'],
                    )


    def read_isomers_file(self):
        with open(self.options['isomers_file']) as f:
            lines = f.readlines()
        driving_coordinates=[]
        for line in lines[1:]:
            dc = []
            twoInts=False
            threeInts=False
            fourInts=False
            for i,elem in enumerate(line.split()):
                if i==0:
                    dc.append(elem)
                    if elem=="ADD" or elem=="BREAK":
                        twoInts =True
                    elif elem=="ANGLE":
                        threeInts =True
                    elif elem=="TORSION" or elem=="OOP":
                        threeInts =True
                else:
                    if twoInts and i>2:
                        dc.append(float(elem))
                    elif threeInts and i>3:
                        dc.append(float(elem))
                    elif fourInts and i>4:
                        dc.append(float(elem))
                    else:
                        dc.append(int(elem))
            driving_coordinates.append(dc)

        nifty.printcool("driving coordinates {}".format(driving_coordinates))
        return driving_coordinates

    def go_gsm(self,max_iters=50,opt_steps=3,rtype=2):
        self.gsm.go_gsm(max_iters=max_iters,opt_steps=opt_steps,rtype=rtype)
        self.post_processing()

    def post_processing(self):
        plot(fx = gsm.energies,x=range(len(gsm.energies)),title=self.gsm.ID,color='b')


if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Parse GSM ")   
    parser.add_argument('-xyzfile', help='XYZ file',  required=True)
    parser.add_argument('-isomers', help='driving coordinate file', type=str, required=False)
    parser.add_argument('-mode', default="DE_GSM",help='GSM Type', type=str, required=True)
    parser.add_argument('-package',default="QChem",type=str,help="Electronic structure theory package",required=False)
    parser.add_argument('-lot_inp_file',type=str,default='qstart', help='qstart,gstart,etc',required=True)
    parser.add_argument('-ID',default=0, type=int,help='string identification',required=False)
    parser.add_argument('-num_nodes',type=int,default=9,help='number of nodes for string',required=False)
    args = parser.parse_args()

    print 'Argument List:', str(sys.argv)
    inpfileq = {
               # LOT
              'states': [(1,0)],
              'job_data' : {},
              'lot_inp_file': args.lot_inp_file,
              'xyzfile' : args.xyzfile,
              'EST_Package': args.package,

              # PES 
              'PES_type': 'PES', 
              'adiabatic_index': 0,
              'multiplicity': 1,
              'FORCE': None,
              
              # Molecule
              'coordinate_type': 'TRIC',

              # Optimizer
              'optimizer': 'eigenvector_follow', # lbfgs, conjugate_gradient
              'linesearch': 'NoLineSearch', # backtrack
              'opt_print_level': 1 ,

              # GSM
              'gsm_type': args.mode, # SE_GSM, SE_Cross
              'num_nodes' : args.num_nodes,
              'isomers_file': args.isomers,
              'ADD_NODE_TOL': 0.05,
              'BDIST_RATIO': 0.5,
              'DQMAG_MAX': 0.4,
              'gsm_print_level': 1,
              'CONV_TOL': 0.0005,
              'ID':args.ID,
              'growth_direction': 0,
              'product_geom_fixed': False,
              }

    gsm = GSM.from_options(**inpfileq)
    gsm.go_gsm()

