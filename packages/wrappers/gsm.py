# standard library imports
import sys
import os
from os import path
import importlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#third party
import argparse
import numpy as np

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

    @staticmethod
    def default_options():
        """ GSM default options. """

        if hasattr(GSM, '_default_options'): return GSM._default_options.copy()
        opt = options.Options() 
        opt.add_option(
                key='gsm_type',
                value='DE_GSM',
                required=False,
                allowed_values=['DE_GSM','SE_GSM','SE_Cross'],
                doc='The type of GSM to use'
                )

        opt.add_option(
            key='states',
            value=[(1,0)],
            required=False,
            doc='list of tuples (multiplicity,state) state is 0-indexed')

        opt.add_option(
                key='job_data',
                value={},
                required=False,
                allowed_types=[dict],
                doc='extra key-word arguments to define level of theory object. e.g.\
                     TeraChem Cloud requires a TeraChem client and options dictionary.'
                )

        opt.add_option(
                key='EST_Package',
                value="QChem",
                required=True,
                allowed_values=['QChem','Molpro','Orca','PyTC','OpenMM','DFTB','TCC'],
                doc='the electronic structure theory package that is being used.'
                )

        opt.add_option(
                key='lot_inp_file',
                required=False,
                value=None,
                )

        opt.add_option(
                key='xyzfile',
                required=True,
                allowed_types=[str],
                doc='reactant xyzfile name.'
                )

        opt.add_option(
                key='PES_type',
                value='PES',
                allowed_values=['PES','Penalty_PES','Avg_PES'],
                required=False,
                doc='PES type'
                )
        opt.add_option(
                key='adiabatic_index',
                value=0,
                required=False if len(opt['states'])==1 else True,
                doc='adiabatic index')

        opt.add_option(
                key='multiplicity',
                value=1,
                required=False if len(opt['states'])==1 else True,
                doc='multiplicity')

        opt.add_option(
                key="FORCE",
                value=None,
                required=False,
                doc='Apply a spring force between atoms in units of AU, e.g. [(1,2,0.1214)]. Negative is tensile, positive is compresive',
                )

        opt.add_option(
                key='coordinate_type',
                value='TRIC',
                required=False,
                allowed_values=['TRIC','DLC','HDLC'],
                doc='the type of coordinate system, TRIC is recommended.'
                )

        opt.add_option(
                key='optimizer',
                value='eigenvector_follow',
                required=False,
                allowed_values=['eigenvector_follow','lbfgs','conjugate_gradient'],
                doc='The type of optimizer, eigenvector follow is recommended for non-condensed phases. L-BFGS is recommended for condensed phases.'
                )
        opt.add_option(
                key='linesearch',
                value='NoLineSearch',
                required=False,
                allowed_values=['NoLineSearch','backtrack'],
                doc='line search for the optimizer.'
                )

        opt.add_option(
                key='opt_print_level',
                value=1,
                required=False,
                doc='1 prints normal, 2 prints almost everything for optimization.'
                )


        opt.add_option(
                key='num_nodes',
                value=9,
                required=False,
                doc='The number of nodes for GSM -- choose large for SE-GSM or SE-CROSS.'
                )

        opt.add_option(
                key='isomers_file',
                value=None,
                required=False if opt.options['gsm_type']=='DE_GSM' else True,
                doc='The file containing the driving coordinate starting with the word NEW'
                )

        #opt.add_option(
        #        key='rp_type',
        #        value='Exact_TS',
        #        required=True if opt.options['gsm_type']=='SE-CROSS' else False,
        #        allowed_values=['Exact_TS','Climb','Opt_Only','SE-MECI','SE-MESX']
        #        doc='How to optimize the string. Exact TS does an exact TS optimization,\
        #                Climb only does climbing image, Opt only does not optimize the TS,\
        #                SE-MECI and SE-MESX are for conical intersections/ISC'
        #                )

        opt.add_option(
                key='DQMAG_MAX',
                value=0.4,
                required=False,
                doc='maximum step size during growth phase for SE methods.'
                )

        opt.add_option(
                key='CONV_TOL',
                value=0.0005,
                doc='convergence tolerance for a single node.'
                )

        opt.add_option(
                key='gsm_print_level',
                value=1,
                allowed_values=[0,1,2],
                doc='1-- normal, 2-- ?'
                )

        opt.add_option(
            key='ADD_NODE_TOL',
            value=0.1,
            required=False,
            allowed_types=[float],
            doc='Convergence threshold before adding a new node'
            )

        opt.add_option(
                key='reactant_geom_fixed',
                value=True,
                required=False,
                doc="Fix last node?"
                )

        opt.add_option(
                key="product_geom_fixed",
                value=True,
                required=False,
                doc="Fix last node?"
                )

        opt.add_option(
                key="growth_direction",
                value=0,
                required=False,
                doc="how to grow string,0=Normal,1=from reactant"
                )


        opt.add_option(
                key="BDIST_RATIO",
                value=0.5,
                required=False,
                doc="SE-Crossing uses this \
                        bdist must be less than 1-BDIST_RATIO of initial bdist in order to be \
                        to be considered grown.",
                        )

        opt.add_option(
                key='ID',
                value=1,
                required=False,
                doc='String identifier'
                )

        GSM._default_options = opt
        return GSM._default_options.copy()

    @classmethod
    def from_options(cls,**kwargs):
        nifty.printcool_dictionary(kwargs)
        return cls(cls.default_options().set_values(kwargs))

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

        if not self.options['reactant_geom_fixed']:
            optimizer.optimize(
               molecule = reactant,
               refE = reactant.energy,
               opt_steps=100,
               )

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
        self.cleanup_scratch()

    def cleanup_scratch(self):
        cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(self.gsm.ID)
        os.system(cmd)
        cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(self.gsm.ID)
        os.system(cmd)
        if self.options['EST_Package']=="DFTB":
            for i in range(self.gsm.nnodes):
                cmd = 'rm -rf scratch/{}'.format(i)
                os.system(cmd)

    def post_processing(self):
        self.plot(fx=self.gsm.energies,x=range(len(self.gsm.energies)),title=self.gsm.ID)

        # TS energy
        minnodeR = np.argmin(self.gsm.energies[:self.gsm.TSnode])
        TSenergy = self.gsm.energies[self.gsm.TSnode] - self.gsm.energies[minnodeR]

        # Delta E
        minnodeP = self.gsm.TSnode + np.argmin(self.gsm.energies[self.gsm.TSnode:])
        deltaE = self.gsm.energies[minnodeR] - self.gsm.energies[minnodeP]
        print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR,minnodeP,self.gsm.TSnode))
        print(" TS energy: %5.4f" % TSenergy)
        print(" Delta E is %5.4f" % deltaE)

    def plot(self,fx,x,title):
        plt.figure(1)
        plt.title("String {:04d}".format(title))
        plt.plot(x,fx,color='b', label = 'Energy',linewidth=2,marker='o',markersize=12)
        plt.xlabel('Node Number')
        plt.ylabel('Energy (kcal/mol)')
        plt.legend(loc='best')
        plt.savefig('{:04d}_string.png'.format(title),dpi=600)


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
              'lot_inp_file': args.lot_inp_file,
              'xyzfile' : args.xyzfile,
              'EST_Package': args.package,
              'reactant_geom_fixed' : False,

              # GSM
              'gsm_type': args.mode, # SE_GSM, SE_Cross
              'num_nodes' : args.num_nodes,
              'isomers_file': args.isomers,
              'ADD_NODE_TOL': 0.01,
              'ID':args.ID,
              }

    gsm = GSM.from_options(**inpfileq)
    gsm.go_gsm()

