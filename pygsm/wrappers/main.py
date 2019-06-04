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
from wrappers import Molecule
from optimizers import *
from growing_string_methods import *



def main():
    parser = argparse.ArgumentParser(description="Parse GSM ")   
    parser.add_argument('-xyzfile', help='XYZ file',  required=True)
    parser.add_argument('-isomers', help='driving coordinate file', type=str, required=False)
    parser.add_argument('-mode', default="DE_GSM",help='GSM Type', type=str, required=True)
    parser.add_argument('-package',default="QChem",type=str,help="Electronic structure theory package",required=False)
    parser.add_argument('-lot_inp_file',type=str,default='qstart', help='qstart,gstart,etc',required=True)
    parser.add_argument('-ID',default=0, type=int,help='string identification',required=False)
    parser.add_argument('-num_nodes',type=int,default=9,help='number of nodes for string',required=False)
    parser.add_argument('-states',type=list,default=[(1,0)],help='',required=False)
    parser.add_argument('-job_data',type=dict,default={},help='',required=False)
    parser.add_argument('-pes_type',type=str,default='PES',help='',required=False)
    parser.add_argument('-adiabatic_index',type=int,default=0,help='',required=False)
    parser.add_argument('-multiplicity',type=int,default=1,help='',required=False)
    parser.add_argument('-FORCE',type=list,default=None,help='',required=False)
    parser.add_argument('-optimizer',type=str,default='eigenvector_follow',help='',required=False)
    parser.add_argument('-opt_print_level',type=int,default=1,help='',required=False)
    parser.add_argument('-linesearch',type=str,default='NoLineSearch',help='',required=False)
    parser.add_argument('-coordinate_type',type=str,default='TRIC',help='',required=False)
    parser.add_argument('-ADD_NODE_TOL',type=float,default=0.05,help='',required=False)
    parser.add_argument('-CONV_TOL',type=float,default=0.0005,help='',required=False)
    parser.add_argument('-growth_direction',type=int,default=0,help='',required=False)
    parser.add_argument('-reactant_geom_fixed',action='store_true',help='')
    parser.add_argument('-product_geom_fixed',action='store_true',help='')


    args = parser.parse_args()


    inpfileq = {
               # LOT
              'lot_inp_file': args.lot_inp_file,
              'xyzfile' : args.xyzfile,
              'EST_Package': args.package,
              'reactant_geom_fixed' : args.reactant_geom_fixed,
              'states': args.states,
              'job_data': args.job_data,
              
              #PES
              'PES_type': args.pes_type,
              'adiabatic_index': args.adiabatic_index,
              'multiplicity': args.multiplicity,
              'FORCE': args.FORCE,

              #optimizer
              'optimizer' : args.optimizer,
              'opt_print_level' : args.opt_print_level,
              'linesearch' : args.linesearch,

              #molecule
              'coordinate_type' : args.coordinate_type,

              # GSM
              'gsm_type': args.mode, # SE_GSM, SE_Cross
              'num_nodes' : args.num_nodes,
              'isomers_file': args.isomers,
              'ADD_NODE_TOL': args.ADD_NODE_TOL,
              'CONV_TOL': args.CONV_TOL,
              'growth_direction': args.growth_direction,
              'ID':args.ID,
              'product_geom_fixed' : args.product_geom_fixed,
              }


    #LOT
    nifty.printcool("Build the pyGSM level of theory (LOT) object")
    est_package=importlib.import_module("level_of_theories."+inpfileq['EST_Package'].lower())
    lot_class = getattr(est_package,inpfileq['EST_Package'])

    geoms = manage_xyz.read_xyzs(inpfileq['xyzfile'])
    lot = lot_class.from_options(
            lot_inp_file=inpfileq['lot_inp_file'],
            states=inpfileq['states'],
            job_data=inpfileq['job_data'],
            geom=geoms[0],
            )

    #PES
    nifty.printcool("Building the PES objects")
    pes_class = getattr(sys.modules[__name__], inpfileq['PES_type'])
    if inpfileq['PES_type']=='PES':
        pes = pes_class.from_options(
                lot=lot,
                ad_idx=inpfileq['adiabatic_index'],
                multiplicity=inpfileq['multiplicity'],
                FORCE=inpfileq['FORCE']
                )
    else:
        pes1 = PES.from_options(
                lot=lot,multiplicity=inpfileq['states'][0][0],
                ad_idx=inpfileq['states'][0][1],
                FORCE=inpfileq['FORCE']
                )
        pes2 = PES.from_options(
                lot=lot,
                multiplicity=inpfileq['states'][1][0],
                ad_idx=inpfileq['states'][1][1],
                FORCE=inpfileq['FORCE']
                )
        pes = pes_class.from_options(PES1=pes1,PES2=pes2,lot=lot)

    # Molecule
    nifty.printcool("Building the reactant object")
    Form_Hessian = True if inpfileq['optimizer']=='eigenvector_follow' else False
    form_primitives = True if inpfileq['gsm_type']!='DE_GSM' else False

    reactant = Molecule.from_options(
            geom=geoms[0],
            PES=pes,
            coordinate_type=inpfileq['coordinate_type'],
            Form_Hessian=Form_Hessian,
            top_settings = {'form_primitives': form_primitives},
            )

    if inpfileq['gsm_type']=='DE_GSM':
        nifty.printcool("Building the product object")
        product = Molecule.from_options(
                geom=geoms[1],
                PES=pes,
                coordinate_type=inpfileq['coordinate_type'],
                Form_Hessian=Form_Hessian,
                node_id=inpfileq['num_nodes']-1,
                top_settings = {'form_primitives': form_primitives},
                )
   
    # optimizer
    nifty.printcool("Building the Optimizer object")
    opt_class = getattr(sys.modules[__name__], inpfileq['optimizer'])
    optimizer = opt_class.from_options(print_level=inpfileq['opt_print_level'],Linesearch=inpfileq['linesearch'])

    # GSM
    nifty.printcool("Building the GSM object")
    gsm_class = getattr(sys.modules[__name__], inpfileq['gsm_type'])
    if inpfileq['gsm_type']=="DE_GSM":
        gsm = gsm_class.from_options(
                reactant=reactant,
                product=product,
                nnodes=inpfileq['num_nodes'],
                CONV_TOL=inpfileq['CONV_TOL'],
                ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
                growth_direction=inpfileq['growth_direction'],
                product_geom_fixed=inpfileq['product_geom_fixed'],
                optimizer=optimizer,
                ID=inpfileq['ID'],
                #print_level=inpfileq['gsm_print_level'],
                )
    else:
        driving_coordinates = read_isomers_file(inpfileq['isomers_file'])
        gsm = gsm_class.from_options(
                reactant=reactant,
                nnodes=inpfileq['num_nodes'],
                DQMAG_MAX=inpfileq['DQMAG_MAX'],
                BDIST_RATIO=inpfileq['BDIST_RATIO'],
                CONV_TOL=inpfileq['CONV_TOL'],
                ADD_NODE_TOL=inpfileq['ADD_NODE_TOL'],
                optimizer=optimizer,
                print_level=inpfileq['gsm_print_level'],
                driving_coords=driving_coordinates,
                ID=inpfileq['ID'],
                )

    if not inpfileq['reactant_geom_fixed']:
        optimizer.optimize(
           molecule = reactant,
           refE = reactant.energy,
           opt_steps=100,
           )

    gsm.go_gsm()

    return


def read_isomers_file(isomers_file):
    with open(isomers_file) as f:
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

def go_gsm(gsm,max_iters=50,opt_steps=3,rtype=2):
    gsm.go_gsm(max_iters=max_iters,opt_steps=opt_steps,rtype=rtype)
    self.post_processing(gsm)
    self.cleanup_scratch(gsm.ID)

def cleanup_scratch(ID):
    cmd = "rm scratch/growth_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    cmd = "rm scratch/opt_iters_{:03d}_*.xyz".format(ID)
    os.system(cmd)
    #if inpfileq['EST_Package']=="DFTB":
    #    for i in range(self.gsm.nnodes):
    #        cmd = 'rm -rf scratch/{}'.format(i)
    #        os.system(cmd)

def plot(fx,x,title):
    plt.figure(1)
    plt.title("String {:04d}".format(title))
    plt.plot(x,fx,color='b', label = 'Energy',linewidth=2,marker='o',markersize=12)
    plt.xlabel('Node Number')
    plt.ylabel('Energy (kcal/mol)')
    plt.legend(loc='best')
    plt.savefig('{:04d}_string.png'.format(title),dpi=600)

def post_processing(gsm):
    plot(fx=gsm.energies,x=range(len(gsm.energies)),title=gsm.ID)

    # TS energy
    minnodeR = np.argmin(gsm.energies[:gsm.TSnode])
    TSenergy = gsm.energies[gsm.TSnode] - gsm.energies[minnodeR]

    # Delta E
    minnodeP = gsm.TSnode + np.argmin(gsm.energies[gsm.TSnode:])
    deltaE = gsm.energies[minnodeR] - gsm.energies[minnodeP]
    print(" min reactant node: %i min product node %i TS node is %i" % (minnodeR,minnodeP,gsm.TSnode))
    print(" TS energy: %5.4f" % TSenergy)
    print(" Delta E is %5.4f" % deltaE)


if __name__=='__main__':
    main()

