# standard library imports
from utilities import manage_xyz, nifty, units
import sys
import os
from os import path
import numpy as np

# third party
import pMolecule as pM
import pCore as pC
from pScientific.Geometry3 import Coordinates3
import pBabel as pB
import glob
#
## local application imports
#from Definitions import *

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
try:
    from .base_lot import Lot
except:
    from base_lot import Lot


class pDynamo(Lot):
    """
    Level of theory is a wrapper object to do QM/MM DFT  calculations 
    Requires a system object. lot_inp_file must create  a pdynamo object
    called system
    """

    def __init__(self, options):

        super(pDynamo, self).__init__(options)

        #pDynamo
        self.system = self.options['job_data'].get('system', None)

        print(" making folder scratch/{}".format(self.node_id))
        os.system('mkdir -p scratch/{}'.format(self.node_id))

        # if simulation doesn't exist create it
        if self.lot_inp_file is not None and self.simulation is None:
            # Now go through the logic of determining which FILE options are activated.
            # DO NOT DUPLICATE OPTIONS WHICH ARE ALREADY PART OF LOT OPTIONS (e.g. charge)
            # ORCA options
            self.file_options.set_active('use_orca', False, bool, "Use ORCA to evaluate energies and gradients")
            self.file_options.set_active('orca_method', "B3LYP", str, "Method to use in ORCA e.g. HF, MP2, or density functional",
                                         depend=(self.file_options.use_orca),
                                         msg="Must use ORCA to specify density functional")
            self.file_options.set_active('basis', "6-31g", str, "Basis set for wavefunction or density functional",
                                         depend=(self.file_options.use_orca),
                                         msg="Must use ORCA to specify density functional")
            self.file_options.set_active('tole', 1e-4, float, "Energy tolerance for convergence")
            self.file_options.set_active('maxiter', 100, int, "Number of SCF cycles")
            self.file_options.set_active('slowconv', False, bool, "Convergence option for ORCA")
            self.file_options.set_active('scfconvtol', 'NormalSCF', str, "Convergence option for ORCA", allowed=['NormalSCF', 'TightSCF', 'ExtremeSCF'])
            self.file_options.set_active('d3', False, bool, "Use Grimme's D3 dispersion")

            # QM/MM CHARMM
            self.file_options.set_active('qmatom_file', None, str, '')
            self.file_options.set_active('use_charmm_qmmm', False, bool, 'Use CHARMM molecular mechanics parameters to perform QMMM',
                                         depend=(self.file_options.qmatom_file is not None),
                                         msg="Must define qm atoms")
            self.file_options.set_active('path_to_prm', None, str, 'path to folder containing Charmm parameter files')
            self.file_options.set_active('path_to_str', None, str, 'path to folder containing to Charmm str files')
            self.file_options.set_active('psf_file', None, str, 'Path to file containing CHARMM PSF')
            self.file_options.set_active('crd_file', None, str, 'Path to file containing CHARMM CRD')

            # DFTB options
            self.file_options.set_active('use_dftb', False, bool, "Use DFTB to evaluate energies and gradients",
                                         clash=(self.file_options.use_orca),
                                         msg="We're not using DFTB+")
            self.file_options.set_active('path_to_skf', None, str, 'path to folder containing skf files')
            self.file_options.set_active('use_scc', True, bool, "Use self-consistent charge")

            # General options
            self.file_options.set_active('command', None, str, 'pDynamo requires a path to an executable like ORCA or DFTB+')
            self.file_options.set_active('scratch', None, str, 'Folder to store temporary files')

            self.file_options.force_active('scratch', 'scratch/{}'.format(self.node_id), 'Setting scratch folder')
            nifty.printcool(" Options for pdynamo")

            for line in self.file_options.record():
                print(line)

            # Build system
            self.build_system()

        # set all active values to self for easy access
        for key in self.file_options.ActiveOptions:
            setattr(self, key, self.file_options.ActiveOptions[key])

    def build_system(self):

        # save xyz file
        manage_xyz.write_xyz('scratch/{}/tmp.xyz'.format(self.node_id), self.geom)

        # ORCA
        if self.use_orca:
            parsed_keywords = []
            # Use these keywords in ORCA
            for key in [self.orca_method, self.basis, self.slowconv, self.scfconvtol, self.d3]:
                if key is not None and key is not False:
                    parsed_keywords.append(key)
            print(parsed_keywords)

            qcmodel = pM.QCModel.QCModelORCA.WithOptions(keywords=parsed_keywords,
                                                         deleteJobFiles=False,
                                                         command=self.command,
                                                         scratch=self.scratch,
                                                         )

            # assuming only one state for now
            qcmodel.electronicState = pM.QCModel.ElectronicState.WithOptions(charge=self.charge, multiplicity=self.states[0][0])
            nbModel = pM.NBModel.NBModelORCA.WithDefaults()

            if self.use_charmm_qmmm:
                # Get PRM
                prm_files = []
                for name in glob.glob(self.path_to_prm+'/*.prm'):
                    prm_files.append(name)
                for name in glob.glob(self.path_to_str+'/*.str'):
                    prm_files.append(name)
                print(prm_files)

                # Build parameters object
                parameters = pB.CHARMMParameterFiles_ToParameters(
                    [x for x in prm_files])
                system = pB.CHARMMPSFFile_ToSystem(
                    self.psf_file,
                    isXPLOR=True,
                    log='scratch/{}/logfile'.format(self.node_id),
                    parameters=parameters)

                # Get qm atoms
                with open(self.qmatom_file) as f:
                    qmatom_indices = f.read().splitlines()
                qmatom_indices = [int(x) for x in qmatom_indices]

                system.DefineQCModel(qcmodel, qcSelection=pC.Selection(qmatom_indices))
                system.DefineNBModel(nbModel)
            else:
                # Define System
                system = pB.XYZFile_ToSystem('scratch/{}/tmp.xyz'.format(self.node_id))
                system.DefineQCModel(qcmodel)
            # system.Summary ( )

            self.system = system

        elif self.use_dftb:
            electronicState = pM.QCModel.ElectronicState.WithOptions(charge=self.charge, multiplicity=self.states[0][0])
            qcModel = pM.QCModel.QCModelDFTB.WithOptions(deleteJobFiles=False,
                                                         electronicState=electronicState,
                                                         randomScratch=True,
                                                         scratch='scratch/{}'.format(self.node_id),
                                                         skfPath=self.path_to_skf,
                                                         command=self.command,
                                                         useSCC=self.use_scc)
            system = pB.XYZFile_ToSystem('scratch/{}/tmp.xyz'.format(self.node_id))
            system.DefineQCModel(qcModel)
            self.system = system

    def run(self, geom, multiplicity=1):
        self.E = []
        self.grada = []
        coordinates3 = Coordinates3.WithExtent(len(geom))
        xyz = manage_xyz.xyz_to_np(geom)
        for (i, (x, y, z)) in enumerate(xyz):
            coordinates3[i, 0] = x
            coordinates3[i, 1] = y
            coordinates3[i, 2] = z
        self.system.coordinates3 = coordinates3
        energy = self.system.Energy(doGradients=True)  # KJ
        energy *= units.KJ_MOL_TO_AU * units.KCAL_MOL_PER_AU  # KCAL/MOL

        self.E.append((multiplicity, energy))
        print(energy)

        gradient = []
        for i in range(len(geom)):
            for j in range(3):
                gradient.append(self.system.scratch.gradients3[i, j] * units.KJ_MOL_TO_AU / units.ANGSTROM_TO_AU)  # Ha/Bohr
        gradient = np.asarray(gradient)
        # print(gradient)
        self.grada.append((multiplicity, gradient))
        # print(gradient)

    @property
    def system(self):
        return self.options['job_data']['system']

    @system.setter
    def system(self, value):
        self.options['job_data']['system'] = value


if __name__ == "__main__":

    # QMMM
    #filepath='/export/zimmerman/craldaz/kevin2/tropcwatersphere.xyz'
    #geom = manage_xyz.read_xyz(filepath)
    #filepath='tropcwatersphere.xyz'
    #lot = pDynamo.from_options(states=[(5,0)],charge=-1,nproc=16,fnm=filepath,lot_inp_file='pdynamo_options_qmmm.txt')
    #lot.run(geom)

    # DFTB
    filepath = '../../data/ethylene.xyz'
    geom = manage_xyz.read_xyz(filepath)
    lot = pDynamo.from_options(states=[(1, 0)], charge=0, nproc=16, fnm=filepath, lot_inp_file='pdynamo_options_dftb.txt')
    lot.run(geom)
