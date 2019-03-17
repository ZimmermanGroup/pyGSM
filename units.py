"""
Units and constants pulled from NIST website 01/03/2017 (CODATA 2014)

Usage goes as
    y(bohr radii) = x(Angstrom)   * ANGSTROM_TO_AU
    x(Angstrom)   = y(bohr radii) / ANGSTROM_TO_AU
"""
import re

# Constants
AVOGADROS_NUMBER = 6.022140857E+23          # (mol^{-1})
BOLTZMANN_CONSTANT_SI = 1.38064852E-23      # (J / K)
HBAR_SI = 1.054571800E-34                   # (J s)
SPEED_OF_LIGHT_SI = 299792458.0             # (m / s)

# Length
M_PER_AU = 0.52917721067E-10                # (m / a_0)
M_TO_AU = 1.0/M_PER_AU                      # (a_0 / m)
ANGSTROM_TO_AU = 1.0E-10*M_TO_AU            # (a_0 / A)

# Dipole
DEBYE_TO_AU = 0.393430307
AU_TO_DEBYE = 1.0 / DEBYE_TO_AU

# Mass
KG_PER_AU = 9.10938356E-31                  # (kg / m_e)
KG_TO_AU = 1.0/KG_PER_AU                    # (m_e / kg)
AMU_TO_AU = 1.0E-3/AVOGADROS_NUMBER*KG_TO_AU  # (m_e / amu)

# Time
S_PER_AU = 2.418884326509E-17               # (s / aut)
S_TO_AU = 1.0/S_PER_AU                      # (aut / s)
FS_TO_AU = S_TO_AU * 1.0E-15                # (aut / fs)
PS_TO_AU = S_TO_AU * 1.0E-12                # (aut / ps)

# Energy/temperature
J_PER_AU = 4.359744650E-18                  # (J / E_h)
J_TO_AU = 1.0/J_PER_AU                      # (E_h / J)

K_TO_AU = BOLTZMANN_CONSTANT_SI*J_TO_AU     # (E_h / K)
K_PER_AU = 1.0/K_TO_AU                      # (K / E_h)

EV_PER_AU = 27.21138602                     # (eV / E_h)
EV_TO_AU = 1.0/EV_PER_AU                    # (E_h / eV)

KJ_MOL_PER_AU = J_PER_AU * 1.0E-3 * AVOGADROS_NUMBER  # ((kJ/mol) / E_h)
KJ_MOL_TO_AU = 1.0/KJ_MOL_PER_AU            # (E_h / (kJ/mol))

KJ_PER_KCAL = 4.184                         # (kJ / kcal)
KCAL_MOL_PER_AU = KJ_MOL_PER_AU / KJ_PER_KCAL  # ((kcal/mol) / E_h)
KCAL_MOL_TO_AU = 1.0/KCAL_MOL_PER_AU        # (E_h / (kcal/mol))

INV_CM_PER_AU = 219474.6313702              # (cm^{-1} / E_h)
INV_CM_TO_AU = 1.0/INV_CM_PER_AU            # (E_h / cm^{-1})

# Force
NEWTON_PER_AU = 8.23872336E-8               # (N / auf)
NEWTON_TO_AU = 1.0 / NEWTON_PER_AU            # (auf / N)
NANONEWTON_TO_AU = 1e-9 * NEWTON_TO_AU      # (auf / nN)

# AMBER Units (http://ambermd.org/Questions/units.html)
AMBERLENGTH_TO_AU = ANGSTROM_TO_AU
AMBERMASS_TO_AU = AMU_TO_AU
AMBERENERGY_TO_AU = KCAL_MOL_TO_AU
AMBERTIME_TO_AU = 1.0/20.455*PS_TO_AU
AMBERVELOCITY_TO_AU = AMBERLENGTH_TO_AU/AMBERTIME_TO_AU
AMBERCHARGE_TO_AU = 1.0/18.2223

# Print all unit conversions
if __name__ == '__main__':
    conversions = dict(locals())
    for key, val in conversions.items():
        if key[0] != '_':
            print((" Conversion: % 22s, Value: %11.11E" % (key, val)))


units = {
    'au_per_amu'   : 1.8228884855409500E+03,        # mass
    'au_per_cminv' : 1.0 / 219474.6305,             # ???
    'au_per_ang'   : 1.0 / 0.5291772109217,         # length
    'au_per_K'     : 1.0 / 3.1577464E5,             # temperature
    'au_per_fs'    : 1.0 / 2.418884326505E-2,       # time
}
for k in list(units.keys()):
    v = units[k]
    mobj = re.match('(\S+)_per_(\S+)',k)
    units['%s_per_%s' % (mobj.group(2),mobj.group(1))] = 1.0 / v
