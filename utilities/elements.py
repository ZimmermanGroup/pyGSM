"""All element data through the whole period table.

    Citation:
        NIST Standard Reference Database 144
        Online: September 1999 | Last update: January 2015
        URL: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&ascii=ascii&isotype=all
        Values obtained: September 15, 2017

"""
from collections import namedtuple

# Data entries for an element
# This is only a tuple to make it light weight
Element = namedtuple("Element", ["symbol", "name", "atomic_num", "mass_amu",
                                 "covalent_radius", "vdw_radius", "bond_radius",
                                 "electronegativity", "max_bonds"])


class ElementData(object):
    """Contains full periodic table element data and methods to access them
    """
    data = [Element(symbol='Xx', name='Dummy', atomic_num=0, mass_amu=0.0,
                    covalent_radius=0.0, vdw_radius=0.0, bond_radius=0.0,
                    electronegativity=0.0, max_bonds=0),
            Element(symbol='H', name='Hydrogen', atomic_num=1, mass_amu=1.007825032,
                    covalent_radius=0.31, vdw_radius=1.1, bond_radius=0.31,
                    electronegativity=2.2, max_bonds=1),
            Element(symbol='He', name='Helium', atomic_num=2, mass_amu=4.002603254,
                    covalent_radius=0.28, vdw_radius=1.4, bond_radius=0.28,
                    electronegativity=0.0, max_bonds=0),
            Element(symbol='Li', name='Lithium', atomic_num=3, mass_amu=7.01600455,
                    covalent_radius=1.28, vdw_radius=1.81, bond_radius=1.28,
                    electronegativity=0.98, max_bonds=1),
            Element(symbol='Be', name='Beryllium', atomic_num=4, mass_amu=9.0121822,
                    covalent_radius=0.96, vdw_radius=1.53, bond_radius=0.96,
                    electronegativity=1.57, max_bonds=2),
            Element(symbol='B', name='Boron', atomic_num=5, mass_amu=11.0093054,
                    covalent_radius=0.84, vdw_radius=1.92, bond_radius=0.84,
                    electronegativity=2.04, max_bonds=4),
            Element(symbol='C', name='Carbon', atomic_num=6, mass_amu=12.0,
                    covalent_radius=0.76, vdw_radius=1.7, bond_radius=0.76,
                    electronegativity=2.55, max_bonds=4),
            Element(symbol='N', name='Nitrogen', atomic_num=7, mass_amu=14.003074005,
                    covalent_radius=0.71, vdw_radius=1.55, bond_radius=0.71,
                    electronegativity=3.04, max_bonds=4),
            Element(symbol='O', name='Oxygen', atomic_num=8, mass_amu=15.99491462,
                    covalent_radius=0.66, vdw_radius=1.52, bond_radius=0.66,
                    electronegativity=3.44, max_bonds=2),
            Element(symbol='F', name='Fluorine', atomic_num=9, mass_amu=18.99840322,
                    covalent_radius=0.57, vdw_radius=1.47, bond_radius=0.57,
                    electronegativity=3.98, max_bonds=1),
            Element(symbol='Ne', name='Neon', atomic_num=10, mass_amu=19.992440175,
                    covalent_radius=0.58, vdw_radius=1.54, bond_radius=0.58,
                    electronegativity=0.0, max_bonds=0),
            Element(symbol='Na', name='Sodium', atomic_num=11, mass_amu=22.989769281,
                    covalent_radius=1.66, vdw_radius=2.27, bond_radius=1.66,
                    electronegativity=0.93, max_bonds=1),
            Element(symbol='Mg', name='Magnesium', atomic_num=12, mass_amu=23.9850417,
                    covalent_radius=1.41, vdw_radius=1.73, bond_radius=1.41,
                    electronegativity=1.31, max_bonds=2),
            Element(symbol='Al', name='Aluminium', atomic_num=13, mass_amu=26.98153863,
                    covalent_radius=1.21, vdw_radius=1.84, bond_radius=1.21,
                    electronegativity=1.61, max_bonds=6),
            Element(symbol='Si', name='Silicon', atomic_num=14, mass_amu=27.976926532,
                    covalent_radius=1.11, vdw_radius=2.1, bond_radius=1.11,
                    electronegativity=1.9, max_bonds=6),
            Element(symbol='P', name='Phosphorus', atomic_num=15, mass_amu=30.97376163,
                    covalent_radius=1.07, vdw_radius=1.8, bond_radius=1.07,
                    electronegativity=2.19, max_bonds=6),
            Element(symbol='S', name='Sulfur', atomic_num=16, mass_amu=31.972071,
                    covalent_radius=1.05, vdw_radius=1.8, bond_radius=1.05,
                    electronegativity=2.58, max_bonds=6),
            Element(symbol='Cl', name='Chlorine', atomic_num=17, mass_amu=34.96885268,
                    covalent_radius=1.02, vdw_radius=1.75, bond_radius=1.02,
                    electronegativity=3.16, max_bonds=1),
            Element(symbol='Ar', name='Argon', atomic_num=18, mass_amu=39.962383123,
                    covalent_radius=1.06, vdw_radius=1.88, bond_radius=1.06,
                    electronegativity=0.0, max_bonds=0),
            Element(symbol='K', name='Potassium', atomic_num=19, mass_amu=38.96370668,
                    covalent_radius=2.03, vdw_radius=2.75, bond_radius=2.03,
                    electronegativity=0.82, max_bonds=1),
            Element(symbol='Ca', name='Calcium', atomic_num=20, mass_amu=39.96259098,
                    covalent_radius=1.76, vdw_radius=2.31, bond_radius=1.76,
                    electronegativity=1.0, max_bonds=2),
            Element(symbol='Sc', name='Scandium', atomic_num=21, mass_amu=44.9559119,
                    covalent_radius=1.7, vdw_radius=2.3, bond_radius=1.7,
                    electronegativity=1.36, max_bonds=6),
            Element(symbol='Ti', name='Titanium', atomic_num=22, mass_amu=47.9479463,
                    covalent_radius=1.6, vdw_radius=2.15, bond_radius=1.6,
                    electronegativity=1.54, max_bonds=6),
            Element(symbol='V', name='Vanadium', atomic_num=23, mass_amu=50.9439595,
                    covalent_radius=1.53, vdw_radius=2.05, bond_radius=1.53,
                    electronegativity=1.63, max_bonds=6),
            Element(symbol='Cr', name='Chromium', atomic_num=24, mass_amu=51.9405075,
                    covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
                    electronegativity=1.66, max_bonds=6),
            Element(symbol='Mn', name='Manganese', atomic_num=25, mass_amu=54.9380451,
                    covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
                    electronegativity=1.55, max_bonds=8),
            Element(symbol='Fe', name='Iron', atomic_num=26, mass_amu=55.9349375,
                    covalent_radius=1.32, vdw_radius=2.05, bond_radius=1.32,
                    electronegativity=1.83, max_bonds=6),
            Element(symbol='Co', name='Cobalt', atomic_num=27, mass_amu=58.933195,
                    covalent_radius=1.26, vdw_radius=2.0, bond_radius=1.26,
                    electronegativity=1.88, max_bonds=6),
            Element(symbol='Ni', name='Nickel', atomic_num=28, mass_amu=57.9353429,
                    covalent_radius=1.24, vdw_radius=2.0, bond_radius=1.24,
                    electronegativity=1.91, max_bonds=6),
            Element(symbol='Cu', name='Copper', atomic_num=29, mass_amu=62.9295975,
                    covalent_radius=1.32, vdw_radius=2.0, bond_radius=1.32,
                    electronegativity=1.9, max_bonds=6),
            Element(symbol='Zn', name='Zinc', atomic_num=30, mass_amu=63.929142,
                    covalent_radius=1.22, vdw_radius=2.1, bond_radius=1.22,
                    electronegativity=1.65, max_bonds=6),
            Element(symbol='Ga', name='Gallium', atomic_num=31, mass_amu=68.925573,
                    covalent_radius=1.22, vdw_radius=1.87, bond_radius=1.22,
                    electronegativity=1.81, max_bonds=3),
            Element(symbol='Ge', name='Germanium', atomic_num=32, mass_amu=73.921177,
                    covalent_radius=1.2, vdw_radius=2.11, bond_radius=1.2,
                    electronegativity=2.01, max_bonds=4),
            Element(symbol='As', name='Arsenic', atomic_num=33, mass_amu=74.921596,
                    covalent_radius=1.19, vdw_radius=1.85, bond_radius=1.19,
                    electronegativity=2.18, max_bonds=3),
            Element(symbol='Se', name='Selenium', atomic_num=34, mass_amu=79.916521,
                    covalent_radius=1.2, vdw_radius=1.9, bond_radius=1.2,
                    electronegativity=2.55, max_bonds=2),
            Element(symbol='Br', name='Bromine', atomic_num=35, mass_amu=78.918337,
                    covalent_radius=1.2, vdw_radius=1.83, bond_radius=1.2,
                    electronegativity=2.96, max_bonds=1),
            Element(symbol='Kr', name='Krypton', atomic_num=36, mass_amu=83.911507,
                    covalent_radius=1.16, vdw_radius=2.02, bond_radius=1.16,
                    electronegativity=3.0, max_bonds=0),
            Element(symbol='Rb', name='Rubidium', atomic_num=37, mass_amu=84.911789,
                    covalent_radius=2.2, vdw_radius=3.03, bond_radius=2.2,
                    electronegativity=0.82, max_bonds=1),
            Element(symbol='Sr', name='Strontium', atomic_num=38, mass_amu=87.905612,
                    covalent_radius=1.95, vdw_radius=2.49, bond_radius=1.95,
                    electronegativity=0.95, max_bonds=2),
            Element(symbol='Y', name='Yttrium', atomic_num=39, mass_amu=88.905848,
                    covalent_radius=1.9, vdw_radius=2.4, bond_radius=1.9,
                    electronegativity=1.22, max_bonds=6),
            Element(symbol='Zr', name='Zirconium', atomic_num=40, mass_amu=89.904704,
                    covalent_radius=1.75, vdw_radius=2.3, bond_radius=1.75,
                    electronegativity=1.33, max_bonds=6),
            Element(symbol='Nb', name='Niobium', atomic_num=41, mass_amu=92.906378,
                    covalent_radius=1.64, vdw_radius=2.15, bond_radius=1.64,
                    electronegativity=1.6, max_bonds=6),
            Element(symbol='Mo', name='Molybdenum', atomic_num=42, mass_amu=97.905408,
                    covalent_radius=1.54, vdw_radius=2.1, bond_radius=1.54,
                    electronegativity=2.16, max_bonds=6),
            Element(symbol='Tc', name='Technetium', atomic_num=43, mass_amu=97.907216,
                    covalent_radius=1.47, vdw_radius=2.05, bond_radius=1.47,
                    electronegativity=1.9, max_bonds=6),
            Element(symbol='Ru', name='Ruthenium', atomic_num=44, mass_amu=101.904349,
                    covalent_radius=1.46, vdw_radius=2.05, bond_radius=1.46,
                    electronegativity=2.2, max_bonds=6),
            Element(symbol='Rh', name='Rhodium', atomic_num=45, mass_amu=102.905504,
                    covalent_radius=1.42, vdw_radius=2.0, bond_radius=1.42,
                    electronegativity=2.28, max_bonds=6),
            Element(symbol='Pd', name='Palladium', atomic_num=46, mass_amu=105.903486,
                    covalent_radius=1.39, vdw_radius=2.05, bond_radius=1.39,
                    electronegativity=2.2, max_bonds=6),
            Element(symbol='Ag', name='Silver', atomic_num=47, mass_amu=106.905097,
                    covalent_radius=1.45, vdw_radius=2.1, bond_radius=1.45,
                    electronegativity=1.93, max_bonds=6),
            Element(symbol='Cd', name='Cadmium', atomic_num=48, mass_amu=113.903358,
                    covalent_radius=1.44, vdw_radius=2.2, bond_radius=1.44,
                    electronegativity=1.69, max_bonds=6),
            Element(symbol='In', name='Indium', atomic_num=49, mass_amu=114.903878,
                    covalent_radius=1.42, vdw_radius=2.2, bond_radius=1.42,
                    electronegativity=1.78, max_bonds=3),
            Element(symbol='Sn', name='Tin', atomic_num=50, mass_amu=119.902194,
                    covalent_radius=1.39, vdw_radius=1.93, bond_radius=1.39,
                    electronegativity=1.96, max_bonds=4),
            Element(symbol='Sb', name='Antimony', atomic_num=51, mass_amu=120.903815,
                    covalent_radius=1.39, vdw_radius=2.17, bond_radius=1.39,
                    electronegativity=2.05, max_bonds=3),
            Element(symbol='Te', name='Tellurium', atomic_num=52, mass_amu=129.906224,
                    covalent_radius=1.38, vdw_radius=2.06, bond_radius=1.38,
                    electronegativity=2.1, max_bonds=2),
            Element(symbol='I', name='Iodine', atomic_num=53, mass_amu=126.904473,
                    covalent_radius=1.39, vdw_radius=1.98, bond_radius=1.39,
                    electronegativity=2.66, max_bonds=1),
            Element(symbol='Xe', name='Xenon', atomic_num=54, mass_amu=131.904153,
                    covalent_radius=1.4, vdw_radius=2.16, bond_radius=1.4,
                    electronegativity=2.6, max_bonds=0),
            Element(symbol='Cs', name='Caesium', atomic_num=55, mass_amu=132.905451,
                    covalent_radius=2.44, vdw_radius=3.43, bond_radius=2.44,
                    electronegativity=0.79, max_bonds=1),
            Element(symbol='Ba', name='Barium', atomic_num=56, mass_amu=137.905247,
                    covalent_radius=2.15, vdw_radius=2.68, bond_radius=2.15,
                    electronegativity=0.89, max_bonds=2),
            Element(symbol='La', name='Lanthanum', atomic_num=57, mass_amu=138.906353,
                    covalent_radius=2.07, vdw_radius=2.5, bond_radius=2.07,
                    electronegativity=1.1, max_bonds=12),
            Element(symbol='Ce', name='Cerium', atomic_num=58, mass_amu=139.905438,
                    covalent_radius=2.04, vdw_radius=2.48, bond_radius=2.04,
                    electronegativity=1.12, max_bonds=6),
            Element(symbol='Pr', name='Praseodymium', atomic_num=59, mass_amu=140.907652,
                    covalent_radius=2.03, vdw_radius=2.47, bond_radius=2.03,
                    electronegativity=1.13, max_bonds=6),
            Element(symbol='Nd', name='Neodymium', atomic_num=60, mass_amu=141.907723,
                    covalent_radius=2.01, vdw_radius=2.45, bond_radius=2.01,
                    electronegativity=1.14, max_bonds=6),
            Element(symbol='Pm', name='Promethium', atomic_num=61, mass_amu=144.912749,
                    covalent_radius=1.99, vdw_radius=2.43, bond_radius=1.99,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Sm', name='Samarium', atomic_num=62, mass_amu=151.919732,
                    covalent_radius=1.98, vdw_radius=2.42, bond_radius=1.98,
                    electronegativity=1.17, max_bonds=6),
            Element(symbol='Eu', name='Europium', atomic_num=63, mass_amu=152.92123,
                    covalent_radius=1.98, vdw_radius=2.4, bond_radius=1.98,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Gd', name='Gadolinium', atomic_num=64, mass_amu=157.924103,
                    covalent_radius=1.96, vdw_radius=2.38, bond_radius=1.96,
                    electronegativity=1.2, max_bonds=6),
            Element(symbol='Tb', name='Terbium', atomic_num=65, mass_amu=158.925346,
                    covalent_radius=1.94, vdw_radius=2.37, bond_radius=1.94,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Dy', name='Dysprosium', atomic_num=66, mass_amu=163.929174,
                    covalent_radius=1.92, vdw_radius=2.35, bond_radius=1.92,
                    electronegativity=1.22, max_bonds=6),
            Element(symbol='Ho', name='Holmium', atomic_num=67, mass_amu=164.930322,
                    covalent_radius=1.92, vdw_radius=2.33, bond_radius=1.92,
                    electronegativity=1.23, max_bonds=6),
            Element(symbol='Er', name='Erbium', atomic_num=68, mass_amu=165.930293,
                    covalent_radius=1.89, vdw_radius=2.32, bond_radius=1.89,
                    electronegativity=1.24, max_bonds=6),
            Element(symbol='Tm', name='Thulium', atomic_num=69, mass_amu=168.934213,
                    covalent_radius=1.9, vdw_radius=2.3, bond_radius=1.9,
                    electronegativity=1.25, max_bonds=6),
            Element(symbol='Yb', name='Ytterbium', atomic_num=70, mass_amu=173.938862,
                    covalent_radius=1.87, vdw_radius=2.28, bond_radius=1.87,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Lu', name='Lutetium', atomic_num=71, mass_amu=174.940771,
                    covalent_radius=1.87, vdw_radius=2.27, bond_radius=1.87,
                    electronegativity=1.27, max_bonds=6),
            Element(symbol='Hf', name='Hafnium', atomic_num=72, mass_amu=179.94655,
                    covalent_radius=1.75, vdw_radius=2.25, bond_radius=1.75,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Ta', name='Tantalum', atomic_num=73, mass_amu=180.947995,
                    covalent_radius=1.7, vdw_radius=2.2, bond_radius=1.7,
                    electronegativity=1.5, max_bonds=6),
            Element(symbol='W', name='Tungsten', atomic_num=74, mass_amu=183.950931,
                    covalent_radius=1.62, vdw_radius=2.1, bond_radius=1.62,
                    electronegativity=2.36, max_bonds=6),
            Element(symbol='Re', name='Rhenium', atomic_num=75, mass_amu=186.955753,
                    covalent_radius=1.51, vdw_radius=2.05, bond_radius=1.51,
                    electronegativity=1.9, max_bonds=6),
            Element(symbol='Os', name='Osmium', atomic_num=76, mass_amu=191.96148,
                    covalent_radius=1.44, vdw_radius=2.0, bond_radius=1.44,
                    electronegativity=2.2, max_bonds=6),
            Element(symbol='Ir', name='Iridium', atomic_num=77, mass_amu=192.962926,
                    covalent_radius=1.41, vdw_radius=2.0, bond_radius=1.41,
                    electronegativity=2.2, max_bonds=6),
            Element(symbol='Pt', name='Platinum', atomic_num=78, mass_amu=194.964791,
                    covalent_radius=1.36, vdw_radius=2.05, bond_radius=1.36,
                    electronegativity=2.28, max_bonds=6),
            Element(symbol='Au', name='Gold', atomic_num=79, mass_amu=196.966568,
                    covalent_radius=1.36, vdw_radius=2.1, bond_radius=1.36,
                    electronegativity=2.54, max_bonds=6),
            Element(symbol='Hg', name='Mercury', atomic_num=80, mass_amu=201.970643,
                    covalent_radius=1.32, vdw_radius=2.05, bond_radius=1.32,
                    electronegativity=2.0, max_bonds=6),
            Element(symbol='Tl', name='Thallium', atomic_num=81, mass_amu=204.974427,
                    covalent_radius=1.45, vdw_radius=1.96, bond_radius=1.45,
                    electronegativity=1.62, max_bonds=3),
            Element(symbol='Pb', name='Lead', atomic_num=82, mass_amu=207.976652,
                    covalent_radius=1.46, vdw_radius=2.02, bond_radius=1.46,
                    electronegativity=2.33, max_bonds=4),
            Element(symbol='Bi', name='Bismuth', atomic_num=83, mass_amu=208.980398,
                    covalent_radius=1.48, vdw_radius=2.07, bond_radius=1.48,
                    electronegativity=2.02, max_bonds=3),
            Element(symbol='Po', name='Polonium', atomic_num=84, mass_amu=208.98243,
                    covalent_radius=1.4, vdw_radius=1.97, bond_radius=1.4,
                    electronegativity=2.0, max_bonds=2),
            Element(symbol='At', name='Astatine', atomic_num=85, mass_amu=209.987148,
                    covalent_radius=1.5, vdw_radius=2.02, bond_radius=1.5,
                    electronegativity=2.2, max_bonds=1),
            Element(symbol='Rn', name='Radon', atomic_num=86, mass_amu=222.017577,
                    covalent_radius=1.5, vdw_radius=2.2, bond_radius=1.5,
                    electronegativity=0.0, max_bonds=0),
            Element(symbol='Fr', name='Francium', atomic_num=87, mass_amu=223.019735,
                    covalent_radius=2.6, vdw_radius=3.48, bond_radius=2.6,
                    electronegativity=0.7, max_bonds=1),
            Element(symbol='Ra', name='Radium', atomic_num=88, mass_amu=226.025409,
                    covalent_radius=2.21, vdw_radius=2.83, bond_radius=2.21,
                    electronegativity=0.9, max_bonds=2),
            Element(symbol='Ac', name='Actinium', atomic_num=89, mass_amu=227.027752,
                    covalent_radius=2.15, vdw_radius=2.0, bond_radius=2.15,
                    electronegativity=1.1, max_bonds=6),
            Element(symbol='Th', name='Thorium', atomic_num=90, mass_amu=232.038055,
                    covalent_radius=2.06, vdw_radius=2.4, bond_radius=2.06,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Pa', name='Protactinium', atomic_num=91, mass_amu=231.035884,
                    covalent_radius=2.0, vdw_radius=2.0, bond_radius=2.0,
                    electronegativity=1.5, max_bonds=6),
            Element(symbol='U', name='Uranium', atomic_num=92, mass_amu=238.050788,
                    covalent_radius=1.96, vdw_radius=2.3, bond_radius=1.96,
                    electronegativity=1.38, max_bonds=6),
            Element(symbol='Np', name='Neptunium', atomic_num=93, mass_amu=237.048173,
                    covalent_radius=1.9, vdw_radius=2.0, bond_radius=1.9,
                    electronegativity=1.36, max_bonds=6),
            Element(symbol='Pu', name='Plutonium', atomic_num=94, mass_amu=244.064204,
                    covalent_radius=1.87, vdw_radius=2.0, bond_radius=1.87,
                    electronegativity=1.28, max_bonds=6),
            Element(symbol='Am', name='Americium', atomic_num=95, mass_amu=243.061381,
                    covalent_radius=1.8, vdw_radius=2.0, bond_radius=1.8,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Cm', name='Curium', atomic_num=96, mass_amu=247.070354,
                    covalent_radius=1.69, vdw_radius=2.0, bond_radius=1.69,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Bk', name='Berkelium', atomic_num=97, mass_amu=247.070307,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Cf', name='Californium', atomic_num=98, mass_amu=251.079587,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Es', name='Einsteinium', atomic_num=99, mass_amu=252.08298,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Fm', name='Fermium', atomic_num=100, mass_amu=257.095105,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Md', name='Mendelevium', atomic_num=101, mass_amu=258.098431,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='No', name='Nobelium', atomic_num=102, mass_amu=259.10103,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=1.3, max_bonds=6),
            Element(symbol='Lr', name='Lawrencium', atomic_num=103, mass_amu=262.10963,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Rf', name='Rutherfordium', atomic_num=104, mass_amu=261.10877,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Db', name='Dubnium', atomic_num=105, mass_amu=262.11408,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Sg', name='Seaborgium', atomic_num=106, mass_amu=263.11832,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Bh', name='Bohrium', atomic_num=107, mass_amu=264.1246,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Hs', name='Hassium', atomic_num=108, mass_amu=265.13009,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Mt', name='Meitnerium', atomic_num=109, mass_amu=268.13873,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Ds', name='Darmstadtium', atomic_num=110, mass_amu=281.162061,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Rg', name='Roentgenium', atomic_num=111, mass_amu=280.164473,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Cn', name='Copernicium', atomic_num=112, mass_amu=285.174105,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Uut', name='Ununtrium', atomic_num=113, mass_amu=284.17808,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Fl', name='Flerovium', atomic_num=114, mass_amu=289.187279,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Uup', name='Ununpentium', atomic_num=115, mass_amu=288.192492,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Lv', name='Livermorium', atomic_num=116, mass_amu=292.199786,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Uus', name='Ununseptium', atomic_num=117, mass_amu=292.20755,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6),
            Element(symbol='Uuo', name='Ununoctium', atomic_num=118, mass_amu=293.21467,
                    covalent_radius=1.6, vdw_radius=2.0, bond_radius=1.6,
                    electronegativity=0.0, max_bonds=6)]

    def __init__(self):
        self.symbol_lookup = {}
        for element in self.data:
            self.symbol_lookup[element.symbol] = element

    @classmethod
    def from_atomic_number(cls, atomic_num):
        """Retrieve element data with atomic number

        Args:
            atomic_num: Atomic number of the element

        Returns:
            An Element object containing all available element data

        Example:
            >>> cs = ElementData().from_atomic_number(55)
            >>> print cs.name
            Caesium
            >>> print ElementData().from_atomic_number(1000)
            Traceback (most recent call last):
              ...
            ValueError: Atomic number 1000 is out of bound
        """
        if not isinstance(atomic_num, int):
            raise TypeError("Atomic number must be an integer")
        if atomic_num >= len(cls.data) or atomic_num < 0:
            raise ValueError("Atomic number {} is out of bound".format(atomic_num))
        return cls.data[atomic_num]

    def from_symbol(self, symbol):
        """Retrieve element data with element symbol

        Args:
            symbol: Element symbol. Case insensitive

        Returns:
            An Element object containing all available element data
        Example:
            >>> fluorine = ElementData().from_symbol('f')
            >>> print fluorine.name
            Fluorine
            >>> print fluorine.atomic_num
            9
        """
        if not isinstance(symbol, str):
            raise TypeError("Element symbol must be a string")
        symbol = symbol.strip().capitalize()
        if symbol not in self.symbol_lookup:
            raise ValueError("Element symbol {} is not valid".format(symbol))
        return self.symbol_lookup[symbol]

    @classmethod
    def num_elements(cls):
        """Return the total number of available elements"""
        return len(cls.data)

    @classmethod
    def get_element_list(cls):
        """Returns the list of symbols of all available elements"""
        return [element.symbol for element in cls.data]
 
if __name__ == '__main__':
    from . import manage_xyz

    filepath="tests/fluoroethene.xyz"
    geom=manage_xyz.read_xyz(filepath,scale=1)
    E = ElementData()
    #geom[0]
    C =E.from_symbol(geom[0][0])
    print(C.vdw_radius)


