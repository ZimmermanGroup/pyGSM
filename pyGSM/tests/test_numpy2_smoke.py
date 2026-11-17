"""End-to-end smoke tests: SE-GSM and DE-GSM with a cheap ASE calculator (EMT).

They only check that the string methods run without crashing. On NumPy >= 2.4 converting
an ndim > 0 array to a Python scalar is a TypeError; on older NumPy it is the
DeprecationWarning below, which is turned into an error so the CI catches it everywhere.
"""
import pytest

pytest.importorskip("ase")
from ase.build import molecule  # noqa: E402
from ase.calculators.emt import EMT  # noqa: E402

from pyGSM.coordinate_systems import DelocalizedInternalCoordinates, PrimitiveInternalCoordinates, Topology  # noqa: E402
from pyGSM.growing_string_methods import DE_GSM, SE_GSM  # noqa: E402
from pyGSM.level_of_theories.ase import ASELoT  # noqa: E402
from pyGSM.molecule import Molecule  # noqa: E402
from pyGSM.optimizers import beales_cg, conjugate_gradient, eigenvector_follow, lbfgs  # noqa: E402
from pyGSM.potential_energy_surfaces import PES  # noqa: E402
from pyGSM.utilities import elements  # noqa: E402

pytestmark = pytest.mark.filterwarnings("error:Conversion of an array with ndim:DeprecationWarning")

OPTIMIZERS = [eigenvector_follow, lbfgs, conjugate_gradient, beales_cg]


def ethane(rotate=0.0):
    atoms = molecule("C2H6")
    if rotate:
        # rotate the methyl group on C1 about the C0-C1 axis
        mask = [i in (1, 5, 6, 7) for i in range(len(atoms))]
        atoms.set_dihedral(2, 0, 1, 5, atoms.get_dihedral(2, 0, 1, 5) + rotate, mask=mask)
    return atoms


def make_molecule(atoms, node_id=0):
    symbols = atoms.get_chemical_symbols()
    xyz = atoms.positions.copy()
    geom = [[s, *p] for s, p in zip(symbols, xyz)]
    table = elements.ElementData()
    els = [table.from_symbol(s) for s in symbols]
    lot = ASELoT.from_options(EMT(), geom=geom, states=[(1, 0)], gradient_states=[(1, 0)])
    pes = PES.from_options(lot=lot, ad_idx=0, multiplicity=1)
    top = Topology.build_topology(xyz, els)
    prims = PrimitiveInternalCoordinates.from_options(xyz=xyz, atoms=els, addtr=True, topology=top)
    prims.newMakePrimitives(xyz)
    coords = DelocalizedInternalCoordinates.from_options(xyz=xyz, atoms=els, addtr=True, primitives=prims)
    return Molecule.from_options(geom=geom, PES=pes, coord_obj=coords, Form_Hessian=True, node_id=node_id)


@pytest.mark.parametrize("opt_class", OPTIMIZERS, ids=lambda c: c.__name__)
def test_se_gsm_torsion(opt_class, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # GSM writes scratch/ and string files in the cwd
    atoms = ethane()
    target = atoms.get_dihedral(2, 0, 1, 5) + 120.0
    target = (target + 180.0) % 360.0 - 180.0
    gsm = SE_GSM.from_options(
        reactant=make_molecule(atoms),
        nnodes=9,
        optimizer=opt_class.from_options(Linesearch="NoLineSearch", update_hess_in_bg=opt_class is not lbfgs),
        driving_coords=[("TORSION", 3, 1, 2, 6, target)],
        DQMAG_MAX=0.5,
    )
    gsm.go_gsm(max_iters=10, opt_steps=5, rtype=2)
    assert gsm.nnodes >= 2


def test_de_gsm(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    nnodes = 7
    reactant = make_molecule(ethane())
    product = Molecule.copy_from_options(reactant, xyz=ethane(rotate=60.0).positions, new_node_id=nnodes - 1)
    gsm = DE_GSM.from_options(
        reactant=reactant,
        product=product,
        nnodes=nnodes,
        optimizer=eigenvector_follow.from_options(Linesearch="NoLineSearch", update_hess_in_bg=True),
    )
    gsm.go_gsm(max_iters=10, opt_steps=3, rtype=2)
    assert len(gsm.energies) == gsm.nnodes >= nnodes  # DE-GSM may add nodes


@pytest.mark.parametrize("opt_class", OPTIMIZERS, ids=lambda c: c.__name__)
def test_optimizer_lowers_energy(opt_class, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the level of theory writes scratch/ in the cwd
    atoms = ethane()
    atoms.positions[2] += [0.0, 0.0, 0.3]  # stretch one C-H bond
    mol = make_molecule(atoms)
    e0 = mol.energy
    opt = opt_class.from_options(Linesearch="NoLineSearch", update_hess_in_bg=opt_class is not lbfgs)
    opt.optimize(molecule=mol, opt_type="UNCONSTRAINED", opt_steps=20, path=str(tmp_path))
    assert mol.energy < e0 - 1.0  # kcal/mol
