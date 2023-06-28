import matplotlib.pyplot as plt
from pyGSM.coordinate_systems import Angle, Dihedral, Distance, OutOfPlane

def get_driving_coord_prim(dc):
    prim = None
    if "ADD" in dc or "BREAK" in dc:
        if dc[1] < dc[2]:
            prim = Distance(dc[1] - 1, dc[2] - 1)
        else:
            prim = Distance(dc[2] - 1, dc[1] - 1)
    elif "ANGLE" in dc:
        if dc[1] < dc[3]:
            prim = Angle(dc[1] - 1, dc[2] - 1, dc[3] - 1)
        else:
            prim = Angle(dc[3] - 1, dc[2] - 1, dc[1] - 1)
    elif "TORSION" in dc:
        if dc[1] < dc[4]:
            prim = Dihedral(dc[1] - 1, dc[2] - 1, dc[3] - 1, dc[4] - 1)
        else:
            prim = Dihedral(dc[4] - 1, dc[3] - 1, dc[2] - 1, dc[1] - 1)
    elif "OOP" in dc:
        # if dc[1]<dc[4]:
        prim = OutOfPlane(dc[1] - 1, dc[2] - 1, dc[3] - 1, dc[4] - 1)
        # else:
        #    prim = OutOfPlane(dc[4]-1,dc[3]-1,dc[2]-1,dc[1]-1)
    return prim


def plot(fx, x, title):
    plt.figure(1)
    plt.title("String {:04d}".format(title))
    plt.plot(x, fx, color='b', label='Energy', linewidth=2, marker='o', markersize=12)
    plt.xlabel('Node Number')
    plt.ylabel('Energy (kcal/mol)')
    plt.legend(loc='best')
    plt.savefig('{:04d}_string.png'.format(title), dpi=600)


