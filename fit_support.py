"""
Created:                        June 6th 2024
Substantial revision:           October 18th 2024
Minor updates:                  March 2025


This script contains many useful functions for Ag/MgO
    supports gasphase NPs
    checks for sanity of structures
    modifies structures
    classifies structures

    and many more


    Some updates in the Active_Learning module. Check the "deprecated" function below
"""
#!/usr/bin/env python
from ase.build.surface import fcc111, fcc100, fcc110, fcc211
import statistics
import functools
from ase.io import (read, write, iread)
import numpy as np
from tqdm import tqdm
from ase import (Atoms, Atom)
import math
from io import StringIO
from sys import argv, exit
from typing import Tuple, Literal, List, Union
from ase.neighborlist import (natural_cutoffs, NeighborList)
from ase.constraints import FixAtoms
import warnings
from copy import deepcopy
from os import system
from utilities.print_bonds import setup_neighborlist, setup_analyzer

LATERAL_SPACING = 14.5 #spacing between between each nanoparticles repeated image. Note, though that
                    #this is a little lower than the spacing, due to the ceil function being used
Z_SPACING = 13.5 #spacing in z direction
ADSORPTION_HEIGHT = 2.2 #adsorption height
LAYER_HEIGHT = 1.5 #heigth of first MgO layer for constraining
LAYERS = "two" #Number of layers for the unit support
INTERFACE_SCALING = 1.3 #scaling factor for interfacial NP atoms' covalent radii
NANOPARTICLE_ELEMENT = "Ag" #what kind of atom is the NP made of?
LOW_INDEX_FACET_BUILDERS = {
    (1,1,1): (fcc111, (4, 4)),
    (1,0,0): (fcc100, (4, 4)),
    (1,1,0): (fcc110, (4, 4)),
    (2,1,1): (fcc211, (6, 4)),
}
#AG_MGO_INTERFACE = (1,0,0)

#def backup() -> None:
#    NEVER MIND
#    print("backup complete")

def deprecated(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"""{function.__name__} is less functional than that in
            the Active_Learning suite. Please use that instead""",
            DeprecationWarning,
            stacklevel = 2
        )
        return function(*args, **kwargs)
    return wrapper


def create_unit_support(layers: str = LAYERS) -> Atoms:
    """Creates a unit cell of the MgO support
    Either 2 layers (lower fixed), or 4 layers (lower three fixed)

    layers must be either 'two' or 'four'

    This uses PBE-D3BJ's lattice constant for MgO; works whether D3 is on Mg or not
    (i.e. 4.26 to 4.27A lattice constant
    Done in GPAW
    """

    #2 layers, the bottom layer fixed
    two_layers = """Mg O
1.0000000000000000
    4.2602238819630642    0.0000000000000000    0.0000000000000000
    0.0000000000000003    4.2602238819630642    0.0000000000000000
    0.0000000000000000    0.0000000000000000   24.9107835868707248
Mg  O
  4   4
Selective dynamics
Cartesian
0.0000000000000021  0.0000000000000009 17.7806716458891927   F   F   F
2.1301119409815343  2.1301119409815330 17.7806716458891927   F   F   F
0.0000000000000026  2.1301119409815330 19.9107835868707248   T   T   T
2.1301119409815348  0.0000000000000011 19.9107835868707248   T   T   T
0.0000000000000023  2.1301119409815330 17.7806716458891927   F   F   F
2.1301119409815343  0.0000000000000009 17.7806716458891927   F   F   F
0.0000000000000025  0.0000000000000011 19.9107835868707248   T   T   T
2.1301119409815348  2.1301119409815334 19.9107835868707248   T   T   T
 """
    #4 layers with the lower 3 layers fixed
    four_layers = """Mg O
1.0000000000000000
    4.2602238819630642    0.0000000000000000    0.0000000000000000
    0.0000000000000003    4.2602238819630642    0.0000000000000000
    0.0000000000000000    0.0000000000000000   24.9107835868707248
Mg  O
  8   8
Selective dynamics
Cartesian
0.0000000000000014  0.0000000000000006 13.5204477639261285   F   F   F
2.1301119409815334  2.1301119409815326 13.5204477639261285   F   F   F
0.0000000000000019  2.1301119409815326 15.6505597049076606   F   F   F
2.1301119409815339  0.0000000000000008 15.6505597049076606   F   F   F
0.0000000000000021  0.0000000000000009 17.7806716458891927   F   F   F
2.1301119409815343  2.1301119409815330 17.7806716458891927   F   F   F
0.0000000000000026  2.1301119409815330 19.9107835868707248   T   T   T
2.1301119409815348  0.0000000000000011 19.9107835868707248   T   T   T
0.0000000000000016  2.1301119409815326 13.5204477639261285   F   F   F
2.1301119409815334  0.0000000000000006 13.5204477639261285   F   F   F
0.0000000000000018  0.0000000000000008 15.6505597049076606   F   F   F
2.1301119409815339  2.1301119409815330 15.6505597049076606   F   F   F
0.0000000000000023  2.1301119409815330 17.7806716458891927   F   F   F
2.1301119409815343  0.0000000000000009 17.7806716458891927   F   F   F
0.0000000000000025  0.0000000000000011 19.9107835868707248   T   T   T
2.1301119409815348  2.1301119409815334 19.9107835868707248   T   T   T
 """

    if layers not in ("two", "four"):
        warnings.warn(f"""You have requested {layers} MgO layers
        but that option is invalid.
        Defaulting to two layers with the lower fixed""",
        category = UserWarning)
        layers = "two"

    basic = StringIO(two_layers if layers == "two" else four_layers)
    unit_mgo = read(basic, format = "vasp")

    return unit_mgo


def divider(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> Union[None, Tuple[Atoms]]:
    """
    Divide the structure into NP and surface
    Preserve the constraints of the support; assume the NP has none

    Inputs:
                Supported_NP
                element         (what element is the NP?)
                                default = 'Ag'
    Returns:    NP
                Bare_Surface
    """
    if not isinstance(atoms, Atoms):
        return None

    element = element.capitalize()

    constrained_indices = atoms.constraints[0].get_indices() if atoms.constraints else list()
    silver_indices = [index for index, atom in enumerate(atoms) if atom.symbol == element]
    support_indices = [index for index, atom in enumerate(atoms) if atom.symbol != element]

    silvers, support = atoms[silver_indices], atoms[support_indices]
    silvers.set_cell(atoms.get_cell())
    support.set_cell(atoms.get_cell())
    silvers.pbc, support.pbc = True, True

    old_to_new_support_indices = dict()
    for new_index, old_index in enumerate(support_indices):
        old_to_new_support_indices[old_index] = new_index

    new_constrained_support_indices = [old_to_new_support_indices[old_i]
            for old_i in constrained_indices if old_i in old_to_new_support_indices
            ]

    if new_constrained_support_indices:
        support.set_constraint(FixAtoms(indices = new_constrained_support_indices))

    return silvers, support


def concatenate(nanoparticle: Atoms, support: Atoms,
        adsorption_height: float = ADSORPTION_HEIGHT) -> Atoms:
    """
    Join an NP to a Surface
    Inputs:     NP
                Bare_Surface
                Adsorption height (Optional)
    Returns:    Supported_NP
    """
    max_support_z = max(support.positions[:,2])
    min_np_z = min(nanoparticle.positions[:,2])

    nanoparticle.translate(displacement = (0, 0,
        adsorption_height - min_np_z + max_support_z))
    support.extend(nanoparticle)
#    support = centralize(support)

    return support


def centralize(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> Atoms:
    """
    Centralize the NP to make viewing easier
    Inputs:     Supported_NP
                element the NP is made of
    Returns:    Supported_NP (with NP centralized)
    """
    silvers, support = divider(atoms, element = element)

    center_x = support.cell.lengths()[0] / 2
    center_y = support.cell.lengths()[1] / 2

    silvers_com_x = silvers.get_center_of_mass()[0]
    silvers_com_y = silvers.get_center_of_mass()[1]

    x_move = center_x - silvers_com_x
    y_move = center_y - silvers_com_y
    silvers.translate(displacement = (x_move, y_move , 0))

    support.extend(silvers)
    support.pbc = atoms.pbc
    support.set_constraint(atoms.constraints)

    support.pbc = True

    return support


def calculate_current_lateral_spacing(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> Tuple[float, float]:
    """
    Calculate the current lateral spacing of NPs
    Inputs:     Supported_NP
                element the NP is made of
    Returns:    X- and Y-spacing of periodic NP images
    """
    silvers, support = divider(atoms, element = element)

    min_x = min(silvers.positions[:,0])
    min_y = min(silvers.positions[:,1])
    max_x = max(silvers.positions[:,0])
    max_y = max(silvers.positions[:,1])

    x_spacing = min_x + atoms.cell.lengths()[0] - max_x
    y_spacing = min_y + atoms.cell.lengths()[1] - max_y

    return x_spacing, y_spacing


def constrain_lower_MgO(atoms: Atoms,
        layer_height: float = LAYER_HEIGHT) -> Atoms:
    """
    Translate system to bottom of cell, then constrain the lower MgO layer
    Inputs:     Supported_NP
    Returns:    Supported_NP (at bottom of cell, with lower MgO constrained)
    """
    minz = min(atoms.positions[:,2])
    atoms.translate(displacement = (0,0,-minz))
    fixed = FixAtoms([atom.index for atom in atoms if atom.z < layer_height])
    atoms.set_constraint(fixed)

    return atoms


def scaler(
        image: Atoms, element: str = NANOPARTICLE_ELEMENT,
        adsorption_height: float = ADSORPTION_HEIGHT,
        z_spacing: float = Z_SPACING, lateral_spacing: float = LATERAL_SPACING,
        layers: str = LAYERS, unit_support: Union[Atoms, None] = None
        ) -> Atoms:
    """
    Return an NP supported upon MgO of a good size
    Inputs:
        Image:              NP or Supported NP
        element:            What element the NP is made of. default = 'Ag'
        adsorption_height:  Desired adsorption height (Optional)
        z_spacing:          Desired Z-spacing of periodic images (Optional)
        lateral_spacing:    Desired X- and Y-spacing of periodic NP images (Optional)
        layers:             How many MgO layers. Must be 'two' or 'four'
        unit_support:       Atoms object of the unit support.
                            Defaults to that given here (GPAW-D3(BJ) for 2 or 4 layers)

    Returns:
                            Supported_NP
    """
    if not unit_support:
        unit_support = create_unit_support(layers = layers)
    unit_cell = unit_support.cell
    unit_cell_x = unit_cell[0,0]
    unit_cell_y = unit_cell[1,1]
    unit_cell_z = unit_cell[2,2]
    unit_cell_max_z = max(unit_support.positions[:,2])

    silvers, support = divider(image, element = element)
    silvers = Atoms(silvers)
    silvers.center(vacuum = 10)

    min_x = min(silvers.positions[:,0])
    min_y = min(silvers.positions[:,1])
    max_x = max(silvers.positions[:,0])
    max_y = max(silvers.positions[:,1])
    min_z = min(silvers.positions[:,2])

    x_diameter = max_x - min_x
    y_diameter = max_y - min_y

    required_x = x_diameter + lateral_spacing
    required_y = y_diameter + lateral_spacing

    ratio_x = math.ceil(required_x / unit_cell_x)
    ratio_y = math.ceil(required_y / unit_cell_y)

    adsorption_height = adsorption_height + unit_cell_max_z
    new_support = unit_support * (ratio_x,ratio_y,1)
    silvers_displacement = adsorption_height - min_z
    silvers.translate(displacement = (0,0,silvers_displacement))

    new_support.cell[2,2] = 10 #deliberately set too low so that the logic within the if statement (below) will play out fine
    new_support.extend(silvers)

    cell_bottom = 0
    min_z = min(new_support.positions[:,2])
    new_cell_displacement = cell_bottom - min_z
    new_support.translate(displacement = (0,0,new_cell_displacement))

    max_height = max(new_support.positions[:,2])
    cell_top = new_support.cell[2,2]
    distance = cell_top - max_height

    if distance < z_spacing:
        new_support.cell[2,2] += (lateral_spacing - cell_top + max_height)

#    new_support = centralize(new_support)
    new_support.info.update(image.info)

    return new_support


@deprecated
def check_exploded(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> float:
    """
    Indicate that the system might have exploded
    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    Max X- or Y-spacing between any two atoms in the system

    Example usage:
    it_exploded = check_exploded(atoms) > 70 #angstrom
    """
    silvers, surface = divider(atoms, element = element)
    min_x, max_x = min(silvers.positions[:,0]),  max(silvers.positions[:,0])
    min_y, max_y = min(silvers.positions[:,1]),  max(silvers.positions[:,1])

    return max(max_x - min_x, max_y - min_y)


def check_inversion(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> bool:
    """
    Check if the system got inverted, which I have noticed
    sometimes happens, long after the calculation has exploded

    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    True if system is inverted, else False
    """
    silvers, surface = divider(atoms, element = element)
    silvers_min_z = min(silvers.positions[:,2])
    surface_max_z = max(surface.positions[:,2])

    return silvers_min_z < surface_max_z


@deprecated
def check_for_uncoordinated_atoms(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT
        ) -> bool:
    """
    Check to see if any atom/atom-groups has flown off the NP
    Inputs:     NP or Supported_NP
                element the NP is made of. default to Ag
    Returns:    True if so, else False
    """
    silvers, surface = divider(atoms, element = element)
    nl = NeighborList(natural_cutoffs(silvers, mult = 1.07),
            self_interaction = False, bothways = True)
    nl.update(silvers)
    bonds = nl.get_connectivity_matrix(sparse = False).sum(axis = 0)

    return 0 in bonds


def classify_traj(traj: List[Atoms], element: str = NANOPARTICLE_ELEMENT,
        coverage: float = 0.75, buffer: float = 0.05,
        return_lattices: bool = False) -> List:
    """
    Divides a list of structures into their sublists, which are:
        Bulk NP
        Supported NP
        Gasphase NP
        Pristine Support
        Unknown/Miscellaneous systems

    also gives you the lattice constant of the support
    (assuming it is MgO) and of the Bulk NP (somewhat)

    Inputs:
        traj: List[Atoms]       List of atoms objects to classify
        element (str)           element the NP is made of. defaults to Ag
        coverage: float         What fraction of the cell would bulk Ag fill?
                                This is required to distinguish
                                between gasphase and bulk Ag
                                Default of 0.75 seems to work well
        buffer: float           Buffer used in logic for calculating lattice constants.
                                Default of 0.05 A should be universally
                                valid for a fixed, crytalline layer
        return_lattices: bool   Whether or not to calculate the bulk Ag's
                                'pseudo' lattice constants  ('pseudo',
                                since it might not be crystalline)
                                and the support's lattice constants

    Returns:
        in the following order:
            supported_NP, pristine_support, bulk_NP, gasphase_NP, miscellaneous,
            NP_pseudolattice, support_lattice

        note that miscellaneous and-or support_lattice_constant
        and-or NP_pseudolattice may be empty lists

    Example usage:
    images = read('traj.traj', ':')
    ag_mgo, pristine_mgo, bulk_ag, gas_ag, misc, ag_plattices,\
            mgo_lattices = classify_traj(images,
            return_lattices = True)
    """

    gas_ag, pristine_mgo, bulk_ag,\
            ag_mgo, miscellaneous = [], [], [], [], []
    mgo_lattices, ag_pseudolattices = [], []
    element = element.capitalize()

    if return_lattices: ##get the lattice constant
        warnings.warn(f"""You have requested the lattice constants of the support
        The calculations assume that:
            1. MgO is the support
            2. The lowest layer is constrained

        You have also asked for {element}'s pseudo-lattice constant
        Note that since the {element} might not be perfectly crystalline,
        the values returned are only a sort of 'average' lattice constant
        Also note that the formula used ONLY applies to FCC metals
            """, category = UserWarning)

    warnings.warn(f"""An extent/coverage of {coverage} may result in
    imperfect discrimination between gasphase and bulk {element}.
    Check results carefully!""", category = UserWarning)

    for atoms in traj:

        ag = Atoms([atom for atom in atoms if atom.symbol == element])
        mgo = Atoms([atom for atom in atoms if atom.symbol != element])

        if len(mgo) == 0: #gas or bulk Ag
            x_positions, y_positions, z_positions = atoms.positions.T
            x_extent = max(x_positions) - min(x_positions)
            y_extent = max(y_positions) - min(y_positions)
            z_extent = max(z_positions) - min(z_positions)

            x_cell, y_cell, z_cell = atoms.cell.lengths()

            if ((x_extent >= coverage * x_cell) and
                    (y_extent >= coverage * y_cell) and
                    (z_extent >= coverage * z_cell)): #bulk Ag virtually fills the cell

                atoms.info["Class"] = "Bulk_NP"
                bulk_ag.append(atoms)

                if return_lattices: ##get the pseudolattice constants
                    volume = atoms.get_volume()
                    ag_pseudolattices.append((4 * volume / len(atoms)) ** (1 / 3))

            else:
                atoms.info["Class"] = "Gas_NP"
                gas_ag.append(atoms)

        elif len(ag) == 0: #pristine mgo
            atoms.info["Class"] = "Support"
            pristine_mgo.append(atoms)

            if return_lattices: ##get the lattice constants
                new_atoms = deepcopy(atoms)
                new_atoms = constrain_lower_MgO(new_atoms)
                lower_layer = [atom.z < LAYER_HEIGHT for atom in new_atoms]
                new_atoms = new_atoms[lower_layer]

                x_slice_anchor = min(new_atoms.positions[:,0])
                x_slice = [(x_slice_anchor - buffer) < atom.x <
                        (x_slice_anchor + buffer) for atom in new_atoms
                        ]

                x_slice_atoms = new_atoms[x_slice]
                y_s = new_atoms.positions[:,1]

                y_length = max(y_s) - min(y_s)
                number = len(x_slice_atoms) / 2
                mg_o = y_length / (number - 0.5)
                mgo_lattices.append(mg_o)

        elif len(ag) != 0: #Ag/MgO
            atoms.info["Class"] = "Supported_NP"
            ag_mgo.append(atoms)

        else: #unknown
            atoms.info["Class"] = "Unknown"
            miscellaneous.append(atoms)

    if miscellaneous:
        warnings.warn(f"{len(miscellaneous)} images have been missed!",
                category = UserWarning)

    return ag_mgo, pristine_mgo, bulk_ag, gas_ag,\
            miscellaneous, ag_pseudolattices, mgo_lattices


@deprecated
def remove_uncoordinated_atoms(atoms: Atoms, element: str = NANOPARTICLE_ELEMENT,
        coord_cutoff: int = 4) -> Atoms:
    """
    Remove uncoordinated atoms or atom-groups
    Inputs:     NP or Supported_NP
                element the NP is made of. defaults to Ag
                Min. no. of bonds for an atom/atom-group to be considered uncoordinated
                default = 4
    Returns:    NP or Supported_NP with uncoordinated atom/atom-group removed
    """
    silvers, surface = divider(atoms, element = element)
    nl = NeighborList(natural_cutoffs(silvers, mult = 1.07),
            self_interaction = False, bothways = True)
    nl.update(silvers)
    bonds = nl.get_connectivity_matrix(sparse = False).sum(axis = 0)
    sakujo = []
    for index in range(len(silvers)):
        if bonds[index] < coord_cutoff:
            sakujo.append(index)
    del silvers[sakujo]

    if len(sakujo) > 0:
        warnings.warn(f"""{len(sakujo)} atoms deleted from this NP.
        I hope you know what you are doing!""",
        category = UserWarning)

    if len(silvers) == 0:
        print(f"No {element} atoms left! Something is very wrong")
        print(f"Error from remove_uncoordinated_atoms() in fit_support.py")
        exit(1)

    surface.extend(silvers)
    surface.set_cell(atoms.get_cell())
    surface.set_constraint(atoms.constraints)

    surface.pbc = True

    return surface


if __name__ == "__main__":
    if len(argv) < 7:
        print(f"run as {argv[0]} gas-phase-np.traj <adsorption_height> <z_spacing> <lateral_spacing> <layers: 'two' or 'four' <element NP is made of>")
        exit(1)

    auto_backup()

    data = iread(argv[1],":")
    adsorption_height, z_spacing, lateral_spacing, num_layers, element = float(argv[2]), float(argv[3]), float(argv[4]), argv[5], argv[6]
    data = [scaler(image, element = element,
        adsorption_height = adsorption_height,
        z_spacing = z_spacing, lateral_spacing = lateral_spacing,
        layers = num_layers, unit_support = None) for image in tqdm(data)]

    write("supported-NPs.traj", data)
    print("Processing completed.\nFile saved in 'supported-NPs.traj'")



