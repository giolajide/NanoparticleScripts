"""
Initialized:            from fit_support.py
                        in May 2025
Extended to other supported NPs:    June 2025

Refined:                May & June 2025 for the NP atom count paper

Slight touches:         August/Sept 2025

Use:                    Potential catalytic descriptors for supported NPs

Functions:              interfacial area and radius
                        interfacial atoms
                        perimeter atoms
                        NP surface atoms

formatted with black
"""
#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy
from ase.build.surface import (
    fcc111,
    fcc100,
    fcc110,
    fcc211,
    bcc111,
    bcc100,
    bcc110,
    hcp0001,
)  # , hcp10m10)
from scipy.spatial import ConvexHull
from ase.visualize import view
from matplotlib import pyplot as plt
from ase.build import bulk
from tqdm import tqdm
import statistics
import functools
from ase.io import read, write
from ascii_colors import ASCIIColors
import numpy as np
from ase import Atoms, Atom
import math
from sys import argv, exit, stdout
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from ase.geometry.analysis import Analysis
from typing import Tuple, Literal, List, Union
from os.path import basename, splitext
from ase.neighborlist import natural_cutoffs, NeighborList
import warnings
from os import system
from joblib import Parallel, delayed

# ~/npscripts/fit_support.py and ~/utilities/print_bonds.py
from fit_support import divider, NANOPARTICLE_ELEMENT
from utilities.print_bonds import setup_neighborlist, setup_analyzer  # , SKIN
from contact_angle import calculate_contact_angle

##try to import cupy for GPU acceleration
"""
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() == 0:
        cp = None
except ImportError:
    cp = None
"""
##TODO: implement all the ways for cupy to run
# Suppose `atoms` is an ASE Atoms object (or any collection); we check its length:
# n_atoms = len(atoms)
# Decide which backend to use:
# if (cp is not None) and (n_atoms > 1000):
#    xp = cp
# (Optional) pre‐allocate any large arrays on GPU:
# e.g. some_array_gpu = cp.asarray(some_array_cpu)
# else:
#    xp = np

# Now, everywhere in your script where you would have done:
#    arr = np.some_function(…)
# replace with:
#    arr = xp.some_function(…)

# Note to self: get tags with atoms.get_tags()
TAGS_DICT = {
    "bulk": 0,
    "surface": 1,
    "interface": 2,
    "perimeter": 3,
    "substrate": 4,
}

INTERFACE_SCALING = 1.3  # 1.3 scaling factor for interfacial NP atoms' covalent radii
CN_SCALING = 1.05  # 1.2 scaling factor for discriminating bulk from NP surface atoms
FCC_AND_HCP_COORD_CUTOFF = 12  # any atom with coord < COORD_CUTOFF is a surface atom
BCC_COORD_CUTOFF = 8  # any atom with coord < COORD_CUTOFF is a surface atom
FCC_FACET_BUILDERS = {
    (1, 1, 1): (fcc111, (4, 4)),
    (1, 0, 0): (fcc100, (4, 4)),
    (1, 1, 0): (fcc110, (4, 4)),
    (2, 1, 1): (fcc211, (6, 4)),
}
BCC_FACET_BUILDERS = {
    (1, 1, 1): (bcc111, (4, 4)),
    (1, 0, 0): (bcc100, (4, 4)),
    (1, 1, 0): (bcc110, (4, 4)),
}
HCP_FACET_BUILDERS = {
    (0, 0, 0, 1): (hcp0001, (4, 4)),
    (0, 0, 1): (hcp0001, (4, 4)),  # 0001 = 001
}  # can add hcp10m10 later
INTERFACIAL_FACET = (1, 1, 1)  # facet of NP that faces the support
DO_ALPHA_SHAPE = False  # don't use alpha shape to get perimeter atoms
SHOW_ALPHASHAPE = False  # don't display the plot of the interface's alphashape
ALPHA_SHAPE_NAME = "nameless"  # name of alphashape plot
VOXEL_RES = (
    0.26  # resolution for voxel grid that discriminates the NP surface from bulk
)
PROCESSES = -1  # How many processes to run in parallel.
NL_SKIN = 0.05  # Ang; nl skin for distinguishing surface NP atoms; set low because we normally use this on perfect crystals, not structures from MD
# this value seems to work well


def cross_sectional_area(atoms: Atoms) -> float:
    """
    Calculates the contact area of an ML
    !!!NOTE!!!
    It assumes the ML is continguous, i.e. it does not break up,
    as a highly strained ML might

    For a non-ML, use get_interfacial_area_and_radius
    """
    warnings.warn(
        """If the structure is not a CONTINGUOUS MONOLAYER,
            you should use get_interfacial_area_and_radius() instead""",
        category=UserWarning,
    )
    return np.linalg.norm(np.cross(atoms.get_cell()[0], atoms.get_cell()[1]))


def calculate_atomic_density(
    facet: Tuple[int, int, int],
    a: Union[float, Tuple[float, float]],
    np_element: str = NANOPARTICLE_ELEMENT,
) -> float:
    """
    Calculates surface atomic density of an NP given the required substrate-facing facet
    I am implementing ONLY for cubic systems

    Requires:
        facet (Tuple)       e.g. (1,0,0)
        np_element (str)    what element the NP is made of. defaults to Ag
        a (float)           lattice (or pseudo-lattice) constant(s).
                            Optional. If not given, will use ASE's default
    Returns:
        surface area per atom (inverse of surface atomic density) [A^2/atom]
    """
    np_element = np_element.capitalize()
    lattice = bulk(np_element).cell.get_bravais_lattice().__class__.__name__
    if lattice == "FCC":
        surface_ = FCC_FACET_BUILDERS
    elif lattice == "BCC":
        surface_ = BCC_FACET_BUILDERS
    elif lattice in ["HCP", "HEX"]:
        surface_ = HCP_FACET_BUILDERS
    else:
        warnings.warn(
            f"""NP element's crystal system is neither fcc, hcp, nor bcc!
        Going to just assume FCC
        """,
            category=UserWarning,
        )
        surface_ = FCC_FACET_BUILDERS

    facet = tuple(facet)
    if facet not in surface_.keys():
        warnings.warn(
            f"{facet} not implemented. Defaulting to {INTERFACIAL_FACET}",
            category=UserWarning,
        )
        facet = INTERFACIAL_FACET
    if not a:
        a = None
    builder, (nx, ny) = surface_[facet]

    if isinstance(a, float):
        slab = builder(np_element, (nx, ny, 1), a=a)
    elif isinstance(a, tuple):
        a, c = a
        slab = builder(np_element, (nx, ny, 1), a=a, c=c)
    cell_area = np.linalg.norm(np.cross(slab.get_cell()[0], slab.get_cell()[1]))
    #    print(cell_area, cell_area / len(slab))
    return cell_area / len(slab)  # A^^2 / atom


def get_interface(
    atoms: Atoms,
    nl: NeighborList = None,
    analyzer: Analysis = None,
    support_elements: Union[str, List[str]] = None,
    np_element: str = NANOPARTICLE_ELEMENT,
    scaling_factor: float = INTERFACE_SCALING,
) -> Union[List[None], Tuple[Atoms, Atoms, List[int], List[int]]]:
    """
    Get indices of interfacial atoms between NP and Support

    Requires:
        atoms (Atoms):                      atoms object of NP + Support
        nl (NeighborList):                  neighborlist object of whole system (i.e. NP + Support).
                                            Optional but strongly recommended to provide analyzer or nl
        analyzer (Analysis):                Analysis object  of whole system (i.e. NP + Support).
                                            Optional but strongly recommended to provide analyzer or nl
        support_elements (Union[str, List[str]]):
                                            what element(s) the support is made of.
        np_element (str):                   what element the NP is made of. default = {NANOPARTICLE_ELEMENT}
        scaling_factor (float):             scaling factor for covalent radii. Optional
                                            Advised to be > 1.0 to fully capture the interface
    Returns:
        np_interfacial_indices (List[int]):         indices of interfacial (+ perimeter) NP atoms
        support_interfacial_indices (List[int]):    indices of interfacial support atoms

    If no NP-Support bonds found,
        Returns:    None, None
    """
    elements = np.unique(atoms.get_chemical_symbols())
    np_element = np_element.capitalize()
    if not support_elements:
        support_elements = [i for i in elements if i != np_element]
    if isinstance(support_elements, str):
        support_elements = [support_elements]

    if not analyzer:
        if not nl:  # reuse neighborlist if given
            natural_cutoff = (
                np.max([natural_cutoffs(Atoms(element)) for element in elements]) * 2
            )  # no need to search beyond this distance
            nl = setup_neighborlist(atoms, scaling_factor, radial_cutoff=natural_cutoff)
            nl.update(atoms)

        analyzer = setup_analyzer(
            atoms, neighborlist=nl
        )  # reuse Analysis object if given

    NP_Support_bonds = {
        e.capitalize(): analyzer.get_bonds(e.capitalize(), np_element, unique=True)[0]
        for e in support_elements
    }
    support_interfacial_indices, np_interfacial_indices = [], []
    for support_e, bonds in NP_Support_bonds.items():
        if bonds:
            np_interfacial_indice, support_interface_indice = zip(
                *[(j, i) for (i, j) in bonds]
            )
            np_interfacial_indice = list(np_interfacial_indice)
            support_interface_indice = list(support_interface_indice)
        else:
            warnings.warn(
                f"""There are no {support_e}-{np_element} bonds in your system!
            This may be a very small cluster""",
                category=RuntimeWarning,
            )
            np_interfacial_indice = list()
            support_interface_indice = list()

        support_interfacial_indices.extend(support_interface_indice)
        np_interfacial_indices.extend(np_interfacial_indice)

    if not (support_interfacial_indices and np_interfacial_indices):
        print(f"No NP-Support bonds found! Failed!")
        return None, None

    support_interfacial_indices = list(np.unique(support_interfacial_indices))
    np_interfacial_indices = list(np.unique(np_interfacial_indices))

    # In case there's gas adsorbates that have same element as support (e.g. O2 gas and MgO support)
    # Remove any support_interfacial_indices or np_interfacial_indices whose z falls far
    # outside of the MEDIAN of its peers' z-position
    # TODO: only if/when you decide to extend this script's functionality to involve adsorbates

    # CAUTION!! np_interfacial_indices includes perimeter atoms.
    # To separate out the perimeter, use get_perimeter on results from this function
    return np_interfacial_indices, support_interfacial_indices


##the former implementation
# def get_perimeter(atoms: Atoms, interfacial_results: [List[int], List[int]],
#         do_alphashape: bool = DO_ALPHA_SHAPE, plotname: str = "nameless",
#         show: bool = False,) -> Union[List[None], Tuple[Atoms, Atoms, List[int], List[int], float]]:
#     f"""
#     Get indices of perimeter atoms of the NP
#     By:
#         1. Getting the interface
#         2. Convex hull of interfacial atoms to trace the perimeter (misses concave regions);
#         from experience, the bigger the NP, the more it'll undercount the perimeter
#         3. To correct (2), also gets CNmax (the most-coordinated atom amongst those flagged by (2)
#         as perimeter) minus 1, and defines that any atom of coordination less than the max of CNmax and
#         CNmedian (median coordination of all atoms flagged by (2) as perimeter).
#         i.e.:
#             threshold_CN = max(CNmedian, CNmax - 1)
#
#         Note: simply setting threshold_CN = CNmax would be good but a little risky in rare cases
#
#     Requires:
#         atoms (Atoms):                      atoms object of NP + Support
#         interfacial_results: Tuple[List[int], List[int]]. Required; results of the get_interface function,
#         do_alphashape (bool):               whether or not to get perimeter atoms by alphashape rather than only by CN
#                                             default = {DO_ALPHA_SHAPE}
#                                             It's very expensive to optimize alpha for large NPs
#         show (bool):                        whether to display the plot (of the interface's alpha shape) or not.
#                                             default = False
#         plotname (str):                     name of png file (of the interface's alpha shape) to be saved
#                                             Optional. default: 'nameless'
#
#     Returns:
#         perimeter_indices (List[int])      indices of perimeter atoms
#
#     If get_interface function failed,
#         Returns:    None
#     """
#     np_interfacial_indices, support_interfacial_indices = interfacial_results
#     np_interface = atoms[np_interfacial_indices] #NPinterface (i.e. no support)
#
#     #get perimeter atoms
#     perimeter_indices = list()
#
#     #criterion 1: concave hull vertices
#     if do_alphashape:
#         positions_2D = np_interface.positions[:,:2]
#         optimum_alpha = alphashape.optimizealpha(positions_2D)
#         print(f"Optimal alpha for {plotname} = {optimum_alpha}")
#         alpha_shape = alphashape.alphashape(positions_2D, optimum_alpha*.88) #scale down optimum_alpha because it'll tend to overcount
#         perim_set = {tuple(pt) for pt in alpha_shape.exterior.coords[:-1]}
#         perimeter_indices = [i for (i, p) in enumerate(positions_2D) if tuple(p) in perim_set]
#
#         plot_2D_concave_hull(positions_2D, alpha_shape, show = show, index = plotname)
#
#     #criterion 2: coordination threshold
#     ##we have to create a new nl because we are now using a subset of the provided Atoms object
#     neighbors = NeighborList(cutoffs = natural_cutoffs(np_interface, mult = INTERFACE_SCALING),
#             self_interaction = False, bothways = True)
#     neighbors.update(np_interface)
#     CNs = [(atom_index, len(neighbors.get_neighbors(atom_index)[0]))
#             for (atom_index, _) in enumerate(np_interface)] #get number of neighbors for each interfacial NP atom
#                                                              #remember: only interface-interface bonds are considered
#     neighbor_counts = [num_neighbors for (_, num_neighbors) in CNs]
#     medianCN, maxCN = np.median(neighbor_counts), max(neighbor_counts)
#     if do_alphashape:
#         thresholdCN = max(medianCN, maxCN - 1)
#     else:
#         thresholdCN = max(medianCN, maxCN) # - 1)
#     missed_perimeter_indices = [atom_index for (atom_index, num_neighbors)
#             in CNs if (num_neighbors < thresholdCN and atom_index not in perimeter_indices)]
#     perimeter_indices.extend(missed_perimeter_indices)
#
#     #convert the perimeter indices to the indices in the Atoms object
#     perimeter_indices = [np_interfacial_indices[i] for i in perimeter_indices]
#
#     return perimeter_indices


def get_perimeter(
    atoms: Atoms,
    interfacial_results: [List[int], List[int]],
    do_alphashape: bool = DO_ALPHA_SHAPE, #useless; will always run
    plotname: str = "nameless",
    show: bool = False,
) -> Union[List[None], Tuple[Atoms, Atoms, List[int], List[int], float]]:
    f"""
    Get indices of perimeter atoms of the NP
    By:
        1. Getting the interface
        2. Convex hull of interfacial atoms to trace the perimeter (misses concave regions);
        from experience, the bigger the NP, the more it'll undercount the perimeter

    !!!WARNING!!! It's very expensive to optimize alpha for large NPs

    Requires:
        atoms (Atoms):                      atoms object of NP + Support
        interfacial_results: Tuple[List[int], List[int]]. Required; results of the get_interface function,
        show (bool):                        whether to display the plot (of the interface's alpha shape) or not.
                                            default = False
        plotname (str):                     name of png file (of the interface's alpha shape) to be saved
                                            Optional. default: 'nameless'

    Returns:
        perimeter_indices (List[int])      indices of perimeter atoms

    If get_interface function failed,
        Returns:    None
    """
    np_interfacial_indices, support_interfacial_indices = interfacial_results
    np_interface = atoms[np_interfacial_indices]  # NPinterface (i.e. no support)

    # get perimeter atoms
    perimeter_indices = list()
    #for tolerance
    n_dp = 2

    if True:
        positions_2D = np_interface.positions[:, :2]
        try:
            optimum_alpha = alphashape.optimizealpha(positions_2D)
            alpha_shape = alphashape.alphashape(
                positions_2D, optimum_alpha * 0.88
            )  # scale down optimum_alpha because it'll tend to overcount
            #get perimeter; also apply a tolerance because of floats
            perim_set = {tuple(np.round(pt, n_dp)) for pt in alpha_shape.exterior.coords[:-1]}
            perimeter_indices = [
                i for (i, p) in enumerate(positions_2D) if tuple(np.round(p, n_dp)) in perim_set
            ]

            plot_2D_concave_hull(positions_2D, alpha_shape, show=show, index=plotname)

        except AttributeError: #perimeter is one atom; get it by the CN method instead
#            view(atoms)
            perimeter_indices = get_perimeter_CN(
                    atoms,
                    interfacial_results=interfacial_results,
                    )

            #I think the below is almost certainly correct in every case,
            #but let's be safe and not use it
            #perimeter_indices = list(np_interfacial_indices) #the entire interface is the perimeter

            return perimeter_indices

        except Exception as unknown_err:
            print(f"Unexpected error occured:\t{unknown_err}\n\nReturning interfacial indices")

            return np_interfacial_indices


        #convert the perimeter indices to the indices in the Atoms object
        perimeter_indices = [np_interfacial_indices[i] for i in perimeter_indices]

    return perimeter_indices


def get_perimeter_CN(
        atoms: Atoms,
        interfacial_results: [List[int], List[int]],
        ) -> Union[List[None], Tuple[Atoms, Atoms, List[int], List[int], float]]:
    f"""
    Get indices of perimeter atoms of the NP
    By:
       1. Getting the interface
       2. Gets CNmax (the most-coordinated atom amongst those flagged by (2)
       as perimeter) minus 1, and defines that any atom of coordination less than the max of CNmax and
       CNmedian (median coordination of all atoms flagged by (2) as perimeter).
       i.e.:
           threshold_CN = max(CNmedian, CNmax - 1)
        Note: simply setting threshold_CN = CNmax would be good but a little risky in rare cases

    Requires:
       atoms (Atoms):                      atoms object of NP + Support
       interfacial_results: Tuple[List[int], List[int]]. Required; results of the get_interface function,

    Returns:
       perimeter_indices (List[int])      indices of perimeter atoms

    If get_interface function failed,
         Returns:    None
    """
    warnings.warn("Getting perimeter by CN. May be risky sometimes",
            category = UserWarning)
    np_interfacial_indices, support_interfacial_indices = interfacial_results
    np_interface = atoms[np_interfacial_indices] #NPinterface (i.e. no support)

    #get perimeter atoms
    perimeter_indices = list()

    #The only criterion: coordination threshold
    ##we have to create a new nl because we are now using a subset of the provided Atoms object
    neighbors = NeighborList(cutoffs = natural_cutoffs(np_interface, mult = INTERFACE_SCALING),
            self_interaction = False, bothways = True)
    neighbors.update(np_interface)
    CNs = [(atom_index, len(neighbors.get_neighbors(atom_index)[0]))
            for (atom_index, _) in enumerate(np_interface)] #get number of neighbors for each interfacial NP atom
                                                             #remember: only interface-interface bonds are considered
    neighbor_counts = [num_neighbors for (_, num_neighbors) in CNs]
    medianCN, maxCN = np.median(neighbor_counts), max(neighbor_counts)
    thresholdCN = max(medianCN, maxCN) # - 1)
    missed_perimeter_indices = [atom_index for (atom_index, num_neighbors)
            in CNs if (num_neighbors < thresholdCN and atom_index not in perimeter_indices)]
    perimeter_indices.extend(missed_perimeter_indices)
     #convert the perimeter indices to the indices in the Atoms object
    perimeter_indices = [np_interfacial_indices[i] for i in perimeter_indices]

    return perimeter_indices



def get_np_surface_by_CN(
    atoms: Atoms,
    nl: NeighborList = None,
    support_elements: Union[str, List[str]] = None,
    np_element: str = NANOPARTICLE_ELEMENT,
    scaling_factor: float = CN_SCALING,
    coord_cutoff: int = None,
) -> Tuple[List[int], List[int]]:
    f"""
    Separate NP surface's from bulk based on CN (coordination numbers)

    Requires:
        atoms (Atoms):                      atoms object of NP + Support
        nl (NeighborList):                  neighborlist object of whole system (i.e. NP + Support).
                                            Optional but strongly recommended to provide analyzer or nl
        support_elements (Union[str, List[str]]):
                                            what element(s) the support is made of.
        np_element (str):                   what element the NP is made of. Default = {NANOPARTICLE_ELEMENT}
        scaling_factor (float):             scaling factor for covalent radii.
                                            optional. default = {CN_SCALING}
        coord_cutoff (int):                 coordination cutoff for discriminating surface from bulk atoms
                                            optional. default = {FCC_AND_HCP_COORD_CUTOFF} for FCC and {BCC_COORD_CUTOFF} for BCC
    Returns
        surface_plus_interface_indices (List[int]):
                                            indices of nanoparticle surface atoms.
                                            Note: This INCLUDES THE INTERFACE + PERIMETER ATOMS
    """
    if not support_elements:
        support_elements = [i for i in elements if i != np_element]
    if isinstance(support_elements, str):
        support_elements = [support_elements]
    if not nl:
        # create neighborlist object
        cutoff_kwargs = {element.capitalize(): 0 for element in support_elements}
        scaling_factor = 1.0
        cutoffs = natural_cutoffs(
            atoms, mult=scaling_factor, **cutoff_kwargs
        )  # set non NP atoms to zero cutoffs
        nl = NeighborList(
            cutoffs=cutoffs, self_interaction=False, bothways=True, skin=NL_SKIN
        )
        nl.update(atoms)

    # apply criterion
    if not coord_cutoff:
        lattice = bulk(np_element).cell.get_bravais_lattice().__class__.__name__
        if lattice in [
            "FCC",
            "HCP",
            "HEX",
        ]:  # should confirm later, but it seems ASE has only hex not hcp (they mean the same)
            coord_cutoff = FCC_AND_HCP_COORD_CUTOFF
        elif lattice == "BCC":
            coord_cutoff = BCC_COORD_CUTOFF
        else:
            warnings.warn(
                f"""NP element's crystal system is neither fcc, hcp, nor bcc!
            Setting coordination cutoff to that of fcc/hcp ({FCC_AND_HCP_COORD_CUTOFF})!
            """,
                category=UserWarning,
            )

    CN = [
        (index, len(nl.get_neighbors(index)[0]))
        for index, i in enumerate(atoms)
        if i.symbol == np_element
    ]
    bulk_indices = [i for i, j in CN if j >= coord_cutoff]
    surface_plus_interface_indices = [i for i, j in CN if i not in bulk_indices]

    return surface_plus_interface_indices


def get_interfacial_area_and_radius(
    atoms: Atoms,
    interfacial_results: Tuple[List[int], List[int]],
    interfacial_facet: Tuple[int, int, int],
    np_element: str = NANOPARTICLE_ELEMENT,
) -> Union[float, None]:
    f"""
    Calculate interfacial area (aka footprint_area) and footprint radius of a supported NP
    To convert from area to radius: Assumes the interface is flat and a CIRCLE

    Requires:
        atoms (Atoms):                              atoms object of NP + Support
        interfacial_results: Tuple[List[int], List[int]]
                                                    (Required) results of the get_interface function,
        interfacial_facet (Tuple[int, int, int]):   bracketed comma-separated ints to define
                                                    the NP facet facing the interface, e.g. (1,0,0)
                                                    Default = {INTERFACIAL_FACET}
        np_element (str):                           what element the NP is made of. Default = {NANOPARTICLE_ELEMENT}

    Returns:
        footprint_area (float):             interfacial area in A^2
        footprint_radius (float):   footprint radius. in A
    """
    if not interfacial_facet:
        warnings.warn(
            f"Facet not supplied. Defaulting to {INTERFACIAL_FACET}",
            category=UserWarning,
        )
        interfacial_facet = INTERFACIAL_FACET
    np_interfacial_indices, support_interfacial_indices = interfacial_results
    np_interface = atoms[np_interfacial_indices]
    ##we have to set this up because we're treating a subset of the provided Atoms object
    analyzer = setup_analyzer(np_interface, neighborlist=None)
    bonds = analyzer.get_bonds(np_element, np_element, unique=True)
    try:
        mean_bond_length = statistics.mean(analyzer.get_values(bonds)[0])
    except IndexError:  # no Np-Np bonds
        mean_bond_length = natural_cutoffs(Atoms(np_element))[0] * 2

    #    warnings.warn(f"""Converting interfacial area to footprint radius is assuming the interface
    #    is flat and a CIRCLE. Untrustworthy for small NPs and clusters!

    #    Lastly, for a (1,0,0) facet, interface will be square or rectangle,
    #    so, radius should strictly be undefined""", category = UserWarning)

    lattice = bulk(np_element).cell.get_bravais_lattice().__class__.__name__
    if lattice == "FCC":
        pseudo_lattice = mean_bond_length * math.sqrt(2)
    elif lattice == "BCC":
        pseudo_lattice = mean_bond_length * 2 / math.sqrt(3)
    elif lattice in ["HCP", "HEX"]:
        pseudo_lattice = (mean_bond_length, mean_bond_length * math.sqrt(8 / 3))

    footprint_area = calculate_atomic_density(
        interfacial_facet, a=pseudo_lattice, np_element=np_element
    ) * len(np_interfacial_indices)

    return footprint_area, (footprint_area / np.pi) ** (1 / 2)  # A^2, A


def spherical_cap_footprint_radius(
    atoms: Atoms,
    np_element: str = NANOPARTICLE_ELEMENT,
    min_resolution: float = VOXEL_RES,
) -> float:
    f"""
    Get footprint radius by fitting the NP to a spherical cap rather than calculating
    from interfacial atoms like get_interfacial_area_and_radius()

    Requires:
        atoms (Atoms):              atoms object. NP on support
        np_element (str):                           what element the NP is made of. Default = {NANOPARTICLE_ELEMENT}
        min_resolution (float):     resolution for voxel grid that discriminates the NP surface from bulk.
                                    default = {VOXEL_RES}
    """
    angle, footprint_area, footprint_radius = calculate_contact_angle(
        atoms=atoms, np_element=np_element, min_resolution=min_resolution
    )

    return footprint_radius  # A^2


def plot_2D_concave_hull(
    positions: np.ndarray,
    hull,
    index: Union[str, int, float],
    show: bool = SHOW_ALPHASHAPE,
) -> None:
    """
    Plot 2D alpha shape of the NP
    Requires:
        positions:  positions array
        hull:       concave hull object
        index:      name of png file to be saved
        show:       whether to display (and save) the plot or do neither
    Returns:
        None, or a saved and displayed plot
    """
    plt.close("all")
    ##2D convex hull
    ##below is written by ChatGPT o3; confirmed to work
    if not isinstance(hull, (Polygon, MultiPolygon)):
        raise TypeError(
            "plot_2D_concave_hull now supports only shapely Polygons / MultiPolygons. "
            "Ensure you passed the result from alphashape.alphashape()."
        )

    def _plot_polygon(poly: Polygon):
        x, y = poly.exterior.xy
        plt.plot(x, y, "b-", alpha=0.6)

    #    plt.close("all")
    plt.figure()
    plt.plot(positions[:, 0], positions[:, 1], "ro", markersize=3)

    if isinstance(hull, Polygon):
        _plot_polygon(hull)
    else:  # MultiPolygon
        for poly in hull.geoms:
            _plot_polygon(poly)

    plt.xlabel("x / Å")
    plt.ylabel("y / Å")
    plt.title("2‑D α‑shape of NP interface")
    plt.savefig(f"{index}_hull2D.png", dpi=250, bbox_inches="tight")
    if show:
        plt.show()


# The spirit of this function fits more in fit_support
# but then having it there would cause a circular import
def get_adsorption_height(
    atoms: Atoms, element: str = NANOPARTICLE_ELEMENT, both_methods: bool = False
) -> Union[Tuple[float, float], float]:
    f"""
    Calculate current height of NP above surface
    Uses two methods:
        1. Min(Ag) - Max(MgO)
        2. Mean(Ag interface) - Mean(MgO interface)

    Will use just the first, if the user specifies so

    Inputs:     Supported_NP
                element the NP is made of. Default = {NANOPARTICLE_ELEMENT}
                both_methods (whether or not to use both methods above). Default = False
    Returns:    Adsorption height by method 1,
                by method 2 if requested
    """
    # method 1
    silvers, surface = divider(atoms, element=element)

    np_depth = min(silvers.positions[:, 2])
    surface_height = max(surface.positions[:, 2])
    height_1 = np_depth - surface_height

    if both_methods:
        # method 2
        np_interfacial_indices, support_interfacial_indices = get_interface(
            atoms, np_element=element, support_elements=None
        )
        height_2 = np.mean(atoms[np_interfacial_indices].positions[:, 2]) - np.mean(
            atoms[support_interfacial_indices].positions[:, 2]
        )

        return height_1, height_2

    return height_1


def discriminate(
    atoms: Atoms,
    nl: NeighborList,
    analyzer: Analysis,
    support_elements: Union[str, List[str]],
    np_element: str = NANOPARTICLE_ELEMENT,
    interface_scaling: float = INTERFACE_SCALING,
    surface_scaling: float = CN_SCALING,
    coord_cutoff: int = None,
    do_alphashape: bool = DO_ALPHA_SHAPE,
    plotname: str = ALPHA_SHAPE_NAME,
    show: bool = SHOW_ALPHASHAPE,
) -> Tuple[
    List[int],
    List[int],
    List[int],
    List[int],
    List[int],
]:
    f"""Main function for separating a supported NP into NP surface, NP Bulk, NP interface (non-perimeter),
    NP perimeter, and Support

    Requires:
        atoms (Atoms):                      atoms object of NP + Support
        nl (NeighborList):                  neighborlist object of whole system (i.e. NP + Support).
                                            Optional but strongly recommended to provide analyzer or nl
        analyzer (Analysis):                Analysis object  of whole system (i.e. NP + Support).
                                            Optional but strongly recommended to provide analyzer or nl
        support_elements (Union[str, List[str]]):
                                            what element(s) the support is made of.
        np_element (str):                   what element the NP is made of. default = {NANOPARTICLE_ELEMENT}
        interface_scaling (float):          scaling factor for covalent radii to discriminate interface from rest of system.
                                            default = {INTERFACE_SCALING}.
        surface_scaling (float):             scaling factor for covalent radii. Default = {CN_SCALING}
        coord_cutoff (int):                 coordination cutoff for discriminating surface from bulk atoms
                                            optional. default = {FCC_AND_HCP_COORD_CUTOFF} for FCC and {BCC_COORD_CUTOFF} for BCC
        do_alphashape (bool):               whether or not to get perimeter atoms by alphashape rather than only by CN
                                            default = {DO_ALPHA_SHAPE}
                                            !!!It's very expensive to optimize alpha for large NPs!!!
        plotname (str):                     name of png file (of the interface's alpha shape) to be saved
                                            Optional. default: {ALPHA_SHAPE_NAME}
        show (bool):                        whether to display the plot (of the interface's alpha shape) or not.
                                            default = {SHOW_ALPHASHAPE}

    Returns:
        np_surface (List[int]):             indices of NP surface atoms (excluding interface)
        np_bulk (List[int]):                indices of bulk NP atoms
        np_interface (List[int]):           indices of NP interfacial atoms (excluding perimeter)
        np_perimeter (List[int]):           indices of NP perimeter atoms
        substrate_interface (List[int]):     indices of support interfacial atoms
        substrate (List[int]):              indices of support (including interfacial support) atoms

    Returns [0,0,0,0,0,0] if get_interface didnt get any NP-support bonds
    """
    elements = np.unique(atoms.get_chemical_symbols())
    np_element = np_element.capitalize()
    if not support_elements:
        support_elements = [i for i in elements if i != np_element]
    if isinstance(support_elements, str):
        support_elements = [support_elements.capitalize()]
    else:
        support_elements = [i.capitalize() for i in support_elements]
    np_and_support_elements = support_elements + [np_element]
    check_elements = [i in elements for i in np_and_support_elements]
    if not np.all(check_elements):
        raise ValueError(
            f"Some of the given elements ({np_and_support_elements}) not in Atoms object!"
        )

    if not analyzer:
        if not nl:  # reuse neighborlist if given
            natural_cutoff = (
                np.max([natural_cutoffs(Atoms(element)) for element in elements]) * 2
            )  # no need to search beyond this distance
            nl = setup_neighborlist(
                atoms, scaling_factor=interface_scaling, radial_cutoff=natural_cutoff
            )  # , skin = NL_SKIN) curse this!
            nl.update(atoms)

        analyzer = setup_analyzer(
            atoms, neighborlist=nl
        )  # reuse Analysis object if given

    nps = [index for index, _ in enumerate(atoms) if atoms[index].symbol == np_element]
    substrate = [
        index
        for index, _ in enumerate(atoms)
        if atoms[index].symbol in support_elements
    ]

    try:
        np_interface_plus_perimeter, substrate_interface = get_interface(
            atoms,
            nl=nl,
            analyzer=analyzer,
            np_element=np_element,
            support_elements=support_elements,
            scaling_factor=interface_scaling,
        )

        np_perimeter = get_perimeter(
            atoms,
            interfacial_results=(np_interface_plus_perimeter, substrate_interface),
            do_alphashape=do_alphashape,
            plotname=plotname,
            show=show,
        )

        np_interface = [i for i in np_interface_plus_perimeter if i not in np_perimeter]
        nl = None  # unfortunately we have to reset in order to set skin to zero for the surface discriminaion
        np_surface_plus_interface = get_np_surface_by_CN(
            atoms,
            nl=nl,
            np_element=np_element,
            support_elements=support_elements,
            scaling_factor=surface_scaling,
            coord_cutoff=coord_cutoff,
        )

        np_surface = [
            i for i in np_surface_plus_interface if i not in np_interface_plus_perimeter
        ]
        np_bulk = [i for i in nps if i not in np_surface_plus_interface]

    except TypeError:  # get_interface returned None, None
        (
            np_surface,
            np_bulk,
            np_interface,
            np_perimeter,
            substrate_interface,
            substrate,
        ) = (0, 0, 0, 0, 0, 0)

    return (
        np_surface,
        np_bulk,
        np_interface,
        np_perimeter,
        substrate_interface,
        substrate,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="""Splits up a supported NP into surface, substrate, bulk, perimeter, and interface
    Also calculates interfacial radius by any/both of two methods if requested"""
    )

    parser.add_argument(
        "--traj",
        "-t",
        type=str,
        required=True,
        help="Traj file on which to run the script. Required",
    )
    parser.add_argument(
        "--np_element",
        "-ne",
        type=str,
        default=None,  # NANOPARTICLE_ELEMENT,
        help=f"Element of which the NP is composed. by default we'll read atoms.info['np_element']",
    )
    parser.add_argument(
        "--interfacial_facet",
        "-if",
        type=str,
        nargs="+",
        default=None,
        help=f"Facet of NP that faces the support. by default we'll read atoms.info['interfacial_facet']",
    )
    parser.add_argument(
        "--support_elements",
        "-se",
        type=str,
        nargs="+",
        default=None,
        help="""List of elements of which
    the substrate is composed. In the format: a b c d. Defaults to elements absent from the NP""",
    )
    parser.add_argument(
        "--interface_scaling",
        "-is",
        type=float,
        default=INTERFACE_SCALING,
        help=f"""scaling factor for interfacial atoms' covalent radii. Default = {INTERFACE_SCALING}""",
    )
    parser.add_argument(
        "--surface_scaling",
        "-ss",
        type=float,
        default=CN_SCALING,
        help=f"""scaling factor for discriminating NP surface from bulk. Default = {CN_SCALING}""",
    )
    parser.add_argument(
        "--do_alphashape",
        action="store_const",
        const=True,
        default=DO_ALPHA_SHAPE,
        help="""To also use alpha shape to get perimeter atoms""",
    )
    parser.add_argument(
        "--show_alphashape",
        action="store_const",
        const=True,
        default=SHOW_ALPHASHAPE,
        help="""To display the plot of the interface's alphashape.""",
    )
    parser.add_argument(
        "--plotname",
        "-an",
        type=str,
        default=ALPHA_SHAPE_NAME,
        help="""Name of alphashape plot (if used).
            Default = {ALPHA_SHAPE_NAME}""",
    )
    parser.add_argument(
        "--processes",
        "-p",
        type=int,
        default=PROCESSES,
        help=f"How many processes to run in parallel. default = {PROCESSES}",
    )
    parser.add_argument(
        "--get_contact_angles",
        action="store_const",
        const=True,
        default=False,
        help="""return contact angles from atoms.info['Theta']""",
    )
    parser.add_argument(
        "--get_footprint_radius_by_SCM",
        action="store_const",
        const=True,
        default=False,
        help="""calculate footprint radius by fitting a spherical cap to the NP""",
    )
    parser.add_argument(
        "--get_footprint_radius_by_count",
        action="store_const",
        const=True,
        default=False,
        help="""calculate footprint radius by counting the interfacial atoms""",
    )

    args = parser.parse_args()
    atoms = read(args.traj, ":")
    atoms = atoms if isinstance(atoms, list) else [atoms]
    processes = args.processes
    try:
        np_element_list = [
            image.info["np_element"] if not args.np_element else args.np_element
            for image in atoms
        ]
    except KeyError:
        warnings.warn(f"Assuming np_element == 'Ag'", category=UserWarning)
        np_element_list = ["Ag" for image in atoms]

    if args.get_footprint_radius_by_count or args.get_footprint_radius_by_SCM:
        interfacial_facet_list = [
            (
                image.info["interfacial_facet"]
                if not args.interfacial_facet
                else args.interfacial_facet
            )
            for image in atoms
        ]
    else:
        interfacial_facet_list = [None] * len(atoms)

    with Parallel(n_jobs=processes) as parallel:
        results = parallel(
            delayed(discriminate)(
                image,
                np_element=np_element_list[index],
                support_elements=args.support_elements,
                interface_scaling=args.interface_scaling,
                surface_scaling=args.surface_scaling,
                do_alphashape=args.do_alphashape,
                plotname=args.plotname,
                show=args.show_alphashape,
                nl=None,
                analyzer=None,
            )
            for index, image in enumerate(
                tqdm(atoms, desc="Discriminating..", total=len(atoms))
            )
        )

    np_surface, np_bulk, np_interface, np_perimeter, substrate_interface, substrate = (
        zip(*results)
    )
    np_total = [
        (j + np_bulk[i] + np_perimeter[i] + np_interface[i])
        for i, j in enumerate(np_surface)
    ]

    np_interface_plus_peri = [j + np_perimeter[i] for i, j in enumerate(np_interface)]
    interfacial_results_list = [
        (j, substrate_interface[i]) for i, j in enumerate(np_interface_plus_peri)
    ]

    if args.get_contact_angles:
        print("Getting contact angles")
        contact_angles = [
            image.info["Theta"] if "Theta" in image.info.keys() else None
            for image in atoms
        ]
        if np.any(contact_angles is None):
            print(
                "At least one image doesn't have contact angle stored as i.info['Theta']"
            )

    if args.get_footprint_radius_by_SCM:
        with Parallel(n_jobs=processes) as parallel:
            radii_scm = parallel(
                delayed(spherical_cap_footprint_radius)(
                    image, np_element=np_element_list[index]
                )
                for index, image in enumerate(
                    tqdm(atoms, desc="SCM footprint radii..", total=len(atoms))
                )
            )

    if args.get_footprint_radius_by_count:
        with Parallel(n_jobs=processes) as parallel:
            results = parallel(
                delayed(get_interfacial_area_and_radius)(
                    image,
                    interfacial_results=interfacial_results_list[index],
                    np_element=np_element_list[index],
                    interfacial_facet=interfacial_facet_list[index],
                )
                for index, image in enumerate(
                    tqdm(
                        atoms,
                        desc="Footprint radii by counting interfacial atoms..",
                        total=len(atoms),
                    )
                )
            )

        radii = [j for i, j in results]  # results = areas, radii

    tags_key = f"""\n\nKEY FOR TAGS:\n
    Bulk:\t{TAGS_DICT["bulk"]}
    Surface:\t{TAGS_DICT["surface"]}
    Interface:\t{TAGS_DICT["interface"]}
    Perimeter:\t{TAGS_DICT["perimeter"]}
    Substrate:\t{TAGS_DICT["substrate"]}
    """
    ASCIIColors.print(
        tags_key,
        color=ASCIIColors.color_green,
        style=ASCIIColors.style_bold,
        background=ASCIIColors.color_black,
        end="\n\n",
        flush=True,
        file=stdout,
    )

    # return Traj but with atoms' identities switched
    # return atoms object with tags modified
    atoms_copied = deepcopy(atoms)
    atoms_interface: List[Atoms] = []
    for index, image in enumerate(
        tqdm(
            atoms_copied,
            total=len(atoms_copied),
            desc="Changing identities and setting tags..",
        )
    ):

        substrate_indices = [
            i for i, atom in enumerate(image) if atom.symbol != np_element_list[index]
        ]

        np_surf, np_int, np_peri = (
            np_surface[index],
            np_interface[index],
            np_perimeter[index],
        )
        tags = np.zeros(shape=len(image))

        for i in np_int:
            image[i].symbol = "Cu"
        for i in np_peri:
            image[i].symbol = "Mo"
        for i in np_surf:
            image[i].symbol = "Pd"

        ##for the interfacial TRAJ
        interfacial_image = deepcopy(image)
        throw_away_indices = [i for i, atom in enumerate(interfacial_image)\
                if atom.symbol not in ["Cu", "Mo"]]
        del interfacial_image[throw_away_indices]
        atoms_interface.append(interfacial_image)


        tags[np_int] = TAGS_DICT["interface"]
        tags[np_peri] = TAGS_DICT["perimeter"]
        tags[np_surf] = TAGS_DICT["surface"]
        tags[substrate_indices] = TAGS_DICT["substrate"]

        atoms[index].set_tags(tags)


    write("switched.traj", atoms_copied)
    ##Also, to see if we correctly discriminated the interface from perimeter
    ##write out the interface (of NP only)
    print("Writing out just the NP's interface")
    write("interface.traj", atoms_interface)

    output_traj = splitext(basename(args.traj))[0] + "_tagged.traj"
    write(output_traj, atoms)

    print("Saving results")
    ##time to write results and plot
    # write the input for the atom_counter
    if args.get_footprint_radius_by_count:
        radius_of_curvature = np.zeros(len(radii))  # R not r
        # write radii by counting interfacial atoms
        df = pd.DataFrame(
            {
                "r (A)": radii,
                "R (A)": radius_of_curvature,
                "Theta": contact_angles,
                "Element": np_element_list,
                "Facet": interfacial_facet_list,
            }
        )
        df.to_csv("input1.csv", index=False)

    if args.get_footprint_radius_by_SCM:
        # write radii by sphere fitting
        df = pd.DataFrame(
            {
                "r (A)": radii_scm,
                "R (A)": radius_of_curvature,
                "Theta": contact_angles,
                "Element": np_element_list,
                "Facet": interfacial_facet_list,
            }
        )
        df.to_csv("input2.csv", index=False)

    # write atomistic results
    np_perimeter_len = [len(i) for i in np_perimeter]
    np_interface_len = [len(i) for i in np_interface]
    np_surface_len = [len(i) for i in np_surface]
    np_total_len = [len(i) for i in np_total]

    df = pd.DataFrame(
        {
            "Perimeter": np_perimeter_len,
            "Interface": np_interface_len,
            "Surface": np_surface_len,
            "Total": np_total_len,
        }
    )
    df.to_csv("output_atomistic.csv", index=False)
