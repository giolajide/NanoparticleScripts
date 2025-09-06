#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from copy import deepcopy
from scipy.spatial import ConvexHull
from ase.visualize import view
from matplotlib import pyplot as plt
from ase.build import bulk
from tqdm import tqdm
import statistics
import functools
from ase.io import (read, write)
import numpy as np
from ase import (Atoms, Atom)
import math
from sys import argv, exit
import alphashape
from shapely.geometry import Polygon, MultiPolygon
from ase.geometry.analysis import Analysis
from typing import (Tuple, Literal, List, Union)
from ase.neighborlist import (natural_cutoffs, NeighborList)
import warnings
from os import system
from joblib import Parallel, delayed
import trimesh
from descartes import PolygonPatch
from scipy.spatial import ConvexHull
from ovito.modifiers import ConstructSurfaceModifier
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
import mdtraj as md
#from scipy.ndimage import center_of_mass
from utilities.print_bonds import setup_neighborlist, setup_analyzer#, SKIN
from utilities.sphere_and_scm_conversions import (
        _natoms_to_sphere_diameter, MIN_RES,
        get_SCM_diameter_and_theta, _alpha
        )
from contact_angle import calculate_contact_angle
from catalytic_descriptors import (
        get_interfacial_area_and_radius, get_interface,
        NANOPARTICLE_ELEMENT, INTERFACE_SCALING,
        INTERFACIAL_FACET, CN_SCALING, DO_ALPHA_SHAPE,
        SHOW_ALPHASHAPE, ALPHA_SHAPE_NAME
        )
#import vg #pip install vg
from utilities.utilities import unsigned_vector_angle

"""
July-August 2025

Here we gather together all methods to calculate interfacial and surface areas
of supported nanoparticles.

Said methods are:

    1. For interfacial areas:
        a. AlphaShape (or ConvexHull) as implemented in AlphaShape Toolbox
        b. shoelace formula + 'perimeter-covalent-radii'

    2. For outer surface areas:
        a. AlphaShape (or ConvexHull) as implemented in AlphaShape Toolbox
        b. AlphaShape as implemented in OVITO
        c. Modelling the Nanoparticle with a spherical cap
        d. Modelling the Nanoparticle with a sphere
        e. Rolling probe (Shake-Rupley algorithm) as implemented in mdtraj
"""
NP_CUTOFF = 1000
FALL_BACK = 0.5 #optimal alpha, I've noticed, is almost always about 0.48
SMOOTHING_LEVEL = 0 #default = 8
ALPHA = None #Ang; following the rule of thumb
FILENAME = "nameless"
PROBE_RADIUS = 1. #Ang


class AlphaShapeToolboxError(ValueError):
    """
    bare-bones class for errors in alpha shape (from alphashape toolbox) related tasks
    """

class AlphaShapeOvitoError(ValueError):
    """
    bare-bones class for errors in alpha shape (by OVITO) related tasks
    """

class MDTRAJRollingProbeError(ValueError):
    """
    bare-bones class for errors in the rolling probe method of mdtraj
    """

def alpha_shape_toolbox_area(
        positions: np.ndarray,
        area: Literal["interface", "surface"],
        method: Literal["convex", "concave"],
        np_cutoff: int = NP_CUTOFF,
        fall_back: float = FALL_BACK
        ) -> Tuple[float, float]:
    f"""
    Calculates an alpha shape and returns the area enclosed by it
    returns the volume too, which will be None if area was set to 'interface'
    Uses the alphashape toolbox, solving for an optimum alpha except in the below case:
        will refuse to optimize alpha is NP size is really large but will use 'fall_back' instead

    Requires:
        positions (np.ndarray)                  Array of positions of nanoparticle ALONE
                                                Should be 2D (i.e. just x and y coords) if you want interfacial area
                                                and 3D for surface area
        area (Literal["interface", "surface"])  whether to calculate surface or interfacial area
        method (Literal["convex", "concave"])   whether to calculate a convex or concave hull
        np_cutoff (int)                         nanoparticles of size < np_cutoff will not have alpha optimized,
                                                will instead fall back to alpha of {FALL_BACK}
        fall_back (float)                       alpha for overly-big particles will be this value

    Returns:
        area (float)                            surface or interfacial area
        volume (float or None)                  enclosed volume. will be None if you request interfacial area
    """
    area = area.lower()
    method = method.lower()
    positions = np.array(positions)
    np_size, dimensions = positions.shape
    if dimensions not in [2, 3]:
        raise ValueError(f"Positions array should be 2D or 3D; currently it is {dimensions}")
    if (dimensions == 2 and area == "surface"):
        raise ValueError("You can't supply 2D positions yet request outer surface areas")
    if (dimensions == 3 and area == "interface"):
        warnings.warn("3D positions supplied! Will delete 3rd dimension",
                category = UserWarning)
        positions = positions[:,:2]
        dimensions = 2
    if area == "interface":
        warnings.warn(f"Hope the positions you supplied are of ONLY the interface?",
                category = UserWarning)
    elif area == "surface":
        warnings.warn(f"Hope the positions you supplied are of the ENTIRE NP, not ONLY the interface?",
                category = UserWarning)

    area, volume = None, None

    if method == "concave":
        if np_size > np_cutoff:
            warnings.warn(f"NP size > {NP_CUTOFF}; will NOT optimize alpha but shall fall back to {fall_back}"
                    , category = UserWarning)
#            fall_backs = np.linspace(fall_back, 0., 5)
#            for fall_back in fall_back:
#                try:
            optimum_alpha = fall_back
        else:
            optimum_alpha = alphashape.optimizealpha(positions)

        try:
            alpha_shape = alphashape.alphashape(positions, optimum_alpha*.88) #scale down optimum_alpha because it'll tend to overcount
        except Exception as e:
            raise AlphaShapeToolboxError(f"Alphashape could not be generated. See following error:\t{e}")

        if dimensions == 2:
            area = alpha_shape.area
            if (isinstance(area, float) and area > 0):
                return area, volume
            else:
                raise AlphaShapeToolboxError("Alphashape area calculation failed")

        elif dimensions == 3:
            mesh = trimesh.Trimesh(vertices = alpha_shape.vertices, faces = alpha_shape.faces)
            volume = abs(mesh.volume)
            if (isinstance(volume, float) and volume > 0):
                area = mesh.area
            else:
                raise AlphaShapeToolboxError("Alphashape area calculation failed")

    elif method == "convex":
        hull = ConvexHull(positions)
        area = hull.area
        if dimensions == 3:
            volume = hull.volume

    return area, volume


def alpha_shape_ovito_area(
        atoms: Atoms,
        area: Literal["interface", "surface"],
        method: Literal["convex", "concave"],
        alpha: int = ALPHA,
        smoothing_level: int = SMOOTHING_LEVEL,
        ) -> Tuple[float, float]:
    """
    Calculate surface area by OVITO's implementation of the alphashape
    returns the volume too

    Note: Right now (July 18th, 2025), can't calculate interfacial area

    Requires:
        atoms (Atoms)                           Atoms object of UNSUPPORTED nanoparticle
        area (Literal["interface", "surface"])  whether to calculate surface or interfacial area
        method (Literal["convex", "concave"])   whether to calculate a convex or concave hull
        alpha (int)                             alpha (aka. probe radius) to use, in ANGSTROM. default = {ALPHA}
        smoothing_level (int)                   how many iterations of the smoothing algorithm to perform.
                                                default = {SMOOTHING_LEVEL}

    Returns:
        area (float)                            surface or interfacial area
        volume (float or None)                  enclosed volume.
    """
    area = area.lower()
    method = method.lower()
    positions = np.array(positions)

    if area == "interface":
        raise NotImplementedError(f"!!Interfacial area calculation not implemented!!! Check back later")

    safe_cutoff = max(natural_cutoff(atoms)) * 2.1
    if not alpha:
        alpha = safe_cutoff
    elif alpha >= safe_cutoff:
        warnings.warn(f"Alpha of {alpha} is too high! Reducing to {safe_cutoff} Ang",
                category = RuntimeWarning)
        alpha = safe_cutoff
    elif alpha <= safe_cutoff / 2:
        safe_cutoff /= 1.5
        warnings.warn(f"Alpha of {alpha} is too low! Increasing to {safe_cutoff} Ang",
                category = RuntimeWarning)
        alpha = safe_cutoff
    if dimensions not in [2, 3]:
        raise ValueError(f"Positions array should be 2D or 3D; currently it is {dimensions}")
    if (dimensions == 2 and area == "surface"):
        raise ValueError("You can't supply 2D positions yet request outer surface areas")

    #https://docs.ovito.org/python/modules/ovito_modifiers.html#ovito.modifiers.ConstructSurfaceModifier
    atoms = ase_to_ovito(atoms)
    try:
        pipeline = Pipeline(source = StaticSource(data = atoms))
        pipeline.modifiers.append(ConstructSurfaceModifier(
            method = ConstructSurfaceModifier.Method.AlphaShape,
            radius = alpha,
            identify_regions = True,
            smoothing_level = smoothing_level
            ))
        data = pipeline.compute()
    except Exception as e:
        raise AlphaShapeOvitoError("Alphashape failed in OVITO. See following error:\t{e}")

    area = data.attributes['ConstructSurfaceMesh.surface_area']
    volume = data.attributes['ConstructSurfaceMesh.filled_volume']

    return area, volume


def rolling_probe_surface_area(
        atoms: Atoms,
        filename: str = FILENAME,
        probe_rad: float = PROBE_RADIUS
        ) -> float:
    """
    Calculate surface area using the Shake-Rupley algorithm, aka. rolling probe,
    as implemented in mdtraj

    Requires:
        atoms (Atoms)                           Atoms object of UNSUPPORTED nanoparticle
        filename (str)                          Filename of PDB file that'll be created to be read by mdtraj
        probe_rad (float)                       probe radius to use, in ANGSTROM. default = {PROBE_RADIUS}

    Returns:
        area (float)                            surface area
    """
    ##save atoms object as PDB or GRO for mdtraj to be able to load
    file = filename + ".pdb"
    write(file, atoms)

    probe_rad /= 10 #converted to nm
    try:
        trajectory = md.load(file)
        sasa = md.shrake_rupley(trajectory,
                mode = "atom",
                n_sphere_points = 1500, #may need to generalize this sometime in future
                probe_radius = probe_rad
                )
        total_sasa = sasa.sum(axis = 1)[0]
    except Exception as e:
        raise MDTRAJRollingProbeError(f"Rolling probe calculation failed with error:\t{e}")

    return total_sasa * 100 #A^2


###This section models the nanoparticle with a sphere
def spherical_surface_area(
        n_atoms: int,
        element: str,
        a: float = None,
        ) -> float:
    """
    Calculate NP surface area assuming a sphere
    receives in Ang and returns in A^2
    given element, n_atoms, and lattice constant (optionally)

    returns area in A^2
    """
    diameter = _natoms_to_sphere_diameter(
            n_atoms = n_atoms,
            element = element,
            a = a
            )
    return np.pi * 4 * ((diameter/2) ** 2)


##THis section models the nanoparticle with a spherical cap
def SCM_areas(
        atoms: Atoms,
        element: str = "Ag",
        resolution: float = MIN_RES
        ) -> Tuple[float, float]:
    f"""
    Calculate NP surface and interfacial areas assuming a spherical cap

    Requires:
        atoms (Atoms)                           Atoms object of supported or unsupported nanoparticle
        element (str):                          element the nanoparticle is made of
        resolution (float):                     resolution of the voxel grid for discriminating
                                                surface from bulk, in Ang. default = {MIN_RES} A

    Returns:
        interfacial_area (float):               interfacial area, in A^2
        surface_area (float):                   surface area, in A^2
    """
    curvature_diameter, angle = get_SCM_diameter_and_theta(
            atoms = atoms,
            np_element = element,
            min_resolution = resolution
            )
    footprint_radius = (curvature_diameter / 2) * np.sin(np.radians(angle))
    interfacial_area = np.pi * (footprint_radius ** 2)
    alpha = _alpha(angle)

    return interfacial_area, interfacial_area * (1 + (2 * alpha))


#calculate interfacial area by my method implemented in 'catalytic_descriptors.py'
def my_interface_area(
        atoms: Atoms,
        interfacial_facet: Tuple[int, int, int] = None,
        np_element: str = NANOPARTICLE_ELEMENT,
        scaling_factor: float = INTERFACE_SCALING,
        nl: NeighborList = None, analyzer: Analysis = None,
        support_elements: Union[str, List[str]] = None
        do_alphashape: bool = DO_ALPHA_SHAPE,
        plotname: str = ALPHA_SHAPE_NAME,
        show: bool = SHOW_ALPHASHAPE
        ) -> Tuple[float, Atoms, Atoms]:
    f"""
    Calculates interface area by extracting the interface, taking the user-supplied facet that
    aligns to the support, and getting its atomic density, which is then multiplied by the
    number of interfacial atoms

    Implemented in catalytic_descriptors.py; summarized here

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
        interfacial_facet (Tuple[int, int, int]):   bracketed comma-separated ints to define
                                                    the NP facet facing the interface, e.g. (1,0,0)
                                                    Default = {INTERFACIAL_FACET}
        do_alphashape (bool):               whether or not to get perimeter atoms by alphashape rather than only by CN
                                            default = {DO_ALPHA_SHAPE}
                                            !!!It's very expensive to optimize alpha for large NPs!!!
        plotname (str):                     name of png file (of the interface's alpha shape) to be saved
                                            Optional. default: {ALPHA_SHAPE_NAME}
        show (bool):                        whether to display the plot (of the interface's alpha shape) or not.
                                            default = {SHOW_ALPHASHAPE}

    Returns:
        interfacial_area (float):           interfacial area

        interfacial_atoms (Atoms):          interfacial atoms (includes perimeter)
        perimeter_atoms (Atoms):            perimeter atoms
    """
    if not interfacial_facet:
        warnings.warn(
                f"Facet not supplied. Defaulting to {INTERFACIAL_FACET}",
                category=UserWarning
                )
        interfacial_facet = INTERFACIAL_FACET
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
        raise ValueError(f"Some of the given elements ({np_and_support_elements}) not in Atoms object!")

    if not analyzer:
        if not nl: #reuse neighborlist if given
            natural_cutoff = np.max([natural_cutoffs(Atoms(element))
                for element in elements]) * 2 #no need to search beyond this distance
            nl = setup_neighborlist(
                    atoms,
                    scaling_factor = scaling_factor,
                    radial_cutoff = natural_cutoff
                    ) #, skin = NL_SKIN) curse this!
            nl.update(atoms)

        analyzer = setup_analyzer(atoms, neighborlist = nl) #reuse Analysis object if given

    nps = [index for index, _ in enumerate(atoms) if atoms[index].symbol == np_element]
    substrate = [index for index, _ in enumerate(atoms) if atoms[index].symbol in support_elements]

    np_interface_plus_perimeter, substrate_interface = get_interface(
            atoms,
            nl = nl,
            analyzer = analyzer,
            np_element = np_element,
            support_elements = support_elements,
            scaling_factor = scaling_factor
            )
    
    ##also return the perimeter, so the excluded_perimeter_area function can use it
    ##also return the interfacial atoms, so I can view it
    np_perimeter = get_perimeter(
            atoms,
            interfacial_results = (np_interface_plus_perimeter, substrate_interface),
            )
    interfacial_atoms = Atoms(atoms[np_interface_plus_perimeter])
    perimeter_atoms = Atoms(atoms[np_perimeter])

    interface_area, interface_radius = get_interfacial_area_and_radius(
            atoms = atoms,
            interfacial_results = (np_interface_plus_perimeter, substrate_interface),
            interfacial_facet = interfacial_facet,
            np_element = np_element
            )

    return interface_area, interfacial_atoms, perimeter_atoms


def excluded_perimeter_area(
        atoms: Atoms,
        interfacial_facet: Tuple[int, int, int] = None,
        np_element: str = NANOPARTICLE_ELEMENT,
        scaling_factor: float = INTERFACE_SCALING,
        nl: NeighborList = None, analyzer: Analysis = None,
        support_elements: Union[str, List[str]] = None
        do_alphashape: bool = DO_ALPHA_SHAPE,
        plotname: str = ALPHA_SHAPE_NAME,
        show: bool = SHOW_ALPHASHAPE
        ) -> Tuple[float, Dict[int, Tuple[float, float]]:

    f"""
    Estimates the total area that a shoelace formula (or a concave hull which perfectly traces
    the perimeter) will miss, due to its assumption that atoms are point-like objects

    As always, the interface should be quite flat, and uniform in its atomic density

    Methodology:
        Take an array of positions of perimeter atoms, henceforth simply called "atom"
        For each atom, run a vector to the two atoms adjacent it,
        find the angle between these vectors,
        use that to estimate the enclosed area, i.e. the area the alpha shape would include:
            enclosed_area = (theta/360) * covalent_radius ** 2     ---- (1)
            note that this is the area of a square, as each atom 'controls' such that shape
        the excluded area, i.e. the area the alpha shape would miss, is:
            excluded_area = covalent_radius ** 2 - enclosed_area ---- (2)

    Note that there is a question of whether covalent_radius or vdw radius is proper here
    probably the conversion from circle to square implicitly converts from covalent to vdW (july 18th: I don't think so!)
    
    August 2025:    I go with the covalent radius

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
        interfacial_facet (Tuple[int, int, int]):   bracketed comma-separated ints to define
                                                    the NP facet facing the interface, e.g. (1,0,0)
                                                    Default = {INTERFACIAL_FACET}
                                                    For this function, we don't care about this, so, you can supply anything
        do_alphashape (bool):               whether or not to get perimeter atoms by alphashape rather than only by CN
                                            default = {DO_ALPHA_SHAPE}
                                            !!!It's very expensive to optimize alpha for large NPs!!!
                                            Yes I recommend it for this function
        plotname (str):                     name of png file (of the interface's alpha shape) to be saved
                                            Optional. default: {ALPHA_SHAPE_NAME}
        show (bool):                        whether to display the plot (of the interface's alpha shape) or not.
                                            default = {SHOW_ALPHASHAPE}

    Returns:
        total_excluded_area (float):        the total area that a shoelace formula will miss
        peri_indices_and_excluded_areas_and_angles (Dict[int, Tuple[float, float]]:
                                            key: perimeter atom index
                                            value:
                                                1. excluded area in Ang^2
                                                2. enclosed angle (pointing inward to the COM of the perimeter)
    """
    ##here, I was trying to 
#    ##flatten the perimeter, i.e. set Zs to zero
#    dimensions = perimeter_positions.shape[0]
#    if dimensions == 3:
#        perimeter_positions[:,2] = 0
#        ##find the center of mass of the perimeter
#    com = center_of_mass(perimeter_positions) ##this function gives wrong answers; insyead use: all_positions.mean(axis=0)
#    ##draw a line for which we say angle = 0
#    arbitrary_number = 32
#    endpoint = com + arbitrary_number
#    zero_line = endpoint - com
    #for each perimeter atom, find the angle between that zero-line and a line drawn
    #from the center of mass to it
#    points_and_angles = {tuple(point): val for point in perimeter_positions}
#    for point in perimeter_positions:
#        line = point - com
#        angle = vg.signed_angle(line, zero_line, look = vg.basis.z) #unsigned_vector_angle
        
    
    #alphashape must be used here!!!
    if do_alphashape is False:
        warnings.warn(
        f"""You have elected not to use the alphashape to get the perimeter;
        however, our results for this function are extremely sensitive to the correct identification
        of perimeter atoms.
        Therefore, I am sorry, but I will be OVERRIDING this. Sorry!!""",
        category = RuntimeWarning
        )
        do_alphashape = True

    #get perimeter atoms
    _, _, perimeter_atoms = my_interface_area(
        atoms = atoms,
        interfacial_facet = interfacial_facet,
        np_element = np_element,
        scaling_factor = scaling_factor,
        nl = nl,
        support_elements = support_elements
        do_alphashape = do_alphashape,
        plotname = plotname,
        show = show
        )

    total_excluded_area = 0
    ##find the two closest perimeter atoms to each perimeter atom, i.e. what it is bonded to
    elements = np.unique(perimeter_atoms.get_chemical_symbols())
    natural_cutoff = np.max([natural_cutoffs(Atoms(element))\
            for element in elements]) * 2 #no need to search beyond this distance
    nl = setup_neighborlist(
            perimeter_atoms,
            scaling_factor = scaling_factor,
            radial_cutoff = natural_cutoff
            ) #, skin = NL_SKIN) curse this!
    nl.update(perimeter_atoms)

    #treat hand-wavingwly, any atom that doesnt have exactly 2 perimeter neighbors
    neighbors = [nl.get_neighbors(i)[0] for i in range(len(perimeter_atoms))]
    mask = [len(i) == 2 for i in neighbors]
    num_unaccounted_atoms, unaccounted_indices = mask.count(False), [index for index, val in enumerate(mask) if not val]
#    unaccounted_for_area = 0.
#    default_unaccounted_for_area = 1. #100% of the atom's area. makes sense
    if num_unaccounted_atoms > 0:
        warnings.warn(
        f"""There are {num_unaccounted_atoms} unaccounted-for atoms;
        we will assume a default excluded area for these""",
        category = UserWarning
        )
#        unaccounted_for_area = num_unaccounted_atoms * default_unaccounted_for_area * natural_cutoff**2

    all_positions = perimeter_atoms.positions[:,:2] #remove the z dimension
    com = all_positions.mean(axis = 0)

    peri_indices_and_excluded_areas_and_angles = {index: [] for index in range(len(perimeter_atoms))}
    for index in range(len(perimeter_atoms)):
        if index not in unaccounted_indices:
            central_position = all_positions[index]
            neighbor_0_position = perimeter_atoms[neighbors[index][0]].position[:2]
            neighbor_1_position = perimeter_atoms[neighbors[index][1]].position[:2]
            vector_to_neighbor_0 = central_position - neighbor_0_position
            vector_to_neighbor_1 = central_position - neighbor_1_position
            com_to_central_atom = com - central_position
            com_to_neighbor_0 = com - neighbor_0_position
            com_to_neighbor_1 = com - neighbor_1_position
            dist_to_neighbor_0 = np.linalg.norm(com_to_neighbor_0)
            dist_to_neighbor_1 = np.linalg.norm(com_to_neighbor_1)
            dist_to_central_atom = np.linalg.norm(com_to_central_atom)

            enclosed_angle = unsigned_vector_angle(
                    vector_to_neighbor_0,
                    vector_to_neighbor_1,
                    )
            if dist_to_central_atom < max(dist_to_neighbor_0, dist_to_neighbor_1):
                #that means the region we're interested in is going to be an angle < 180
                enclosed_angle = min(360 - enclosed_angle, enclosed_angle)
            else:
                #that means the region we're interested in is going to be an angle > 180
                enclosed_angle = max(360 - enclosed_angle, enclosed_angle)

        else:
            enclosed_angle = 0 #default_unaccounted_for_area = 1.

            excluded_area = (360 - (enclosed_angle / 360)) * natural_cutoff**2
            total_excluded_area += excluded_area
            peri_indices_and_excluded_areas_and_angles[index].extend([excluded_area, enclosed_angle])

    return total_excluded_area, peri_indices_and_excluded_areas_and_angles



##this place needs to be adapted
if __name__ == "__main__":
    parser = ArgumentParser(description = """Estimates interfacial and surface areas of supported
    nanoparticles using a sleuth of methods""")

    parser.add_argument("--traj", "-t", type = str, required = True, help = "Traj file on which to run the script. Required")

    parser.add_argument("--np_element", "-ne", type = str, default = None, #NANOPARTICLE_ELEMENT,
            help = f"Element of which the NP is composed. by default we'll read atoms.info['np_element']")
    parser.add_argument("--interfacial_facet", "-if", type = str, nargs = "+", default = None,
            help = f"Facet of NP that faces the support. by default we'll read atoms.info['interfacial_facet']")
    parser.add_argument("--support_elements", "-se", type = str, nargs = "+", default = None, help = """List of elements of which
    the substrate is composed. In the format: a b c d. Defaults to elements absent from the NP""")
    parser.add_argument("--interface_scaling", "-is", type = float, default = INTERFACE_SCALING,
            help = f"""scaling factor for interfacial atoms' covalent radii. Default = {INTERFACE_SCALING}""")
    parser.add_argument("--surface_scaling", "-ss", type = float, default = CN_SCALING,
            help = f"""scaling factor for discriminating NP surface from bulk. Default = {CN_SCALING}""")

    parser.add_argument("--do_alphashape", action = "store_const", const = True, default = DO_ALPHA_SHAPE,
            help = """To also use alpha shape to get perimeter atoms""")
    parser.add_argument("--show_alphashape", action = "store_const", const = True, default = SHOW_ALPHASHAPE,
            help = """To display the plot of the interface's alphashape.""")
    parser.add_argument("--plotname", "-an", type = str, default = ALPHA_SHAPE_NAME, help = """Name of alphashape plot (if used).
            Default = {ALPHA_SHAPE_NAME}""")

    parser.add_argument("--processes", "-p", type = int, default = PROCESSES,
            help = f"How many processes to run in parallel. default = {PROCESSES}")

    parser.add_argument("--alphashape_toolbox", action = "store_const", const = True, default = True,
            help = """calculate surface and interfacial areas using alphashape in the toolbox implementation""")

    parser.add_argument("--alphashape_ovito", action = "store_const", const = True, default = True,
            help = """calculate surface area using alphashape in the OVITO implementation""")
    parser.add_argument("--rolling_probe", action = "store_const", const = True, default = True,
            help = """calculate surface area using the Shake-Rupley algorithm (aka 'rolling probe')""")
    parser.add_argument("--spherical", action = "store_const", const = True, default = True,
            help = """calculate surface area using a spherical model""")
    parser.add_argument("--scm", action = "store_const", const = True, default = True,
            help = """calculate surface area using a spherical cap model""")

    parser.add_argument("--atomic_density", action = "store_const", const = True, default = True,
            help = """calculate interfacial area by my 1st method:
            from the atomic density and interfacial facet type""")
    parser.add_argument("--excluded_angles", action = "store_const", const = True, default = True,
            help = """calculate interfacial area by my 2nd method:
            from estimating the enclosed areas and combining with the concave hull""")


    #gather inputs
    args = parser.parse_args()
    atoms = read(args.traj, ":")
    atoms = atoms if isinstance(atoms, list) else [atoms]
    processes = args.processes
    np_element_list = [image.info["np_element"] if not args.np_element else args.np_element for image in atoms]
    interfacial_facet_list = [image.info["interfacial_facet"] if not args.interfacial_facet
            else args.interfacial_facet for image in atoms]

    mask_list = [[atom.symbol == np_element_list[index] for atom in image] for index, image in enumerate(atoms)]
    np_positions_list = [[image[mask[index]] for index, image in enumerate(atoms)]]
    ##need to then get positions list of only interfacial atoms
    
    ##start calculating
    #interface and surface areas with alphashapetoolbox
    #concave hull
    if alphashape_toolbox:
        with Parallel(n_jobs = processes) as parallel:
            toolbox_interface_concave_results = parallel(delayed(alpha_shape_toolbox_area)(
                positions = np_positions_list[index],
                area = "surface",
                method = "concave",
                np_cutoff = None,
                fall_back = FALL_BACK
                ) for index, image in enumerate(
                tqdm(atoms, desc = "AlphaShape Toolbox Concave Interfacial Areas...", total = len(atoms)
                    )))
        with Parallel(n_jobs = processes) as parallel:
            toolbox_interface_convex_results = parallel(delayed(alpha_shape_toolbox_area)(
                positions = np_positions_list[index],
                area = "surface",
                method = "convex",
                np_cutoff = None,
                fall_back = None,
                ) for index, image in enumerate(
                tqdm(atoms, desc = "AlphaShape Toolbox Convex Interfacial Areas...", total = len(atoms)
                    )))

        toolbox_interface_concave_A, toolbox_interface_concave_V = zip(*[i for i in toolbox_interface_concave_results])
        toolbox_interface_convex_A, toolbox_interface_convex_V = zip(*[i for i in toolbox_interface_convex_results])

        for index,image in enumerate(atoms):
            image.info["toolbox_interface_concave_Area"] = toolbox_interface_concave_A[index]
            image.info["toolbox_interface_convex_Area"] = toolbox_interface_convex_A[index]

    ###stopped here

    np_surface, np_bulk, np_interface, np_perimeter, substrate_interface, substrate = zip(*results)
    np_total = [(j + np_bulk[i] + np_perimeter[i] + np_interface[i]) for i, j in enumerate(np_surface)]

    np_interface_plus_peri = [j + np_perimeter[i] for i, j in enumerate(np_interface)]
    interfacial_results_list = [(j,substrate_interface[i]) for i,j in enumerate(np_interface_plus_peri)]

    if args.get_contact_angles:
        print("Getting contact angles")
        contact_angles = [image.info["Theta"] if "Theta" in
                image.info.keys() else None for image in atoms]
        if np.any(contact_angles is None):
            print("At least one image doesn't have contact angle stored as i.info['Theta']")

    if args.get_footprint_radius_by_SCM:
        with Parallel(n_jobs = processes) as parallel:
            radii_scm = parallel(delayed(spherical_cap_footprint_radius)(
                image, np_element = np_element_list[index])
                for index, image in enumerate(tqdm(atoms, desc = "SCM footprint radii..", total = len(atoms))))

    if args.get_footprint_radius_by_count:
        with Parallel(n_jobs = processes) as parallel:
            results = parallel(delayed(get_interfacial_area_and_radius)(
                image,
                interfacial_results = interfacial_results_list[index],
                np_element = np_element_list[index],
                interfacial_facet = interfacial_facet_list[index])
                for index, image in enumerate(tqdm(
                    atoms, desc = "Footprint radii by counting interfacial atoms..", total = len(atoms))))

        radii = [j for i,j in results] #results = areas, radii

    #return Traj but with atoms' identities switched
    for index, image in enumerate(tqdm(atoms, total = len(atoms), desc = "Changing identities..")):
        np_surf, np_int, np_peri = np_surface[index], np_interface[index], np_perimeter[index]
        for i in np_int:
            image[i].symbol = "Cu"
        for i in np_peri:
            image[i].symbol = "Mo"
        for i in np_surf:
            image[i].symbol = "Pd"

    write("switched.traj", atoms)


    print("Saving results")
    ##time to write results and plot
    #write the input for the atom_counter
    if args.get_footprint_radius_by_count:
        radius_of_curvature = np.zeros(len(radii)) #R not r
        #write radii by counting interfacial atoms
        df = pd.DataFrame({
            "r (A)"  : radii,
            "R (A)"  : radius_of_curvature,
            "Theta"  : contact_angles,
            "Element": np_element_list,
            "Facet"  : interfacial_facet_list,
        })
        df.to_csv("input1.csv", index=False)


    if args.get_footprint_radius_by_SCM:
        #write radii by sphere fitting
        df = pd.DataFrame({
            "r (A)"  : radii_scm,
            "R (A)"  : radius_of_curvature,
            "Theta"  : contact_angles,
            "Element": np_element_list,
            "Facet"  : interfacial_facet_list,
        })
        df.to_csv("input2.csv", index=False)

    #write atomistic results
    np_perimeter_len = [len(i) for i in np_perimeter]
    np_interface_len = [len(i) for i in np_interface]
    np_surface_len = [len(i) for i in np_surface]
    np_total_len = [len(i) for i in np_total]

    df = pd.DataFrame({
        "Perimeter"  : np_perimeter_len,
        "Interface"  : np_interface_len,
        "Surface"  : np_surface_len,
        "Total": np_total_len,
    })
    df.to_csv("output_atomistic.csv", index=False)



