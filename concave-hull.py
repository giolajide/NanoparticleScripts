from tqdm import tqdm
import matplotlib.pyplot as plt
#from alpha_shapes import Alpha_Shaper, plot_alpha_shape
import pyvista as pv #mamba activate puchik
import numpy as np
#import cupy as cp
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import ConvexHull
from ase.io import read, write
from ase.visualize import view
from sys import argv, exit
import trimesh
import alphashape
from descartes import PolygonPatch
from typing import List, Tuple, Union, Literal
from ase import Atoms
import warnings
try:
    from fit_support import cross_sectional_area
except ImportError:
    print(f"Add /mnt/public/giolajide/npscripts/ to your PYTHON_PATH")
    exit(1)



ALPHA = 0.3
GRID = 0.02 #Ang
MIN_ALPHA = 0.05


def default_to_convex_hull(positions: np.ndarray, show: bool = False
        ) -> Union[Tuple[float, float], float, None]:
    """
    Calculate area (and volume, if 3D) by a convex hull
    Requires:
        positions (numpy array)
    Returns:
        enclosed volume and total outer surface area
    """
    warnings.warn(f"""Using a convex hull (alpha = 0) should
    overestimate the enclosed volume and underestimate surface area.
    I hope you know what you are doing!""", category = UserWarning)
    hull = ConvexHull(positions)
    plot_convex_hull(positions, hull, show = show)

    if positions.shape[1] == 3:
        return hull.volume, hull.area
    elif positions.shape[1] == 2:
        return hull.area
    else:
        print(f"Shape of positions ({positions.shape}) won't work")
        return None


def plot_toolbox_alpha_shape(alpha_shape, index: int, show: bool = False) -> None:
    """
    Plot alpha shape (2D or 3D) given by alpha shape toolbox
    Requires:
        alpha_shape:    2D or 3D Alpha shape object from alpha shape toolbox
        show:           whether to display the plot or not
    Returns:
        a saved (and optionally displayed) plot
    """
    plt.close('all')
    
    # Check for the 2D case: Shapely Polygons have an "exterior" attribute.
    if hasattr(alpha_shape, 'exterior'):
        # 2D plotting using matplotlib
        fig, ax = plt.subplots()
        x, y = alpha_shape.exterior.xy
        ax.plot(x, y, color='blue', lw=2, label='Alpha Shape Boundary')
        ax.scatter(x, y, color='red', s=10, label='Boundary Points')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.title("Alpha Shape (2D)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"{index}_alpha_shape_toolbox_2d.png")
        if show:
            plt.show()
        plt.close()
    
    # Check for the 3D case: Expecting attributes vertices and faces.
    elif hasattr(alpha_shape, 'vertices') and hasattr(alpha_shape, 'faces'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        # Unpack the vertices into x, y, z sequences.
        ax.plot_trisurf(*zip(*alpha_shape.vertices), triangles=alpha_shape.faces,
                        color='lightblue', alpha=0.5, edgecolor='k')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Alpha Shape (3D)")
        plt.tight_layout()
        plt.savefig(f"{index}_alpha_shape_toolbox_3d.png")
        if show:
            plt.show()
        plt.close()
    
    else:
        print("Unknown alpha_shape format. Cannot plot.")


##TODO: Combine with the plot_toolbox_alpha_shape function above
def plot_pyvista_alpha_shape(alpha_shape, index: int, show: bool = False) -> None:
    """
    Plot alpha shape (2D or 3D) given by pyvista
    Requires:
        alpha_shape:    2D or 3D Alpha shape object from pyvista
        show:           whether to display the plot or not
    Returns:
        a saved (and optionally displayed) plot
    """
    plt.close("all")
    plotter = pv.Plotter()
    plotter.add_mesh(alpha_shape, color = 'lightblue', opacity = 0.5, show_edges = True)
    plotter.add_points(cloud, color = 'red', point_size = 10)
    plotter.add_legend([("Alpha Shape Boundary", "lightblue"), ("Input Points", "red")])
    plotter.screenshot("pyvista.png")
    if show:
        plotter.show()
    plt.close()


def plot_convex_hull(positions: np.ndarray, hull, index: int, show: bool) -> None:
    """
    Plot 2D or 3D alpha shape (alpha = 0)
    Requires:
        positions:  positions array
        hull:    convex hull object
        show:           whether to display the plot or not
    Returns:
        a saved (and optionally displayed) plot
    """
    plt.close("all")
    ##2D convex hull
    if positions.shape[1] == 2:
        plt.figure()
        plt.plot(positions[:, 0], positions[:, 1], 'ro')
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])
            plt.plot(positions[simplex, 0], positions[simplex, 1], 'b-', alpha = 0.6)
        plt.xlabel("x_dimension")
        plt.ylabel("y_dimension")
        plt.title("Convex Hull (2D)")
        plt.savefig(f"{index}_convex_hull2D.png")
        if show:
            plt.show()
        plt.close()

    elif positions.shape[1] == 3:
        ##3D convex hull
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for simplex in hull.simplices:
            triangle = positions[simplex]
            tri_points = np.vstack([triangle, triangle[0]])
            ax.plot(tri_points[:, 0], tri_points[:, 1], tri_points[:, 2], 'b-', alpha=0.6)
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='r', marker='o')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.title("Convex Hull (3D)")
        plt.savefig(f"{index}_convex_hull3D.png")
        if show:
            plt.show()
        plt.close()

    else:
        print("Someting is wrong with the positions array")
        return None


def plot_np_skeleton(np_positions: np.ndarray, index: int, show: bool = False) -> None:
    """
    Plot array of NP atoms' positions
    Requires:
        positions:  array of positions
    Returns:
        plot of the positions, in 2D or 3D
    """
    plt.close("all")
    ##3D plot
    if np_positions.shape[1] == 3:
        x, y, z = np_positions.T
        projection = "3d"
    elif np_positions.shape[1] == 2:
        x, y = np_positions.T
        projection = "rectilinear"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = projection)
    try:
        ax.scatter(x, y, z, c = "r", marker = "o")
    except:
        ax.scatter(x, y, c = "r", marker = "o")
    ax.set_xlabel('x_dimension')
    ax.set_ylabel('y_dimension')
    try:
        ax.set_zlabel('z_dimension')
    except:
        pass
    plt.tight_layout()
    plt.title(f"Original NP coordinates ({projection})")
    plt.savefig(f"{index}_original_{projection}.png")
    if show:
        plt.show()
    plt.close()



def nanoparticle_volume_and_area(atoms: Atoms, index: int, element: str = "Ag", alpha: float = ALPHA,
        show: bool = False, min_alpha: float = MIN_ALPHA) -> Tuple[float, float]:
    """
    Calculate the enclosed volume within an NP
    Using the alpha shape method
    Requires:
        atoms (ase.Atoms)   Atoms object
        element (str)       Symbol of the NP
        alpha (float)       Value of alpha. Default = 0 (convex hull)
        min_alpha (float)   Minimum value of alpha to try (when looping, if first tested alpha value fails)
    Returns:
        enclosed volume and total outer surface area
    """
    print("Calculating Volume and Area")
    alpha = max(alpha, 0)
    np_indices = [i.symbol == element for i in atoms]
    np_positions = atoms[np_indices].positions
    if np_positions.ndim != 2:
        raise ValueError("I dont think the element is here!")

    steps = 5
    alpha_values = np.arange(alpha, min_alpha, min(int(-(alpha - min_alpha) / 5), -2))

    plot_np_skeleton(np_positions, show = show, index=index)

    if alpha == 0: #CASE 1: user requests convex hull 
        volume, area = default_to_convex_hull(np_positions[:,:2])
        return volume, area
    else: #CASE 2: alpha shape toolbox from https://alphashape.readthedocs.io/en/latest/installation.html
        print("CASE 2: Alpha Shape Toolbox")
        try:
            ##CASE 2.1: optimize alpha
            optimum_alpha = alphashape.optimizealpha(np_positions)
            optimum_alpha *= 0.85 #optimum_alpha will tend to be too high
            alpha_shape = alphashape.alphashape(np_positions, optimum_alpha) #will probably fail.

            #UPDATE: June 2025; the above line should now work. Here was the solution:
                #pip uninstall alphashape
                #pip install git+https://github.com/bellockk/alphashape.git
                #pip install trimesh==4.2.0
                #pip install numpy==1.26.4

            print(f"optimal alpha = {optimum_alpha}") ##is this right?
                                                    #No, it isn't. It won't return the alpha value as it doesn't save it
                                                    #you can 
            mesh = trimesh.Trimesh(vertices = alpha_shape.vertices,
                    faces = alpha_shape.faces)
            volume = abs(mesh.volume)
            if volume > 0: #check if volume == 0
                area = mesh.area
                print(f"Area > 0! Success!")
                plot_toolbox_alpha_shape(alpha_shape, show = show,index=index)

                return volume, area
            else:
                print("Zero volume! Continueing")
                pass

        except Exception as toolbox_opt_error:
            toolbox_worked = False
            print(f"CASE 2.1\nCould not optimize alpha. See following error:\n{toolbox_opt_error}")
            for value in alpha_values: #CASE 2.2: try increasingly smaller alpha values till one works
                try:
                    alpha_shape = alphashape.alphashape(np_positions, value)
                    mesh = trimesh.Trimesh(vertices = alpha_shape.vertices,
                            faces = alpha_shape.faces)
                    volume = abs(mesh.volume)
                    if volume > 0: #check if volume == 0
                        print(f"CASE 2.2\nOne of the alpha values ({value}) finally worked!")
                        area = mesh.area
                        toolbox_worked = True
                        break
                except Exception as toolbox_alpha_error:
                    print(f"CASE 2.2\nalpha = {value} did not work. See error:\n{toolbox_alpha_error}")
                    continue

        if toolbox_worked:
            plot_toolbox_alpha_shape(alpha_shape, show = show,index=index)

            return volume, area

        ##CASE 3: if all the above have failed, try the pyvista version:
        ##https://docs.pyvista.org/api/core/_autosummary/pyvista.datasetfilters.delaunay_3d
        print("CASE 2 failed.\n\nCASE 3: Pyvista Method")
        pyvista_worked = False
        cloud = pv.PolyData(np_positions)
        ##Delaunay 3D triangulation
        for value in alpha_values:
            try:
                delaunay = cloud.delaunay_3d(alpha = alpha)
                alpha_shape = delaunay.extract_geometry()
                volume = alpha_shape.volume
                if volume > 0: #check that the computed shape isn't empty
                    area = alpha_shape.area
                    print(f"CASE 3\nOne of the alpha values ({value}) finally worked!")
                    pyvista_worked = True
                    break
                else:
                    print("zero volume. continuing")
                    continue
            except Exception as pyvista_alph:
                print(f"CASE 3\nalpha = {value} did not work. See error:\n{pyvista_alph}")
                continue

        if pyvista_worked:
            plot_pyvista_alpha_shape(alpha_shape, show = show,index=index)

            return volume, area

        ##CASE 4: if everything failed, default to a convex hull
        warnings.warn("""Both the alpha shape toolbox and pyvista implementations have failed
        Defaulting to a convex hull""", category = RuntimeWarning)
        volume, area = default_to_convex_hull(np_positions)

        return volume, area


def nanoparticle_interfacial_area(atoms: Atoms, index: int, element: str = "Ag", alpha: float = ALPHA,
        show: bool = False, min_alpha: float = MIN_ALPHA, grid_spacing: float = GRID) -> Tuple[float, float]:
    """
    Calculate the enclosed volume within an NP
    Using the alpha shape method
    Requires:
        atoms (ase.Atoms)   Atoms object
        element (str)       Symbol of the NP
        alpha (float)       Value of alpha. Default = 0 (convex hull)
        min_alpha (float)   Minimum value of alpha to try (when looping, if first tested alpha value fails)
    Returns:
        NP footprint area, and fraction of cell that area occupies
    """
    print("Calculating interfacial area")
    total_area = cross_sectional_area(atoms)
    alpha = max(alpha, 0)
    np_indices = [i.symbol == element for i in atoms]
    np_positions = atoms[np_indices].positions

    ##get the lowest NP atom within each slice of a grid to get the interfacial atoms
    max_y, min_y = max(np_positions[:,1]), min(np_positions[:,1])
    max_x, min_x = max(np_positions[:,0]), min(np_positions[:,0])
    x_grid, y_grid = np.arange(min_x, max_x, grid_spacing), np.arange(min_y, max_y, grid_spacing)
    x_indices = np.digitize(np_positions[:, 0], bins=x_grid) - 1
    y_indices = np.digitize(np_positions[:, 1], bins=y_grid) - 1

    lowest_atoms = {}
    ##can add njit here eventually
    for i in range(len(x_grid) - 1):
        for j in range(len(y_grid) - 1):
            mask = (x_indices == i) & (y_indices == j)
            if np.any(mask):
                cell_points = np_positions[mask]
                min_index = np.argmin(cell_points[:, 2])
                lowest_atom = cell_points[min_index]
                lowest_atoms[(i, j)] = lowest_atom

    positions_3d = np.array(list(lowest_atoms.values()))
    plot_np_skeleton(positions_3d, show = show,index=index)
    positions_2d = positions_3d[:,:2]
    plot_np_skeleton(positions_2d, show = show,index=index)

    if alpha == 0: # CASE 1: If alpha is zero, default to convex hull area.
        area = default_to_convex_hull(positions_2d)
        return area, area / total_area
    else: #CASE 2: Try alpha shape toolbox implementation (for 2D)
        ##https://alphashape.readthedocs.io/en/latest/installation.html
        print("CASE 2: Alpha Shape Toolbox")
        try:
            ##CASE 2.1: optimize alpha
            print("CASE 2.1: oPTIMIZE alpha")
            alpha_shape = alphashape.alphashape(positions_2d) #will probably fail
            print(f"optimal alpha = {alpha_shape}") ##is this right?
            area = alpha_shape.area
            if (isinstance(area, float) and area > 0):
                print("Optimization succeeded!")
                plot_toolbox_alpha_shape(alpha_shape, show = show,index=index)

                return area, area / total_area

        except Exception as toolbox_opt_error:
            print(f"CASE 2.1\nToolbox alpha shape failed: {toolbox_opt_error}")

    ##CASE 2.2 Loop over different alpha values till one works
    print("CASE 2.2: Looping over values")
    alpha_values = np.arange(alpha, min_alpha, min(int(-(alpha - min_alpha) / 5), -2))
    toolbox_worked = False
    for value in alpha_values:
        try:
            alpha_shape = alphashape.alphashape(positions_2d, value)
            area = alpha_shape.area
            if (isinstance(area, float) and area > 0):
                toolbox_worked = True
                print(f"CASE 2.2: alpha = {value} worked!")
                break
            else:
                print(f"CASE 2.2: zero volume with {value}. continuing")
                continue

        except Exception as toolbox_alpha_error:
            print(f"CASE 2.2: alpha = {value} did not work. See error:\n{toolbox_alpha_error}")
            continue

    if toolbox_worked:
        plot_toolbox_alpha_shape(alpha_shape, show = show,index=index)

        return area, area / total_area

    ##CASE 3. if alpha shape toolbox failed, try the pyvista implementation:
    ##https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgridfilters.delaunay_2d
    print("CASE 3: PyVista's Implementation")
    points_3d = np.hstack([positions_2d, np.zeros((positions_2d.shape[0], 1))]) # Lift points to 3D (z=0)
    cloud = pv.PolyData(points_3d)
    pyvista_worked = False
    print("Looping over alpha values")
    for value in alpha_values:
        try:
            mesh_2d = cloud.delaunay_2d(alpha = value)
            alpha_shape = mesh_2d.extract_geometry()
            boundary = mesh_2d.extract_feature_edges(
                    boundary_edges = True, non_manifold_edges = False,
                    feature_edges = False, manifold_edges = False)
            # Get the boundary points (if the boundary is closed, these should define the outer contour)
            boundary_coords = boundary.points[:, :2]  # take only x and y
            # Order the boundary points (optional but generally improves polygon formation)
            center = boundary_coords.mean(axis = 0)
            angles = np.arctan2(boundary_coords[:, 1] - center[1], boundary_coords[:, 0] - center[0])
            order = np.argsort(angles)
            ordered_boundary = boundary_coords[order]
            polygon = Polygon(ordered_boundary)
            area = polygon.area
            if (isinstance(area, float) and area > 0):
                pyvista_worked = True
                print(f"CASE 3: alpha = {value} worked!")
                break
            else:
                print(f"CASE 3: zero area with {value}. continuing")
                continue
        except Exception as pyvista_alph:
            print(f"CASE 3: alpha = {value} did not work. See error:\n{pyvista_alph}")
            continue

    ##CASE 4: if everything failed, default to a convex hull
    warnings.warn("""Both the alpha shape toolbox and pyvista implementations have failed
    Defaulting to a convex hull""", category = RuntimeWarning)
    area = default_to_convex_hull(positions_2d)

    return area, area / total_area


if __name__ == "__main__":
#    images=read("/mnt/public/giolajide/Silver_MgO/MLIP/I6/training/Full/Without_D3/evaluation/"
#            "Winterbottom_Curve/OtherMethods/MethodPBE_With_Contact_Angles_Tracked/"
#            "Centered_90/d3bj_no_Mg/d3bj_no_Mg_optimized-winter.xyz",":")
    images=read("../AgMgO_optimized.traj",":")
    alpha = 0.3
    results= []
    show = False 
    successfull=[]
    for index, atoms in enumerate(tqdm(images)):
        if atoms.symbols.count("Ag") > 0:
            try:
                V, A = nanoparticle_volume_and_area(atoms, element = "Ag", alpha = alpha,
                        show = show, index = index)
                IA, Frxn = nanoparticle_interfacial_area(atoms, element = "Ag", alpha = alpha,
                        show = show, index = index)
                results.append([V,A,IA,Frxn])
                atoms.info["Volume"] =V
                atoms.info["Surface_Area"] =A
                atoms.info["Interfacial_Area"]=IA
                atoms.info["Area_Fraction"]=Frxn
                successfull.append(atoms)
            except Exception as e:
                print(f"Image index {index} gave this error:\t{e}")

    write("calculated.traj", successfull)

    volumes, areas, interfaces, fractions = zip(*results)
    print(volumes)
    print(areas)
    print(interfaces)
    print(fractions)

    print("\nSaving...")
    np.savetxt("volumes.csv", np.array(volumes), delimiter = ",", fmt = "%s")
    np.savetxt("areas.csv", np.array(areas), delimiter = ",", fmt = "%s")
    np.savetxt("interfaces.csv", np.array(interfaces), delimiter = ",", fmt = "%s")
    np.savetxt("fractions.csv", np.array(fractions), delimiter = ",", fmt = "%s")

    print("Saved!")


