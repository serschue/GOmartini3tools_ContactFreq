# Import packages
import glob
import MDAnalysis as mda
import os
import numpy as np
from natsort import natsorted

def load_trajectory(topology="equil.gro", directory=".", trajectories_pattern="MD_step_*.xtc"):
    """
    Load MD trajectories from a specified directory using MDAnalysis,
    print basic trajectory info, and return a NumPy array of all coordinates.

    Parameters:
    - topology: str, path to the topology file (e.g., .gro)
    - directory: str, path to the folder containing the trajectories
    - trajectories_pattern: str, glob pattern for trajectory files

    Returns:
    - u: MDAnalysis Universe object
    - traj: NumPy array of shape (n_frames, n_atoms, 3)
    """
    # Combine directory and pattern
    full_pattern = os.path.join(directory, trajectories_pattern)
    
    # Find all matching trajectory files
    trajectories = natsorted(glob.glob(full_pattern))
    
    if not trajectories:
        raise FileNotFoundError(f"No trajectories found in {directory} matching {trajectories_pattern}")
    
    # Combine directory and topology
    full_topology = os.path.join(directory,topology)

    if not full_topology:
        raise FileNotFoundError(f"No topology found in {directory} matching {topology}")

    
    # Load Universe
    u = mda.Universe(full_topology, trajectories)

    # Basic info
    nframe = len(u.trajectory)
    natom = len(u.atoms)
    traj_shape = (nframe, natom, 3)  # x, y, z
    
    print("*Basic info*")
    print("Trajectory shape: ", traj_shape)
    print("Number of frames: ", nframe)
    print("Number of atoms:  ", natom)
    print("Coords: x, y, z = 3\n")
    
    return u

def get_universe_context(u):
    """
    Extracts context data from an MDAnalysis Universe.

    Parameters:
    - u: MDAnalysis Universe object

    Returns:
    - context: dict with keys:
        - 'residues': list of residue names
        - 'atoms': list of atom names
        - 'n_atoms': total number of atoms
        - 'masses': list/array of atom masses
        - 'carbon_alpha_ids': list of indices of CA atoms
    """
    residues = [res.resname for res in u.residues]
    atoms = [atom.name for atom in u.atoms]
    n_atoms = u.atoms.n_atoms
    masses = u.atoms.masses
    carbon_alpha_ids = [atom.index for atom in u.select_atoms("name CA")]

    context = {
        "residues": residues,
        "atoms": atoms,
        "n_atoms": n_atoms,
        "masses": masses,
        "carbon_alpha_ids": carbon_alpha_ids
    }

    # Print summary
    print("*Universe context*")
    print("Number of residues:", len(residues))
    print("Number of atoms:", n_atoms)
    print("Number of masses:", len(masses))
    print("Cα indices:", carbon_alpha_ids, "\n")

    return context

def parse_mdp(mdp_file, directory="."):
    """Parse een GROMACS .mdp file naar een dict"""
    params = {}
    # Combine directory and pattern
    full_mdp_file = os.path.join(directory, mdp_file)
    with open(full_mdp_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                key, value = parts[0], ' '.join(parts[1:])
            params[key.strip()] = value.strip()
    return params

def get_time_axis(u, mdp_params):
    """
    Generate a time axis (in ns) for an MD trajectory.

    Parameters:
    - u: MDAnalysis Universe object
    - mdp_params: dict, parsed MD parameters (from .mdp file)

    Returns:
    - t: np.array of length n_frames, time of each frame in ns
    - time_between_frames: float, time between saved frames in ns
    """
    dt = float(mdp_params.get("dt")) * 1e-3  # ns
    nstxout_compressed = int(mdp_params.get("nstxout-compressed"))

    # Time between frames
    time_between_frames = dt * nstxout_compressed

    # Create time axis
    nframe = len(u.trajectory)
    t = np.arange(nframe) * time_between_frames

    # Informative print
    print("*Info time axis*")
    print(f"Time between saved frames: {time_between_frames:.3f} ns")
    print(f"Trajectory length: {nframe} frames ({t[0]:.3f} ns → {t[-1]:.3f} ns)\n")

    return t




