#!/usr/bin/env python3
"""
Extract frames from trajectory and write PDB by residue ranges per chain.
Then postprocess PDBs: renumber residues, assign chain IDs, standardize residue names,
and drop specific atom names. Also fixes ILE CD -> CD1.

Usage:
  python traj_to_pdb.py --trajectory trajectory.xtc --topology topology.pdb \
      --ranges 1-123,124-246,247-369 --outdir . --stride 1 [--keepH]

Notes:
  - By default hydrogens are removed (use --keepH to keep them).
  - Residue renames:
        CYX -> CYS
        HSE/HSD/HID/HIE/HSP -> HIS
  - Atom names dropped entirely: CY, OY, NT, CAY, CAT.
  - Isoleucine atom name 'CD' is standardized to 'CD1'.
"""

import argparse
import os
import glob
import MDAnalysis as mda
from string import ascii_uppercase
from tqdm import tqdm
from io_functions import load_trajectory

# --- configuration ---

# Residue-level replacements (3-char field, right-justified)
RESNAME_MAP = {
    "CYX": "CYS",
#    "HSE": "HIS",
#    "HSD": "HIS",
#    "HID": "HIS",
#    "HIE": "HIS",
#    "HSP": "HIS",
}

# Atom names to drop (match core atom name without spaces)
ATOMNAME_DROP = {"CY", "OY", "NT", "CAY", "CAT"}


# --- helpers ---

def is_hydrogen(atom) -> bool:
    """Return True if the atom is a hydrogen using element or name heuristics."""
    try:
        element = (atom.element or "").strip()
    except Exception:
        element = ""
    if element.upper() == "H":
        return True
    return atom.name.strip().upper().startswith("H")


def infer_element(atom) -> str:
    """Infer element symbol (1-2 chars, upper case)."""
    try:
        element = (atom.element or '').strip()
    except Exception:
        element = ''
    if not element:
        nm = atom.name.strip()
        if not nm:
            return "X"
        if nm[0].isdigit() and len(nm) >= 2:
            return nm[1].upper()
        return nm[0].upper()
    return element.upper()[:2]


def atom_core_from_field(name_field_4: str) -> str:
    """
    Extract the core atom name from a 4-char PDB name field (cols 13–16),
    removing spaces and uppercasing.
    """
    return name_field_4.strip().upper()


def format_atom_name_4(name: str) -> str:
    """Return a 4-char atom name field, left-justified."""
    return name.ljust(4)[:4]


def standardize_resname_3(resname_3: str) -> str:
    """Return standardized 3-char residue name, right-justified."""
    key = resname_3.strip().upper()
    if key in RESNAME_MAP:
        return f"{RESNAME_MAP[key]:>3s}"
    return f"{key:>3s}"


def ile_fix_name(resname_std: str, core: str, current_field: str) -> str:
    """Return corrected 4-char atom name field for ILE CD -> CD1; otherwise return current."""
    if resname_std == "ILE" and core == "CD":
        return " CD1"
    return current_field


# --- core writers ---

def write_frame_by_ranges(universe, frame_index, residue_ranges, output_dir, keep_h=False):
    """Extract one frame and write a pdb with specified residue ranges as chains."""
    universe.trajectory[frame_index]
    atoms = universe.atoms
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"frame_{frame_index:04d}.pdb")
    atom_serial = 1

    with open(filename, 'w') as f:
        for chain_idx, (start, end) in enumerate(residue_ranges):
            chain_id = ascii_uppercase[chain_idx % len(ascii_uppercase)]
            selection = atoms.select_atoms(f"resid {start}:{end}")
            # Map original residue ids to sequential ids per chain
            res_map = {res.resindex: idx + 1 for idx, res in enumerate(selection.residues)}

            for residue in selection.residues:
                resname_std = standardize_resname_3(residue.resname[:3])
                new_resid = res_map[residue.resindex]

                for atom in residue.atoms:
                    if not keep_h and is_hydrogen(atom):
                        continue  # drop hydrogens unless keepH is True

                    # build 4-char atom name field and decide if we drop it
                    name_field = format_atom_name_4(atom.name)
                    core = atom_core_from_field(name_field)
                    if core in ATOMNAME_DROP:
                        continue  # drop this atom entirely

                    # standardize ILE CD -> CD1
                    name_field = ile_fix_name(resname_std, core, name_field)

                    element = infer_element(atom)

                    line = (
                        "{:<6s}{:>5d} {:<4s}{:1s}{:>3s} {:1s}"
                        "{:>4d}    "
                        "{:>8.3f}{:>8.3f}{:>8.3f}"
                        "{:>6.2f}{:>6.2f}          {:>2s}\n"
                    ).format(
                        "ATOM",
                        atom_serial,
                        name_field,
                        "",  # altLoc
                        resname_std,
                        chain_id,
                        new_resid,
                        atom.position[0], atom.position[1], atom.position[2],
                        1.00, 0.00,
                        element.rjust(2)
                    )
                    f.write(line)
                    atom_serial += 1
            f.write("TER\n")
        f.write("END\n")
    return filename


def parse_ranges(ranges_string):
    """Parse comma-separated 'start-end' strings into tuples of ints."""
    ranges = []
    for r in ranges_string.split(','):
        start, end = r.split('-')
        ranges.append((int(start), int(end)))
    return ranges


def process_pdb_file(file_path):
    """Renumber residues sequentially, assign chain IDs and reset per chain."""
    with open(file_path, 'r') as infile:
        lines = infile.readlines()

    new_lines = []
    current_chain = 'A'
    residue_counter = 0
    prev_resnum = None

    for line in lines:
        if line.startswith(('ATOM', 'HETATM')):
            resnum = line[22:26].strip()
            if resnum != prev_resnum:
                residue_counter += 1
                prev_resnum = resnum
            new_line = (
                line[:21]
                + current_chain
                + f"{residue_counter:4d}"
                + line[26:]
            )
            new_lines.append(new_line)
        elif line.startswith('TER'):
            new_lines.append(line)
            current_chain = chr(ord(current_chain) + 1)
            residue_counter = 0
            prev_resnum = None
        else:
            new_lines.append(line)

    with open(file_path, 'w') as outfile:
        outfile.writelines(new_lines)


# --- standardization and cleanup on disk ---

def filter_and_standardize_pdb_line(line: str):
    """Return standardized line or None if it must be dropped."""
    if not line.startswith(("ATOM", "HETATM")):
        return line

    # Drop by atom name
    name_field = line[12:16]
    core = atom_core_from_field(name_field)
    if core in ATOMNAME_DROP:
        return None

    # Standardize residue name
    resname_field = line[17:20]
    resname_std = standardize_resname_3(resname_field)
    line = line[:17] + resname_std + line[20:]

    # Fix ILE CD -> CD1
    if resname_std == "ILE" and core == "CD":
        line = line[:12] + " CD1" + line[16:]

    return line


def clean_pdb_files(directory):
    """Apply atom dropping and residue name standardization to all *.pdb files in directory."""
    for fname in glob.glob(os.path.join(directory, "*.pdb")):
        with open(fname, "r") as f:
            lines = f.readlines()
        new_lines = []
        for ln in lines:
            out = filter_and_standardize_pdb_line(ln)
            if out is not None:
                new_lines.append(out)
        with open(fname, "w") as f:
            f.writelines(new_lines)


def standardize_text_like_files(directory, extensions=None):
    """Replace residue names in text-like files (*.txt, *.map)."""
    if extensions is None:
        extensions = ['txt', 'map']
    for ext in extensions:
        for fname in glob.glob(os.path.join(directory, f"*.{ext}")):
            with open(fname, 'r') as f:
                content = f.read()
            for old, new in RESNAME_MAP.items():
                content = content.replace(old, new)
            with open(fname, 'w') as f:
                f.write(content)


# --- main ---

def main():
    parser = argparse.ArgumentParser(
        description="Convert trajectory frames to PDB by chain ranges, then postprocess."
    )
    parser.add_argument('--trajectory', help='Trajectory file (xtc, dcd, trr, etc.)')
    parser.add_argument('--topology', default = "./equil/equil.gro",  help='Topology file (pdb, gro, psf, etc.)')
    parser.add_argument(
        '--ranges',
        required=True,
        help='Residue ranges per chain, e.g., 1-123,124-246,247-369'
    )
    parser.add_argument('--outdir', default='.', help='Output directory')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride')
    parser.add_argument(
        '--keepH',
        action='store_true',
        help='Keep hydrogen atoms (default: remove hydrogens)'
    )
    parser.add_argument(
        '--dir',
        default='.',
        help='Gromacs directory (/gromacs) (default: current directory)'
    )
    parser.add_argument(tra_pattern='trajectory_pattern', default = "./COM_corrected/com_MD_step_*.xtc", help='Trajectory file (xtc, dcd, trr, etc.)')
    args = parser.parse_args()

    residue_ranges = parse_ranges(args.ranges)
    u = load_trajectory(topology=args.topology, directory=args.dir, trajectories_pattern = args.tra_pattern) #u = mda.Universe(args.topology, args.trajectory)

    total_frames = len(u.trajectory)
    frame_indices = range(0, total_frames, args.stride)

    pdb_files = []
    for idx in tqdm(frame_indices, desc='Writing pdb frames'):
        pdb = write_frame_by_ranges(u, idx, residue_ranges, args.outdir, keep_h=args.keepH)
        pdb_files.append(pdb)

    # Renumber and chain-assign per PDB
    for pdb in pdb_files:
        process_pdb_file(pdb)

    # Cleanup and residue standardization on disk
    clean_pdb_files(args.outdir)
    standardize_text_like_files(args.outdir)  # only residue names in txt/map

    kept = "kept" if args.keepH else "removed"
    print(
        f"Processed {len(pdb_files)} frames into {args.outdir} "
        f"(hydrogens {kept}; residues {', '.join([f'{k}->{v}' for k,v in RESNAME_MAP.items()])}; "
        f"atoms dropped: {sorted(ATOMNAME_DROP)}; ILE CD->CD1 applied)"
    )


if __name__ == '__main__':
    main()

