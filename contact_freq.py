#!/usr/bin/env python3
# updated: 23-12-2025
"""
Comprehensive contact analysis pipeline including martinize2.

Now supports frames in PDB or CIF natively, without conversion.
If CIF frames are present, they are used directly so chain IDs are preserved.

This script performs the following steps:
  1. Generate contact maps for each frame (.pdb or .cif)
  2. Clean and filter contacts by distance and flags (distance thresholds in nm via --go-low and --go-up)
  3. Annotate intra and inter chain contacts
  4. Compute contact frequencies and identify high-frequency pairs
  5. Select the single reference frame with the most high-frequency contacts
  6. Run martinize2 to build coarse-grained topology and structure
  7. Build bead index, write mock ITP and filter real ITP
  8. Measure distances for missing contacts and write them to a separate ITP
  9. Write per-frame counts of high-frequency contacts and Go contacts
 10. Move final .txt, .map and frame files into an output_files folder

Usage:
  python contact_analysis.py [options]
  e.g. python contact_calculation.py --type both --merge all --dssp mkdssp --go-eps 15 --from charmm --cm /home/phoenix/software/

Run `python contact_analysis.py -h` to see all available flags.
"""

import os
import glob
import shutil
import re
import argparse
import subprocess
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="MDAnalysis.topology.PDBParser")

# ---------------- frame discovery ----------------

FRAME_RE = re.compile(r"^frame_(\d+)\.(pdb|cif)$", re.IGNORECASE)

def list_frames() -> Dict[int, str]:
    """Return {frame_index: path} for frame_####.(pdb|cif), preferring PDB if both exist for same index."""
    candidates: Dict[int, Tuple[str, str]] = {}
    for fn in glob.glob("frame_*.*"):
        m = FRAME_RE.match(os.path.basename(fn))
        if not m:
            continue
        idx, ext = int(m.group(1)), m.group(2).lower()
        if idx not in candidates:
            candidates[idx] = (fn, ext)
        else:
            # prefer pdb over cif when both exist
            if candidates[idx][1] == "cif" and ext == "pdb":
                candidates[idx] = (fn, ext)
    return {k: candidates[k][0] for k in sorted(candidates)}

# ---------------- minimal mmCIF reader (no MDAnalysis dependency for CIF) ----------------

def read_cif_atoms(path: str) -> List[Dict[str, str]]:
    """
    Minimal mmCIF atom_site reader without using file tell/seek.
    Returns list of dicts with keys: chain, resid, name, x, y, z
    Prefers auth_* fields; falls back to label_*.
    Assumes no whitespace-containing values for needed columns.
    """
    with open(path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]

    rows: List[Dict[str, str]] = []
    i = 0
    n = len(lines)

    while i < n:
        s = lines[i]
        if s != "loop_":
            i += 1
            continue

        # collect headers
        i += 1
        headers: List[str] = []
        while i < n and lines[i].startswith("_"):
            headers.append(lines[i])
            i += 1

        # need atom_site with required columns
        if not headers or not any(h.startswith("_atom_site.") for h in headers):
            while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
                i += 1
            continue

        idx = {h: k for k, h in enumerate(headers)}
        def pick(name_list):
            for nm in name_list:
                if nm in idx:
                    return idx[nm]
            return None

        i_chain = pick(["_atom_site.auth_asym_id", "_atom_site.label_asym_id"])
        i_resid = pick(["_atom_site.auth_seq_id", "_atom_site.label_seq_id"])
        i_name  = pick(["_atom_site.auth_atom_id", "_atom_site.label_atom_id"])
        i_x = idx.get("_atom_site.Cartn_x")
        i_y = idx.get("_atom_site.Cartn_y")
        i_z = idx.get("_atom_site.Cartn_z")

        if None in (i_chain, i_resid, i_name, i_x, i_y, i_z):
            while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
                i += 1
            continue

        # consume data rows for this loop
        while i < n and not (lines[i].startswith("loop_") or lines[i].startswith("_")):
            tokens = lines[i].split()
            if len(tokens) >= len(headers):
                try:
                    chain = tokens[i_chain]
                    resid = tokens[i_resid]
                    name  = tokens[i_name]
                    x = float(tokens[i_x]); y = float(tokens[i_y]); z = float(tokens[i_z])
                    rows.append({"chain": chain, "resid": resid, "name": name, "x": x, "y": y, "z": z})
                except Exception:
                    pass
            i += 1

    return rows


def get_cif_chains(path: str) -> List[str]:
    atoms = read_cif_atoms(path)
    return sorted({a["chain"] for a in atoms if a["chain"]})

def get_cif_ca_coords(path: str) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Return {(resid_str, chain): np.array([x,y,z])} for CA atoms from a CIF frame (angstroms).
    """
    out: Dict[Tuple[str, str], np.ndarray] = {}
    for a in read_cif_atoms(path):
        if a["name"].upper() == "CA":
            try:
                resid_str = str(int(float(a["resid"])))
            except Exception:
                resid_str = a["resid"]
            out[(resid_str, a["chain"])] = np.array([a["x"], a["y"], a["z"]], dtype=float)
    return out

# ---------------- core steps ----------------

def process_contact_map(args):
    in_file, cm_dir = args
    exe = os.path.join(cm_dir, "contact_map")
    out_map = f"{os.path.splitext(in_file)[0]}.map"
    subprocess.run([exe, in_file],
                   stdout=open(out_map, "w"),
                   stderr=subprocess.DEVNULL)

def run_contact_map(frames, cm_dir, cpus):
    with Pool(cpus) as pool:
        for _ in tqdm(pool.imap_unordered(
                        process_contact_map,
                        [(p, cm_dir) for p in frames]
                    ),
                      total=len(frames),
                      desc="Mapping"):
            pass

def clean_maps(src, backup, header_regex):
    """
    Keep rows after the header line using a regex so spacing differences do not break parsing.
    """
    os.makedirs(backup, exist_ok=True)
    hdr_re = re.compile(header_regex)
    for m in glob.glob(os.path.join(src, "*.map")):
        bkp = os.path.join(backup, os.path.basename(m))
        shutil.move(m, bkp)
        with open(bkp) as inp, open(m, "w") as out:
            hit_header = False
            for line in inp:
                if not hit_header:
                    if hdr_re.search(line):
                        hit_header = True
                        out.write(line)
                else:
                    if "UNMAPPED" not in line:
                        out.write(line)

def filter_map(map_file, low_nm, up_nm, out_txt):
    """
    Keep contacts with distance between low_nm and up_nm inclusive.
    Map distances are in angstroms, so thresholds in nm are converted to angstroms.
    """
    low_a = low_nm * 10.0
    up_a = up_nm * 10.0
    ov = re.compile(r"1 [01] [01] [01]")
    rz = re.compile(r"[01] [01] [01] 1")
    with open(map_file) as f, open(out_txt, "w") as out:
        for line in f:
            if not line.startswith("R"):
                continue
            parts = line.split()
            try:
                i1, i2 = int(parts[5]), int(parts[9])       # I(PDB)
                dist_a = float(parts[10])                   # angstroms from contact_map
                flags = " ".join(parts[11:15])
                r1, c1 = parts[3], parts[4]                 # resname, chain
                r2, c2 = parts[7], parts[8]
            except (IndexError, ValueError):
                continue
            if (abs(i2 - i1) >= 4 and
                low_a <= dist_a <= up_a and
                (ov.search(flags) or rz.search(flags))):
                out.write(f"{r1}\t{c1}\t{i1}\t{r2}\t{c2}\t{i2}\t{dist_a:.4f}\t{flags}\n")

def annotate(inp, outp, keep_same, keep_diff):
    seen = set()
    with open(inp) as fin, open(outp, "w") as out:
        for line in fin:
            cols = line.strip().split("\t")
            if len(cols) < 7:
                continue
            ch1, ch2 = cols[1], cols[4]
            if ((keep_same and ch1 == ch2) or
                (keep_diff and ch1 != ch2)):
                pair = (ch1, cols[2], ch2, cols[5])  # (c1, i1, c2, i2)
                inv = (ch2, cols[5], ch1, cols[2])
                if pair not in seen and inv not in seen:
                    seen.add(pair)
                    rel = ("same_chain" if ch1 == ch2 else "different_chains")
                    out.write(line.strip() + f"\t{rel}\n")

def analyze_frequency(pattern, out_norm, out_high, thr):
    """
    Build per-pair frequency over all annotated_* files.

    The normalized/high files have columns:
      Res1  Res2  Freq  Chain1  Chain2  Resname1  Resname2
    where Res1 and Res2 are i1 and i2 from the annotated lines,
    Chain1 and Chain2 are c1 and c2, and Resname1/2 are r1 and r2.
    """
    counts = defaultdict(int)
    records = []
    for fn in glob.glob(pattern):
        lines = open(fn).read().splitlines()
        records.append([l for l in lines if l and not l.startswith("Res1")])
    total = len(records) if records else 1
    for rec in records:
        for l in rec:
            c = l.split()
            if len(c) < 6:
                continue
            key = (c[2], c[5], c[1], c[4], c[0], c[3])  # (i1, i2, c1, c2, r1, r2)
            counts[key] += 1

    with open(out_norm, "w") as out:
        out.write("Res1\tRes2\tFreq\tChain1\tChain2\tResname1\tResname2\n")
        for k, v in counts.items():
            i1, i2, c1, c2, r1, r2 = k
            out.write(f"{i1}\t{i2}\t{v/total:.2f}\t{c1}\t{c2}\t{r1}\t{r2}\n")

    with open(out_high, "w") as out:
        header = open(out_norm).readline()
        out.write(header)
        for l in open(out_norm).read().splitlines()[1:]:
            if float(l.split()[2]) >= thr:
                out.write(l + "\n")

# ---------------- helpers for keys and per-frame counting ----------------

def _key_from_annotated_line(line):
    """
    From annotated_* line with columns:
      0:rname1 1:c1 2:i1_resid 3:rname2 4:c2 5:i2_resid ...
    Return orientation independent key (i_resid1, c1, i_resid2, c2).
    """
    p = line.split()
    if len(p) < 6:
        return None
    a = (p[2], p[1], p[5], p[4])
    b = (p[5], p[4], p[2], p[1])
    return a if a <= b else b

def write_counts_per_frame(ref_pairs, annotated_pattern, out_path, label="RefSet"):
    total_ref = len(ref_pairs) if ref_pairs else 1
    files = []
    for fn in glob.glob(annotated_pattern):
        m = re.search(r"frame_(\d+)", os.path.basename(fn))
        if m:
            files.append((int(m.group(1)), fn))
    files.sort(key=lambda x: x[0])

    with open(out_path, "w") as out:
        out.write(f"Frame\t{label}\tFractionOfRefSet\tFile\n")
        for idx, fn in files:
            cnt = 0
            with open(fn) as f:
                for L in f:
                    if not L.strip() or L.startswith("Res1"):
                        continue
                    k = _key_from_annotated_line(L)
                    if k and k in ref_pairs:
                        cnt += 1
            out.write(f"{idx}\t{cnt}\t{(cnt/total_ref):.4f}\t{os.path.basename(fn)}\n")

def go_pairs_as_resid_chain(itp_path, inv_rev):
    ref = set()
    for line in open(itp_path):
        if not line.startswith("molecule_0_"):
            continue
        a, b = line.split()[:2]
        i1 = int(a.rsplit("_", 1)[1])
        i2 = int(b.rsplit("_", 1)[1])
        if i1 in inv_rev and i2 in inv_rev:
            (res1, ch1) = inv_rev[i1]
            (res2, ch2) = inv_rev[i2]
            t1 = (res1, ch1, res2, ch2)
            t2 = (res2, ch2, res1, ch1)
            ref.add(t1 if t1 <= t2 else t2)
    return ref

def _key_from_high_line(line):
    """
    From high_* line with columns:
      0:i1 1:i2 2:freq 3:c1 4:c2 5:r1 6:r2
    Build an orientation-independent tuple (i1_resid, c1, i2_resid, c2).
    """
    p = line.split()
    if len(p) < 5:
        return None
    a = (p[0], p[3], p[1], p[4])
    b = (p[1], p[4], p[0], p[3])
    return a if a <= b else b

def write_high_counts_per_frame(highfile, annotated_pattern, out_path):
    high_keys = set()
    with open(highfile) as fh:
        next(fh, None)
        for line in fh:
            k = _key_from_high_line(line)
            if k:
                high_keys.add(k)
    write_counts_per_frame(high_keys, annotated_pattern, out_path, label="HighContacts")

# ---------------- martinize2 runner ----------------

def run_martinize_from_atom(atom_path,
                            go_map_path,
                            merge,
                            dssp,
                            goeps,
                            src,
                            posres,
                            ss,
                            nter_list,
                            cter_list,
                            neutral_termini,
                            *,
                            go_low,
                            go_up,
                            go_res_dist,
                            go_write_file,
                            go_backbone,
                            go_atomname,
                            water_bias,
                            water_bias_eps,
                            id_regions,
                            idr_tune,
                            noscfix,
                            scfix,
                            cys,
                            mutate,
                            modify,
                            write_graph,
                            write_repair,
                            write_canon,
                            vcount,
                            maxwarn_list,
                            to_ff=None,
                            extra_ff_dir=None,
                            extra_map_dir=None
):
    atom = atom_path
    base = os.path.splitext(os.path.basename(atom))[0]
    atom_dir = os.path.dirname(atom) or "."
    cg = os.path.join(atom_dir, f"{base}_CG.pdb")

    cmd = ["martinize2", "-f", atom]

    if merge:
        cmd += ["-merge", merge]
    if dssp:
        cmd += ["-dssp", dssp]
    if ss:
        cmd += ["-ss", ss]
        
    # force field selection
    if to_ff:
        cmd += ["-ff", to_ff]

    # additional ff dirs
    if extra_ff_dir:
        for d in extra_ff_dir:
            cmd += ["-ff-dir", d]

    # additional map dirs
    if extra_map_dir:
        for d in extra_map_dir:
            cmd += ["-map-dir", d]    

    # Go model from external map file plus tunables
    cmd += ["-go", go_map_path, "-go-eps", str(goeps)]
    if go_low is not None:
        cmd += ["-go-low", str(go_low)]
    if go_up is not None:
        cmd += ["-go-up", str(go_up)]
    if go_res_dist is not None:
        cmd += ["-go-res-dist", str(go_res_dist)]
    if go_write_file is not None:
        if go_write_file == "":
            cmd += ["-go-write-file"]
        else:
            cmd += ["-go-write-file", go_write_file]
    if go_backbone is not None:
        cmd += ["-go-backbone", go_backbone]
    if go_atomname is not None:
        cmd += ["-go-atomname", go_atomname]

    # Water bias related
    if water_bias:
        cmd += ["-water-bias"]
    if water_bias_eps:
        cmd += ["-water-bias-eps", *water_bias_eps]
    if id_regions:
        cmd += ["-id-regions", *id_regions]
    if idr_tune:
        cmd += ["-idr-tune"]

    # Protein description and modifications
    if noscfix:
        cmd += ["-noscfix"]
    if scfix:
        cmd += ["-scfix"]
    if cys is not None:
        cmd += ["-cys", cys]
    if mutate:
        cmd += ["-mutate", *mutate]
    if modify:
        cmd += ["-modify", *modify]

    # Termini patches
    for mod in (nter_list or []):
        cmd += ["-nter", mod]
    for mod in (cter_list or []):
        cmd += ["-cter", mod]
    if neutral_termini:
        cmd += ["-nt"]

    # Debug and limits
    if write_graph:
        cmd += ["-write-graph", write_graph]
    if write_repair:
        cmd += ["-write-repair", write_repair]
    if write_canon:
        cmd += ["-write-canon", write_canon]
    if vcount and vcount > 0:
        cmd += ["-v"] * vcount

    # core output and settings
    cmd += [
        "-o", "topol.top",
        "-x", cg,
        "-p", posres,
        "-cys", "auto" if cys is None else cys,
        "-ignh",
        "-name", "molecule_0",
    ]
    if src is not None:
        cmd += ["-from", src]
        
    if maxwarn_list:
        cmd += ["-maxwarn", *[str(x) for x in maxwarn_list]]
    else:
        cmd += ["-maxwarn", "100"]

    print("Running martinize2:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return atom

# ---------------- index builder for PDB and CIF ----------------

def build_index(struct_path: str):
    """
    Map (resid_str, chain_id) -> sequential bead index.
    PDB is parsed by text. CIF is parsed with read_cif_atoms.
    """
    ext = os.path.splitext(struct_path)[1].lower()
    inv, offset = {}, 0

    if ext == ".pdb":
        by_chain = defaultdict(list)
        with open(struct_path) as fh:
            for l in fh:
                if l.startswith(("ATOM", "HETATM")):
                    ch = l[21]
                    try:
                        resi = int(l[22:26])
                    except ValueError:
                        continue
                    by_chain[ch].append(resi)
        for ch in sorted(by_chain):
            uniq = sorted(set(by_chain[ch]))
            for i, r in enumerate(uniq, 1):
                inv[(str(r), ch)] = i + offset
            offset += len(uniq)
        return inv

    # CIF
    by_chain = defaultdict(list)
    for a in read_cif_atoms(struct_path):
        ch = a["chain"] or "A"
        try:
            resi = int(float(a["resid"]))
        except ValueError:
            continue
        by_chain[ch].append(resi)
    for ch in sorted(by_chain):
        uniq = sorted(set(by_chain[ch]))
        for i, r in enumerate(uniq, 1):
            inv[(str(r), ch)] = i + offset
        offset += len(uniq)
    return inv

def load_itp(path):
    s = set()
    for line in open(path):
        if line.startswith("molecule_0_"):
            a, b = line.split()[:2]
            i, j = map(int, [a.rsplit("_", 1)[1], b.rsplit("_", 1)[1]])
            s.add((min(i, j), max(i, j)))
    return s

def write_mock(highfile, struct_path, itp_out):
    """
    Write a mock Go ITP using residue indices (i1, i2) and chain IDs (c1, c2).
    Works for PDB or CIF, using build_index.
    """
    inv = build_index(struct_path)  # keys: (str(resid), chain) -> sequential bead index
    with open(itp_out, "w") as out:
        out.write("[ nonbond_params ]\n")
        with open(highfile) as hf:
            next(hf, None)  # skip header
            for line in hf:
                p = line.split()
                if len(p) < 5:
                    continue
                resid1, resid2 = p[0], p[1]
                ch1, ch2 = p[3], p[4]
                i1 = inv.get((resid1, ch1))
                i2 = inv.get((resid2, ch2))
                if i1 and i2:
                    out.write(f"molecule_0_{i1} molecule_0_{i2} 1 0.00000000 0.00000000 ; mock\n")

# ---------------- main ----------------

def main():
    import sys

    # Log the command used to run the script
    with open("run.log", "a") as log_file:
        log_file.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Command: {' '.join(sys.argv)}\n")

    parser = argparse.ArgumentParser(
        description="Run full contact analysis and build coarse-grained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--cm", default=".", help="Path to contact_map executable directory")
    parser.add_argument("--type", choices=["both","intra","inter"], default="both", help="Contact type")
    parser.add_argument("--cpus", type=int, default=15, help="Number of parallel processes")
    parser.add_argument("--threshold", type=float, default=0.7, help="Frequency threshold for high-frequency contacts")

    # martinize2 related arguments
    parser.add_argument("--merge", type=str, default=None, help="Chains to merge or 'all'")

    # optional DSSP
    parser.add_argument("--dssp", dest="dssp_path", default=None, help="Path to dssp executable")

    # position restraints
    parser.add_argument("--posres", choices=["none", "all", "backbone"], default="none",
                        help="Output position restraints")

    # manual secondary structure
    parser.add_argument("--ss", type=str, default=None, help="Manual secondary structure string or single letter")

    # Go model controls and contact thresholds in nm
    parser.add_argument("--go-eps", dest="go_eps", type=float, default=9.414, help="Epsilon for go potential")
    parser.add_argument("--go-low", dest="go_low", type=float, default=0.3,
                        help="Minimum contact distance threshold in nm")
    parser.add_argument("--go-up", dest="go_up", type=float, default=1.1,
                        help="Maximum contact distance threshold in nm")
    parser.add_argument("--go-res-dist", dest="go_res_dist", type=int, default=None,
                        help="Minimum graph distance below which contacts are removed")
    parser.add_argument("--go-write-file", dest="go_write_file", nargs="?", const="", default=None,
                        help="Write contact map when Martinize2 calculates it; optional output path")
    parser.add_argument("--go-backbone", dest="go_backbone", type=str, default="BB",
                        help="Backbone bead name for Go site")
    parser.add_argument("--go-atomname", dest="go_atomname", type=str, default="CA",
                        help="Virtual Go site atom name")
                        
    parser.add_argument("--ff", dest="to_ff", default="martini3001",
                    help="Coarse-grained force field for martinize2")

    parser.add_argument("--ff-dir", dest="extra_ff_dir", nargs="+", default=None,
                    help="Additional repository paths for custom force fields")

    parser.add_argument("--map-dir", dest="extra_map_dir", nargs="+", default=None,
                    help="Additional repository paths for mapping files")


    # Water bias options
    parser.add_argument("--water-bias", dest="water_bias", action="store_true",
                        help="Apply water bias to secondary structure elements")
    parser.add_argument("--water-bias-eps", dest="water_bias_eps", nargs="+", default=None,
                        help="Water bias strengths like H:3.6 C:2.1 idr:2.1")
    parser.add_argument("--id-regions", dest="id_regions", nargs="+", default=None,
                        help="Disordered regions as [chain-]start:end tokens")
    parser.add_argument("--idr-tune", dest="idr_tune", action="store_true",
                        help="Tune IDR regions with specific bonded potentials (deprecated)")

    # Protein description / modifications
    parser.add_argument("--noscfix", dest="noscfix", action="store_true",
                        help="Do not apply side chain corrections")
    parser.add_argument("--scfix", dest="scfix", action="store_true",
                        help="Legacy scfix flag")
    parser.add_argument("--cys", dest="cys", default=None, help="Cystein bonds setting")
    parser.add_argument("--mutate", dest="mutate", nargs="+", default=None,
                        help="Mutations like A-PHE45:ALA PHE30:ALA")
    parser.add_argument("--modify", dest="modify", nargs="+", default=None,
                        help="Residue modifications like A-ASP45:ASP0 ASP:ASP0 +HSE")

    # Termini patches
    parser.add_argument("--nter", dest="nter", action="append", default=None,
                        help="Patch for N-termini")
    parser.add_argument("--cter", dest="cter", action="append", default=None,
                        help="Patch for C-termini")
    parser.add_argument("--nt", dest="neutral_termini", action="store_true",
                        help="Set neutral termini")

    # source force field
    parser.add_argument("--from", dest="md_source", choices=["amber","charmm"], default=None,
                        help="Source force field for martinize2")

    # Debugging / diagnostics passthrough
    parser.add_argument("--write-graph", dest="write_graph", default=None, help="Write graph after MakeBonds")
    parser.add_argument("--write-repair", dest="write_repair", default=None, help="Write graph after RepairGraph")
    parser.add_argument("--write-canon", dest="write_canon", default=None, help="Write graph after CanonicalizeModifications")
    parser.add_argument("-v", dest="vcount", action="count", default=0, help="Increase Martinize2 verbosity")
    parser.add_argument("--maxwarn", dest="maxwarn_list", nargs="+", default=None,
                        help="Maximum allowed warnings for Martinize2")

    # Append missing high-frequency contacts
    parser.add_argument("--add-missing", dest="add_missing", action="store_true",
                        help="Append entries from missing_high_freq.itp into go_nbparams.itp to include all high-frequency contacts")

    # optional: force a specific frame index
    parser.add_argument("--force-frame", type=int, default=None,
                        help="Use this specific frame index for martinize2")

    # NEW FLAG: sigma recalculation
    parser.add_argument("--sigma", dest="sigma", action="store_true",
                        help="Recalculate sigma values from selected frame and replace them in go_nbparams.itp")
    
    # NEW FLAG: skip CG model generation and just do contact analysis
    parser.add_argument("--skip-cg", default=False, dest="skip_cg", action="store_true",
                        help="Skip coarse-grained model generation and just perform contact analysis")

    args = parser.parse_args()

    # discover frames
    frames_map = list_frames()
    if not frames_map:
        raise FileNotFoundError("No frames found. Expected frame_####.pdb or frame_####.cif")
    frames = [frames_map[i] for i in sorted(frames_map.keys())]

    # optional merge all chains
    if args.merge == "all" and frames:
        first = frames[0]
        if first.lower().endswith(".pdb"):
            uni = mda.Universe(first)
            chains = sorted({(seg.segid or "").strip() for seg in uni.segments if (seg.segid or "").strip()})
        else:
            chains = get_cif_chains(first)
        args.merge = ",".join(chains)

    # run external contact mapper and clean maps
    run_contact_map(frames, args.cm, args.cpus)
    clean_maps(".", "orig_maps", header_regex=r"ID\s+I1\s+AA\s+C\s+I\(PDB\)")

    # filter and annotate using go-low and go-up in nm
    filtered = []
    for mfile in glob.glob("*.map"):
        base, _ = os.path.splitext(mfile)
        out_txt = f"filtered_{os.path.basename(base)}.txt"
        filter_map(mfile, args.go_low, args.go_up, out_txt)
        filtered.append(out_txt)

    for f in filtered:
        suffix = "_intra" if args.type == "intra" else "_inter" if args.type == "inter" else ""
        outp = f"annotated_{f.replace('.txt', suffix + '.txt')}"
        annotate(f, outp,
                 keep_same=(args.type in ("both", "intra")),
                 keep_diff=(args.type in ("both", "inter")))

    # frequency over all annotated files
    norm_file = f"normalized_{args.type}.txt"
    high_file = f"high_{args.type}.txt"
    analyze_frequency("annotated_*.txt", norm_file, high_file, args.threshold)

    # per-frame counts against the high set
    write_high_counts_per_frame(high_file, "annotated_*.txt", "high_counts_per_frame.txt")

    if (args.skip_cg):
        print(f"High-frequency contacts written to {high_file}")
        print("Analysis completed without CG model generation.")
        return

    # determine available frame files for selection and martinize2
    available_map = {}
    for root in (".", "output_files"):
        if os.path.isdir(root):
            for path in glob.glob(os.path.join(root, "frame_*.*")):
                m = FRAME_RE.match(os.path.basename(path))
                if m:
                    available_map[int(m.group(1))] = path
    available_map.update(frames_map)

    # choose frame
    if args.force_frame is not None:
        if args.force_frame not in available_map:
            raise FileNotFoundError(f"--force-frame {args.force_frame} has no frame file in . or output_files/")
        frame_idx = int(args.force_frame)
        atom_path = available_map[frame_idx]
    else:
        high_keys = set()
        with open(high_file) as fh:
            next(fh, None)
            for line in fh:
                k = _key_from_high_line(line)
                if k:
                    high_keys.add(k)
        candidates = []
        for fn in glob.glob("annotated_*.txt"):
            m = re.search(r"frame_(\d+)", os.path.basename(fn))
            if not m:
                continue
            idx = int(m.group(1))
            if idx not in available_map:
                continue
            cnt = 0
            with open(fn) as f:
                for L in f:
                    if not L.strip() or L.startswith("Res1"):
                        continue
                    q = _key_from_annotated_line(L)
                    if q and q in high_keys:
                        cnt += 1
            candidates.append((idx, cnt, fn))
        if not candidates:
            raise FileNotFoundError("No frame available that matches annotated_*.txt")
        max_cnt = max(c for _, c, _ in candidates)
        best_idx = min(i for i, c, _ in candidates if c == max_cnt)
        frame_idx = best_idx
        atom_path = available_map[frame_idx]

    print(f"Using frame {frame_idx} -> {atom_path}")

    # locate the corresponding .map for the selected frame
    base = os.path.splitext(os.path.basename(atom_path))[0]
    map_candidate_same_dir = os.path.join(os.path.dirname(atom_path) or ".", f"{base}.map")
    map_candidate_out = os.path.join("output_files", f"{base}.map")
    if os.path.isfile(map_candidate_same_dir):
        go_map_path = map_candidate_same_dir
    elif os.path.isfile(map_candidate_out):
        go_map_path = map_candidate_out
    else:
        raise FileNotFoundError(f"Map file for selected frame not found: {base}.map")

    # run martinize2
    atom_path = run_martinize_from_atom(
        atom_path,
        go_map_path,
        args.merge,
        args.dssp_path,
        args.go_eps,
        args.md_source,
        args.posres,
        args.ss,
        args.nter,
        args.cter,
        args.neutral_termini,
        go_low=args.go_low,
        go_up=args.go_up,
        go_res_dist=args.go_res_dist,
        go_write_file=args.go_write_file,
        go_backbone=args.go_backbone,
        go_atomname=args.go_atomname,
        water_bias=args.water_bias,
        water_bias_eps=args.water_bias_eps,
        id_regions=args.id_regions,
        idr_tune=args.idr_tune,
        noscfix=args.noscfix,
        scfix=args.scfix,
        cys=args.cys,
        mutate=args.mutate,
        modify=args.modify,
        write_graph=args.write_graph,
        write_repair=args.write_repair,
        write_canon=args.write_canon,
        vcount=args.vcount,
        maxwarn_list=args.maxwarn_list,
        to_ff=args.to_ff,
        extra_ff_dir=args.extra_ff_dir,
        extra_map_dir=args.extra_map_dir
    )

    # build index and reverse map
    inv_map = build_index(atom_path)                          # (resid_str, chain) -> seq_idx
    inv_rev_full = {seq_idx: key for key, seq_idx in inv_map.items()}  # seq_idx -> (resid_str, chain)
    inv_map_inv = {v: k[1] for k, v in inv_map.items()}      # seq_idx -> chain

    # collect high-frequency pairs mapped into sequential bead indices
    high_pairs = set()
    with open(high_file) as hf:
        next(hf)
        for line in hf:
            p = line.split()
            if len(p) < 5:
                continue
            resid1, resid2 = p[0], p[1]
            ch1, ch2 = p[3], p[4]
            i1 = inv_map.get((resid1, ch1))
            i2 = inv_map.get((resid2, ch2))
            if i1 and i2:
                high_pairs.add((min(i1, i2), max(i1, i2)))

    # write mock using residue indices and chains
    mock_itp = f"go_nbparams_mock_{args.type}.itp"
    write_mock(high_file, atom_path, mock_itp)

    # --- rewrite go_nbparams.itp with proper header and filtering ---
    src_itp = "go_nbparams.itp"
    bak_itp = "go_nbparams.itp.bak"
    shutil.copy(src_itp, bak_itp)

    header_re = re.compile(r'^\s*\[\s*nonbond_params\s*\]\s*$', re.IGNORECASE)

    with open(bak_itp, "r") as rf, open(src_itp, "w") as wf:
        # always write exactly one header
        wf.write("[ nonbond_params ]\n")

        for line in rf:
            ls = line.strip()

            # skip any existing section headers to avoid duplicates
            if header_re.match(ls):
                continue

            # pass through comments and blanks unchanged
            if not ls or ls.startswith(";"):
                wf.write(line)
                continue

            # process only pair lines; pass through anything else
            if not ls.startswith("molecule_0_"):
                wf.write(line)
                continue

            # parse bead indices
            try:
                a, b = ls.split()[:2]
                i1 = int(a.rsplit("_", 1)[1])
                i2 = int(b.rsplit("_", 1)[1])
            except Exception:
                wf.write(line)
                continue

            pair_key = (min(i1, i2), max(i1, i2))
            c1 = inv_map_inv.get(i1)
            c2 = inv_map_inv.get(i2)
            same_chain = (c1 is not None and c2 is not None and c1 == c2)

            # decide which Go contacts to keep depending on type
            if args.type == "both":
                # keep only high-frequency (intra and inter)
                keep = (pair_key in high_pairs)

            elif args.type == "intra":
                if same_chain:
                    # for intra contacts: keep only high-frequency intra
                    keep = (pair_key in high_pairs)
                else:
                    # for inter contacts: do not touch them, always keep
                    keep = True

            else:  # args.type == "inter"
                if same_chain:
                    # for intra contacts: do not touch them, always keep
                    keep = True
                else:
                    # for inter contacts: keep only high-frequency inter
                    keep = (pair_key in high_pairs)

            if keep:
                wf.write(line)
        # if not keep: drop this Go pair

    print("ITP filtering done:",
          f"type={args.type}, kept_high_pairs={len(load_itp('go_nbparams.itp'))}, high_pairs_total={len(high_pairs)}",
          flush=True)

    # --- optional sigma recalculation from selected frame ---
    if getattr(args, "sigma", False):
        print("Recalculating sigma values from selected frame distances...", flush=True)

        # Load the selected structure
        u = mda.Universe(atom_path)

        # Collect current pairs from the filtered go_nbparams.itp
        pairs = []
        for line in open("go_nbparams.itp"):
            if line.startswith("molecule_0_"):
                a, b = line.split()[:2]
                try:
                    i1 = int(a.rsplit("_", 1)[1])
                    i2 = int(b.rsplit("_", 1)[1])
                    pairs.append((i1, i2))
                except Exception:
                    continue

        # Measure distances on the selected frame and compute sigma = distance / 2^(1/6)
        dist_data = {}
        for i1, i2 in tqdm(pairs, desc="Computing sigma from frame"):
            (r1, c1) = inv_rev_full.get(i1, (None, None))
            (r2, c2) = inv_rev_full.get(i2, (None, None))
            if not all([r1, r2, c1, c2]):
                continue
            sel1 = u.select_atoms(f"segid {c1} and resid {r1} and name CA")
            sel2 = u.select_atoms(f"segid {c2} and resid {r2} and name CA")
            if sel1.n_atoms > 0 and sel2.n_atoms > 0:
                d_nm = distance_array(sel1.positions, sel2.positions)[0, 0] / 10.0
                sigma = d_nm / (2 ** (1 / 6))
                dist_data[(min(i1, i2), max(i1, i2))] = sigma

        # Rewrite go_nbparams.itp replacing only sigma values
        tmp_out = "go_nbparams_sigma.itp"
        with open("go_nbparams.itp", "r") as rf, open(tmp_out, "w") as wf:
            for line in rf:
                if line.startswith("molecule_0_"):
                    parts = line.split()
                    if len(parts) < 5:
                        wf.write(line)
                        continue
                    a, b = parts[:2]
                    try:
                        i1 = int(a.rsplit("_", 1)[1])
                        i2 = int(b.rsplit("_", 1)[1])
                    except Exception:
                        wf.write(line)
                        continue
                    pkey = (min(i1, i2), max(i1, i2))
                    sigma = dist_data.get(pkey)
                    if sigma is not None:
                        # keep epsilon from the line if it exists, otherwise fallback to args.go_eps
                        try:
                            eps = float(parts[4])
                        except Exception:
                            eps = args.go_eps
                        wf.write(f"{a} {b} 1 {sigma:.8f} {eps:.8f} ; sigma from frame {frame_idx}\n")
                    else:
                        wf.write(line)
                else:
                    wf.write(line)

        shutil.move(tmp_out, "go_nbparams.itp")
        print("Sigma recalculation complete.", flush=True)

    # measure distances for missing high-frequency pairs and write a separate ITP
    mock_pairs = load_itp(mock_itp)
    real_pairs_after = load_itp("go_nbparams.itp")
    missing = mock_pairs - real_pairs_after

    # prepare mapping info for missing
    missing_info = []
    with open(high_file) as hf:
        next(hf, None)
        for line in hf:
            p = line.split()
            if len(p) < 5:
                continue
            r1_resid, r2_resid = p[0], p[1]
            ch1, ch2 = p[3], p[4]
            i1 = inv_map.get((r1_resid, ch1))
            i2 = inv_map.get((r2_resid, ch2))
            if i1 and i2 and (min(i1, i2), max(i1, i2)) in missing:
                missing_info.append((r1_resid, ch1, r2_resid, ch2))

    # collect all frame files for distance measurement (PDB and CIF)
    frame_files = sorted(set(glob.glob("frame_*.pdb") + glob.glob("frame_*.cif")))

    # distances are accumulated in nanometers to match Martinize2 Go parameters
    dist_dict = {mi: [] for mi in missing_info}
    for fpath in tqdm(frame_files, desc="Measuring missing distances"):
        if fpath.lower().endswith(".pdb"):
            u = mda.Universe(fpath)
            u.guess_TopologyAttrs(to_guess=["elements"])  # optional; no impact on distances
            for r1, c1, r2, c2 in missing_info:
                sel1 = u.select_atoms(f"segid {c1} and resid {r1} and name CA")
                sel2 = u.select_atoms(f"segid {c2} and resid {r2} and name CA")
                if sel1.n_atoms > 0 and sel2.n_atoms > 0:
                    d_nm = distance_array(sel1.positions, sel2.positions)[0, 0] / 10.0  # A -> nm
                    dist_dict[(r1, c1, r2, c2)].append(d_nm)
        else:
            coords = get_cif_ca_coords(fpath)  # angstroms
            for r1, c1, r2, c2 in missing_info:
                k1 = (r1, c1); k2 = (r2, c2)
                if k1 in coords and k2 in coords:
                    d_ang = np.linalg.norm(coords[k1] - coords[k2])
                    dist_dict[(r1, c1, r2, c2)].append(d_ang / 10.0)  # nm

    missing_itp = "missing_high_freq.itp"
    with open(missing_itp, "w") as wf:
        wf.write("; missing high-frequency contacts\n")
        for (r1, c1, r2, c2), ds in dist_dict.items():
            if not ds:
                continue
            avg = np.mean(ds)                 # nm
            if avg > args.go_up:  # keep only if within the go_up threshold (nm)
                continue
            rmin = avg / (2 ** (1 / 6))       # nm, Lennard-Jones minimum
            i1, i2 = inv_map[(r1, c1)], inv_map[(r2, c2)]
            wf.write(f"molecule_0_{i1} molecule_0_{i2} 1 {rmin:.8f} {args.go_eps:.8f} ; go bond {avg:.4f}\n")

    # optionally append missing high-frequency contacts into the selected ITP
    if args.add_missing and os.path.isfile(missing_itp):
        with open("go_nbparams.itp", "a") as out, open(missing_itp, "r") as addf:
            for ln in addf:
                ls = ln.strip()
                if not ls:
                    continue
                if ls.startswith(";") or ls.startswith("molecule_0_"):
                    out.write(ln)
        print("Appended missing high-frequency contacts into go_nbparams.itp", flush=True)

    # build reference sets for per-frame counting
    high_ref = set()
    with open(high_file) as fh:
        next(fh, None)
        for line in fh:
            k = _key_from_high_line(line)
            if k:
                high_ref.add(k)

    go_ref = go_pairs_as_resid_chain("go_nbparams.itp", inv_rev_full)

    write_counts_per_frame(high_ref, "annotated_*.txt", "high_counts_per_frame.txt", label="HighContacts")
    write_counts_per_frame(go_ref, "annotated_*.txt", "go_counts_per_frame.txt", label="GoContacts")

    # move outputs
    outdir = "output_files"
    os.makedirs(outdir, exist_ok=True)

    for ext in ("*.txt", "*.map"):
        for fn in glob.glob(ext):
            shutil.move(fn, os.path.join(outdir, fn))

    for path in frames:
        base = os.path.basename(path)
        if base.endswith("_CG.pdb"):
            continue
        if os.path.exists(path):
            shutil.move(path, os.path.join(outdir, base))

    print("Done.")

if __name__ == "__main__":
    main()
