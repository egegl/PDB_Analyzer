import os
import time
import numpy as np
import pandas as pd
from math import sqrt, pi, radians
from Bio.PDB import PDBParser, PDBIO, NeighborSearch, Selection
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Chain import Chain
from Bio.PDB.vectors import Vector
from cleaner import clean

# CX constants
cx_radius = 10  # Angstrom
v_atom = 20.1  # Angstrom^3
v_sphere = (4 / 3) * pi * cx_radius ** 3  # Angstrom^3

# roughness constants
r_min_radius = 0.2  # Angstrom
r_radius_step = 0.1  # Angstrom
r_max_radius = 4.0 + r_radius_step  # Angstrom
radii = np.arange(r_min_radius, r_max_radius, r_radius_step)

# patch constants
patch_radius = 9  # Angstrom

# structure constants
next_chain_id = 'A'

residue_max_asas = {  # Angstrom^2
    'ALA': 129.0, 'ARG': 274.0,
    'ASN': 195.0, 'ASP': 193.0,
    'CYS': 167.0, 'GLN': 225.0,
    'GLU': 223.0, 'GLY': 104.0,
    'HIS': 224.0, 'ILE': 197.0,
    'LEU': 201.0, 'LYS': 236.0,
    'MET': 224.0, 'PHE': 240.0,
    'PRO': 159.0, 'SER': 155.0,
    'THR': 172.0, 'TRP': 285.0,
    'TYR': 263.0, 'VAL': 174.0
}

residue_hydrophobicities = {
    "ALA": 1.8, "ARG": -4.5,
    "ASN": -3.5, "ASP": -3.5,
    "CYS": 2.5, "GLN": -3.5,
    "GLU": -3.5, "GLY": -0.4,
    "HIS": -3.2, "ILE": 4.5,
    "LEU": 3.8, "LYS": -3.9,
    "MET": 1.9, "PHE": 2.8,
    "PRO": -1.6, "SER": -0.8,
    "THR": -0.7, "TRP": -0.9,
    "TYR": -1.3, "VAL": 4.2
}

standard_volumes_of_atoms = {  # Angstrom^3
    'ALA N': 14.4, 'ALA CA': 12.4, 'ALA C': 8.6, 'ALA O': 22.8, 'ALA CB': 32.7,
    'ARG N': 14.0, 'ARG CA': 11.7, 'ARG C': 8.4, 'ARG O': 22.2, 'ARG CB': 20.4, 'ARG CG': 20.9, 'ARG CD': 20.3,
    'ARG NE': 16.3, 'ARG CZ': 9.4, 'ARG NH1': 23.1, 'ARG NH2': 24.1,
    'ASN N': 13.8, 'ASN CA': 11.2, 'ASN C': 8.5, 'ASN O': 22.0, 'ASN CB': 20.2, 'ASN CG': 9.2, 'ASN OD1': 21.7,
    'ASN ND2': 25.5,
    'ASP N': 13.9, 'ASP CA': 11.3, 'ASP C': 8.3, 'ASP O': 22.2, 'ASP CB': 20.8, 'ASP CG': 9.1, 'ASP OD1': 20.7,
    'ASP OD2': 21.6,
    'CYS N': 14.3, 'CYS CA': 11.8, 'CYS C': 8.5, 'CYS O': 22.8, 'CYS CB': 22.8, 'CYS SG': 34.2,
    'GLN N': 13.9, 'GLN CA': 11.5, 'GLN C': 8.4, 'GLN O': 22.0, 'GLN CB': 20.0, 'GLN CG': 20.4, 'GLN CD': 9.6,
    'GLN OE1': 23.4, 'GLN NE2': 24.6,
    'GLU N': 14.0, 'GLU CA': 11.7, 'GLU C': 8.4, 'GLU O': 22.1, 'GLU CB': 20.3, 'GLU CG': 21.1, 'GLU CD': 9.1,
    'GLU OE1': 22.8, 'GLU OE2': 23.5,
    'GLY N': 14.9, 'GLY CA': 20.0, 'GLY C': 9.3, 'GLY O': 22.8,
    'HIS N': 14.3, 'HIS CA': 11.8, 'HIS C': 8.4, 'HIS O': 22.1, 'HIS CB': 21.5, 'HIS CG': 10.4, 'HIS ND1': 16.5,
    'HIS CD2': 19.7, 'HIS CE1': 19.2, 'HIS NE2': 17.8,
    'ILE N': 14.1, 'ILE CA': 11.4, 'ILE C': 8.2, 'ILE O': 22.5, 'ILE CB': 13.0, 'ILE CG1': 22.4, 'ILE CG2': 33.2,
    'ILE CD1': 35.3,
    'LEU N': 14.1, 'LEU CA': 11.6, 'LEU C': 8.5, 'LEU O': 22.3, 'LEU CB': 20.8, 'LEU CG': 13.7, 'LEU CD1': 35.2,
    'LEU CD2': 35.0,
    'LYS N': 14.0, 'LYS CA': 11.6, 'LYS C': 8.5, 'LYS O': 22.3, 'LYS CB': 20.6, 'LYS CG': 21.0, 'LYS CD': 21.7,
    'LYS CE': 21.8, 'LYS NZ': 23.0,
    'MET N': 14.0, 'MET CA': 11.7, 'MET C': 8.5, 'MET O': 22.6, 'MET CB': 21.0, 'MET CG': 23.0, 'MET SD': 27.8,
    'MET CE': 34.8,
    'PHE N': 14.0, 'PHE CA': 11.6, 'PHE C': 8.4, 'PHE O': 22.5, 'PHE CB': 21.1, 'PHE CG': 10.2, 'PHE CD1': 20.3,
    'PHE CD2': 20.8, 'PHE CE1': 22.0, 'PHE CE2': 22.2, 'PHE CZ': 22.0,
    'PRO N': 9.4, 'PRO CA': 12.1, 'PRO C': 8.4, 'PRO O': 22.8, 'PRO CB': 23.0, 'PRO CG': 24.1, 'PRO CD': 20.7,
    'SER N': 14.3, 'SER CA': 11.6, 'SER C': 8.4, 'SER O': 22.1, 'SER CB': 20.9, 'SER OG': 23.4,
    'THR N': 14.0, 'THR CA': 11.3, 'THR C': 8.3, 'THR O': 21.9, 'THR CB': 12.9, 'THR OG1': 23.1, 'THR CG2': 31.8,
    'TRP N': 14.3, 'TRP CA': 11.8, 'TRP C': 8.4, 'TRP O': 22.1, 'TRP CB': 21.5, 'TRP CG': 10.4, 'TRP CD1': 20.1,
    'TRP CD2': 10.8, 'TRP NE1': 18.3, 'TRP CE2': 10.2, 'TRP CE3': 20.8, 'TRP CZ2': 20.9, 'TRP CZ3': 22.1,
    'TRP CH2': 21.4,
    'TYR N': 13.9, 'TYR CA': 11.5, 'TYR C': 8.5, 'TYR O': 22.1, 'TYR CB': 21.3, 'TYR CG': 10.2, 'TYR CD1': 20.0,
    'TYR CD2': 20.1, 'TYR CE1': 20.3, 'TYR CE2': 20.3, 'TYR CZ': 10.0, 'TYR OH': 25.1,
    'VAL N': 14.1, 'VAL CA': 11.4, 'VAL C': 8.4, 'VAL O': 22.6, 'VAL CB': 13.3, 'VAL CG1': 33.5, 'VAL CG2': 33.4,
    'VAL SD': 27.8
}

# global lists by patch
all_patches = []
surface_patches = []
interface_patches = []
patch_chains = []
patch_nums = []
patch_types = []
patch_cx = []
patch_asa = []
patch_hydrophobicity = []
patch_planarity = []
patch_roughness = []

# global lists by residue
surface_residues = []
surface_residues_by_chain = {}
interface_residues = []
interface_residues_by_chain = {}
res_chains = []
res_nums = []
res_types = []
res_cx = []
res_rasa = []
res_surface = []
res_hydrophobicity = []


def calculate_roughness_by_patch(patch):
    surface_areas = []
    residues = Selection.unfold_entities(patch, 'R')

    # create dummy chain object with only the patch residues
    patch_chain = Chain(next_chain_id)
    for residue in patch:
        patch_chain.add(residue)

    # calculate the surface area of patch for each radius (0.2-4.0 Angstrom)
    for radius in radii:
        sr = ShrakeRupley(probe_radius=radius)
        sr.compute(patch_chain, level="R")
        area = sum(residue.sasa for residue in residues)
        surface_areas.append(area)

    # calculate the slope of the log of surface area vs. radius
    roughness = 0
    num_radii = len(radii) - 1
    for i in range(num_radii):
        log_area1 = np.log(surface_areas[i])
        log_area2 = np.log(surface_areas[i + 1])
        log_radius1 = np.log(radii[i])
        log_radius2 = np.log(radii[i + 1])
        diff = (log_area2 - log_area1) / (log_radius2 - log_radius1)
        roughness += diff

    # get mean roughness
    roughness /= num_radii

    # normalize roughness
    roughness = 2 - roughness
    return roughness


def calculate_planarity_by_patch(patch):
    # get the coordinates of the atoms in patch
    coords = []
    for residue in patch:
        coords.extend(atom.coord for atom in residue if atom.element != "H")
    coords = np.array(coords)

    # center the coordinates by subtracting the mean
    coords = coords - coords.mean(axis=0)

    # calculate the singular value decomposition of the centered coordinates
    u, s, v = np.linalg.svd(coords)

    # the third principal component is the normal vector to the plane of best fit (defined by the first & second PCs)
    normal = v[2]

    # calculate the distance of each point from the plane
    distances = np.abs(np.dot(coords, normal))

    # calculate the root mean squared distance & return planarity (higher = flatter)
    rmsd = sqrt(np.mean(distances ** 2))
    return 1 / rmsd


def calculate_asa_by_patch(domain):
    # Calculate mean ASA for the domain
    asa_sum = 0
    n_atom = 0
    domain_atoms = get_domain_atoms(domain)
    for atom in domain_atoms:
        asa_sum += atom.sasa
        n_atom += 1
    return asa_sum / n_atom


def calculate_hydrophobicity_by_patch(patch):
    hydrophobicity_sum = 0
    n_atom = 0
    chain_id = patch[0].get_parent().id
    for residue in patch:
        res_index = surface_residues.index(residue)
        hydrophobicity_sum += res_hydrophobicity[res_index]
        n_atom += 1
    return hydrophobicity_sum / n_atom


# Calculate the mean cx value for a given domain
def calculate_cx_by_domain(domain):
    cx_sum = 0
    n_atom = 0
    domain_atoms = get_domain_atoms(domain)
    for atom in domain_atoms:
        if atom.element != "H":
            cx_sum += atom.get_bfactor()
            n_atom += 1
    return cx_sum / n_atom


# Check if domain is a patch, get domain atoms
def get_domain_atoms(domain):
    if not isinstance(domain, list):
        return domain.get_atoms()
    domain_atoms = []
    for residue in domain:
        domain_atoms.extend(iter(residue.get_list()))
    return domain_atoms


# get the solvent vector of a residue
def get_sv(residue, chain):
    # Get the central residue's location
    cr = residue["CA"].coord

    # Find the 10 nearest neighbors of the central residue
    ns = NeighborSearch(list(chain.get_atoms()))
    near = ns.search(cr, 10, level="A")  # radius = 10 Angstroms
    near = [atom for atom in near if atom.id == "CA" and atom.get_parent() != residue]  # only get CA atoms of residues
    near.sort(key=lambda a: a - cr)  # sort near atoms by distance from central residue
    nn = near[:10]  # get the nearest 10

    # find the center of geometry of the neighbors
    cg_x = sum(a.coord[0] * a.mass for a in nn) / sum(a.mass for a in nn)
    cg_y = sum(a.coord[1] * a.mass for a in nn) / sum(a.mass for a in nn)
    cg_z = sum(a.coord[2] * a.mass for a in nn) / sum(a.mass for a in nn)
    cg = Vector(cg_x, cg_y, cg_z)
    cr = Vector(cr)

    # Vector from the central residue to the center of geometry
    vi = cr - cg

    # Solvent vector (vs) is the inverse of vi
    return -vi


# Check if two residues are surface neighbors
def check_neighbor(r1, r2, chain, radius):
    ca1 = r1["CA"]
    ca2 = r2["CA"]
    distance = ca1 - ca2
    if distance < radius:
        sv1 = get_sv(r1, chain)
        sv2 = get_sv(r2, chain)
        # Check if the angle between the solvent vectors is less than 110 degrees
        if sv1.angle(sv2) < radians(110):
            return True
    return False


def get_patches_from_residues(residues_by_chain, patch_type):
    sr = ShrakeRupley()
    res_patches = {}
    patch_index = 1
    for chain in residues_by_chain:
        # calculate ASA before complexation
        sr.compute(chain, level="R")
        chain_id = chain.id
        res_patches[chain] = []
        potential_neighbors = residues_by_chain[chain]
        for i in range(0, len(potential_neighbors), 3):
            this_res = potential_neighbors[i]
            this_patch = [this_res]
            for other in potential_neighbors:
                if other != this_res and check_neighbor(this_res, other, chain, patch_radius):
                    this_patch.append(other)
            res_patches[chain].append(this_patch)

            # set global patch info
            if patch_type == "Surface":
                surface_patches.append(this_patch)
            elif patch_type == "Interface":
                interface_patches.append(this_patch)
            all_patches.append(this_patch)
            patch_chains.append(chain_id)
            patch_nums.append(patch_index)
            patch_types.append(patch_type)
            patch_cx.append(calculate_cx_by_domain(this_patch))
            patch_asa.append(calculate_asa_by_patch(this_patch))
            patch_hydrophobicity.append(calculate_hydrophobicity_by_patch(this_patch))
            patch_planarity.append(calculate_planarity_by_patch(this_patch))
            this_patch_roughness = calculate_roughness_by_patch(this_patch)
            patch_roughness.append(this_patch_roughness)

            patch_index += 1
        patch_index = 1
    return res_patches


# Categorize patches of a given structure
def categorize_patches(structure):
    # Get patches from surface & interface residues by chain
    get_patches_from_residues(surface_residues_by_chain, "Surface")
    get_patches_from_residues(interface_residues_by_chain, "Interface")


def res_to_df():
    res_df = pd.DataFrame({
        'Chain': res_chains,
        'Number': res_nums,
        'Type': res_types,
        'CX': res_cx,
        'rASA': res_rasa,
        'Surface': res_surface,
        'Hydrophobicity': res_hydrophobicity
    })
    return res_df.sort_values(by=['Chain', 'Number'])


def patches_to_df():
    patches_df = pd.DataFrame({
        'Chain': patch_chains,
        'Number': patch_nums,
        'Type': patch_types,
        'CX': patch_cx,
        'ASA': patch_asa,
        'Hydrophobicity': patch_hydrophobicity,
        'Planarity': patch_planarity,
        'Roughness': patch_roughness
    })
    return patches_df.sort_values(by=['Chain', 'Type', 'Number'])


def get_interface_res(structure):
    # Get ASA before complexation
    asa_before = {}
    asa_after = {}
    for chain in surface_residues_by_chain:
        asa_before[chain] = []
        for residue in surface_residues_by_chain[chain]:
            asa_before[chain].append(residue.sasa)

    # Get ASA after complexation
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    for chain in surface_residues_by_chain:
        asa_after[chain] = []
        for residue in surface_residues_by_chain[chain]:
            asa_after[chain].append(residue.sasa)

    # Find interface residues by ASA difference
    for chain in surface_residues_by_chain:
        for i in range(len(surface_residues_by_chain[chain])):
            asa_diff = asa_after[chain][i] - asa_before[chain][i]
            if asa_diff < -1:
                interface_residues_by_chain[chain].append(surface_residues_by_chain[chain][i])


def categorize_residues(structure):
    asa_before_by_chain = {}
    for chain in structure.get_chains():
        if chain not in surface_residues_by_chain:
            surface_residues_by_chain[chain] = []
            interface_residues_by_chain[chain] = []
        chain_id = chain.id

        # get ASA before complexation
        sr = ShrakeRupley()
        sr.compute(chain, level="R")
        asa_before_by_chain[chain_id] = [residue.sasa for residue in chain.get_residues()]
        chain_res = list(chain.get_residues())
        for i in range(len(chain_res)):
            residue = chain_res[i]
            residue_num = residue.get_full_id()[3][1]
            residue_type = residue.get_resname()
            residue_rasa = asa_before_by_chain[chain_id][i] / residue_max_asas[residue_type]
            residue_cx = calculate_cx_by_domain(residue)
            residue_hydrophobicity = residue_hydrophobicities[residue_type]

            # get surface residues
            residue_surface = residue_rasa >= 0.25
            # set global residue info if surface residue
            surface_residues.append(residue)
            surface_residues_by_chain[chain].append(residue)
            res_chains.append(chain_id)
            res_nums.append(residue_num)
            res_types.append(residue_type)
            res_cx.append(residue_cx)
            res_rasa.append(residue_rasa)
            res_surface.append(residue_surface)
            res_hydrophobicity.append(residue_hydrophobicity)
    get_interface_res(structure)


def set_as_bfactor(structure, values):
    for i in range(len(all_patches)):
        patch = all_patches[i]
        for residue in patch:
            for atom in residue.get_list():
                if atom.element != "H":
                    atom.set_bfactor(values[i])
    return structure


def set_ip_as_bfactor(structure):
    for atom in structure.get_atoms():
        if atom.element != "H":
            atom.set_bfactor(0)
    for patch in interface_patches:
        for residue in patch:
            for atom in residue.get_list():
                if atom.element != "H":
                    atom.set_bfactor(1)
    return structure


def set_hydrophobicity_as_bfactor(structure):
    for i in range(len(all_patches)):
        patch = all_patches[i]
        h = patch_hydrophobicity[i]
        for residue in patch:
            for atom in residue.get_list():
                if atom.element != "H":
                    atom.set_bfactor(h)
    return structure


def set_cx_as_bfactor(structure):
    for chain in structure.get_chains():
        ns = NeighborSearch(list(chain.get_atoms()))
        for residue in chain.get_list():
            residue_name = residue.get_resname()
            for atom in residue.get_list():
                v_int = 0
                if atom.element != "H":
                    neighbors = ns.search(atom.get_coord(), cx_radius, level='A')
                    for neighbor in neighbors:
                        try:
                            v_int += standard_volumes_of_atoms[f"{residue_name} {neighbor.id}"]
                        except KeyError:
                            v_int += v_atom
                v_ext = v_sphere - v_int
                cx_val = np.clip(v_ext / v_int, 0, 15)
                atom.set_bfactor(cx_val)
    return structure


def save_structure_to_pdb(structure, out_dir, file_name):
    io = PDBIO()
    io.set_structure(structure)
    io.save(f"{out_dir}/{file_name}.pdb")


def get_protein_from_pdb(pdb_file):
    protein_id = pdb_file[3:-4]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_id, pdb_file)
    return protein_id, structure


def clear_global_variables():
    global surface_residues, res_chains, res_nums, res_types, res_cx, res_rasa, res_surface, res_hydrophobicity
    global all_patches, surface_patches, interface_patches, patch_chains, patch_nums, patch_types, patch_cx, patch_asa, patch_hydrophobicity, patch_planarity, patch_roughness
    surface_residues, res_chains, res_nums, res_types, res_cx, res_rasa, res_surface, res_hydrophobicity = [], [], [], [], [], [], [], []
    all_patches, surface_patches, interface_patches, patch_chains, patch_nums, patch_types, patch_cx, patch_asa, patch_hydrophobicity, patch_planarity, patch_roughness = [], [], [], [], [], [], [], [], [], [], []
    global surface_residues_by_chain, interface_residues_by_chain
    surface_residues_by_chain, interface_residues_by_chain = {}, {}


def get_all_bfactors(structure):
    return [
        atom.get_bfactor()
        for atom in structure.get_atoms()
        if atom.element != "H"
    ]


def main():  # sourcery skip: sum-comprehension
    for file in os.listdir("in"):
        if file.endswith(".pdb"):
            # get pdb file
            pdb_file = f"in/{file}"

            # clean pdb file
            clean(pdb_file)

            # get protein id and structure
            protein_id, structure = get_protein_from_pdb(pdb_file)

            # create output directory
            out_dir = f"out/{protein_id}"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            # get next chain id
            global next_chain_id
            for chain_id in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if not structure[0].has_id(chain_id):
                    next_chain_id = chain_id
                    break

            # start timer
            start_time = time.time()

            # categorize residues as surface & interface, remove internal residues
            categorize_residues(structure)

            # set cx as b-factor & save structure to pdb
            save_structure_to_pdb(set_cx_as_bfactor(structure), out_dir, f"{protein_id}_cx")

            # categorize patches as interface & surface
            categorize_patches(structure)

            # set interface patches as b-factor & save structure to pdb
            save_structure_to_pdb(set_ip_as_bfactor(structure), out_dir, f"{protein_id}_ip")

            bfactors_old = get_all_bfactors(structure)

            # set hydrophobicities of patches as b-factor & save structure to pdb
            save_structure_to_pdb(set_hydrophobicity_as_bfactor(structure), out_dir, f"{protein_id}_hydrophobicity")

            bfactors_new = get_all_bfactors(structure)

            # set planarity & roughness values of patches as b-factor & save structure to pdb
            save_structure_to_pdb(set_as_bfactor(structure, patch_planarity), out_dir, f"{protein_id}_planarity")
            save_structure_to_pdb(set_as_bfactor(structure, patch_roughness), out_dir, f"{protein_id}_roughness")

            # save global residue info to csv
            res_to_df().to_csv(f"{out_dir}/{protein_id}_residues.csv", index=False)

            # save global patch info to csv
            patches_to_df().to_csv(f"{out_dir}/{protein_id}_patches.csv", index=False)

            # get time of execution & print
            end_time = time.time()
            print(f"{protein_id} - processed in: {round(end_time - start_time, 2)} seconds")

            # print the total number of residues in structure and residues in patches
            print(f"#surface residues in structure: {len(surface_residues)}")
            res = sum(len(patch) for patch in surface_patches)
            print(f"#residues in surface patches: {res}")

            # compare old and new b-factors
            same = 0
            for i in range(len(bfactors_old)):
                if bfactors_old[i] == bfactors_new[i]:
                    same += 1
            print(f"atoms not present in patches: {same}\n")

            # clear global variables for next iteration
            clear_global_variables()


if __name__ == "__main__":
    main()
