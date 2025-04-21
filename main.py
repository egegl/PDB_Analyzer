import os
import time
import numpy as np
import pandas as pd
from math import sqrt, pi, radians, log10
from numpy import mean
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from Bio.PDB import PDBParser, PDBIO, NeighborSearch
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.vectors import Vector
from cleaner import clean

# CX constants
cx_radius = 10  # Angstrom
v_atom = 20.1  # Angstrom^3
v_sphere = (4 / 3) * pi * cx_radius ** 3  # Angstrom^3

# roughness constants
r_min_radius = 0.2  # Angstrom
r_radius_step = 0.1  # Angstrom
r_max_radius = 2.0 + r_radius_step  # Angstrom
radii = np.arange(r_min_radius, r_max_radius, r_radius_step)
num_radii = len(radii)
res_radii_dict = {}

# patch constants
patch_radius = 11  # Angstrom (9 was a bit too small for certain proteins)
central_residue_step = 3

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
    'ALA N': 14.4, 'ALA CA': 12.4, 'ALA C': 8.6, 'ALA O': 22.8, 'ALA CB': 32.7, 'ALA OXT': 18.1,
    'ARG N': 14.0, 'ARG CA': 11.7, 'ARG C': 8.4, 'ARG O': 22.2, 'ARG CB': 20.4, 'ARG CG': 20.9, 'ARG CD': 20.3, 'ARG NE': 16.3, 'ARG CZ': 9.4, 'ARG NH1': 23.1, 'ARG NH2': 24.1,
    'ASN N': 13.8, 'ASN CA': 11.2, 'ASN C': 8.5, 'ASN O': 22.0, 'ASN CB': 20.2, 'ASN CG': 9.2, 'ASN OD1': 21.7, 'ASN ND2': 25.5,
    'ASP N': 13.9, 'ASP CA': 11.3, 'ASP C': 8.3, 'ASP O': 22.2, 'ASP CB': 20.8, 'ASP CG': 9.1, 'ASP OD1': 20.7, 'ASP OD2': 21.6,
    'CYS N': 14.3, 'CYS CA': 11.8, 'CYS C': 8.5, 'CYS O': 22.8, 'CYS CB': 22.8, 'CYS SG': 34.2,
    'GLN N': 13.9, 'GLN CA': 11.5, 'GLN C': 8.4, 'GLN O': 22.0, 'GLN CB': 20.0, 'GLN CG': 20.4, 'GLN CD': 9.6, 'GLN OE1': 23.4, 'GLN NE2': 24.6,
    'GLU N': 14.0, 'GLU CA': 11.7, 'GLU C': 8.4, 'GLU O': 22.1, 'GLU CB': 20.3, 'GLU CG': 21.1, 'GLU CD': 9.1, 'GLU OE1': 22.8, 'GLU OE2': 23.5,
    'GLY N': 14.9, 'GLY CA': 20.0, 'GLY C': 9.3, 'GLY O': 22.4, 'GLY OXT': 26.8,
    'HIS N': 13.8, 'HIS CA': 11.4, 'HIS C': 8.4, 'HIS O': 21.6, 'HIS CB': 20.5, 'HIS CG': 10.0, 'HIS ND1': 16.5, 'HIS CD2': 19.7, 'HIS CE1': 19.2, 'HIS NE2': 17.8,
    'ILE N': 14.1, 'ILE CA': 11.4, 'ILE C': 8.2, 'ILE O': 22.5, 'ILE CB': 13.0, 'ILE CG1': 22.4, 'ILE CG2': 33.2, 'ILE CD1': 35.3,
    'LEU N': 14.1, 'LEU CA': 11.6, 'LEU C': 8.5, 'LEU O': 22.3, 'LEU CB': 20.8, 'LEU CG': 13.7, 'LEU CD1': 35.2, 'LEU CD2': 35.0,
    'LYS N': 14.0, 'LYS CA': 11.6, 'LYS C': 8.5, 'LYS O': 22.3, 'LYS CB': 20.6, 'LYS CG': 21.0, 'LYS CD': 21.7, 'LYS CE': 21.8, 'LYS NZ': 23.0,
    'MET N': 14.0, 'MET CA': 11.7, 'MET C': 8.5, 'MET O': 22.6, 'MET CB': 21.0, 'MET CG': 23.0, 'MET SD': 27.8, 'MET CE': 34.8,
    'PHE N': 14.0, 'PHE CA': 11.6, 'PHE C': 8.4, 'PHE O': 22.5, 'PHE CB': 21.1, 'PHE CG': 10.2, 'PHE CD1': 20.3, 'PHE CD2': 20.8, 'PHE CE1': 22.0, 'PHE CE2': 22.2, 'PHE CZ': 22.0, 'PHE OXT': 19.3,
    'PRO N': 9.4, 'PRO CA': 12.1, 'PRO C': 8.4, 'PRO O': 22.8, 'PRO CB': 23.0, 'PRO CG': 24.1, 'PRO CD': 20.7,
    'SER N': 14.3, 'SER CA': 11.6, 'SER C': 8.4, 'SER O': 22.1, 'SER CB': 20.9, 'SER OG': 23.4,
    'THR N': 14.0, 'THR CA': 11.3, 'THR C': 8.3, 'THR O': 21.9, 'THR CB': 12.9, 'THR OG1': 23.1, 'THR CG2': 31.8,
    'TRP N': 14.3, 'TRP CA': 11.8, 'TRP C': 8.4, 'TRP O': 22.1, 'TRP CB': 21.5, 'TRP CG': 10.4, 'TRP CD1': 20.1, 'TRP CD2': 10.8, 'TRP NE1': 18.3, 'TRP CE2': 10.2, 'TRP CE3': 20.8, 'TRP CZ2': 20.9, 'TRP CZ3': 22.1, 'TRP CH2': 21.4,
    'TYR N': 13.9, 'TYR CA': 11.5, 'TYR C': 8.5, 'TYR O': 22.1, 'TYR CB': 21.3, 'TYR CG': 10.2, 'TYR CD1': 20.0, 'TYR CD2': 20.1, 'TYR CE1': 20.3, 'TYR CE2': 20.3, 'TYR CZ': 10.0, 'TYR OH': 25.1,
    'VAL N': 14.1, 'VAL CA': 11.4, 'VAL C': 8.4, 'VAL O': 22.6, 'VAL CB': 13.3, 'VAL CG1': 33.5, 'VAL CG2': 33.4,
}

# global lists by patch
all_patches = []
surface_patches_by_chain = {}
surface_patches_residues_by_chain = {}
interface_patches_by_chain = {}
interface_patches_residues_by_chain = {}

# global lists by residue
all_residues = []
surface_residues_by_chain = {}
interface_residues_by_chain = {}
interior_residues_by_chain = {}


def calculate_chain_asa_by_radii(chain):
    res_radii_dict.clear()

    for residue in chain.get_residues():
        res_radii_dict[residue] = []

    for radius in radii:
        sr = ShrakeRupley(probe_radius=radius)
        sr.compute(chain, level='R')
        for residue in chain.get_residues():
            res_radii_dict[residue].append(residue.sasa)


def calculate_roughness_by_patch(patch, patch_type):
    # calculate the surface area of patch for each radius (0.2-4.0 Angstrom)
    surface_areas = []
    for i in range(num_radii):
        patch_area = 0
        for residue in patch:
            patch_area += res_radii_dict[residue.residue][i]
        surface_areas.append(patch_area)

    # calculate the log of surface areas and radii
    log_radii = [log10(radius) for radius in radii]  # x-values
    log_surface_areas = [log10(area) for area in surface_areas]  # y-values

    surface_areas_divided = [area/1000 for area in surface_areas]

    # plot the log of surface areas vs the log of radii
    if patch_type == "Surface":
        plt.plot(radii, surface_areas_divided, color="blue")
    elif patch_type == "Interface":
        plt.plot(radii, surface_areas_divided, color="red")

    # calculate the slope of graph & roughness
    slope = np.polyfit(log_radii, log_surface_areas, 1)[0]
    roughness = 2 - slope
    return roughness


def calculate_planarity_by_patch(patch):
    # get the coordinates of the atoms in patch
    coords = []
    for residue in patch:
        coords.extend(atom.coord for atom in residue.residue if atom.element != "H")
    coords = np.array(coords)

    # center the coordinates by subtracting the mean
    coords = coords - coords.mean(axis=0)

    # calculate the PCA of the centered coordinates
    pca = PCA(n_components=3)  # x, y, z
    pca.fit(coords)

    # the first two principal components define the plane of best fit, so the normal vector is their cross product
    normal = np.cross(pca.components_[0], pca.components_[1])  # x, y

    # make normal a unit vector
    normal /= np.linalg.norm(normal)

    # calculate the distance of each point from the plane
    distances = np.abs(np.dot(coords, normal))

    # calculate the root mean squared distance & return planarity (higher = more planar/flat)
    rmsd = sqrt(mean(distances ** 2))
    return 1 / rmsd


def calculate_asa_by_patch(patch):
    # Calculate mean ASA for the domain
    asa_sum = 0
    for residue in patch:
        asa_sum += residue.residue.sasa
    return asa_sum / len(patch)


def calculate_hydrophobicity_by_patch(patch):
    hydrophobicity_sum = 0
    for residue in patch:
        hydrophobicity_sum += residue.hydrophobicity
    return hydrophobicity_sum / len(patch)


def calculate_cx_by_patch(patch):
    cx_sum = 0
    n_residue = 0
    for residue in patch:
        cx_sum += residue.cx
        n_residue += 1
    return cx_sum / n_residue


def calculate_residue_cx(residue):
    cx_sum = 0
    n_atom = 0
    for atom in residue:
        if atom.element != "H":
            cx_sum += atom.get_bfactor()
            n_atom += 1
    return cx_sum / n_atom


# get the solvent vector of a residue
def get_sv(residue, chain):
    # Get the central residue's location
    cr = residue["CA"]
    cr_coord = cr.coord

    # Find the 10 nearest neighbors of the central residue
    ns = NeighborSearch(list(chain.get_atoms()))
    near = ns.search(cr_coord, 6, level="R")
    for residue in near:
        near[near.index(residue)] = residue["CA"]
    near.sort(key=lambda a: a - cr)  # sort near atoms by distance from central residue
    nn = near[:10]  # get the nearest 10

    # find the center of geometry of the neighbors
    cg_x, cg_y, cg_z = 0, 0, 0
    for atom in nn:
        x, y, z = atom.coord
        cg_x += x
        cg_y += y
        cg_z += z
    cg_x /= 10
    cg_y /= 10
    cg_z /= 10
    cg = Vector(cg_x, cg_y, cg_z)

    cr = Vector(cr_coord)

    # get the vector from the central residue to the center of geometry
    vi = cr - cg

    # Solvent vector (vs) is the inverse of vi
    return -vi


# Check if two residues are surface neighbors
def check_neighbor(r1, r2, chain):
    # get the solvent vectors of the residues & calculate the angle between them
    sv1 = get_sv(r1, chain)
    sv2 = get_sv(r2, chain)
    solvent_angle = sv1.angle(sv2)

    # return true if the angle between the solvent vectors is less than 110 degrees
    return solvent_angle < radians(110)


def get_patches_from_residues(residues_by_chain, patch_type):
    sr = ShrakeRupley()
    patch_index = 1
    for chain in residues_by_chain:
        # create new patch lists for this chain
        if chain not in surface_patches_by_chain:
            surface_patches_by_chain[chain] = []
            interface_patches_by_chain[chain] = []

        potential_neighbors = residues_by_chain[chain]
        potential_neighbor_residues = []
        for residue in potential_neighbors:
            potential_neighbor_residues.append(residue.residue)

        # calculate ASA by radii for this chain (for roughness)
        calculate_chain_asa_by_radii(chain)

        # calculate ASA before complexation
        sr.compute(chain, level="R")
        chain_id = chain.id

        potential_neighbor_atoms = []
        for residue in potential_neighbor_residues:
            potential_neighbor_atoms.extend(list(residue.get_atoms()))
        ns = NeighborSearch(potential_neighbor_atoms)
        for i in range(0, len(potential_neighbors), central_residue_step):
            # add residues to this patch
            this_res = potential_neighbors[i]
            this_patch = [this_res]
            this_neighbors = ns.search(this_res.residue["CA"].coord, patch_radius, level="R")
            this_neighbors.remove(this_res.residue)
            for other in this_neighbors:
                if check_neighbor(this_res.residue, other, chain):
                    # find other as a Residue object in potential_neighbor_residues
                    for j in range(len(potential_neighbors)):
                        if other == potential_neighbors[j].residue:
                            this_patch.append(potential_neighbors[j])
                            break
            if len(this_patch) < 3:
                del this_patch
                i -= 1
                continue

            # create new patch object:
            patch = Patch(
                this_patch,
                chain_id,
                patch_index,
                patch_type,
                calculate_cx_by_patch(this_patch),
                calculate_asa_by_patch(this_patch),
                calculate_hydrophobicity_by_patch(this_patch),
                calculate_planarity_by_patch(this_patch),
                calculate_roughness_by_patch(this_patch, patch_type)
            )
            all_patches.append(patch)

            # categorize patch
            if patch_type == "Surface":
                surface_patches_by_chain[chain].append(patch)
            elif patch_type == "Interface":
                interface_patches_by_chain[chain].append(patch)

            patch_index += 1
        patch_index = 1


def purify_patch_residues(patch_list):
    patch_residues = []
    for patch in patch_list:
        for residue in patch.residues:
            if residue not in patch_residues:
                patch_residues.append(residue)
    return patch_residues


def categorize_patches():
    # Get patches from surface residues by chain
    get_patches_from_residues(surface_residues_by_chain, "Surface")
    for chain in surface_patches_by_chain:
        surface_patches_residues_by_chain[chain] = []
        for residue in purify_patch_residues(surface_patches_by_chain[chain]):
            surface_patches_residues_by_chain[chain].append(residue)

    # Get patches from interface residues by chain
    get_patches_from_residues(interface_residues_by_chain, "Interface")
    for chain in interface_patches_by_chain:
        interface_patches_residues_by_chain[chain] = []
        for residue in purify_patch_residues(interface_patches_by_chain[chain]):
            interface_patches_residues_by_chain[chain].append(residue)


def res_to_df(protein_id, out_dir):
    res_chains = []
    res_nums = []
    res_types = []
    res_cx = []
    res_rasa = []
    res_interface = []
    res_surface = []
    res_hydrophobicity = []
    for residue in all_residues:
        res_chains.append(residue.chain_id)
        res_nums.append(residue.number)
        res_types.append(residue.name)
        res_cx.append(residue.cx)
        res_rasa.append(residue.rasa)
        res_interface.append(residue.interface)
        res_surface.append(residue.surface)
        res_hydrophobicity.append(residue.hydrophobicity)

    res_df = pd.DataFrame({
        'Chain': res_chains,
        'Number': res_nums,
        'Type': res_types,
        'CX': res_cx,
        'RASA': res_rasa,
        'Interface': res_interface,
        'Surface (>%25 rASA)': res_surface,
        'Hydrophobicity': res_hydrophobicity
    })
    res_df.sort_values(by=['Chain', 'Number']).to_csv(f"{out_dir}/{protein_id}_residues.csv", index=False)


def save_to_scatter_plot(x_values_s, y_values_s, x_values_i, y_values_i, out_dir, file_name, x_label, y_label, title):
    plt.scatter(x_values_s, y_values_s, label="Surface")
    plt.scatter(x_values_i, y_values_i, label="Interface")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()

    plt.savefig(f"{out_dir}/{file_name}.png")
    plt.clf()


def save_to_bar_chart(out_dir, file_name, x_label, y_label, title, x_values, y_values):
    plt.bar(x_values, y_values)

    # limit the y-axis if roughness
    if y_label == "Roughness":
        axes = plt.gca()
        axes.set_ylim([2, 2.8])

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.savefig(f"{out_dir}/{file_name}.png")
    plt.clf()


def patches_to_data(structure, protein_id, out_dir):
    # get patch values
    patch_chains = []
    patch_nums = []
    patch_types = []
    patch_cx = []
    patch_asa = []
    patch_hydrophobicity = []
    patch_planarity = []
    patch_roughness = []
    for patch in all_patches:
        patch_chains.append(patch.chain_id)
        patch_nums.append(patch.number)
        patch_types.append(patch.patch_type)
        patch_cx.append(patch.cx)
        patch_asa.append(patch.asa)
        patch_hydrophobicity.append(patch.hydrophobicity)
        patch_planarity.append(patch.planarity)
        patch_roughness.append(patch.roughness)

    # save patch values to pdb
    save_structure_to_pdb(set_as_bfactor(structure, patch_hydrophobicity), out_dir, f"{protein_id}_hydrophobicity")
    save_structure_to_pdb(set_as_bfactor(structure, patch_planarity), out_dir, f"{protein_id}_planarity")
    save_structure_to_pdb(set_as_bfactor(structure, patch_roughness), out_dir, f"{protein_id}_roughness")

    # save patch values to csv
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
    patches_df.sort_values(by=['Chain', 'Type', 'Number']).to_csv(f"{out_dir}/{protein_id}_patches.csv", index=False)


def remove_interior_res(structure):
    for chain in interior_residues_by_chain:
        for residue in interior_residues_by_chain[chain]:
            structure[0][chain.id].detach_child(residue.residue.id)
    return structure


def categorize_residues(structure):
    # Get ASA after complexation
    asa_after = {}
    sr = ShrakeRupley()
    sr.compute(structure, level="R")
    for chain in structure.get_chains():
        asa_after[chain] = []
        for residue in chain.get_residues():
            asa_after[chain].append(residue.sasa)

    # categorize residues
    for chain in structure.get_chains():
        if chain not in surface_residues_by_chain:
            surface_residues_by_chain[chain] = []
            interface_residues_by_chain[chain] = []
            interior_residues_by_chain[chain] = []
        chain_id = chain.id

        # get ASA before complexation
        asa_before = {}
        sr.compute(chain, level="R")
        asa_before[chain] = []
        for residue in chain.get_residues():
            asa_before[chain].append(residue.sasa)

        # get chain residues
        chain_res = list(chain.get_residues())
        for i in range(len(chain_res)):
            # get residue info
            residue = chain_res[i]
            residue_num = residue.get_full_id()[3][1]
            residue_type = residue.get_resname()
            residue_rasa = residue.sasa / residue_max_asas[residue_type]
            asa_diff = asa_after[chain][i] - asa_before[chain][i]
            residue_cx = calculate_residue_cx(residue)
            residue_hydrophobicity = residue_hydrophobicities[residue_type]
            residue_surface = residue_rasa >= 0.25
            residue_interface = asa_diff < -1

            # create residue object & add to global list
            res = Residue(
                residue,
                residue_type,
                chain_id,
                residue_num,
                residue_cx,
                residue_rasa,
                residue_interface,
                residue_surface,
                residue_hydrophobicity
            )
            all_residues.append(res)

            # categorize residue
            if residue_interface:
                interface_residues_by_chain[chain].append(res)
            elif residue_surface:
                surface_residues_by_chain[chain].append(res)
            else:
                interior_residues_by_chain[chain].append(res)


def set_as_bfactor(structure, values):
    for i in range(len(all_patches)):
        patch = all_patches[i]
        for residue in patch.residues:
            for atom in residue.residue:
                if atom.element != "H":
                    atom.set_bfactor(values[i])
    return structure


def remove_rogue_residues(structure):
    for i in range(len(all_patches)):
        patch = all_patches[i]
        for residue in patch.residues:
            residue.residue["CA"].set_bfactor(2)
    rogue_residues = 0
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            bfactor = residue["CA"].get_bfactor()
            if bfactor != 2 or bfactor == 1:
                chain.detach_child(residue.id)
                rogue_residues += 1
    print(f"Removed {rogue_residues} rogue residues")
    return structure


def set_ip_as_bfactor(structure):
    for atom in structure.get_atoms():
        if atom.element != "H":
            atom.set_bfactor(1)
    for chain in interface_patches_by_chain:
        for patch in interface_patches_by_chain[chain]:
            for residue in patch.residues:
                for atom in residue.residue:
                    if atom.element != "H":
                        atom.set_bfactor(0)
    return structure


def set_cx_as_bfactor(structure):
    for chain in structure.get_chains():
        ns = NeighborSearch(list(chain.get_atoms()))
        for residue in chain:
            residue_name = residue.get_resname()
            for atom in residue:
                v_int = 0
                if atom.element != "H":
                    neighbors = ns.search(atom.get_coord(), cx_radius, level='A')
                    for neighbor in neighbors:
                        try:
                            v_int += standard_volumes_of_atoms[f"{residue_name} {neighbor.id}"]
                        except KeyError:
                            v_int += v_atom
                if v_int == 0:
                    chain.detach_child(residue.id)
                    break
                v_ext = v_sphere - v_int
                cx_val = v_ext / v_int
                cx_val = 0 if cx_val < 0 else cx_val
                atom.set_bfactor(cx_val)
    return structure


def save_structure_to_pdb(structure, out_dir, file_name):
    io = PDBIO()
    io.set_structure(structure)
    io.save(f"{out_dir}/{file_name}.pdb")


def get_protein_from_pdb(pdb_file):
    protein_id = pdb_file.split("/")[-1][:4]
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(protein_id, pdb_file)
    return protein_id, structure


def clear_global_variables():
    global all_patches, all_residues
    all_patches, all_residues = [], []
    global surface_residues_by_chain, interface_residues_by_chain, interior_residues_by_chain, surface_patches_by_chain, interface_patches_by_chain, surface_patches_residues_by_chain, interface_patches_residues_by_chain
    surface_residues_by_chain, interface_residues_by_chain, interior_residues_by_chain, surface_patches_by_chain, interface_patches_by_chain, surface_patches_residues_by_chain, interface_patches_residues_by_chain = {}, {}, {}, {}, {}, {}, {}


def main():
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

            # start timer
            start_time = time.time()

            # set cx as b-factor & save structure to pdb
            save_structure_to_pdb(set_cx_as_bfactor(structure), out_dir, f"{protein_id}_cx")

            # categorize residues as surface, interface & non-surface
            categorize_residues(structure)

            # categorize patches as interface, surface, interior
            categorize_patches()

            # remove interior residues (rASA = 0)
            structure = remove_interior_res(structure)

            # plot the patches' roughness plot
            plt.title(f"{protein_id} - interface & surface patch surface area vs. probe radius")
            plt.xlabel("probe radius (A)")
            plt.xscale("log")
            plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0, 2.0], ["0.2", "0.4", "0.6", "0.8", "1.0", "2.0"])
            plt.ylabel("surface area (10^3 A^2)")
            plt.yscale("log")
            plt.yticks([0.4, 0.6, 0.8, 1.0, 2.0], ["0.4", "0.6", "0.8", "1.0", "2.0"])
            plt.savefig(f"out/{protein_id}/sa-radius.png")
            plt.clf()

            # set interface patches as b-factor & save structure to pdb
            save_structure_to_pdb(set_ip_as_bfactor(structure), out_dir, f"{protein_id}_ip")

            # remove rogue residues
            structure = remove_rogue_residues(structure)

            # save global residue info
            res_to_df(protein_id, out_dir)

            # save global patch info
            patches_to_data(structure, protein_id, out_dir)

            # get patch data to graph
            interface_roughness = []
            interface_planarity = []
            interface_asa = []
            interface_cx = []
            interface_hydrophobicity = []
            for patch in interface_patches_by_chain[next(iter(interface_patches_by_chain))]:
                interface_hydrophobicity.append(patch.hydrophobicity)
                interface_cx.append(patch.cx)
                interface_asa.append(patch.asa)
                interface_planarity.append(patch.planarity)
                interface_roughness.append(patch.roughness)
            surface_roughness = []
            surface_planarity = []
            surface_asa = []
            surface_cx = []
            surface_hydrophobicity = []
            for patch in surface_patches_by_chain[next(iter(surface_patches_by_chain))]:
                surface_hydrophobicity.append(patch.hydrophobicity)
                surface_cx.append(patch.cx)
                surface_asa.append(patch.asa)
                surface_planarity.append(patch.planarity)
                surface_roughness.append(patch.roughness)

            # graph patch features to bar chart
            save_to_bar_chart(out_dir, "roughness", "Patch Type", "Roughness", f"{protein_id} - Roughness", ["Surface", "Interface"], [mean(surface_roughness), mean(interface_roughness)])
            save_to_bar_chart(out_dir, "planarity", "Patch Type", "Planarity", f"{protein_id} - Planarity", ["Surface", "Interface"], [mean(surface_planarity), mean(interface_planarity)])
            save_to_bar_chart(out_dir, "asa", "Patch Type", "ASA", f"{protein_id} - ASA", ["Surface", "Interface"], [mean(surface_asa), mean(interface_asa)])
            save_to_bar_chart(out_dir, "cx", "Patch Type", "CX", f"{protein_id} - CX", ["Surface", "Interface"], [mean(surface_cx), mean(interface_cx)])
            save_to_bar_chart(out_dir, "hydrophobicity", "Patch Type", "Hydrophobicity", f"{protein_id} - Hydrophobicity", ["Surface", "Interface"], [mean(surface_hydrophobicity), mean(interface_hydrophobicity)])

            # get time of execution & print
            end_time = time.time()
            print(f"{protein_id} - processed in: {round(end_time - start_time, 2)} seconds")

            # print the total number of residues in structure and residues in patches
            surface_res = sum(len(res) for res in surface_residues_by_chain.values())
            interface_res = sum(len(res) for res in interface_residues_by_chain.values())
            print(f"#total surface:interface residues in structure: {surface_res}:{interface_res}")
            surface_patches_res = 0
            for chain in surface_patches_by_chain:
                surface_patches_res += sum(len(patch.residues) for patch in surface_patches_by_chain[chain])
            interface_patches_res = 0
            for chain in interface_patches_by_chain:
                interface_patches_res += sum(len(patch.residues) for patch in interface_patches_by_chain[chain])
            print(f"#total residues in surface:interface patches: {surface_patches_res}:{interface_patches_res}")

            # clear global variables for the next structure
            clear_global_variables()
            print()


class Residue:
    def __init__(self, residue, name, chain_id, number, cx, rasa, interface, surface, hydrophobicity):
        self.residue = residue
        self.name = name
        self.chain_id = chain_id
        self.number = number
        self.cx = cx
        self.rasa = rasa
        self.interface = interface
        self.surface = surface
        self.hydrophobicity = hydrophobicity


class SmallResidue:
    def __init__(self, residue, asa):
        self.residue = residue
        self.asa = asa


class Patch:
    def __init__(self, residues, chain_id, number, patch_type, cx, asa, hydrophobicity, planarity, roughness):
        self.residues = residues
        self.chain_id = chain_id
        self.number = number
        self.patch_type = patch_type
        self.cx = cx
        self.asa = asa
        self.hydrophobicity = hydrophobicity
        self.planarity = planarity
        self.roughness = roughness


if __name__ == "__main__":
    main()

