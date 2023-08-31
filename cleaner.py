import tempfile
import shutil

amino_acids = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def clean(file):
    input_file = open(file, "r")
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False)
    polypeptide_chains = set()

    for line in input_file:
        if not line.startswith("HETATM"):
            temp_file.write(line)
        else:
            chain = line[21]
            residue = line[17:20]
            if residue in amino_acids and chain in polypeptide_chains:
                temp_file.write(line)

    input_file.close()
    temp_file.close()
    shutil.move(temp_file.name, input_file.name)
