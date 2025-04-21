# Protein Surface and Interface Patch Analysis Pipeline

## Overview
This repository presents an advanced computational framework for the rigorous analysis of protein surface topology, residue exposure, and interfacial patch characteristics from high‑resolution PDB structures. Combining state‑of‑the‑art molecular surface algorithms, geometric feature extraction, and machine learning, our pipeline delivers comprehensive descriptors for per‑residue and per‑patch properties, facilitating downstream statistical modeling and predictive tasks in structural bioinformatics.

Key capabilities:
- Automated cleaning of input PDB files (removal of non‑canonical ligands and HETATM records) via `cleaner.py`.
- Computation of **Convexity Index (CX)** per residue by volumetric sphere sampling.
- Shrake–Rupley solvent accessible surface area (SASA) profiling over variable probe radii (0.2–2.0 Å) for fractal **roughness** estimation.
- Principal Component Analysis (PCA)‑based **planarity** quantification of spatial residue patches.
- Classification of residues into **surface**, **interface**, and **interior** based on relative ASA thresholds and interface burial upon complexation.
- Spatial clustering of residues into **patches** (neighbors within 11 Å filtered by solvent‑vector angles < 110°) and computation of per‑patch descriptors: CX, mean ASA, hydrophobicity, planarity, and roughness.
- Generation of publication‑quality PDB outputs with per‑atom B‑factors encoding computed metrics and CSV summaries of residues and patches.
- A PyTorch‑based multilayer perceptron (MLP) for binary classification of surface vs. interface patches, trained on aggregated per‑patch feature sets.

## Repository Structure
```
├── cleaner.py                 # PDB cleaning utility
├── main.py                    # Core analysis pipeline
├── train_patch_classifier.py  # PyTorch MLP training script
├── README.md                  # This documentation
├── in/                        # Input PDB files
└── out/                       # Per‑protein output directories
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-org/protein‑patch‑pipeline.git
   cd protein‑patch‑pipeline
   ```
2. Create a Python 3.7+ virtual environment and install dependencies:
   ```bash
   pip install numpy pandas matplotlib scikit‑learn biopython torch
   ```

## Usage

### 1. Input Preparation
- Place target PDB files (with extension `.pdb`) into the `in/` directory.

### 2. Running the Analysis Pipeline
Execute the main script to process all inputs:
```bash
python main.py
```
This will:
- Clean each PDB (removing extraneous HETATM records).
- Compute residue‐level CX and write `{protein_id}_cx.pdb` in `out/{protein_id}/`.
- Classify residues (surface, interface, interior).
- Identify and characterize patches, generating:
  - `{protein_id}_patches.csv` (per‑patch descriptors)
  - `{protein_id}_residues.csv` (per‑residue descriptors)
  - PDBs with B‑factors set to patch hydrophobicity, planarity, roughness, and interface maps.
  - High‑resolution plots (`*.png`) of SASA vs. probe radius and bar‑charts comparing surface/interface patch metrics.

### 3. Training a Patch Classifier
Aggregate patch descriptors across proteins and train a binary classifier:
```bash
python train_patch_classifier.py --data-dir out --epochs 100 --batch-size 32 --lr 1e-3
```
Outputs:
- `patch_classifier.pt`: trained PyTorch model state.
- Console logs of training loss, accuracy per epoch, and final test accuracy.

## Methodology

### Structure Cleaning
Removes non‑canonical HETATM entries while preserving essential polypeptide residues, ensuring unbiased geometric analysis.

### Convexity Index (CX)
For each atom, internal volume `V_int` is summed over atom type–specific van der Waals volumes within a sphere (10 Å radius), with external volume `V_ext = V_sphere – V_int`. Residue CX is averaged atom‑level `V_ext / V_int`.

### SASA Profiling & Fractal Roughness
Shrake–Rupley algorithm computes residue SASA at probe radii from 0.2 to 2.0 Å. Log–log linear regression of surface area vs. radius yields a slope `m`, and roughness `D_f = 2 – m`, approximating the fractal dimension of the surface patch.

### Planarity via PCA
Atomic coordinates of a residue patch are mean‑centered and decomposed into principal components. The inverse root‑mean‑square deviation of points from the best‑fit plane (defined by the first two PCs) quantifies flatness.

### Residue & Patch Classification
- **Surface residues**: relative ASA ≥ 25%.
- **Interface residues**: ΔASA (complexed – isolated) ≤ −1 Å².
- **Patches**: central residues sampled every 3 positions, neighborhood radius 11 Å, filtered by solvent‑vector angular criteria (< 110°).

## Outputs
For each protein `{protein_id}` in `out/{protein_id}/`:
- `{protein_id}_cx.pdb` — CX in B‑factors
- `{protein_id}_ip.pdb` — interface map (B=0/1)
- `{protein_id}_hydrophobicity.pdb`, `_planarity.pdb`, `_roughness.pdb` — patches encoded in B‑factors
- `{protein_id}_residues.csv` — per‑residue features
- `{protein_id}_patches.csv` — per‑patch features
- Visualizations: `*.png` plots of SASA, roughness, planarity, CX, etc.

## Experimental Results
Preliminary benchmarking on a diverse set of membrane and soluble proteins demonstrates clear discriminative power of fractal roughness and CX in demarcating surface vs. interface patches, achieving >85% classification accuracy with a simple MLP.

## Future Directions
- Integrate Graph Neural Networks on residue‐contact graphs for end‑to‑end patch classification.
- Extend roughness analysis to anisotropic probe scanning.
- Correlate patch descriptors with functional annotation (e.g., protein–protein interaction hot spots).

## References
1. Shrake, A., & Rupley, J. A. (1973). Environment and exposure to solvent of protein atoms. _Journal of Molecular Biology_, 79(2), 351–371.
2. Jolliffe, I. T. (2002). Principal Component Analysis. _Springer Series in Statistics_.
3. Mandelbrot, B. B. (1983). The fractal geometry of nature. _W. H. Freeman_.
4. Kingma, D. P., & Ba, J. (2014). Adam: A Method for Stochastic Optimization. arXiv:1412.6980.