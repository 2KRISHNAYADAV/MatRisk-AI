"""
src.materials.composition
--------------------------
Composition-derived descriptors for DS1 formulas: element presence,
atomic-mass / atomic-number statistics, and the density identity
rho = n_sites * mean_atomic_mass * u / V_cell.

Atomic numbers and standard atomic weights below cover Z=1..92 (all
elements that appear in DS1 formulas and DS5 element prices), taken
from the standard periodic table (IUPAC standard atomic weights).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.audit import parse_elements

# symbol -> (atomic number Z, standard atomic weight, amu)
ATOMIC_DATA = {
    "H": (1, 1.008), "He": (2, 4.003), "Li": (3, 6.94), "Be": (4, 9.012),
    "B": (5, 10.81), "C": (6, 12.011), "N": (7, 14.007), "O": (8, 15.999),
    "F": (9, 18.998), "Ne": (10, 20.180), "Na": (11, 22.990), "Mg": (12, 24.305),
    "Al": (13, 26.982), "Si": (14, 28.085), "P": (15, 30.974), "S": (16, 32.06),
    "Cl": (17, 35.45), "Ar": (18, 39.948), "K": (19, 39.098), "Ca": (20, 40.078),
    "Sc": (21, 44.956), "Ti": (22, 47.867), "V": (23, 50.942), "Cr": (24, 51.996),
    "Mn": (25, 54.938), "Fe": (26, 55.845), "Co": (27, 58.933), "Ni": (28, 58.693),
    "Cu": (29, 63.546), "Zn": (30, 65.38), "Ga": (31, 69.723), "Ge": (32, 72.630),
    "As": (33, 74.922), "Se": (34, 78.971), "Br": (35, 79.904), "Kr": (36, 83.798),
    "Rb": (37, 85.468), "Sr": (38, 87.62), "Y": (39, 88.906), "Zr": (40, 91.224),
    "Nb": (41, 92.906), "Mo": (42, 95.95), "Tc": (43, 98.0), "Ru": (44, 101.07),
    "Rh": (45, 102.906), "Pd": (46, 106.42), "Ag": (47, 107.868), "Cd": (48, 112.414),
    "In": (49, 114.818), "Sn": (50, 118.710), "Sb": (51, 121.760), "Te": (52, 127.60),
    "I": (53, 126.904), "Xe": (54, 131.293), "Cs": (55, 132.905), "Ba": (56, 137.327),
    "La": (57, 138.905), "Ce": (58, 140.116), "Pr": (59, 140.908), "Nd": (60, 144.242),
    "Pm": (61, 145.0), "Sm": (62, 150.36), "Eu": (63, 151.964), "Gd": (64, 157.25),
    "Tb": (65, 158.925), "Dy": (66, 162.500), "Ho": (67, 164.930), "Er": (68, 167.259),
    "Tm": (69, 168.934), "Yb": (70, 173.045), "Lu": (71, 174.967), "Hf": (72, 178.49),
    "Ta": (73, 180.948), "W": (74, 183.84), "Re": (75, 186.207), "Os": (76, 190.23),
    "Ir": (77, 192.217), "Pt": (78, 195.085), "Au": (79, 196.967), "Hg": (80, 200.592),
    "Tl": (81, 204.38), "Pb": (82, 207.2), "Bi": (83, 208.980), "Po": (84, 209.0),
    "At": (85, 210.0), "Rn": (86, 222.0), "Fr": (87, 223.0), "Ra": (88, 226.0),
    "Ac": (89, 227.0), "Th": (90, 232.038), "Pa": (91, 231.036), "U": (92, 238.029),
}

AMU_TO_G = 1.66053906660e-24  # grams per atomic mass unit


def composition_table(formulas: pd.Series) -> pd.DataFrame:
    """
    Per-row composition descriptors parsed from the chemical formula:
    mean_atomic_mass, mean_Z, max_Z, min_Z, atoms_per_formula, n_elements_parsed.
    Rows with an unrecognised element raise no error; unknown symbols are
    skipped and flagged in `n_unparsed_elements`.
    """
    rows = []
    for f in formulas:
        comp = parse_elements(f)
        masses, Zs, unparsed = [], [], 0
        total_atoms = 0.0
        for el, cnt in comp.items():
            if el in ATOMIC_DATA:
                Z, m = ATOMIC_DATA[el]
                masses.extend([m] * 1)  # weighted below by count
                Zs.extend([Z] * 1)
                total_atoms += cnt
            else:
                unparsed += 1
        if comp:
            atoms_per_formula = sum(comp.values())
            mean_mass = sum(cnt * ATOMIC_DATA[el][1] for el, cnt in comp.items() if el in ATOMIC_DATA) / max(atoms_per_formula, 1e-9)
            mean_Z = sum(cnt * ATOMIC_DATA[el][0] for el, cnt in comp.items() if el in ATOMIC_DATA) / max(atoms_per_formula, 1e-9)
            Z_list = [ATOMIC_DATA[el][0] for el in comp if el in ATOMIC_DATA]
            max_Z = max(Z_list) if Z_list else np.nan
            min_Z = min(Z_list) if Z_list else np.nan
        else:
            atoms_per_formula = np.nan
            mean_mass = mean_Z = max_Z = min_Z = np.nan
        rows.append({
            "mean_atomic_mass": mean_mass,
            "mean_Z": mean_Z,
            "max_Z": max_Z,
            "min_Z": min_Z,
            "atoms_per_formula": atoms_per_formula,
            "n_elements_parsed": len(comp),
            "n_unparsed_elements": unparsed,
        })
    return pd.DataFrame(rows, index=formulas.index)


def density_from_cell(nsites, mean_atomic_mass, volume_A3):
    """rho [g/cm3] implied by nsites atoms of mean atomic mass [amu] in a cell of volume [A^3]."""
    nsites = np.asarray(nsites, dtype=float)
    mean_atomic_mass = np.asarray(mean_atomic_mass, dtype=float)
    volume_A3 = np.asarray(volume_A3, dtype=float)
    mass_g = nsites * mean_atomic_mass * AMU_TO_G
    volume_cm3 = volume_A3 * 1e-24
    with np.errstate(divide="ignore", invalid="ignore"):
        return mass_g / volume_cm3


def element_presence(formulas: pd.Series, min_count: int = 1) -> pd.DataFrame:
    """One boolean column per element that appears in >= min_count formulas."""
    parsed = [parse_elements(f) for f in formulas]
    counts: dict[str, int] = {}
    for comp in parsed:
        for el in comp:
            counts[el] = counts.get(el, 0) + 1
    keep = sorted(el for el, n in counts.items() if n >= min_count)
    data = {el: [el in comp for comp in parsed] for el in keep}
    return pd.DataFrame(data, index=formulas.index)
