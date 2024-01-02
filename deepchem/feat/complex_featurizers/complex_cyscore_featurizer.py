import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile
from scipy.spatial import distance

from deepchem.feat.base_classes import ComplexFeaturizer

logger = logging.getLogger(__name__)

def standardize(features):
    v_min = np.array([-4.872, -6.564, -0.05943, 0.0, -10.86, ])
    v_max = np.array([1.337, 15.1, 0.0, 0.294, 14.01, ])
    v_mean = np.array([-1.149, -1.749, -0.00425, 0.1823, -2.72, ])
    v_std = np.array([0.8394, 1.6, 0.00806, 0.102, 1.979, ])
    features = np.max([np.min([v_max, features], axis=0), v_min], axis=0)
    features = (features - v_mean)/v_std
    return features

class ComplexCyscoreFeaturizer(ComplexFeaturizer):
    """
    Calculate Xscore features for a complex. PDB file for receptor and MOL2 file for ligand must be provided.
    """

    def __init__(
        self,
        override_cyscore_executable_path = None,
    ):
        """
        Parameters
        ----------
        only_atom_type: bool, default False
          Whether to use only one-hot-encoded atom type as feature vector
        """
        if override_cyscore_executable_path is None:
            self.cyscore_executable_path = os.path.join(os.path.dirname(__file__), "bin", "Cyscore")
        else:
            self.cyscore_executable_path = override_cyscore_executable_path
        super().__init__()


    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """
        A tuple of paths to the receptor and ligand structure files must be provided.
        """
        receptor_structure_file_path = datapoint[0]
        ligand_structure_file_path = datapoint[1]
        features = np.array([])
        if os.path.exists(receptor_structure_file_path) and os.path.exists(ligand_structure_file_path):
            ligand_extension = ligand_structure_file_path.split(".")[-1]
            if ligand_extension not in ["mol2", "sdf"]:
                raise ValueError("Ligand file extension must be mol2 or sdf")
            if ligand_extension == "sdf":
                sdf_file = ligand_structure_file_path
                ligand_structure_file_path = ligand_structure_file_path.replace(".sdf", ".mol2")
                if not os.path.exists(ligand_structure_file_path):
                    subprocess.run(f"obabel {sdf_file} -O {ligand_structure_file_path}", shell=True)
            
            receptor_extension = receptor_structure_file_path.split(".")[-1]
            if receptor_extension not in ["pdb"]:
                raise ValueError("Receptor file extension must be pdb")
            command = f"{self.cyscore_executable_path} {receptor_structure_file_path} {ligand_structure_file_path}" 
            try:
                output = subprocess.run(command,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    shell=True)
                
                scores_text = output.stdout.decode('utf-8')
                lines = scores_text.split("\n")[-5:-3]
                terms = lines[0].split()
                scores = lines[1].split()
                if len(terms) == 8 and len(scores) == 2:
                    features = np.array(terms[1::2] + scores[1:2]).astype(float)
                    features = standardize(features)
                elif "Error: The two molecules are not docked!" in scores_text:
                    features = np.zeros(5, dtype=float)
                else:
                    error = output.stderr.decode('utf-8')
                    raise ValueError(f"Could not parse Cyscore features from {scores_text}. {error}")
            except Exception as e:
                msg = f"Error in xscore command {e}. {receptor_structure_file_path} {ligand_structure_file_path}. {e}"
                raise ValueError(msg)
        else:
            raise ValueError(f"Could not find {receptor_structure_file_path} or {ligand_structure_file_path}")
        return features