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

def execlude_features(features):
    features = features[:108] if len(features.shape) == 1 else features[:,108]
    invariant_indices = [15, 16, 19, 23, 27, 28, 29, 30, 31, 32, 33, 34, 35, 55, 64, 65, 66, 67, 68,
                         69, 70, 71, 100, 101, 102, 103, 104, 105, 106, 107]
    selected_features = np.delete(features, invariant_indices)
    return selected_features

def standardize(features):
    v_min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 46.0, 15.0, 10.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 165.67, 54.67, 57.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                      0.0, 0.0, 0.0, 0.0, 0.0,])
    v_max = np.array([128.0, 33.33, 45.0, 12.0, 37.0, 12.0, 19.0, 4.0, 53.0, 26.0, 21.0, 4.0, 
                      30.33, 7.0, 8.0, 8.0, 5.0, 9.0, 4.0, 4.0, 16.0, 4.0, 5.0, 2348.66, 553.0,
                      616.0, 85.0, 601.33, 160.0, 199.0, 19.0, 902.66, 278.0, 263.0, 22.0, 419.33,
                      93.33, 92.0, 17.0, 156.33, 56.0, 56.0, 140.0, 37.0, 35.0, 5.0, 204.0, 44.0,
                      47.0, 9.0, 9487.33, 2386.0, 2586.67, 196.33, 2496.0, 648.0, 761.33, 46.0,
                      3647.0, 1037.33, 1060.33, 50.0, 1485.0, 372.0, 408.0, 30.0, 586.67, 177.0,
                      183.0, 9.0, 502.67, 138.0, 140.33, 9.0, 804.0, 186.0, 189.0, 20.0, ])
    v_mean = np.array([25.87, 5.153, 9.911, 0.6877, 5.642, 1.13, 2.41, 0.1397, 9.977, 3.184, 3.391,
                       0.1483, 0.9669, 0.1608, 0.2133, 0.1823, 0.1197, 0.3607, 0.1097, 0.1398, 0.5023,
                       0.08085, 0.1194, 756.8, 180.75, 198.77, 10.84, 140.01, 34.04, 38.44, 1.99,
                       170.99, 45.72, 48.03, 1.927, 14.17, 3.294, 3.55, 0.2531, 6.16, 1.992, 1.984,
                       11.55, 2.852, 2.989, 0.1423, 7.628, 1.726, 1.912, 0.1262, 2912.07, 761.95,
                       820.41, 32.24, 539.67, 142.43, 154.52, 5.864, 657.95, 178.41, 189.06, 6.38,
                       51.38, 13.02, 13.89, 0.6764, 24.73, 7.237, 7.471, 0.2079, 44.38, 11.84,
                       12.46, 0.4304, 27.94, 6.975, 7.523, 0.3515, ])
    v_std = np.array([16.63, 4.352, 6.733, 1.542, 5.96, 1.641, 2.769, 0.4712, 8.636, 3.987, 3.594, 
                      0.466, 3.38, 0.6976, 0.8337, 0.796, 0.5213, 1.112, 0.4381, 0.4757, 1.825,
                      0.3988, 0.5047, 330.6, 77.41, 88.95, 13.47, 104.94, 26.73, 30.96, 3.035,
                      133.02, 41.03, 40.74, 2.876, 47.15, 10.88, 11.52, 1.316, 20.68, 6.932, 6.92,
                      23.58, 5.973, 6.164, 0.5172, 24.57, 5.531, 6.054, 0.6397, 1306.14, 324.64, 357.8,
                      30.39, 404.13, 106.34, 117.93, 7.011, 523.52, 149.34, 154.95, 7.212, 169.56, 42.45,
                      45.49, 2.895, 82.62, 24.46, 25.27, 0.9038, 88.74, 23.74, 24.83, 1.165, 89.53, 22.08,
                      23.7, 1.472, ])
    features = np.max([np.min([v_max, features], axis=0), v_min], axis=0)
    features = (features - v_mean)/v_std
    return features

class ComplexRFScoreFeaturizer(ComplexFeaturizer):
    """
    Calculate RFScore features for a complex. PDB file for receptor and SDF file for ligand must be provided.
    """

    def __init__(
        self,
        override_rfscore_executable_path = None,
    ):
        """
        Parameters
        ----------
        only_atom_type: bool, default False
          Whether to use only one-hot-encoded atom type as feature vector
        """
        if override_rfscore_executable_path is None:
            self.rfscore_executable_path = os.path.join(os.path.dirname(__file__), "bin", "rfscore")
        else:
            self.rfscore_executable_path = override_rfscore_executable_path
        super().__init__()


    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """
        """
        receptor_structure_file_path = datapoint[0]
        ligand_structure_file_path = datapoint[1]
        features = np.array([])
        if os.path.exists(receptor_structure_file_path) and os.path.exists(ligand_structure_file_path):
            ligand_extension = ligand_structure_file_path.split(".")[-1]
            if ligand_extension not in ["sdf", "pdb"]:
                raise ValueError("Ligand file extension must be sdf or pdb")
            
            receptor_extension = receptor_structure_file_path.split(".")[-1]
            if receptor_extension not in ["pdb"]:
                raise ValueError("Receptor file extension must be pdbqt")
            command = f"{self.rfscore_executable_path} -p {receptor_structure_file_path} -l {ligand_structure_file_path}"
            try:
                output = subprocess.run(command,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    shell=True)
                
                scores_text = output.stdout.decode('utf-8')
                
                tokens = scores_text.split("\n")[-2].split(",")
                if len(tokens) == 216:
                    features = np.array(tokens).astype(float)
                    features = execlude_features(features)
                    features = standardize(features)
                else:
                    error = output.stderr.decode('utf-8')
                    raise ValueError(f"Could not parse RFScore features from {scores_text}. {error}")
            except Exception as e:
                msg = f"Error in rfscore command {e}. {receptor_structure_file_path} {ligand_structure_file_path}"
                raise ValueError(msg)
        else:
            raise ValueError(f"Could not find {receptor_structure_file_path} or {ligand_structure_file_path}")
        return features 

