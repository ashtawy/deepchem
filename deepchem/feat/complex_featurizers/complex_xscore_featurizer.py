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
    v_min = np.array([86.91, 0.0, 0.0, 0.0, 0.0, 0.0, 2.807, 3.67, 1.967, 2.973, ])
    v_max = np.array([1596.82, 14.15, 315.99, 11.9, 673.87, 48.0, 10.7, 11.0, 10.16, 10.55, ])
    v_mean = np.array([493.85, 1.941, 67.34, 1.706, 173.88, 4.657, 5.843, 5.946, 5.721, 5.837, ])
    v_std = np.array([191.27, 2.261, 52.16, 1.648, 107.48, 6.009, 1.008, 1.002, 0.9602, 0.9642, ])
    features = np.max([np.min([v_max, features], axis=0), v_min], axis=0)
    features = (features - v_mean)/v_std
    return features

class ComplexXScoreFeaturizer(ComplexFeaturizer):
    """
    Calculate Xscore features for a complex. PDB file for receptor and MOL2 file for ligand must be provided.
    """

    def __init__(
        self,
        override_xscore_executable_path = None,
    ):
        """
        Parameters
        ----------
        only_atom_type: bool, default False
          Whether to use only one-hot-encoded atom type as feature vector
        """
        if override_xscore_executable_path is None:
            self.xscore_executable_path = os.path.join(os.path.dirname(__file__), "bin", "xscore")
        else:
            self.xscore_executable_path = override_xscore_executable_path
        super().__init__()


    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """
        Calculate molecule graph features from RDKit mol object.

        Parameters
        ----------
        datapoint: RDKitMol
          RDKit mol object.

        Returns
        -------
        graph: GraphData
          A molecule graph object with features:
          - node_features: Node feature matrix with shape [num_nodes, num_node_features]
          - edge_index: Graph connectivity in COO format with shape [2, num_edges]
          - edge_features: Edge feature matrix with shape [num_edges, num_edge_features]
          - global_features: Array of global molecular features
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
            command = f"{self.xscore_executable_path} -p  {receptor_structure_file_path} -l {ligand_structure_file_path}" 
            try:
                xscore_root_path = os.path.dirname(self.xscore_executable_path)
                env_vars = {"XSCORE_PARAMETER": os.path.join(xscore_root_path, "parameter")}
                output = subprocess.run(command,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    env=env_vars,
                                    shell=True)
                
                scores_text = output.stdout.decode('utf-8')
                tokens = scores_text.split("\n")[-2].split(",")
                if len(tokens) == 10:
                    features = np.array(tokens).astype(float)
                    features = standardize(features)
                elif "Probably the ligand has not been docked with the protein" in scores_text:
                    features = np.zeros(10, dtype=float)
                else:
                    error = output.stderr.decode('utf-8')
                    raise ValueError(f"Could not parse Xscore features from {scores_text}. {error}")
            except Exception as e:
                msg = f"Error in xscore command {e}. {receptor_structure_file_path} {ligand_structure_file_path}. {e}"
                raise ValueError(msg)
        else:
            raise ValueError(f"Could not find {receptor_structure_file_path} or {ligand_structure_file_path}")
        return features