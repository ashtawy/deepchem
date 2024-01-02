import copy
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
from deepchem.feat.graph_data import GraphData
from deepchem.feat.graph_features import bond_features as b_Feats
from deepchem.feat.molecule_featurizers.circular_fingerprint import \
    CircularFingerprint
from deepchem.feat.molecule_featurizers.mol_dmpnn_featurizer import (
    GraphConvConstants, atom_features, bond_features, generate_global_features)
from deepchem.feat.molecule_featurizers.rdkit_descriptors import \
    RDKitDescriptors
from deepchem.utils.molecule_feature_utils import one_hot_encode
from deepchem.utils.typing import RDKitMol

logger = logging.getLogger(__name__)

def standardize(features):
    v_min = np.array([-15.95, -12.72, 14.6, 273.63, 0.08012, 0.0, 0.0, ])
    v_max = np.array([144.04, 8.93, 256.68, 3920.85, 244.25, 163.51, 17.49, ])
    v_mean = np.array([-4.411, -0.5579, 82.76, 1287.43, 7.251, 34.65, 2.907, ])
    v_std = np.array([10.92, 1.725, 32.65, 467.24, 17.26, 27.85, 2.873, ])
    features = np.max([np.min([v_max, features], axis=0), v_min], axis=0)
    features = (features - v_mean)/v_std
    return features

MGLTOOLS_PATH = "/software/mgltools_x86_64Linux2_1.5.7"

def receptor_to_pdbqt(receptor_file_name, receptor_pdbqt_file_name, overwrite=False, use_obabel=True):    
    if not os.path.exists(receptor_pdbqt_file_name) or overwrite:
        env_vars = copy.deepcopy(os.environ)
        env_vars['PYTHONPATH'] = ''
        env_vars['PYTHONPATH'] = f"{MGLTOOLS_PATH}/MGLToolsPckgs/:{env_vars['PYTHONPATH']}"
        try:
            # execute the following command and get the std output and std error
            conv_path = f"{MGLTOOLS_PATH}/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py"
            python_path = "/home/hashtawy/.conda/envs/autodock_tools/bin/python"
            command = f"{python_path} {conv_path} -r {receptor_file_name} -o {receptor_pdbqt_file_name}"
            command = f"{conv_path} -r {receptor_file_name} -o {receptor_pdbqt_file_name}"
            logging.info(f"Converting receptor to pdbqt using prepare_receptor4.py: {receptor_file_name}")
            output = subprocess.run(command,
                                env = env_vars, 
                                stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE,
                                shell=True)
            logging.info(output.stdout.decode('utf-8'))
            logging.error(output.stderr.decode('utf-8'))
        except Exception as E:
            logging.error(f"Failed to convert receptor to pdbqt using prepare_receptor4.py: {receptor_file_name}. {E}")
        if use_obabel and not os.path.exists(receptor_pdbqt_file_name):
            logging.error(f"Could not convert receptor from pdb to the PDBQT format using ADT, trying with obabel: {receptor_file_name}")
            try:
                subprocess.call(["obabel", receptor_file_name, f"-O{receptor_pdbqt_file_name}"])
            except Exception as E:
                msg = (f"Could not convert receptor from pdb to the PDBQT format using obabel: {receptor_file_name}")
                logging.error(msg)
    return os.path.exists(receptor_pdbqt_file_name)

class ComplexSminaFeaturizer(ComplexFeaturizer):
    """
    This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation

    The default node(atom) and edge(bond) representations are based on
    `Analyzing Learned Molecular Representations for Property Prediction paper <https://arxiv.org/pdf/1904.01561.pdf>`_.

    The default node representation are constructed by concatenating the following values,
    and the feature length is 133.

    - Atomic num: A one-hot vector of this atom, in a range of first 100 atoms.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Formal charge: Integer electronic charge, -1, -2, 1, 2, 0.
    - Chirality: A one-hot vector of the chirComplexSminaFeaturizerality tag (0-3) of this atom.
    - Number of Hydrogens: A one-hot vector of the number of hydrogens (0-4) that this atom connected.
    - Hybridization: A one-hot vector of "SP", "SP2", "SP3", "SP3D", "SP3D2".
    - Aromatic: A one-hot vector of whether the atom belongs to an aromatic ring.
    - Mass: Atomic mass * 0.01

    The default edge representation are constructed by concatenating the following values,
    and the feature length is 14.

    - Bond type: A one-hot vector of the bond type, "single", "double", "triple", or "aromatic".
    - Same ring: A one-hot vector of whether the atoms in the pair are in the same ring.
    - Conjugated: A one-hot vector of whether this bond is conjugated or not.
    - Stereo: A one-hot vector of the stereo configuration (0-5) of a bond.

    If you want to know more details about features, please check the paper [1]_ and
    utilities in deepchem.utils.molecule_feature_utils.py.

    Examples
    --------
    >>> smiles = ["C1=CC=CN=C1", "C1CCC1"]
    >>> featurizer = DMPNNFeaturizer()
    >>> out = featurizer.featurize(smiles)
    >>> type(out[0])
    <class 'deepchem.feat.graph_data.GraphData'>
    >>> out[0].num_nodes
    6
    >>> out[0].num_node_features
    133
    >>> out[0].node_features.shape
    (6, 133)
    >>> out[0].num_edge_features
    14
    >>> out[0].num_edges
    12
    >>> out[0].edge_features.shape
    (12, 14)

    References
    ----------
    .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond fingerprints."
       Journal of computer-aided molecular design 30.8 (2016):595-608.

    Note
    ----
    This class requires RDKit to be installed.
    """

    def __init__(
        self,
        override_smina_executable_path = None,
    ):
        """
        Parameters
        ----------
        only_atom_type: bool, default False
          Whether to use only one-hot-encoded atom type as feature vector
        """
        if override_smina_executable_path is None:
            self.smina_executable_path = os.path.join(os.path.dirname(__file__), "bin", "smina.static")
        else:
            self.smina_executable_path = override_smina_executable_path
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
            if ligand_extension not in ["sdf", "pdbqt"]:
                raise ValueError("Ligand file extension must be sdf or pdbqt")
            
            receptor_extension = receptor_structure_file_path.split(".")[-1]
            if receptor_extension not in ["pdbqt", "pdb"]:
                raise ValueError("Receptor file extension must be pdbqt")
            if receptor_extension == "pdb":
                receptor_pdbqt_file_name = receptor_structure_file_path.replace(".pdb", ".pdbqt")
                if receptor_to_pdbqt(receptor_structure_file_path, receptor_pdbqt_file_name,
                                  overwrite=False, use_obabel=True):
                    receptor_structure_file_path = receptor_pdbqt_file_name
                else:
                    return features
            command = f"{self.smina_executable_path} --score_only -r {receptor_structure_file_path} -l {ligand_structure_file_path}" 
            try:
                output = subprocess.run(command,
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    shell=True)
                #print(output.stdout.decode('utf-8'))
                #print(output.stderr.decode('utf-8'))
                
                scores_text = output.stdout.decode('utf-8')
                features = []
                for line in scores_text.split("\n"):#[-8:]:
                    tokens = line.split()
                    if len(line) > 20:
                        if "Affinity: " in line:
                            features.append(tokens[1])
                        if "Intramolecular energy: " in line:
                            features.append(tokens[2])
                        if "## " in line and "## Name " not in line:
                            features += tokens[2:-1] # last term is always "0"
                features = np.array(features).astype(float)
                features = standardize(features)
            except Exception as e:
                msg = f"Error in smina command {e}. {receptor_structure_file_path} {ligand_structure_file_path}"
                raise ValueError(msg)
        else:
            raise ValueError(f"Could not find {receptor_structure_file_path} or {ligand_structure_file_path}")
        return features 

