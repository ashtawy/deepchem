import logging
import os
import sys
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import SDMolSupplier
from rdkit.Chem.rdmolfiles import MolFromMol2File, MolFromPDBFile

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

def load_gnina_type(path, object_type='ligand'):
    gdf = pd.read_parquet(path)
    a2i_map = pd.DataFrame({'atomic_number': [6, 7, 8, 16, 9, 15, 17, 35, 53],
                            'idx': [0, 1, 2, 3, 4, 5, 6, 7, 8]})
    gdf = gdf.merge(a2i_map)
    positions = gdf[["x", "y", "z"]].values
    features = np.squeeze(np.eye(9)[gdf['idx'].values.reshape(-1)])
    if object_type == 'ligand':
        features = np.concatenate([np.zeros([features.shape[0], 1]), features], axis=1)
    else:
        features = np.concatenate([np.ones([features.shape[0], 1]), features], axis=1)
    return {"node_features": features, "node_positions": positions}

def get_inter_bonds(ligand_graph, protein_graph, distance_threshold=5):
    points_l = ligand_graph["node_positions"]
    points_p = protein_graph["node_positions"]
    # Compute the pairwise distance
    diff = points_l[:, np.newaxis, :] - points_p[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    valid_protein_idxs = np.any(distances <= distance_threshold+1, axis=0)
    points_p = points_p[valid_protein_idxs]

    points_lp = np.concatenate([points_l, points_p], axis = 0)


    # Compute the pairwise distance
    diff = points_lp[:, np.newaxis, :] - points_lp[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=-1))
    filtered_distances = (distances > 0.0) & (distances <= distance_threshold)
    close_points_indices = np.argwhere(filtered_distances)
    #close_points_indices[:, 1] = close_points_indices[:, 1] + ligand_graph["num_nodes"]
    intermolecular_bonds = np.zeros([close_points_indices.shape[0]*2, 2], dtype=np.int64)
    intermolecular_bonds[::2, :] = close_points_indices
    intermolecular_bonds[1::2, :] = close_points_indices[:, [1, 0]]
    weight = np.repeat(distances[filtered_distances], 2).reshape([-1, 1])
    step = 0.2
    alphas = np.arange(0, distance_threshold+step, step)
    weight = np.exp(-10*(weight - alphas)**2).T
    return valid_protein_idxs, intermolecular_bonds.T, weight

def protein_to_graph(protein):
    atypes = GraphConvConstants.ATOM_FEATURES["atom_type"]
    atoms_feature_list = [[1] + one_hot_encode(atom.GetSymbol(), atypes) for atom in protein.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype = np.float64)

    c = protein.GetConformer()
    # coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(protein.GetNumAtoms())]

    # node_positions = np.array(coordinates, dtype = np.float64)
    node_positions = c.GetPositions()
    return {"node_features": node_features, "node_positions": node_positions}

def ligand_to_graph(ligand):
    atypes = GraphConvConstants.ATOM_FEATURES["atom_type"]
    atoms_feature_list = [[0] + one_hot_encode(atom.GetSymbol(), atypes) for atom in ligand.GetAtoms()]
    node_features = np.array(atoms_feature_list, dtype = np.float64)

    c = ligand.GetConformer()
    # coordinates = [[c.GetAtomPosition(atom_index)[i] for i in range(3)] for atom_index in range(ligand.GetNumAtoms())]
    # node_positions = np.array(coordinates, dtype = np.float64)
    node_positions = c.GetPositions()
    return {"node_features": node_features, "node_positions": node_positions}

def covalent_and_intermolecular_interactions_graph(ligand_graph, protein_graph, distance_threshold):
    valid_protein_index, intermolecular_edge_index, intermolecular_edge_weight = get_inter_bonds(ligand_graph,
                                                                                                 protein_graph,
                                                                                                 distance_threshold)
    if len(intermolecular_edge_weight) == 0:
        intermolecular_edge_index = np.empty((2, 0), dtype = np.int64)
        intermolecular_edge_weight = np.empty((1, 0), dtype = np.float64)
    protein_node_features = protein_graph["node_features"][valid_protein_index]
    ligand_node_features = ligand_graph["node_features"]
    node_features = np.concatenate([ligand_node_features, protein_node_features], axis = 0)
    # num_nodes = node_features.shape[0]
    # protein_node_positions = protein_graph["node_positions"][valid_protein_index]
    # ligand_node_positions = ligand_graph["node_positions"]
    # node_positions = np.concatenate([ligand_node_positions, protein_node_positions], axis = 0)

    return GraphData(
        node_features=node_features,
        edge_index=intermolecular_edge_index,
        edge_features=intermolecular_edge_weight.T,
        global_features=np.empty(0),
    )
class ComplexDMPNNFeaturizer(ComplexFeaturizer):
    """
    This class is a featurizer for Directed Message Passing Neural Network (D-MPNN) implementation

    The default node(atom) and edge(bond) representations are based on
    `Analyzing Learned Molecular Representations for Property Prediction paper <https://arxiv.org/pdf/1904.01561.pdf>`_.

    The default node representation are constructed by concatenating the following values,
    and the feature length is 133.

    - Atomic num: A one-hot vector of this atom, in a range of first 100 atoms.
    - Degree: A one-hot vector of the degree (0-5) of this atom.
    - Formal charge: Integer electronic charge, -1, -2, 1, 2, 0.
    - Chirality: A one-hot vector of the chirality tag (0-3) of this atom.
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
        distance_threshold=5,
        only_atom_type: bool = False,
        master_atom: bool = False,
        features_generators: Optional[List[str]] = None,
        is_adding_hs: bool = False,
        use_original_atom_ranks: bool = False,
    ):
        """
        Parameters
        ----------
        only_atom_type: bool, default False
          Whether to use only one-hot-encoded atom type as feature vector
        master_atom: bool, default False
          Whether to include central atom connected to all other atoms
        features_generator: List[str], default None
          List of global feature generators to be used.
        is_adding_hs: bool, default False
          Whether to add Hs or not.
        use_original_atom_ranks: bool, default False
          Whether to use original atom mapping or canonical atom mapping
        """
        self.distance_threshold = distance_threshold
        self.only_atom_type = only_atom_type
        self.master_atom = master_atom
        self.features_generators = features_generators
        self.is_adding_hs = is_adding_hs
        super().__init__()

    def _construct_bond_index(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Construct edge (bond) index

        Parameters
        ----------
        datapoint: RDKitMol
          RDKit mol object.

        Returns
        -------
        edge_index: np.ndarray
          Edge (Bond) index
        """
        src: List[int] = []
        dest: List[int] = []
        for bond in datapoint.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]
        return np.asarray([src, dest], dtype=int)

    def _get_bond_features(self, datapoint: RDKitMol) -> np.ndarray:
        """
        Construct bond(edge) features for the given datapoint

        For each bond index, 2 bond feature arrays are added to the main features array,
        for the current bond and its reverse bond respectively.

        Note: This method of generating bond features ensures that the shape of the bond features array
              is always equal to (number of bonds, number of bond features), even if the number of bonds
              is equal to 0.

        Parameters
        ----------
        datapoint: RDKitMol
          RDKit mol object.

        Returns
        -------
        f_bonds: np.ndarray
          Bond features array
        """
        bonds: Chem.rdchem._ROBondSeq = datapoint.GetBonds()

        bond_fdim: int = GraphConvConstants.BOND_FDIM
        number_of_bonds: int = (
            len(bonds) * 2
        )  # Note the value is doubled to account for reverse bonds
        f_bonds: np.ndarray = np.empty((number_of_bonds, bond_fdim))

        for index in range(0, number_of_bonds, 2):
            bond_id: int = index // 2
            bond_feature: np.ndarray = np.asarray(
                bond_features(bonds[bond_id]), dtype=bool
            )
            f_bonds[index] = bond_feature  # bond
            f_bonds[index + 1] = bond_feature  # reverse bond
        return f_bonds.astype(bool)

    def add_master_atom(self, atom_features, edge_features, edge_index):
        n_atoms = atom_features.shape[0]
        n_atom_features = atom_features.shape[1]
        n_edges = edge_features.shape[0]
        n_edge_features = edge_features.shape[1]

        m_atom_features = (
            np.where(atom_features.mean(axis=0) > 0.5, 1, 0)
            .astype(bool)
            .reshape([1, -1])
        )
        # m_atom_features = np.zeros([1, n_atom_features], dtype=bool)
        atom_features = np.concatenate([atom_features, m_atom_features], axis=0)
        atom_features = np.concatenate(
            [atom_features, np.zeros([n_atoms + 1, 1], dtype=bool)], axis=1
        )
        atom_features[n_atoms, n_atom_features] = True

        m_edge_features = np.zeros([n_atoms * 2, n_edge_features], dtype=bool)
        edge_features = np.concatenate([edge_features, m_edge_features], axis=0)
        edge_features = np.concatenate(
            [edge_features, np.zeros([edge_features.shape[0], 1], dtype=bool)], axis=1
        )

        edge_features[n_edges:, n_edge_features] = True
        src = []
        dst = []
        assert edge_index.max() == n_atoms - 1
        m_i = n_atoms
        for a_i in range(n_atoms):
            src += [a_i, m_i]
            dst += [m_i, a_i]
        master_index = np.asarray([src, dst], dtype=int)
        edge_index = np.concatenate([edge_index, master_index], axis=1)
        return atom_features, edge_features, edge_index

    def _featurize(self, datapoint, **kwargs) -> GraphData:
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
        if os.path.exists(receptor_structure_file_path) and os.path.exists(ligand_structure_file_path):
            ligand_extension = ligand_structure_file_path.split(".")[-1]
            if ligand_extension == "sdf":
                ligand = next(SDMolSupplier(ligand_structure_file_path, sanitize = False))
                if ligand is None:
                    raise ValueError(f"Could not parse ligand from {ligand_structure_file_path}")
                try:
                    ligand = Chem.RemoveHs(ligand, sanitize=False)
                except Exception as e:
                    raise ValueError(f"Could not remove hydrogens from ligand at {ligand_structure_file_path}. {e}")
                ligand_graph = ligand_to_graph(ligand)
            elif ligand_extension == "mol2":
                ligand = Chem.MolFromMol2File(ligand_structure_file_path, sanitize=False, cleanupSubstructures=False)
                if ligand is None:
                    raise ValueError(f"Could not parse ligand from {ligand_structure_file_path}")
                try:
                    ligand = Chem.RemoveHs(ligand, sanitize=False)
                except Exception as e:
                    raise ValueError(f"Could not remove hydrogens from ligand at {ligand_structure_file_path}. {e}")
                ligand_graph = ligand_to_graph(ligand)
            elif ligand_extension in ["gninatype", "gninatypes", "parquet"]:
                ligand_graph = load_gnina_type(ligand_structure_file_path, object_type='ligand')
            
            
            receptor_extension = receptor_structure_file_path.split(".")[-1]
            if receptor_extension == "pdb":
                receptor = MolFromPDBFile(receptor_structure_file_path, sanitize = False)
                if receptor is None:
                    raise ValueError(f"Could not parse receptor from {receptor_structure_file_path}")
                try:
                    receptor = Chem.RemoveHs(receptor, sanitize=False)
                except Exception as e:
                    raise ValueError(f"Could not remove hydrogens from receptor at {receptor_structure_file_path}. {e}")
                protein_graph = protein_to_graph(receptor)
            elif receptor_extension in ["gninatype", "gninatypes", "parquet"]:
                protein_graph = load_gnina_type(ligand_structure_file_path, object_type='protein')
            graph = covalent_and_intermolecular_interactions_graph(ligand_graph, protein_graph, self.distance_threshold)
        else:
            raise ValueError(f"Could not find {receptor_structure_file_path} or {ligand_structure_file_path}")
        return graph

