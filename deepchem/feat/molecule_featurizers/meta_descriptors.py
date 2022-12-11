"""
Basic molecular features.
"""

import logging
import sys
from typing import Any, Dict, List, Union

import numpy as np

from deepchem.feat.base_classes import ParallelMolecularFeaturizer
from deepchem.feat.molecule_featurizers.circular_fingerprint import CircularFingerprint
from deepchem.feat.molecule_featurizers.dmpnn_featurizer import DMPNNFeaturizer
from deepchem.feat.molecule_featurizers.frequent_subgraphs import FrequentSubgraphs
from deepchem.feat.molecule_featurizers.maccs_keys_fingerprint import (
    MACCSKeysFingerprint,
)
from deepchem.feat.molecule_featurizers.rdkit_descriptors import RDKitDescriptors
from deepchem.feat.molecule_featurizers.rdkit_properties import RDKitProperties
from deepchem.utils.typing import RDKitMol

logger = logging.getLogger(__name__)


class MetaDescriptors(ParallelMolecularFeaturizer):
    """
    Meta descriptors.

    This class computes a list of features of different types

    Attributes
    ----------
    descriptors: List[str]
      List of RDKit descriptor names used in this class.

    Note
    ----
    This class requires RDKit to be installed.

    Examples
    --------
    >>> import deepchem as dc
    >>> smiles = ['CC(=O)OC1=CC=CC=C1C(=O)O']
    >>> featurizer = dc.feat.MetaDescriptors(descriptor_sets={"ecfp": {1024}, "rdkit-descriptors": {}})
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (208,)
    """

    def __init__(
        self,
        descriptor_sets: Dict[str, Any] = {"ecfp": None, "rdkit-properties": None},
        n_threads=None,
    ):
        """Initialize this featurizer.

        Parameters
        ----------
        descriptor_sets: Dict, optional (default True)
          A dictionary with the feature types to calcualte along with their configurations
        """
        super().__init__(n_threads=n_threads)
        ecfp_def_params = {
            "radius": 2,
            "size": 1024,
            "chiral": False,
            "bonds": True,
            "features": False,
            "sparse": False,
            "smiles": False,
        }
        rdkit_def_params = {
            "use_fragment": True,
            "ipc_avg": True,
            # "is_normalized": False,
            # "use_bcut2d": False,
        }
        def_params = {
            "ecfp": ecfp_def_params,
            "rdkit-descriptors": rdkit_def_params,
            "rdkit-properties": {},
            "dmpnn": {"master_atom": True, "only_atom_type": False},
            "maccs": {},
            "frequent-subgraphs": {},
        }
        str2cls_map = {
            "ecfp": CircularFingerprint,
            "rdkit-descriptors": RDKitDescriptors,
            "rdkit-properties": RDKitProperties,
            "dmpnn": DMPNNFeaturizer,
            "maccs": MACCSKeysFingerprint,
            "frequent-subgraphs": FrequentSubgraphs,
        }
        self.descriptors = []
        self.featurizers = {}
        self.dtype = bool
        for ftype, fparams in descriptor_sets.items():
            if ftype == "rdkit-descriptors":
                self.dtype = np.float32

            ftype = ftype.lower()
            if ftype in str2cls_map:
                if fparams is None:
                    fparams = def_params[ftype]
                self.featurizers[ftype] = str2cls_map[ftype](**fparams)
                if hasattr(self.featurizers[ftype], "descriptors"):
                    self.descriptors += self.featurizers[ftype].descriptors
                elif ftype.lower() == "ecfp":
                    self.descriptors += [
                        f"ECFP_{i}" for i in range(def_params["ecfp"]["size"])
                    ]
                elif ftype.lower() == "maccs":
                    self.descriptors += [f"maccs_{i}" for i in range(166)]
                elif ftype.lower() == "dmpnn":
                    pass
                else:
                    raise ValueError(
                        f"Feature type {ftype} does not seem to have names of the features it generates"
                    )
            else:
                raise ValueError(
                    f"Feature type of {ftype} is not supported. Supported types are {list(str2cls_map)}"
                )

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint: RDKitMol
          RDKit Mol object

        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`.
          The length is `len(self.descriptors)`.
        """
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )

        features: List[Union[int, float]] = []
        graph_features = None
        for featurizer_name, featurizer in self.featurizers.items():
            features_subset = featurizer._featurize(datapoint)
            if features_subset is not None:
                # and (isinstance() and np.isnan(features_subset).sum() == 0):
                if featurizer_name in ["dmpnn"]:
                    graph_features = features_subset
                else:
                    features.append(features_subset)

            else:
                graph_features = None
                features = []
                break

        if graph_features is not None:
            if len(features) > 0 and len(features) == len(self.featurizers) - 1:
                features = np.concatenate(features, axis=0).astype(self.dtype)
                graph_features.global_features = features
                features = graph_features
            elif len(features) > 0 and len(features) < len(self.featurizers) - 1:
                features = np.array([], dtype=self.dtype)
            else:
                features = graph_features
        elif len(features) == len(self.featurizers):
            features = np.concatenate(features, axis=0).astype(self.dtype)
        else:
            features = np.array([], dtype=self.dtype)
        return features
