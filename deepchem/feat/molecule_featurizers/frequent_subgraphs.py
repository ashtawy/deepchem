import os

import numpy as np
from rdkit import Chem

from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.typing import RDKitMol


class FrequentSubgraphs(MolecularFeaturizer):
    """MACCS Keys Fingerprint.

    The MACCS (Molecular ACCess System) keys are one of the most commonly used structural keys.
    Please confirm the details in [1]_, [2]_.

    Examples
    --------
    >>> import deepchem as dc
    >>> smiles = 'CC(=O)OC1=CC=CC=C1C(=O)O'
    >>> featurizer = dc.feat.FrequentSubgraphs()
    >>> features = featurizer.featurize([smiles])
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (167,)

    Note
    ----
    This class requires RDKit to be installed.

    """

    def __init__(self, smarts_fname=None):
        """Initialize this featurizer."""
        self.descriptors = []
        if smarts_fname is None:
            smarts_fname = os.path.join(
                os.path.dirname(__file__), "frequent_subgraphs.csv"
            )
        with open(smarts_fname) as smarts_file:
            lines = smarts_file.readlines()
            for line in lines[1:]:
                line = line.strip()
                if len(line) > 1:
                    subgraph = line.split(",")[0]
                    self.descriptors.append(f"frequent-subgraphs_{subgraph}")
                    self.subgraphs.append(Chem.MolFromSmarts(subgraph))

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        if "mol" in kwargs:
            datapoint = kwargs.get("mol")
            raise DeprecationWarning(
                'Mol is being phased out as a parameter, please pass "datapoint" instead.'
            )
        features = []
        for subgraph in self.subgraphs:
            features.append(len(datapoint.GetSubstructMatches(subgraph)) > 0)

        return np.array(features, dtype=bool)
