from deepchem.feat.base_classes import ParallelMolecularFeaturizer
from deepchem.feat.molecule_featurizers.descriptor_quantization import quantize
from deepchem.utils.typing import RDKitMol
from rdkit.Chem import Descriptors


class RDKitProperties(ParallelMolecularFeaturizer):
    """RDKit descriptors.

    This class computes a list of chemical descriptors like
    molecular weight, number of valence electrons, maximum and
    minimum partial charge, etc using RDKit.

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
    >>> featurizer = dc.feat.RDKitProperties()
    >>> features = featurizer.featurize(smiles)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (20,)

    """

    def __init__(self, quantize=True, n_threads=None):
        """Initialize this featurizer.

        Parameters
        ----------
        use_fragment: bool, optional (default True)
          If True, the return value includes the fragment binary descriptors like 'fr_XXX'.
        ipc_avg: bool, optional (default True)
          If True, the IPC descriptor calculates with avg=True option.
          Please see this issue: https://github.com/rdkit/rdkit/issues/1527.
        """
        super().__init__(n_threads=n_threads)
        self.quantize = quantize
        self.descriptors = []
        self.descList = []
        self.allowed_descriptors = [
            "MolWt",
            "NumValenceElectrons",
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
            "MinAbsPartialCharge",
            "FpDensityMorgan1",
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "FractionCSP3",
            "TPSA",
            "HeavyAtomCount",
            "NHOHCount",
            "NOCount",
            "NumAliphaticCarbocycles",
            "NumAliphaticHeterocycles",
            "NumAliphaticRings",
            "NumAromaticCarbocycles",
            "NumAromaticHeterocycles",
            "NumAromaticRings",
            "NumHAcceptors",
            "NumHDonors",
            "NumHeteroatoms",
            "NumRotatableBonds",
            "NumSaturatedCarbocycles",
            "NumSaturatedHeterocycles",
            "NumSaturatedRings",
            "RingCount",
            "MolLogP",
            "MolMR",
            "qed",
        ]
        n_buckets = 4
        for descriptor, _ in Descriptors.descList:
            if descriptor in self.allowed_descriptors:
                if self.quantize:
                    dnames = [f"{descriptor}_{i}" for i in range(n_buckets)]
                else:
                    dnames = descriptor
                self.descriptors += dnames

    def _featurize(self, datapoint: RDKitMol, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint: rdkit.Chem.rdchem.Mol
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

        features = []
        for descriptor, function in Descriptors.descList:
            if descriptor in self.allowed_descriptors:
                descriptor_value = function(datapoint)
                if self.quantize:
                    descriptor_value = quantize(descriptor_value, descriptor)
                else:
                    descriptor_value = [descriptor_value]
                features += descriptor_value
        return np.asarray(features, dtype=bool)
