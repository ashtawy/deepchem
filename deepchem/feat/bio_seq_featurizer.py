import numpy as np
import deepchem as dc
try:
    import pysam
except ImportError:
    print("Error: Unable to import pysam. Please make sure it is installed.")
from deepchem.feat import Featurizer


class SAMFeaturizer(Featurizer):
    """
    Featurizes SAM files, that store biological sequences aligned to a reference
    sequence. This class extracts Query Name, Query Sequence, Query Length,
    Reference Name,Reference Start, CIGAR and Mapping Quality of each read in
    a SAM file.

    Examples
    --------
    >>> from deepchem.data.data_loader import SAMLoader
    >>> import deepchem as dc
    >>> inputs = 'deepchem/data/tests/example.sam'
    >>> featurizer = dc.feat.SAMFeaturizer()
    >>> features = featurizer.featurize(inputs)
    Information for each read is stored in a 'numpy.ndarray'.
    >>> type(features[0])
    <class 'numpy.ndarray'>

    This is the default featurizer used by SAMLoader, and it extracts the following
    fields from each read in each SAM file in the given order:-
    - Column 0: Query Name
    - Column 1: Query Sequence
    - Column 2: Query Length
    - Column 3: Reference Name
    - Column 4: Reference Start
    - Column 5: CIGAR
    - Column 6: Mapping Quality
    For the given example, to extract specific features, we do the following.
    >>> features[0][0]     # Query Name
    r001
    >>> features[0][1]     # Query Sequence
    TTAGATAAAGAGGATACTG
    >>> features[0][2]     # Query Length
    19
    >>> features[0][3]     # Reference Name
    ref
    >>> features[0][4]     # Reference Start
    6
    >>> features[0][5]     # CIGAR
    [(0, 8), (1, 4), (0, 4), (2, 1), (0, 3)]
    >>> features[0][6]     # Mapping Quality
    30

    Note
    ----
    This class requires pysam to be installed. Pysam can be used with Linux or MacOS X.
    To use Pysam on Windows, use Windows Subsystem for Linux(WSL).
    """

    def __init__(self, max_records=None):
        """
        Initialize SAMFeaturizer.

        Parameters
        ----------
        max_records : int or None, optional
            The maximum number of records to extract from the SAM file. If None, all records will be extracted.
        """
        self.max_records = max_records

    def _featurize(self, datapoint):
        """
        Extract features from a SAM file.

        Parameters
        ----------
        datapoint : str
            Name of SAM file.

        Returns
        -------
        features : numpy.ndarray
        A 2D NumPy array representing the extracted features.
        Each row corresponds to a SAM record, and columns represent different features.
            - Column 0: Query Name
            - Column 1: Query Sequence
            - Column 2: Query Length
            - Column 3: Reference Name
            - Column 4: Reference Start
            - Column 5: CIGAR
            - Column 6: Mapping Quality
        """

        features = []
        record_count = 0

        for record in datapoint:
            feature_vector = [
                record.query_name,
                record.query_sequence,
                record.query_length,
                record.reference_name,
                record.reference_start,
                record.cigar,
                record.mapping_quality,
            ]

            features.append(feature_vector)
            record_count += 1

            # Break the loop if max_records is set
            if self.max_records is not None and record_count >= self.max_records:
                break

        datapoint.close()

        return np.array(features, dtype="object")
