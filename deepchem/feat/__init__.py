"""
Making it easy to import in classes.
"""
# flake8: noqa

from deepchem.feat.atomic_conformation import (
    AtomicConformation,
    AtomicConformationFeaturizer,
)

# base classes for featurizers
from deepchem.feat.base_classes import (
    ComplexFeaturizer,
    DummyFeaturizer,
    Featurizer,
    MaterialCompositionFeaturizer,
    MaterialStructureFeaturizer,
    MolecularFeaturizer,
    UserDefinedFeaturizer,
)
from deepchem.feat.binding_pocket_features import BindingPocketFeaturizer

# complex featurizers
from deepchem.feat.complex_featurizers import (
    AtomicConvFeaturizer,
    CationPiVoxelizer,
    ChargeVoxelizer,
    ComplexNeighborListFragmentAtomicCoordinates,
    ContactCircularFingerprint,
    ContactCircularVoxelizer,
    HydrogenBondCounter,
    HydrogenBondVoxelizer,
    NeighborListAtomicCoordinates,
    NeighborListComplexAtomicCoordinates,
    PiStackVoxelizer,
    RdkitGridFeaturizer,
    SaltBridgeVoxelizer,
    SplifFingerprint,
    SplifVoxelizer,
)
from deepchem.feat.graph_data import GraphData
from deepchem.feat.graph_features import ConvMolFeaturizer, WeaveFeaturizer

# material featurizers
from deepchem.feat.material_featurizers import (
    CGCNNFeaturizer,
    ElementPropertyFingerprint,
    ElemNetFeaturizer,
    LCNNFeaturizer,
    SineCoulombMatrix,
)

# molecule featurizers
from deepchem.feat.molecule_featurizers import (
    AtomicCoordinates,
    BPSymmetryFunctionInput,
    CircularFingerprint,
    CoulombMatrix,
    CoulombMatrixEig,
    DMPNNFeaturizer,
    MACCSKeysFingerprint,
    MATFeaturizer,
    MetaDescriptors,
    Mol2VecFingerprint,
    MolGanFeaturizer,
    MolGraphConvFeaturizer,
    MordredDescriptors,
    OneHotFeaturizer,
    PagtnMolGraphFeaturizer,
    PubChemFingerprint,
    RawFeaturizer,
    RDKitDescriptors,
    RDKitProperties,
    SmilesToImage,
    SmilesToSeq,
    SparseMatrixOneHotFeaturizer,
    create_char_to_idx,
)

# tokenizers
try:
    import transformers
    from deepchem.feat.smiles_tokenizer import BasicSmilesTokenizer, SmilesTokenizer
    from transformers import BertTokenizer
except ModuleNotFoundError:
    pass

try:
    from deepchem.feat.bert_tokenizer import BertFeaturizer
    from transformers import BertTokenizerFast
except ModuleNotFoundError:
    pass

try:
    from deepchem.feat.reaction_featurizer import RxnFeaturizer
    from deepchem.feat.roberta_tokenizer import RobertaFeaturizer
    from transformers import RobertaTokenizerFast
except ModuleNotFoundError:
    pass

# support classes
from deepchem.feat.molecule_featurizers import GraphMatrix
