Utilities
=========
DeepChem has a broad collection of utility functions. Many of these
maybe be of independent interest to users since they deal with some
tricky aspects of processing scientific datatypes.

Data Utilities
--------------

Array Utilities
^^^^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.pad_array

Data Directory
^^^^^^^^^^^^^^^
The DeepChem data directory is where downloaded MoleculeNet datasets are stored.

.. autofunction:: deepchem.utils.data_utils.get_data_dir

URL Handling
^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.download_url

File Handling
^^^^^^^^^^^^^

.. autofunction:: deepchem.utils.data_utils.untargz_file

.. autofunction:: deepchem.utils.data_utils.unzip_file

.. autofunction:: deepchem.utils.data_utils.load_data

.. autofunction:: deepchem.utils.data_utils.load_sdf_files

.. autofunction:: deepchem.utils.data_utils.load_csv_files

.. autofunction:: deepchem.utils.data_utils.load_json_files

.. autofunction:: deepchem.utils.data_utils.load_pickle_files

.. autofunction:: deepchem.utils.data_utils.load_from_disk

.. autofunction:: deepchem.utils.data_utils.save_to_disk

.. autofunction:: deepchem.utils.data_utils.load_dataset_from_disk

.. autofunction:: deepchem.utils.data_utils.save_dataset_to_disk

Molecular Utilities
-------------------

.. autoclass:: deepchem.utils.conformers.ConformerGenerator
  :members:

.. autoclass:: deepchem.utils.rdkit_utils.MoleculeLoadException
  :members:

.. autofunction:: deepchem.utils.rdkit_utils.get_xyz_from_mol

.. autofunction:: deepchem.utils.rdkit_utils.add_hydrogens_to_mol

.. autofunction:: deepchem.utils.rdkit_utils.compute_charges

.. autofunction:: deepchem.utils.rdkit_utils.load_molecule

.. autofunction:: deepchem.utils.rdkit_utils.write_molecule

Molecular Fragment Utilities
----------------------------

It's often convenient to manipulate subsets of a molecule. The :code:`MolecularFragment` class aids in such manipulations.

.. autoclass:: deepchem.utils.fragment_utils.MolecularFragment
  :members:

.. autoclass:: deepchem.utils.fragment_utils.AtomShim
  :members:

.. autofunction:: deepchem.utils.fragment_utils.strip_hydrogens

.. autofunction:: deepchem.utils.fragment_utils.merge_molecular_fragments

.. autofunction:: deepchem.utils.fragment_utils.get_contact_atom_indices

.. autofunction:: deepchem.utils.fragment_utils.reduce_molecular_complex_to_contacts

Coordinate Box Utilities
------------------------

.. autoclass:: deepchem.utils.coordinate_box_utils.CoordinateBox
  :members:

.. autofunction:: deepchem.utils.coordinate_box_utils.intersect_interval

.. autofunction:: deepchem.utils.coordinate_box_utils.union

.. autofunction:: deepchem.utils.coordinate_box_utils.merge_overlapping_boxes

.. autofunction:: deepchem.utils.coordinate_box_utils.get_face_boxes

Evaluation Utils
----------------

.. autoclass:: deepchem.utils.evaluate.Evaluator
  :members:

.. autoclass:: deepchem.utils.evaluate.GeneratorEvaluator
  :members:

.. autofunction:: deepchem.utils.evaluate.relative_difference


Genomic Utilities
-----------------

.. autofunction:: deepchem.utils.genomics_utils.seq_one_hot_encode

.. autofunction:: deepchem.utils.genomics_utils.encode_bio_sequence

.. autofunction:: deepchem.utils.sequence_utils.hhblits

.. autofunction:: deepchem.utils.sequence_utils.hhsearch

.. autofunction:: deepchem.utils.sequence_utils.MSA_to_dataset


Geometry Utilities
------------------

.. autofunction:: deepchem.utils.geometry_utils.unit_vector

.. autofunction:: deepchem.utils.geometry_utils.angle_between

.. autofunction:: deepchem.utils.geometry_utils.generate_random_unit_vector

.. autofunction:: deepchem.utils.geometry_utils.generate_random_rotation_matrix

.. autofunction:: deepchem.utils.geometry_utils.is_angle_within_cutoff

Graph Utilities
---------------

.. autofunction:: deepchem.utils.graph_utils.fourier_encode_dist

.. autofunction:: deepchem.utils.graph_utils.aggregate_mean

.. autofunction:: deepchem.utils.graph_utils.aggregate_max

.. autofunction:: deepchem.utils.graph_utils.aggregate_min

.. autofunction:: deepchem.utils.graph_utils.aggregate_std

.. autofunction:: deepchem.utils.graph_utils.aggregate_var

.. autofunction:: deepchem.utils.graph_utils.aggregate_moment

.. autofunction:: deepchem.utils.graph_utils.aggregate_sum

.. autofunction:: deepchem.utils.graph_utils.scale_identity

.. autofunction:: deepchem.utils.graph_utils.scale_amplification

.. autofunction:: deepchem.utils.graph_utils.scale_attenuation

Hash Function Utilities
-----------------------

.. autofunction:: deepchem.utils.hash_utils.hash_ecfp

.. autofunction:: deepchem.utils.hash_utils.hash_ecfp_pair

.. autofunction:: deepchem.utils.hash_utils.vectorize

Voxel Utils
-----------

.. autofunction:: deepchem.utils.voxel_utils.convert_atom_to_voxel

.. autofunction:: deepchem.utils.voxel_utils.convert_atom_pair_to_voxel

.. autofunction:: deepchem.utils.voxel_utils.voxelize


Graph Convolution Utilities
---------------------------

.. autofunction:: deepchem.utils.molecule_feature_utils.one_hot_encode

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_type_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.construct_hydrogen_bonding_info

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_hydrogen_bonding_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_is_in_aromatic_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_hybridization_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_total_num_Hs_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_chirality_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_formal_charge

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_partial_charge

.. autofunction:: deepchem.utils.molecule_feature_utils.get_atom_total_degree_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_type_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_is_in_same_ring_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_is_conjugated_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_stereo_one_hot

.. autofunction:: deepchem.utils.molecule_feature_utils.get_bond_graph_distance_one_hot

Grover Utilities
----------------

.. autofunction:: deepchem.utils.grover.extract_grover_attributes

Debug Utilities
---------------

Docking Utilities
-----------------

These utilities assist in file preparation and processing for molecular
docking.

.. autofunction:: deepchem.utils.docking_utils.write_vina_conf

.. autofunction:: deepchem.utils.docking_utils.write_gnina_conf

.. autofunction:: deepchem.utils.docking_utils.load_docked_ligands

.. autofunction:: deepchem.utils.docking_utils.prepare_inputs

.. autofunction:: deepchem.utils.docking_utils.read_gnina_log


Print Threshold
^^^^^^^^^^^^^^^

The printing threshold controls how many dataset elements are printed
when :code:`dc.data.Dataset` objects are converted to strings or
represnted in the IPython repl.

.. autofunction:: deepchem.utils.debug_utils.get_print_threshold

.. autofunction:: deepchem.utils.debug_utils.set_print_threshold

.. autofunction:: deepchem.utils.debug_utils.get_max_print_size

.. autofunction:: deepchem.utils.debug_utils.set_max_print_size

Fake Data Generator
-------------------

The utilities here are used to generate random sample data which can be
used for testing model architectures or other purposes.

.. autoclass:: deepchem.utils.fake_data_generator.FakeGraphGenerator
  :members:

Electron Sampler
-------------------

The utilities here are used to sample electrons in a given molecule
and update it using monte carlo methods, which can be used for methods
like Variational Monte Carlo, etc.

.. autoclass:: deepchem.utils.electron_sampler.ElectronSampler
  :members:

Density Functional Theory Utilities
-----------------------------------

The utilites here are used to create an object that contains information about a system's self-consistent iteration steps and other processes.

.. autoclass:: deepchem.utils.dft_utils.Lattice
  :members:

.. autoclass:: deepchem.utils.dft_utils.SpinParam
  :members:

.. autoclass:: deepchem.utils.dft_utils.ValGrad
  :members:

.. autoclass:: deepchem.utils.dft_utils.data.datastruct.CGTOBasis
  :members:

.. autoclass:: deepchem.utils.dft_utils.data.datastruct.AtomCGTOBasis
  :members:

.. autoclass:: deepchem.utils.dft_utils.BaseXC
  :members:

.. autoclass:: deepchem.utils.dft_utils.AddBaseXC
  :members:

.. autoclass:: deepchem.utils.dft_utils.xc.base_xc.MulBaseXC
  :members:

.. autoclass:: deepchem.utils.dft_utils.BaseGrid
  :members:

.. autoclass:: deepchem.utils.dft_utils.df.base_df.BaseDF
  :members:

.. autoclass:: deepchem.utils.dft_utils.hamilton.base_hamilton.BaseHamilton
  :members:

.. autoclass:: deepchem.utils.dftutils.KSCalc
  :members:

.. autofunction:: deepchem.utils.dftutils.hashstr

.. autoclass:: deepchem.utils.dftutils.BaseGrid
  :members:

.. autoclass:: deepchem.utils.dftutils.BaseQCCalc
  :members:

.. autoclass:: deepchem.utils.dftutils.SpinParam
  :members:

.. autoclass:: deepchem.utils.dft_utils.config._Config
  :members:

.. autoclass:: deepchem.utils.dft_utils.BaseOrbParams
  :members:

.. autoclass:: deepchem.utils.dft_utils.QROrbParams
  :members:

.. autoclass:: deepchem.utils.dft_utils.MatExpOrbParams
  :members:

.. autoclass:: deepchem.utils.dft_utils.api.parser.parse_moldesc
  :members:

.. autoclass:: deepchem.utils.dft_utils.system.base_system.BaseSystem
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.RadialGrid
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.get_xw_integration
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.SlicedRadialGrid
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.BaseGridTransform
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.DE2Transformation
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.LogM3Transformation
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.TreutlerM4Transformation
  :members:

.. autoclass:: deepchem.utils.dft_utils.grid.radial_grid.get_grid_transform
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.editable_module.EditableModule
  :members:

.. autofunction:: deepchem.utils.differentiation_utils.normalize_bcast_dims

.. autofunction:: deepchem.utils.differentiation_utils.get_bcasted_dims

.. autofunction:: deepchem.utils.differentiation_utils.match_dim

.. autoclass:: deepchem.utils.differentiation_utils.linop.LinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.linop.AddLinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.linop.MulLinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.linop.AdjointLinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.linop.MatmulLinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.linop.MatrixLinearOperator
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.pure_function.PureFunction
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.pure_function.FunctionPureFunction
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.pure_function.EditableModulePureFunction
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.pure_function.TorchNNPureFunction
  :members:

.. autoclass:: deepchem.utils.differentiation_utils.pure_function.PureFunction
  :members:

.. autofunction:: deepchem.utils.differentiation_utils.pure_function._check_identical_objs

.. autofunction:: deepchem.utils.differentiation_utils.pure_function.get_pure_function

.. autofunction:: deepchem.utils.differentiation_utils.set_default_option

.. autofunction:: deepchem.utils.differentiation_utils.get_and_pop_keys

.. autofunction:: deepchem.utils.differentiation_utils.get_method

.. autofunction:: deepchem.utils.differentiation_utils.dummy_context_manager

.. autofunction:: deepchem.utils.differentiation_utils.assert_runtime

.. autofunction:: deepchem.utils.differentiation_utils._set_initial_v

.. autofunction:: deepchem.utils.differentiation_utils._take_eigpairs

.. autofunction:: deepchem.utils.differentiation_utils.exacteig

.. autofunction:: deepchem.utils.differentiation_utils.degen_symeig

.. autofunction:: deepchem.utils.differentiation_utils.davidson

.. autofunction:: deepchem.utils.differentiation_utils.symeig.lsymeig

.. autofunction:: deepchem.utils.differentiation_utils.symeig.usymeig

.. autofunction:: deepchem.utils.differentiation_utils.symeig.symeig

.. autoclass:: deepchem.utils.differentiation_utils.symeig.symeig_torchfcn
  :members:

.. autofunction:: deepchem.utils.differentiation_utils.symeig._check_degen

.. autofunction:: deepchem.utils.differentiation_utils.symeig.ortho

.. autofunction:: deepchem.utils.differentiation_utils.grad.jac

.. autoclass:: deepchem.utils.differentiation_utils.grad._Jac
  :members:

.. autofunction:: deepchem.utils.differentiation_utils.grad._setup_idxs

.. autofunction:: deepchem.utils.differentiation_utils.grad.connect_graph

.. autofunction:: deepchem.utils.differentiation_utils.solve.wrap_gmres

.. autofunction:: deepchem.utils.differentiation_utils.solve.exactsolve

.. autofunction:: deepchem.utils.differentiation_utils.solve.solve_ABE

.. autofunction:: deepchem.utils.differentiation_utils.solve.get_batchdims

.. autofunction:: deepchem.utils.differentiation_utils.solve.setup_precond

.. autofunction:: deepchem.utils.differentiation_utils.solve.dot

.. autofunction:: deepchem.utils.differentiation_utils.solve.gmres

.. autofunction:: deepchem.utils.differentiation_utils.solve.setup_linear_problem

.. autofunction:: deepchem.utils.differentiation_utils.solve.safedenom

.. autofunction:: deepchem.utils.differentiation_utils.solve.get_largest_eival

.. autofunction:: deepchem.utils.differentiation_utils.solve.solve

.. autofunction:: deepchem.utils.differentiation_utils.solve.cg

.. autofunction:: deepchem.utils.differentiation_utils.solve.bicgstab

.. autoclass:: deepchem.utils.differentiation_utils.solve.solve_torchfcn
  :members:

Attribute Utilities
-------------------

The utilities here are used to modify the attributes of the classes. Used by differentiation_utils.

.. autoclass:: deepchem.utils.attribute_utils.get_attr
  :members:

.. autoclass:: deepchem.utils.attribute_utils.set_get_attr
  :members:

.. autoclass:: deepchem.utils.attribute_utils.del_attr
  :members:

Pytorch Utilities
-----------------

.. autofunction:: deepchem.utils.pytorch_utils.unsorted_segment_sum

.. autofunction:: deepchem.utils.pytorch_utils.segment_sum

.. autofunction:: deepchem.utils.pytorch_utils.chunkify

.. autofunction:: deepchem.utils.pytorch_utils.get_memory

.. autofunction:: deepchem.utils.pytorch_utils.gaussian_integral

.. autofunction:: deepchem.utils.pytorch_utils.TensorNonTensorSeparator

.. autofunction:: deepchem.utils.pytorch_utils.tallqr

.. autofunction:: deepchem.utils.pytorch_utils.to_fortran_order

.. autofunction:: deepchem.utils.pytorch_utils.get_np_dtype

Batch Utilities
---------------

The utilites here are used for computing features on batch of data.
Can be used inside of default_generator function.

.. autofunction:: deepchem.utils.batch_utils.batch_coulomb_matrix_features

.. autofunction:: deepchem.utils.batch_utils.batch_elements

.. autofunction:: deepchem.utils.batch_utils.create_input_array

.. autofunction:: deepchem.utils.batch_utils.create_output_array

Periodic Table Utilities
------------------------

The Utilities here are used to computing atomic mass and radii data.
These can be used by DFT and many other Molecular Models.

.. autofunction:: deepchem.utils.periodictable_utils.get_atomz

.. autofunction:: deepchem.utils.periodictable_utils.test_get_atom_mass

.. autofunction:: deepchem.utils.periodictable_utils.test_get_period

Equivariance Utilities
----------------------

The utilities here refer to equivariance tools that play a vital
role in mathematics and applied sciences. They excel in preserving
the relationships between objects or data points when undergoing transformations
such as rotations or scaling.

You can refer to the `tutorials <https://deepchem.io/tutorials/introduction-to-equivariance/>`_
for additional information regarding equivariance and Deepchem's support for equivariance.

.. autofunction:: deepchem.utils.equivariance_utils.su2_generators

.. autofunction:: deepchem.utils.equivariance_utils.so3_generators

.. autofunction:: deepchem.utils.equivariance_utils.change_basis_real_to_complex

.. autofunction:: deepchem.utils.equivariance_utils.wigner_D

Miscellaneous Utilities
-----------------------

The utilities here are used for miscellaneous purposes.
Initial usecases are for improving the printing format of __repr__.

.. autofunction:: deepchem.utils.misc_utils.indent

.. autofunction:: deepchem.utils.misc_utils.shape2str

.. autoclass:: deepchem.utils.attribute_utils.UnimplementedError
  :members:

.. autoclass:: deepchem.utils.attribute_utils.GetSetParamsError
  :members:

.. autoclass:: deepchem.utils.attribute_utils.ConvergenceWarning
  :members:

.. autoclass:: deepchem.utils.attribute_utils.MathWarning
  :members:

.. autoclass:: deepchem.utils.attribute_utils.Uniquifier
  :members:

SafeOperations Utilities
------------------------

The utilities here are used for safe operations on tensors.
These are used to avoid NaNs and Infs in the output.

.. autofunction:: deepchem.utils.safe_ops.safepow

.. autofunction:: deepchem.utils.safe_ops.safenorm

.. autofunction:: deepchem.utils.safe_ops.occnumber

.. autofunction:: deepchem.utils.safe_ops.get_floor_and_ceil

.. autofunction:: deepchem.utils.safe_ops.safe_cdist