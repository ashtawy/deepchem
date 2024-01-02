"""
Basic molecular features.
"""

import logging
import os
import re
import select
import subprocess
import sys
import time
import uuid
from typing import Any, Dict, List, Union

import numpy as np

from deepchem.feat.base_classes import ParallelComplexFeaturizer
from deepchem.feat.complex_featurizers.binana import binana
from deepchem.feat.complex_featurizers.complex_cyscore_featurizer import \
    ComplexCyscoreFeaturizer
from deepchem.feat.complex_featurizers.complex_dmpnn_featurizer import \
    ComplexDMPNNFeaturizer
from deepchem.feat.complex_featurizers.complex_rfscore_featurizer import \
    ComplexRFScoreFeaturizer
from deepchem.feat.complex_featurizers.complex_smina_featurizer import \
    ComplexSminaFeaturizer
from deepchem.feat.complex_featurizers.complex_xscore_featurizer import \
    ComplexXScoreFeaturizer

logger = logging.getLogger(__name__)

FEATURIZERS_PATH = "/software/featurizers/"
SMINA_PATH = "/software/smina/smina.static"

def isNumber(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def parse_interface(interfaceFname):
    interfaceFile = open(interfaceFname,'r')
    env_vars = os.environ
    if('PATH' not in env_vars):
        env_vars['PATH'] = ''
    ddb_path = os.path.dirname(os.path.realpath(__file__))

    env_vars['PATH'] = env_vars['PATH'] + ":" + ddb_path + '/utils/mgltools/bin'
    env_vars['PATH'] = env_vars['PATH'] + ":" + ddb_path + '/utils/mgltools/MGLToolsPckgs/AutoDockTools/Utilities24'

    if('PYTHONPATH' not in env_vars):
        env_vars['PYTHONPATH'] = ''

    if('LD_LIBRARY_PATH' not in env_vars):
        env_vars['LD_LIBRARY_PATH'] = ''

    cml_args = {}; 
    cntr = 1
    for line in interfaceFile:
        line = line.strip()
        tokens = line.split()
        if(len(tokens)==3 and tokens[0]!='#' and tokens[0]!='Order' and isNumber(tokens[0])):
            if(tokens[0]=='0'):
                var_name = tokens[1]
                var_value = tokens[2]
                if(var_name=='PATH' and var_value!=''):
                    var_value = env_vars['PATH'] + ':' + var_value
                if(var_name=='PYTHONPATH' and var_value!=''):
                    var_value = env_vars['PYTHONPATH'] + ':' + var_value
                if(var_name=='LD_LIBRARY_PATH' and var_value!=''):
                    var_value = env_vars['LD_LIBRARY_PATH'] + ':' + var_value
                env_vars[var_name] = var_value
            else:
                cml_args[cntr] = {'flag':tokens[1],'value':tokens[2]}
                cntr+=1
    interfaceFile.close()
    return([env_vars,cml_args])

def get_interface(toolName):
    interface_path = os.path.join(FEATURIZERS_PATH, toolName, 'interface.txt')
    if(os.path.exists(interface_path)):
        interface = parse_interface(interface_path)
    else:
        logging.error(f"ERROR: unable to access interface file: {interface_path}")
    return(interface)

def process_env_vars(toolName,raw_env_vars):
    tool_path = os.path.join(FEATURIZERS_PATH, toolName)
    env_vars = raw_env_vars

    for key in raw_env_vars:
        value = raw_env_vars[key]
        if('tool' in value.lower()):
            csiv = re.compile(re.escape('{tool}'),re.IGNORECASE)
            new_value = csiv.sub(tool_path,value)
            env_vars[key] = new_value
            #print(key, env_vars[key];)
    return env_vars

def convert_protein_format(prtn_name, new_value):
    subprocess.call(["obabel",prtn_name,f"-O{new_value}"])

def process_cml_args(toolName, prtn_name, ligand_name, oFileName, raw_cml_args):
    tool_path =  os.path.join(FEATURIZERS_PATH, toolName)
    toolBin_path = tool_path + '/bin/' +toolName
    lFileNameBase = ligand_name.split('.')[0]
    pFileNameBase = prtn_name.split('.')[0]

    outPrefixPlaceHolder = '{outputprefix}/'


    if(not os.path.exists(toolBin_path)):
        print("ERROR: unable to access the tool executable file: ")
        #print(toolBin_path)
        sys.exit()
    cml_args = [toolBin_path]

    for i in range(1,len(raw_cml_args)+1):
        iValue = raw_cml_args[i]
        if(iValue['flag'].lower()!= 'null'):
            cml_args.append(iValue['flag'])
        if(iValue['value'].lower()!= 'null' and iValue['value'][0]!='{'):
            cml_args.append(iValue['value'])
        if(iValue['value'].lower()!= 'null' and iValue['value'][0]=='{'):
            new_value = iValue['value']
            if('{tool}' in iValue['value'].lower()):
                csiv = re.compile(re.escape('{tool}'),re.IGNORECASE)
                new_value = csiv.sub(tool_path,iValue['value'])
            if('{receptor}' in iValue['value'].lower()):
                csiv = re.compile(re.escape('{receptor}'),re.IGNORECASE)
                new_value = csiv.sub(pFileNameBase, iValue['value'])
                if not os.path.exists(new_value):
                    subprocess.call(["obabel",prtn_name,f"-O{new_value}"])
            if('{protein}' in iValue['value'].lower()):
                csiv = re.compile(re.escape('{protein}'),re.IGNORECASE)
                new_value = csiv.sub(pFileNameBase,iValue['value']) 
                if not os.path.exists(new_value):
                    subprocess.call(["obabel",prtn_name,f"-O{new_value}"])
            if('{ligand}' in iValue['value'].lower()):
                csiv = re.compile(re.escape('{ligand}'),re.IGNORECASE)
                new_value = csiv.sub(lFileNameBase,iValue['value'])   
                if not os.path.exists(new_value):
                    subprocess.call(["obabel",ligand_name,f"-O{new_value}"])
            if(outPrefixPlaceHolder in iValue['value'].lower()):  
                outputFileCsv = oFileName
                new_value = oFileName
            cml_args.append(new_value)
    return([cml_args,outputFileCsv])

class MetaComplexDescriptors(ParallelComplexFeaturizer):
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
    >>> datapoints = [("/path/to/protein1.pdb", "/path/to/ligand1.pdb"), ("/path/to/protein2.pdb", "/path/to/ligand2.pdb")]
    >>> featurizer = dc.feat.MetaComplexDescriptors(descriptor_sets={"xscore": None, "rfscore": None})
    >>> features = featurizer.featurize(datapoints)
    >>> type(features[0])
    <class 'numpy.ndarray'>
    >>> features[0].shape
    (208,)
    """

    def __init__(
        self,
        descriptor_sets: Dict[str, Any] = {"dcdmpnn": {"distance_threshold": 6, 
                                                      "distance_step_size": 0.25, 
                                                      "discrete_distance_filter":True}},
        n_threads=None,
    ):
        """Initialize this featurizer.

        Parameters
        ----------
        descriptor_sets: Dict, optional (default True)
          A dictionary with the feature types to calcualte along with their configurations
        """
        super().__init__(n_threads=n_threads)
        def_params = {
            "xscore": {},
            "rfscore": {},
            "cyscore": {},
            "binana": {},
            "dcdmpnn": {"distance_threshold": 6, "distance_step_size": 0.25, "discrete_distance_filter":True},
            "ccdmpnn": {"distance_threshold": 6, "distance_step_size": 0.25, "discrete_distance_filter":False},
            "smina": {"smina_executable_path": SMINA_PATH},

        }
        str2cls_map = {
            "binana": binana,
            "dcdmpnn": ComplexDMPNNFeaturizer,
            "ccdmpnn": ComplexDMPNNFeaturizer,
            "smina": ComplexSminaFeaturizer,
            "xscore": ComplexXScoreFeaturizer,
            "rfscore": ComplexRFScoreFeaturizer,
            "cyscore": ComplexCyscoreFeaturizer,
        }
        self.descriptors = []
        self.featurizers = {}
        self.dtype = np.float32
        for ftype, fparams in descriptor_sets.items():
            ftype = ftype.lower()
            if ftype in str2cls_map:
                if fparams is None:
                    fparams = def_params[ftype]
                self.featurizers[ftype] = str2cls_map[ftype](**fparams)
            else:
                raise ValueError(
                    f"Feature type of {ftype} is not supported. Supported types are {list(str2cls_map)}"
                )
    def calc_descriptors(self, descriptor_type, protein_path, ligand_path):
        pid = os.getpid()
        pipe_name = f'/tmp/my_pipe_{pid}_{uuid.uuid4()}'
        os.mkfifo(pipe_name)

        try:
            interface = get_interface(descriptor_type)
            raw_env_vars = interface[0]
            raw_cml_args = interface[1]
            env_vars = process_env_vars(descriptor_type, raw_env_vars)
            cml_args, _ = process_cml_args(descriptor_type, protein_path, ligand_path, pipe_name, raw_cml_args)

            subprocess.Popen(cml_args, env=env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            output = ""
            # Open the pipe in non-blocking mode
            pipe_fd = os.open(pipe_name, os.O_RDONLY | os.O_NONBLOCK)
            with os.fdopen(pipe_fd, 'r') as pipe:
                # Set a timeout in seconds
                timeout = 15

                # Initialize the end time
                end_time = time.time() + timeout

                # Wait for data to become available on the pipe or for the timeout
                while True:
                    current_time = time.time()
                    if current_time >= end_time:
                        logging.error(f"Timeout reached, no data for {timeout} seconds when running {descriptor_type}"
                                      f" on {protein_path} and {ligand_path}")
                        break

                    timeout_remaining = end_time - current_time
                    ready, _, _ = select.select([pipe], [], [], timeout_remaining)
                    if ready:
                        data = os.read(pipe_fd, 65536)  # Read in chunks
                        if not data:
                            break
                        output += data.decode('utf-8')
                    else:
                        logging.error(f"Timeout reached, no data for {timeout} seconds when running {descriptor_type}"
                                      f" on {protein_path} and {ligand_path}")
                        break

            return None if output == "" else np.array(output.split(',')).astype(np.float32)
        finally:
            # Clean up: remove the named pipe
            os.remove(pipe_name)
    
    def _featurize(self, datapoint, **kwargs) -> np.ndarray:
        """
        Calculate RDKit descriptors.

        Parameters
        ----------
        datapoint: protein_path, ligand_path
          RDKit Mol object

        Returns
        -------
        np.ndarray
          1D array of RDKit descriptors for `mol`.
          The length is `len(self.descriptors)`.
        """
        features: List[Union[int, float]] = []
        graph_features = None
        for featurizer_name, featurizer in self.featurizers.items():
            if featurizer_name == "binana":
                try:
                    features_subset = featurizer.featurize(datapoint[0], datapoint[1], "\t", False)
                    features_subset = list(features_subset.features.values())
                except Exception as e:
                    message = (f"Error when running binana on datapoint: {datapoint} {e}")
                    logger.error(message)
                    raise ValueError(message)
                    features_subset = None
            elif featurizer_name in ["dcdmpnn", "ccdmpnn", "smina", "xscore", "rfscore", "cyscore"]:
                features_subset = featurizer.featurize([datapoint])[0]
            else:
                features_subset = self.calc_descriptors(featurizer_name, datapoint[0], datapoint[1])

            if features_subset is not None:
                # and (isinstance() and np.isnan(features_subset).sum() == 0):
                if featurizer_name in ["ccdmpnn", "dcdmpnn"]:
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
