import sys
import os.path
import numpy as np
import pandas
import typing
from typing import List

import hdbscan
from sklearn.cluster import DBSCAN

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame
from .timeseries_formatter import TimeSeriesFormatterPrimitive

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.dataset.Dataset
Outputs = container.dataset.Dataset

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'HDBSCAN', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['DBSCAN', 'HDBSCAN'],
        description = 'type of clustering algorithm to use')
    eps = hyperparams.Uniform(lower=0, upper=sys.maxsize, default = 0.5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'maximum distance between two samples for them to be considered as in the same neigborhood, \
        used in DBSCAN algorithm')
    min_cluster_size = hyperparams.UniformInt(lower=2, upper=sys.maxsize, default = 5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'the minimum size of clusters')  
    min_samples = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default = 5, semantic_types = 
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'The number of samples in a neighbourhood for a point to be considered a core point.')   
    long_format = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether the input dataset is already formatted in long format or not")
    pass

class Hdbscan(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies Hierarchical Density-Based Clustering or Density-Based Clustering 
        algorithms to time series data. This is an unsupervised, clustering primitive, but has been
        representend as a supervised classification problem to produce a compliant primitive. 

        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "ca014488-6004-4b54-9403-5920fbe5a834",
        'version': __version__,
        'name': "hdbscan",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [
             {
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'cython',
                'version': '0.29.7',
             },
             {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
             ),
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.clustering.hdbscan.Hdbscan',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.DBSCAN,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        hp_class = TimeSeriesFormatterPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
        self._hp = hp_class.defaults().replace({'file_col_index':1, 'main_resource_index':'learningData'})
        self.train_sorted = None

        if self.hyperparams['algorithm'] == 'HDBSCAN':
            self.clf = hdbscan.HDBSCAN(min_cluster_size=self.hyperparams['min_cluster_size'],min_samples=self.hyperparams['min_samples'])
        else:
            self.clf = DBSCAN(eps=self.hyperparams['eps'],min_samples=self.hyperparams['min_samples'])
    
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits Kmeans clustering algorithm using training data from set_training_data and hyperparameters
        '''
        # sort training labels to match clusters by size in produce method
        self.train_sorted = pandas.Series(self._y_train).value_counts().index
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: numpy ndarray of size (number_of_time_series, time_series_length) containing training time series
        
        '''
        if not self.hyperparams['long_format']:
            inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
            inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)

        # load and reshape training data
        # 'series_id' and 'value' should be set by metadata
        n_ts = len(inputs.d3mIndex.unique())
        ts_sz = int(inputs.shape[0] / n_ts)
        self._X_train = np.array(inputs.value).reshape(n_ts, ts_sz)
        self._y_train = inputs.label.iloc[::ts_sz]

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : numpy ndarray of size (number_of_time_series, time_series_length) containing new time series 

        Returns
        ----------
        Outputs
            The output is a dataframe containing a single column where each entry is the associated series' cluster number.
        """

        # temporary (until Uncharted adds conversion primitive to repo)
        if not self.hyperparams['long_format']:
            inputs = TimeSeriesFormatterPrimitive(hyperparams = self._hp).produce(inputs = inputs).value['0']
        else:
            hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
            ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
            inputs = d3m_DataFrame(ds2df_client.produce(inputs = inputs).value)

        # parse values from output of time series formatter
        n_ts = len(inputs.d3mIndex.unique())
        ts_sz = int(inputs.shape[0] / n_ts)
        input_vals = np.array(inputs.value).reshape(n_ts, ts_sz)

        # cluster training and test series to produce labels
        values = np.concatenate((self._X_train, input_vals), axis=0)
        preds = self.clf.fit_predict(values)
        preds_sorted = pandas.Series(preds).value_counts().index.drop(-1)
        preds_overflow = preds_sorted[len(self.train_sorted):]
        print(preds_sorted, file = sys.__stdout__)
        labels = [preds_sorted[0] if (pred < 0 or pred in preds_overflow) else pred for pred in preds[self._X_train.shape[0]:]]
        labels = [self.train_sorted[np.where(preds_sorted == p)[0][0]] for p in labels]

        # add metadata to output
        labels = pandas.DataFrame(labels)
        out_df= pandas.concat([pandas.DataFrame(inputs.d3mIndex.unique()), labels], axis = 1)
        # get column names from metadata
        out_df.columns = ['d3mIndex', 'label']
        hdbscan_df = d3m_DataFrame(out_df)
        
        # first column ('d3mIndex')
        col_dict = dict(hdbscan_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        # confirm that this metadata still exists
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        hdbscan_df.metadata = hdbscan_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        
        # second column ('labels')
        col_dict = dict(hdbscan_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        #index = inputs['0'].metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        #col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['name'] = 'label'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', 'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
        hdbscan_df.metadata = hdbscan_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        return CallResult(hdbscan_df)

if __name__ == '__main__':

    # Load data and preprocessing
    hyperparams_class = Hdbscan.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    hdbscan_client = Hdbscan(hyperparams=hyperparams_class.defaults())
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/66_chlorineConcentration/TEST/dataset_TEST/datasetDoc.json')
    results = hdbscan_client.produce(inputs = test_dataset)
    print(results.value)
