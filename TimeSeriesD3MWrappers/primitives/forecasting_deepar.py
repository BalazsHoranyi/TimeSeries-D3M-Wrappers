import sys
import os
import numpy as np
import pandas as pd
import logging
import collections

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError

from deepar.dataset.time_series import TimeSeries, TimeSeriesTest
from deepar.model.learner import DeepARLearner
import tensorflow as tf
import time
from datetime import timedelta
import typing

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:jeffrey.gleason@yonder.co"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)
#logger.setLevel(logging.INFO)


class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    emb_dim = hyperparams.UniformInt(
        lower=8,
        upper=256,
        default=64,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of cells to use in the categorical embedding component of the model",
    )
    lstm_dim = hyperparams.UniformInt(
        lower=8,
        upper=256,
        default=64,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of cells to use in the lstm component of the model",
    )
    epochs = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of training epochs",
    )
    steps_per_epoch = hyperparams.UniformInt(
        lower=5,
        upper=200,
        default=20,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of steps to do per epoch",
    )
    early_stopping_patience = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=10,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of epochs to wait before invoking early stopping criterion",
    )
    early_stopping_delta = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=5,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="""early stopping will interpret change of < delta in desired direction
            will increment early stopping counter state""",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=1e-2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="learning rate",
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=256,
        default=16,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="batch size",
    )
    dropout_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="dropout to use in lstm model (input and recurrent transform)",
    )
    count_data = hyperparams.Union[typing.Union[bool, None]](
        configuration=collections.OrderedDict(
            user_selected=hyperparams.UniformBool(default = True),
            auto_selected=hyperparams.Hyperparameter[None](default=None),
        ),
        default="auto_selected",
        description="Whether we should label the target column as real or count (positive) \
            based on user input or automatic selection. For example, user might want to specify \
            positive only count data if target column is real-valued, but domain is > 0",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
    )
    window_size = hyperparams.UniformInt(
        lower=10,
        upper=sys.maxsize,
        default=20,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="window size of sampled time series in training process",
    )
    negative_obs = hyperparams.UniformInt(
        lower=0,
        upper=10,
        default=1,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="whether to sample time series with padded observations before t=0 in training",
    )
    val_split = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="""proportion of training records to set aside for validation. Ignored 
            if iterations flag in `fit` method is not None""",
    )
    confidence_interval_horizon = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="horizon for confidence interval forecasts. Exposed through auxiliary \
            'produce_confidence_intervals' method",
    )
    confidence_interval_alpha = hyperparams.Uniform(
        lower=0.01,
        upper=1,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="significance level for confidence interval, i.e. alpha = 0.05 \
            returns a 95%% confdience interval from alpha / 2 to 1 - (alpha / 2) . \
            Exposed through auxiliary 'produce_confidence_intervals' method",
    )
    confidence_interval_samples = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=100,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of samples to draw at each timestep, which will be used to calculate \
            confidence intervals",
    )


class DeepAR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Primitive that applies a deep autoregressive forecasting algorithm for time series
        prediction. The implementation is based off of this paper: https://arxiv.org/pdf/1704.04110.pdf
        and is implemented in AWS's Sagemaker interface.

        Training inputs: 1) Feature dataframe, 2) Target dataframe
        Outputs: Dataframe with predictions for specific time series at specific future time instances 
    
        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object
        
        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "3410d709-0a13-4187-a1cb-159dd24b584b",
            "version": __version__,
            "name": "DeepAR",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": [
                "time series",
                "forecasting",
                "convolutional neural network",
                "autoregressive",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    # Unstructured URIs.
                    "https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers",
                ],
            },
            # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
            # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
            # install a Python package first to be even able to run setup.py of another package. Or you have
            # a dependency which is not on PyPi.
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.7"},
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.time_series_forecasting.convolutional_neural_net.DeepAR",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # set seed for reproducibility
        tf.random.set_seed(random_seed)

        self._is_fit = False
        self._new_train_data = False

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self._params = params

    def _drop_multiple_special_cols(self, col_list, col_type):
        """
            private util function that creates list of duplicated special columns (for deletion)

            Arguments:
                col_list {List[int]} -- list of column indices 
                col_type {str} -- D3M semantic type

            Returns:
                int or None -- first column idx in col_list if any column idxs are marked (else None)
        """

        if len(col_list) == 0:
            return None
        elif len(col_list) > 1:
            logger.warn(
                f"""There are more than one {col_type} marked. This primitive will use the first and drop other {col_type}s."""
            )
            self._drop_cols += col_list[1:]
            if col_type != "target column":
                self._drop_cols_no_tgt += col_list[1:]
        return col_list[0]

    def _get_cols(self, input_metadata):
        """ private util function: get indices of important columns from metadata 

            Arguments:
                input_metadata {D3M Metadata object} -- D3M Metadata object for input frame

            Raises:
                ValueError: If Target column is not of type 'Integer' or 'Float'
        """

        self._drop_cols = []
        self._drop_cols_no_tgt = []

        # get target idx (first column by default)
        target_columns = input_metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        if len(target_columns) == 0:
            raise ValueError("At least one column must be marked as a target")
        self._target_column = self._drop_multiple_special_cols(
            target_columns, "target column"
        )

        # get timestamp idx (first column by default)
        timestamp_columns = input_metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        self._timestamp_column = self._drop_multiple_special_cols(
            timestamp_columns, "timestamp column"
        )

        # get grouping idx and add suggested grouping keys to drop_cols list
        grouping_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/GroupingKey",)
        )
        self._grouping_column = self._drop_multiple_special_cols(
            grouping_columns, "grouping column"
        )
        suggested_grouping_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",)
        )
        self._drop_cols += suggested_grouping_columns
        self._drop_cols_no_tgt += suggested_grouping_columns

        # get index_col (first index column by default)
        index_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/PrimaryKey",)
        )
        self._index_column = self._drop_multiple_special_cols(
            index_columns, "index column"
        )

        # determine whether targets are count data
        target_semantic_types = input_metadata.query_column_field(
            self._target_column, "semantic_types"
        )
        if self.hyperparams['count_data'] is not None:
            self._count_data = self.hyperparams['count_data']
        elif "http://schema.org/Integer" in target_semantic_types:
            self._count_data = True
        elif "http://schema.org/Float" in target_semantic_types:
            self._count_data = False
        else:
            raise ValueError("Target column is not of type 'Integer' or 'Float'")
        logger.info(f"count data: {self._count_data}")

    def _update_indices(self):
        """ private util function: 
            subtract length of drop cols from each marked idx to account for smaller df 
        """

        length = len(self._drop_cols)
        if self._target_column is not None:
            self._target_column -= length
        if self._timestamp_column is not None:
            self._timestamp_column -= length
        if self._grouping_column is not None:
            self._grouping_column -= length
        if self._index_column is not None:
            self._index_column -= length

    def _create_data_object_and_learner(self, val_split):
        """ private util function:
            creates (or updates) train ds object and learner 

            Arguments:
                val_split {float} -- proportion of training data to withhold for validation

        """

        # Create TimeSeries dataset objects
        self._ts_object = TimeSeries(
            self._ts_frame,
            target_idx=self._target_column,
            timestamp_idx=self._timestamp_column,
            grouping_idx=self._grouping_column,
            index_col=self._index_column,
            count_data=self._count_data,
            negative_obs=self.hyperparams["negative_obs"],
            val_split=val_split,
            integer_timestamps=self._integer_timestamps,
        )

        # Create learner
        self._learner = DeepARLearner(
            self._ts_object,
            emb_dim=self.hyperparams["emb_dim"],
            lstm_dim=self.hyperparams["lstm_dim"],
            dropout=self.hyperparams["dropout_rate"],
            lr=self.hyperparams["learning_rate"],
            batch_size=self.hyperparams["batch_size"],
            train_window=self.hyperparams["window_size"],
            verbose=0,
        )

        # save weights so we can restart fitting from scratch (if desired by caller)
        self._learner.save_weights("model_initial_weights.h5")

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
                outputs {Outputs} -- D3M dataframe containing targets
            
            Raises:
                ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """

        # save copy of train data so we don't predict for each row in training
        self._output_columns = outputs.columns
        self._train_data = inputs.copy()

        # combine inputs and outputs for internal TimeSeries object
        self._ts_frame = inputs.append_columns(outputs)

        # Parse cols needed for ts object
        # TODO should only find cols to drop once!
        self._get_cols(self._ts_frame.metadata)

        # Mark time difference (between min and min + 1 timestamp)
        if self._grouping_column is None:
            self._max_train = max(self._ts_frame.iloc[:, self._timestamp_column])
            self._train_diff = int(
                np.diff(np.sort(self._ts_frame.iloc[:, self._timestamp_column]))[0]
            )
        else:
            g_col, t_col = (
                self._ts_frame.columns[self._grouping_column],
                self._ts_frame.columns[self._timestamp_column],
            )
            self._max_train = self._ts_frame.groupby(g_col)[t_col].agg("max")
            # making simplifying assumption that difference is the same across all groups
            self._train_diff = int(
                self._ts_frame.groupby(g_col)[t_col]
                .apply(lambda x: np.diff(np.sort(x))[0])
                .iloc[0]
            )

        # assumption is that integer timestamps are days (treated this way by DeepAR objects)
        timestamp_semantic_types = self._ts_frame.metadata.query_column_field(
            self._timestamp_column, "semantic_types"
        )
        if "http://schema.org/Integer" in timestamp_semantic_types:
            self._integer_timestamps = True
        else:
            self._integer_timestamps = False

        # drop cols if multiple grouping columns
        if len(self._drop_cols) > 0:
            self._ts_frame = self._ts_frame.remove_columns(self._drop_cols)
            self._update_indices()

        # Create TimeSeries dataset object and learner
        self._create_data_object_and_learner(self.hyperparams["val_split"])

        # mark that new training data has been set
        self._new_train_data = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits DeepAR model using training data from set_training_data and hyperparameters
            
            Keyword Arguments:
                timeout {float} -- timeout, considered (default: {None})
                iterations {int} -- iterations, considered (default: {None})
            
            Returns:
                CallResult[None]
        """

        # restore initial model weights if new training data
        if self._new_train_data:

            # only create new dataset object / model (w/out val) if new training data
            if iterations is not None:
                self._create_data_object_and_learner(0)
            self._learner.load_weights("model_initial_weights.h5")

        if iterations is None:
            iterations_set = False
            iterations = self.hyperparams["epochs"]
            validation = self.hyperparams["val_split"] > 0
        else:
            iterations_set = True
            validation = False

        # time training for 1 epoch so we can consider timeout argument thoughtfully
        if timeout:
            logger.info(
                """Timing the fitting procedure for one epoch so we
                can consider timeout thoughtfully"""
            )
            start_time = time.time()
            _, iterations_completed = self._learner.fit(
                validation=validation,
                steps_per_epoch=self.hyperparams["steps_per_epoch"],
                epochs=1,
                stopping_patience=self.hyperparams["early_stopping_patience"],
                stopping_delta=self.hyperparams["early_stopping_delta"],
                tensorboard=False,
            )
            epoch_time_estimate = time.time() - start_time
            # subract 1 for epoch that already happened and 1 more to be safe
            timeout_epochs = timeout // epoch_time_estimate - 2
            iters = min(timeout_epochs, iterations)
        else:
            iters = iterations

        # normal fitting
        logger.info(f"Fitting for {iters} iterations")
        start_time = time.time()

        _, iterations_completed = self._learner.fit(
            validation=validation,
            steps_per_epoch=self.hyperparams["steps_per_epoch"],
            epochs=iters,
            stopping_patience=self.hyperparams["early_stopping_patience"],
            stopping_delta=self.hyperparams["early_stopping_delta"],
            tensorboard=False,
        )
        logger.info(
            f"Fit for {iterations_completed} epochs, took {time.time() - start_time}s"
        )

        # maintain primitive state (mark that training data has been used)
        self._new_train_data = False
        self._is_fit = True

        # use fitting history to set CallResult return values
        if iterations_set:
            has_finished = False
        elif iters < iterations:
            has_finished = False
        else:
            has_finished = self._is_fit

        return CallResult(
            None, has_finished=has_finished, iterations_done=iterations_completed
        )

    @classmethod
    def _discretize_time_difference(
        cls, times, initial_time, time_diff, integer_timestamps=False
    ) -> typing.Sequence[int]:
        """method that discretizes sequence of datetimes (for prediction slices) 
        
            Arguments:
                times {Sequence[datetime]} -- sequence of datetime objects
                initial_time {datetime} -- last datetime instance from training set 
                    (to offset test datetimes)
                time_diff {timedelta} -- difference between last and second to last datetime instances
                    in training set, used to calculate granularity for discretization
            
            Keyword Arguments:
                integer_timestamps {bool} -- whether timestamps are integers or datetime values
            
            Returns:
                typing.Sequence[int] -- prediction intervals expressed at specific time granularity
        """

        SECONDS_PER_MINUTE = 60
        MINUTES_PER_HOUR = 60
        HOURS_PER_DAY = 24
        DAYS_PER_WEEK = 7
        DAYS_PER_MONTH = [30, 31]
        DAYS_PER_YEAR = 365

        s_per_year = (
            SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR
        )
        s_per_month_30 = (
            SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH[0]
        )
        s_per_month_31 = (
            SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH[1]
        )
        s_per_day = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY
        s_per_hr = SECONDS_PER_MINUTE * MINUTES_PER_HOUR

        # edge case for integer timestamps
        if integer_timestamps:
            return [t - initial_time - 1 for t in times]

        # take differences to convert to timedeltas
        time_differences = times - initial_time

        # granularity is years
        if time_diff % s_per_year == 0 and time_diff >= s_per_year:
            logger.debug("granularity is years")
            time_differences = [x / s_per_year for x in time_differences]

        # granularity is months
        elif time_diff % s_per_month_31 == 0 and time_diff >= s_per_month_31:
            logger.debug("granularity is months 31")
            time_differences = [x / s_per_month_31 for x in time_differences]
        elif time_diff % s_per_month_30 == 0 and time_diff >= s_per_month_30:
            logger.debug("granularity is months 30")
            time_differences = [x / s_per_month_30 for x in time_differences]

        # granularity is days
        elif time_diff % s_per_day == 0 and time_diff >= s_per_day:
            logger.debug("granularity is days")
            time_differences = [x / s_per_day for x in time_differences]

        # granularity is hours
        elif time_diff % s_per_hr == 0 and time_diff >= s_per_hr:
            logger.debug("granularity is hours")
            time_differences = [x / s_per_hr for x in time_differences]

        # granularity is seconds
        elif time_diff % SECONDS_PER_MINUTE == 0 and time_diff >= SECONDS_PER_MINUTE:
            logger.debug("granularity is seconds")
            time_differences = [x / SECONDS_PER_MINUTE for x in time_differences]

        # we subtract one from list of differences because we want intervals to be 0 indexed
        return [int(t) - 1 for t in time_differences]

    def _get_pred_intervals(self, df, keep_all=False):
        """ private util function that retrieves unevenly spaced prediction intervals from data frame 

            Arguments:
                df {pandas df} -- df of predictions from which to extract prediction intervals

            Keyword Arguments:
                keep_all {bool} -- if True, take every evenly spaced interval, otherwise only take
                    those given by the test df

            Returns:
                pd Series -- series of intervals, indexed by group, granularity of 1 interval 

        """

        # no grouping column
        if self._grouping_column is None:
            intervals = self._discretize_time_difference(
                df.iloc[:, self._timestamp_column],
                self._max_train,
                self._train_diff,
                self._integer_timestamps,
            )
            return pd.Series([intervals])

        # grouping column
        else:
            g_col, t_col = (
                df.columns[self._grouping_column],
                df.columns[self._timestamp_column],
            )
            all_intervals, groups = [], []
            for (group, vals), max_t in zip(df.groupby(g_col)[t_col], self._max_train):
                if keep_all:
                    vals = vals.interpolate(method="time", limit_direction="both")
                all_intervals.append(
                    self._discretize_time_difference(
                        vals, max_t, self._train_diff, self._integer_timestamps
                    )
                )
                groups.append(group)
            return pd.Series(all_intervals, index=groups)

    def _create_new_test_frame(self, df, pred_intervals, max_t_train, granularity):
        """ private util function that creates new test frame from df and pred_intervals 
            to cover whole horizon 

            Arguments:
                df {pd df} -- df to transform
                pred_intervals {pd Series} -- series of intervals, indexed by group, granularity of 1 interval 
                max_t_train {int or float} -- last value of DateTime column in train df
                granularity {int or float} -- difference between last value and second to last value of DateTime
                    column in train df

            Raises:
                ValueError: if df we are trying to transform has covariates and unevenly 
                    spaced time intervals

            Returns:
                pd df -- new df that covers whole horizon for all series
        """

        # add 1 because 0 indexed
        max_h = max(pred_intervals.apply(lambda x: max(x))) + 1
        max_h = int(max_h)
        min_len = min(pred_intervals.apply(lambda x: len(x)))

        if max_h > min_len:

            # throw error if df we are trying to change has covariates
            if df.shape[1] > 2:
                raise ValueError(
                    """Cannot predict at unevenly spaced time intervals, because we won't
                    have all covariates for all timesteps"""
                )

            dfs = [
                pd.DataFrame(
                    [
                        range(
                            int(max_t_train[grp] + granularity),
                            int(max_t_train[grp] + (max_h + 1) * granularity),
                            granularity,
                        ),
                        [grp] * max_h,
                    ]
                ).T
                for grp in pred_intervals.index
            ]
            new_df = pd.concat(dfs).reset_index(drop=True)
            new_df.columns = df.columns
            return new_df
        else:
            return df

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's predictions for specific time series at specific future time instances
            * these specific timesteps / series are specified implicitly by input dataset

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
            
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit
            
            Returns:
                CallResult[Outputs] -- (N, 2) dataframe with d3m_index and value for each prediction slice requested.
                    prediction slice = specific horizon idx for specific series in specific regression 
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        if len(self._drop_cols_no_tgt) > 0:
            test_frame = inputs.remove_columns(self._drop_cols_no_tgt)
        else:
            test_frame = inputs.copy()

        # training
        if self._train_data.equals(inputs):
            include_all_training = False

        # test
        else:
            include_all_training = True

            # function to get prediction slices
            pred_intervals = self._get_pred_intervals(test_frame)

            # function to update frame (throw error if covariates)
            test_frame = self._create_new_test_frame(
                test_frame, pred_intervals, self._max_train, self._train_diff
            )

        # Create TimeSeriesTest object with saved metadata and train object
        ts_test_object = TimeSeriesTest(
            test_frame,
            self._ts_object,
            timestamp_idx=self._timestamp_column,
            grouping_idx=self._grouping_column,
            index_col=self._index_column,
        )

        # make predictions with learner
        start_time = time.time()
        logger.info(f"Making predictions...")
        preds = self._learner.predict(
            ts_test_object, horizon=None, include_all_training=include_all_training
        )
        logger.info(
            f"Predicting {preds.shape[1]} timesteps into the future took {time.time() - start_time} s"
        )

        # slice predictions with learned intervals for testing frame
        if not self._train_data.equals(inputs):
            all_preds = []
            for p, idxs in zip(preds, pred_intervals.values):
                #all_preds.extend(p[: len(idxs)])  # this takes first n predictions
                all_preds.extend(p[idxs]) # this takes predictions at actual indices
            flat_list = np.array([p for pred_list in all_preds for p in pred_list])
        else:
            flat_list = preds.flatten()

        # fill nans with 0s in case model predicted some
        flat_list = np.nan_to_num(flat_list)

        # create output frame
        result_df = container.DataFrame(
            {self._ts_frame.columns[self._target_column]: flat_list},
            generate_metadata=True,
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
        )

        return CallResult(result_df, has_finished=self._is_fit)

    def produce_confidence_intervals(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ produce confidence intervals for each series 'confidence_interval_horizon' periods into
                the future
        
        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})
        
        Raises:
            PrimitiveNotFittedError: 
        
        Returns:
            CallResult[Outputs] -- 

            Ex. 
                series | timestep | mean | 0.05 | 0.95
                --------------------------------------
                a      |    0     |  5   |   3  |   7
                a      |    1     |  6   |   4  |   8
                b      |    0     |  5   |   3  |   7
                b      |    1     |  6   |   4  |   8
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        horizon = self.hyperparams["confidence_interval_horizon"]
        alpha = self.hyperparams["confidence_interval_alpha"]

        if len(self._drop_cols_no_tgt) > 0:
            test_frame = inputs.remove_columns(self._drop_cols_no_tgt)
        else:
            test_frame = inputs.copy()

        # training
        if self._train_data.equals(inputs):
            include_all_training = False

        # test
        else:
            include_all_training = True

            # function to get prediction slices
            pred_intervals = self._get_pred_intervals(test_frame, keep_all=True)

            # function to update frame (throw error if covariates)
            test_frame = self._create_new_test_frame(
                test_frame, pred_intervals, self._max_train, self._train_diff
            )

        # Create TimeSeriesTest object with saved metadata and train object
        ts_test_object = TimeSeriesTest(
            test_frame,
            self._ts_object,
            timestamp_idx=self._timestamp_column,
            grouping_idx=self._grouping_column,
            index_col=self._index_column,
        )

        # make predictions with learner
        start_time = time.time()
        logger.info(f"Making predictions...")
        preds = self._learner.predict(
            ts_test_object,
            horizon=horizon,
            samples=self.hyperparams["confidence_interval_samples"],
            include_all_training=include_all_training,
        )
        logger.info(
            f"Predicting {preds.shape[1]} timesteps into the future took {time.time() - start_time} s"
        )

        # convert samples to percentiles
        means = np.percentile(preds, 50, axis=2).reshape(-1, 1)
        lowers = np.percentile(preds, alpha / 2 * 100, axis=2).reshape(-1, 1)
        uppers = np.percentile(preds, (1 - alpha / 2) * 100, axis=2).reshape(-1, 1)

        # convert to df
        if self._grouping_column is None:
            indices = np.repeat(self._output_columns[0], horizon)
        else:
            indices = np.repeat(
                test_frame[test_frame.columns[self._grouping_column]].unique(), horizon
            )
        interval_df = pd.DataFrame(
            np.concatenate((means, lowers, uppers), axis=1),
            columns=["mean", str(alpha / 2), str(1 - alpha / 2)],
            index=indices,
        )

        # add index column
        interval_df["horizon_index"] = np.tile(
            np.arange(horizon), len(interval_df.index.unique())
        )

        logger.info(interval_df.head(10))
        # structure return df
        # TODO: add metadata to interval_df??
        return CallResult(
            container.DataFrame(interval_df, generate_metadata=True),
            has_finished=self._is_fit,
        )

