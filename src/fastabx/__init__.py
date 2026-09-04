"""Efficient computation of ABX discriminability."""

from fastabx.accessor import Accessor, Batch, InMemoryAccessor
from fastabx.alignment import Alignment, AlignmentName
from fastabx.bootstrap import Bootstrap
from fastabx.cell import Cell
from fastabx.constraints import Constraints, NoConstraintsError, constraints_all_different
from fastabx.dataset import (
    Dataset,
    EmptyFeaturesError,
    FeaturesSizeError,
    FrequencyTypeError,
    InvalidItemFileError,
    InvalidTimesError,
    NonFiniteError,
    TimesArrayDimensionError,
    TimesArrayFrontiersError,
)
from fastabx.distance import (
    Distance,
    DistanceName,
    IdenticalDistanceDimensionError,
    abx_on_cell,
)
from fastabx.pooling import PooledDataset, PoolingName, PoolingNormalizedError, pool_dataset
from fastabx.score import (
    BootstrapConstraintsError,
    BootstrapSubsampledTaskError,
    CollapseError,
    EmptyScoreError,
    IncompatibleNormalizationError,
    NoBootstrapError,
    Score,
)
from fastabx.stats import InvalidResampleError, NestedBootstrapWarning
from fastabx.subsample import Subsampler
from fastabx.task import Task
from fastabx.utils import InvalidEnvironmentVariableError
from fastabx.verify import (
    DuplicateConditionsError,
    EmptyDataPointsError,
    EmptyDatasetError,
    EmptyTaskError,
    InputTypeError,
    InvalidCellError,
    InvalidDatasetError,
    InvalidFeatureDtypeError,
    InvalidFeaturesError,
    InvalidLevelsError,
    LabelReservedNameError,
    LabelSuffixError,
    MissingLabelError,
    NaNDistanceError,
    NonContiguousIndicesError,
    PrecomputedCellsError,
    UnknownConditionError,
)
from fastabx.zerospeech import InvalidSpeakerOrContextError, MissingMaxXAcrossError, zerospeech_abx

__all__ = [
    "Accessor",
    "Alignment",
    "AlignmentName",
    "Batch",
    "Bootstrap",
    "BootstrapConstraintsError",
    "BootstrapSubsampledTaskError",
    "Cell",
    "CollapseError",
    "Constraints",
    "Dataset",
    "Distance",
    "DistanceName",
    "DuplicateConditionsError",
    "EmptyDataPointsError",
    "EmptyDatasetError",
    "EmptyFeaturesError",
    "EmptyScoreError",
    "EmptyTaskError",
    "FeaturesSizeError",
    "FrequencyTypeError",
    "IdenticalDistanceDimensionError",
    "InMemoryAccessor",
    "IncompatibleNormalizationError",
    "InputTypeError",
    "InvalidCellError",
    "InvalidDatasetError",
    "InvalidEnvironmentVariableError",
    "InvalidFeatureDtypeError",
    "InvalidFeaturesError",
    "InvalidItemFileError",
    "InvalidLevelsError",
    "InvalidResampleError",
    "InvalidSpeakerOrContextError",
    "InvalidTimesError",
    "LabelReservedNameError",
    "LabelSuffixError",
    "MissingLabelError",
    "MissingMaxXAcrossError",
    "NaNDistanceError",
    "NestedBootstrapWarning",
    "NoBootstrapError",
    "NoConstraintsError",
    "NonContiguousIndicesError",
    "NonFiniteError",
    "PooledDataset",
    "PoolingName",
    "PoolingNormalizedError",
    "PrecomputedCellsError",
    "Score",
    "Subsampler",
    "Task",
    "TimesArrayDimensionError",
    "TimesArrayFrontiersError",
    "UnknownConditionError",
    "abx_on_cell",
    "constraints_all_different",
    "pool_dataset",
    "zerospeech_abx",
]
