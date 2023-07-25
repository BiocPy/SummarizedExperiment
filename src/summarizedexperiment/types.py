from collections import namedtuple
from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd
from biocframe import BiocFrame
from filebackedarray import H5BackedDenseData, H5BackedSparseData
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[np.ndarray, sp.spmatrix, H5BackedSparseData, H5BackedDenseData]
ArrayTypes = Union[np.ndarray, sp.lil_matrix]
BiocOrPandasFrame = Union[pd.DataFrame, BiocFrame]
MatrixSlicerTypes = Union[Sequence[int], Sequence[bool], slice]
SlicerTypes = Union[Sequence[int], Sequence[bool], Sequence[str], slice]
SlicerArgTypes = Union[Tuple[SlicerTypes], Sequence[SlicerTypes], slice]
SlicerResult = namedtuple(
    "SlicerResult", ["rowData", "colData", "assays", "rowIndices", "colIndices"]
)
