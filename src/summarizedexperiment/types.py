from collections import namedtuple
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from biocframe import BiocFrame
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[np.ndarray, sp.spmatrix]
ArrayTypes = Union[np.ndarray, sp.lil_matrix]
BiocOrPandasFrame = Union[pd.DataFrame, BiocFrame]
MatrixSlicerTypes = Union[List[int], List[bool], slice]
SlicerTypes = Union[List[int], List[bool], List[str], slice]
SlicerArgTypes = Union[Tuple[SlicerTypes], List[SlicerTypes], slice]
SlicerResult = namedtuple(
    "SlicerResult", ["row_data", "col_data", "assays", "row_indices", "col_indices"]
)
