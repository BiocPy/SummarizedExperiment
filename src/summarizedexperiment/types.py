from collections import namedtuple
from typing import List, Tuple, Union

import numpy as np
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixTypes = Union[np.ndarray, sp.spmatrix]
ArrayTypes = Union[np.ndarray, sp.lil_matrix]
MatrixSlicerTypes = Union[List[int], List[bool], slice]
SlicerTypes = Union[List[int], List[bool], List[str], slice]
SlicerArgTypes = Union[Tuple[SlicerTypes], List[SlicerTypes], slice]

SliceResult = namedtuple(
    "SlicerResult", ["rows", "columns", "assays", "row_indices", "col_indices"]
)
