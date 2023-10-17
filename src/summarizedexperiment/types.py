from collections import namedtuple
from typing import List, Tuple, Union

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

MatrixSlicerTypes = Union[List[int], List[bool], slice]
SlicerTypes = Union[List[int], List[bool], List[str], slice]
SlicerArgTypes = Union[Tuple[SlicerTypes], List[SlicerTypes], slice]
SlicerResult = namedtuple(
    "SlicerResult", ["row_data", "col_data", "assays", "row_indices", "col_indices"]
)
