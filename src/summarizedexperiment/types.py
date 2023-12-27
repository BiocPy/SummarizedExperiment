from collections import namedtuple


__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

SliceResult = namedtuple(
    "SlicerResult",
    [
        "rows",
        "columns",
        "assays",
        "row_names",
        "column_names",
        "row_indices",
        "col_indices",
    ],
)
