import itertools

import biocutils as ut

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def merge_assays(assays, by):
    if by not in ["row", "column"]:
        raise ValueError("'by' must be either 'row' or 'column'.")

    _all_keys = [list(x.keys()) for x in assays]
    _all_keys = list(set(itertools.chain.from_iterable(_all_keys)))

    _all_assays = {}
    for k in _all_keys:
        _all_mats = [x[k] for x in assays]

        if by == "row":
            _all_assays[k] = ut.combine_rows(*_all_mats)
        else:
            _all_assays[k] = ut.combine_columns(*_all_mats)

    return _all_assays


def check_assays_are_equal(assays):
    _first = assays[0]
    _first_keys = set(list(_first.keys()))

    for x in assays[1:]:
        if (
            len(list(_first_keys - set(x.keys()))) != 0
            or len(list(set(x.keys()) - _first_keys)) != 0
        ):
            raise ValueError("Not all experiments contain the same assays.")
