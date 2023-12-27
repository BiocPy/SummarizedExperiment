import itertools

import biocutils as ut
import numpy as np
from biocframe import BiocFrame

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


def relaxed_merge_assays(se, by):
    if by not in ["row", "column"]:
        raise ValueError("'by' must be either 'row' or 'column'.")

    _all_keys = [x.assay_names for x in se]
    _all_keys = list(set(itertools.chain.from_iterable(_all_keys)))

    _all_assays = {}
    for k in _all_keys:
        _all_mats = []
        for x in se:
            _txmat = None
            if k not in x.assay_names:
                _txmat = np.ma.array(
                    np.zeros(shape=x.shape),
                    mask=True,
                )
            else:
                _txmat = x.assays[k]

            _all_mats.append(_txmat)

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
            raise ValueError(
                "Not all experiments contain all the assays, try 'relaxed_combine_*' methods."
            )


def merge_frame_names(frames: BiocFrame):
    all_names = None

    if all(x.row_names is None for x in frames) is True:
        return all_names

    all_names = []
    for f in frames:
        if f.row_names is not None:
            all_names += f.row_names
        else:
            all_names += [""] * len(f)

    return all_names


def merge_se_rownames(expts):
    has_row_names = False
    for expt in expts:
        if expt._row_names is not None:
            has_row_names = True
            break

    _new_row_names = None
    if has_row_names:
        _new_row_names = []
        for expt in expts:
            other_names = expt._row_names
            if other_names is None:
                other_names = [""] * expt.shape[0]
            _new_row_names += other_names

    return _new_row_names


def merge_se_colnames(expts):
    has_col_names = False
    for expt in expts:
        if expt._column_names is not None:
            has_col_names = True
            break

    _new_column_names = None
    if has_col_names:
        _new_column_names = []
        for expt in expts:
            other_names = expt._column_names
            if other_names is None:
                other_names = [""] * expt.shape[0]
            _new_column_names += other_names

    return _new_column_names
