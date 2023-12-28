from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
import biocutils
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_SE_relaxed_combine_rows(summarized_experiments):
    with pytest.raises(Exception):
        combined = biocutils.combine_rows(
            summarized_experiments.se_unnamed, summarized_experiments.se_unnamed_2
        )

    combined = biocutils.relaxed_combine_rows(
        summarized_experiments.se_unnamed, summarized_experiments.se_unnamed_2
    )

    assert combined is not None
    assert isinstance(combined, SummarizedExperiment)
    assert combined.shape == (200, 10)
    assert set(combined.assay_names).issubset(["counts", "normalized"])
    assert list(combined.row_data.column_names) == ["A", "B"]
    assert list(combined.column_data.column_names) == ["A"]


def test_SE_combine_rows_with_names_mixed(summarized_experiments):
    combined = biocutils.combine_rows(
        summarized_experiments.se1, summarized_experiments.se3
    )

    assert combined is not None
    assert isinstance(combined, SummarizedExperiment)
    assert combined.shape == (6, 3)
    assert set(combined.assay_names).issubset(["counts", "lognorm"])
    assert list(combined.row_data.column_names) == ["seqnames", "start", "end"]
    assert list(combined.column_data.column_names) == [
        "sample",
        "disease",
        "doublet_score",
    ]
    assert combined.row_names is not None
    assert len(combined.row_names) == 6
    assert combined.column_names is not None
    assert len(combined.column_names) == 3

    combined = biocutils.relaxed_combine_rows(
        summarized_experiments.se1, summarized_experiments.se3
    )

    assert combined is not None
    assert isinstance(combined, SummarizedExperiment)
    assert combined.shape == (6, 3)
    assert set(combined.assay_names).issubset(["counts", "lognorm"])
    assert list(combined.row_data.column_names) == ["seqnames", "start", "end"]
    assert list(combined.column_data.column_names) == [
        "sample",
        "disease",
        "doublet_score",
    ]
    assert combined.row_names is not None
    assert len(combined.row_names) == 6
    assert combined.column_names is not None
    assert len(combined.column_names) == 3


def test_SE_both_combine_rows_with_names(summarized_experiments):
    combined = biocutils.combine_rows(
        summarized_experiments.se1, summarized_experiments.se2
    )

    assert combined is not None
    assert isinstance(combined, SummarizedExperiment)
    assert combined.shape == (6, 3)
    assert set(combined.assay_names).issubset(["counts", "lognorm"])
    assert list(combined.row_data.column_names) == ["seqnames", "start", "end"]
    assert list(combined.column_data.column_names) == [
        "sample",
        "disease",
        "doublet_score",
    ]
    assert combined.row_names is not None
    assert len(combined.row_names) == 6
    assert combined.column_names is not None
    assert len(combined.column_names) == 3

    combined = biocutils.relaxed_combine_rows(
        summarized_experiments.se1, summarized_experiments.se2
    )

    assert combined is not None
    assert isinstance(combined, SummarizedExperiment)
    assert combined.shape == (6, 3)
    assert set(combined.assay_names).issubset(["counts", "lognorm"])
    assert len(combined.row_data.column_names) > 2
    assert len(combined.column_data.column_names) > 2
    assert combined.row_names is not None
    assert combined.column_names is not None
