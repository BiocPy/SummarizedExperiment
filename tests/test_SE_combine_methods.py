from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def checkIdentical(
    se: SummarizedExperiment,
    target_shape: Tuple[int, int],
    target_assay_names: List[str],
    target_row_data: pd.DataFrame,
    target_col_data: pd.DataFrame,
):
    """Check if the SummarizedExperiment object matches the expected output.

    Args:
        se (SummarizedExperiment): the SummarizedExperiment object to be checked.
        target_shape (Tuple[int, int]): the expected shape of the SummarizedExperiment.
        target_assay_names (List[str]): the expected assay names of the SummarizedExperiment.
        target_row_data pd.DataFrame: the expected row_data of the SummarizedExperiment.
        target_col_data pd.DataFrame: the expected col_data of the SummarizedExperiment.
    """
    assert se.shape == target_shape
    assert sorted(list(se.assays)) == sorted(target_assay_names)
    assert se.row_data.shape == target_row_data.shape
    assert se.col_data.shape == target_col_data.shape


def test_basic_combine_cols(summarized_experiments):
    combined = summarized_experiments.se1.combine_cols(summarized_experiments.se2)

    checkIdentical(
        se=combined,
        target_shape=(3, 6),
        target_assay_names=["counts", "lognorm"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2"],
                "start": [500, 300, 200],
                "end": [510, 310, 210],
            },
            index=["HER2", "BRCA1", "TPFK"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_1", "SAM_2", "SAM_3", "SAM_4", "SAM_5", "SAM_6"],
                "disease": ["True", "True", "True", "True", "False", "True"],
                "doublet_score": [np.nan, np.nan, np.nan, 0.05, 0.23, 0.54],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        ),
    )


def test_basic_combine_cols_mismatch_assays_should_fail(summarized_experiments):
    with pytest.raises(Exception):
        combined = summarized_experiments.se1.combine_cols(summarized_experiments.se4)


def test_basic_combine_cols_mismatch_assays_flexible(summarized_experiments):
    summarized_experiments.se2.assays["beta"] = np.empty((3, 3))

    combined = summarized_experiments.se1.combine_cols(
        summarized_experiments.se2, fill_missing_assay=True
    )

    checkIdentical(
        se=combined,
        target_shape=(3, 6),
        target_assay_names=["counts", "lognorm", "beta"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_1", "chr_9"],
                "start": [700.0, 100.0, 900.0],
                "end": [710.0, 110.0, 910.0],
            },
            index=["MYC", "BRCA2", "TPFK"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_7", "SAM_8", "SAM_9", "SAM_10", "SAM_11", "SAM_12"],
                "disease": ["True", "False", "False", "True", "False", "False"],
                "doublet_score": [0.15, 0.62, 0.18, 0.15, 0.62, 0.18],
            },
            index=["cell_7", "cell_8", "cell_9", "cell_10", "cell_11", "cell_12"],
        ),
    )
