from typing import Sequence, Tuple

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
    target_assay_names: Sequence[str],
    target_row_data: pd.DataFrame,
    target_col_data: pd.DataFrame,
):
    """Check if the SummarizedExperiment object matches the expected output.

    Args:
        se (SummarizedExperiment): the SummarizedExperiment object to be checked.
        target_shape (Tuple[int, int]): the expected shape of the SummarizedExperiment.
        target_assay_names (Sequence[str]): the expected assay names of the SummarizedExperiment.
        target_row_data pd.DataFrame: the expected row_data of the SummarizedExperiment.
        target_col_data pd.DataFrame: the expected col_data of the SummarizedExperiment.
    """
    assert se.shape == target_shape
    assert sorted(list(se.assays)) == sorted(target_assay_names)
    assert se.row_data.equals(target_row_data)
    assert se.col_data.equals(target_col_data)


def test_SE_combine_cols_unnamed(summarized_experiments):
    """Test case to verify combine_cols() when the inputs have unnamed rows."""
    combined = summarized_experiments.se_unnamed.combine_cols(
        summarized_experiments.se_unnamed_2, use_names=False
    )

    checkIdentical(
        se=combined,
        target_shape=(100, 20),
        target_assay_names=["counts", "normalized"],
        target_row_data=pd.DataFrame(data={"A": [1] * 100, "B": ["B"] * 100}),
        target_col_data=pd.DataFrame(
            data={
                "A": np.repeat([1, 2], 10),
                "B": np.repeat([np.nan, 3], 10),
            },
            index=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * 2,
        ),
    )


def test_SE_combine_cols_use_names_false(summarized_experiments):
    """Test case to verify combine_cols(..., use_names=False).

    Test Scenarios:
    1. Test with same number of rows and same row names.
    2. Test with same number of rows but different row names.
    3. Test with overlapping sample names.
    4. Test with empty row_data and col_data.
    5. Test with different number of rows.
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combine_cols(
        summarized_experiments.se2, use_names=False
    )

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

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combine_cols(
        summarized_experiments.se3, use_names=False
    )

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
                "sample": ["SAM_4", "SAM_5", "SAM_6", "SAM_7", "SAM_8", "SAM_9"],
                "disease": ["True", "False", "True", "True", "False", "False"],
                "doublet_score": [0.05, 0.23, 0.54, 0.15, 0.62, 0.18],
            },
            index=["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"],
        ),
    )

    # Scenario 3: overlapping sample names
    combined = summarized_experiments.se4.combine_cols(
        summarized_experiments.se6, use_names=False
    )

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm", "beta"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_5", "chr_1", "chr_9", "chr_3"],
                "start": [700, 500, 100, 900, 300],
                "end": [710, 510, 110, 910, 310],
            },
            index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_10", "SAM_11", "SAM_12", "SAM_10", "SAM_11", "SAM_12"],
                "disease": ["True", "False", "False", "True", "False", "False"],
                "doublet_score": [0.15, 0.62, 0.18, np.nan, np.nan, np.nan],
                "qual": [np.nan, np.nan, np.nan, 0.95, 0.92, 0.98],
            },
            index=["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"],
        ),
    )

    # Scenario 4: empty row_data and col_data
    combined = summarized_experiments.se1.combine_cols(
        summarized_experiments.se_nonames, use_names=False
    )

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
                "sample": ["SAM_1", "SAM_2", "SAM_3", np.nan, np.nan, np.nan],
                "disease": ["True", "True", "True", np.nan, np.nan, np.nan],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        ),
    )

    # Scenario 5: different number of rows
    with pytest.raises(ValueError):
        summarized_experiments.se3.combine_cols(
            summarized_experiments.se4, use_names=False
        )


def test_SE_combine_cols_use_names_true(summarized_experiments):
    """Test case to verify combine_cols(..., use_names=True).

    This test case covers functionality of combine_cols(..., use_names=True) by
    testing with "SummarizedExperiment" objects of varying shapes and properties.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combine_cols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with same number of rows and same row names.
    2. Test with same number of rows but different row names.
    3. Test with different number of rows.
    4. Test with null row name.
    5. Test with duplicated row name.
    6. Test with overlapping sample names.
    7. Test with empty row_data and col_data.
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combine_cols(
        summarized_experiments.se2, use_names=True
    )

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

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combine_cols(
        summarized_experiments.se3, use_names=True
    )

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2", np.nan, np.nan],
                "start": [500.0, 300.0, 200.0, np.nan, np.nan],
                "end": [510.0, 310.0, 210.0, np.nan, np.nan],
            },
            index=["HER2", "BRCA1", "TPFK", "MYC", "BRCA2"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_4", "SAM_5", "SAM_6", "SAM_7", "SAM_8", "SAM_9"],
                "disease": ["True", "False", "True", "True", "False", "False"],
                "doublet_score": [0.05, 0.23, 0.54, 0.15, 0.62, 0.18],
            },
            index=["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"],
        ),
    )

    # Scenario 3: different number of rows
    combined = summarized_experiments.se3.combine_cols(
        summarized_experiments.se4, use_names=True
    )

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm", "beta"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_1", "chr_9", np.nan, np.nan],
                "start": [700.0, 100.0, 900.0, np.nan, np.nan],
                "end": [710.0, 110.0, 910.0, np.nan, np.nan],
            },
            index=["MYC", "BRCA2", "TPFK", "BRCA1", "GSS"],
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

    # assert se4 samples are non-nan and other entries are 0 for 'beta' assay
    se4_sample_vals = summarized_experiments.se4.colnames
    se4_sample_idxs = np.argwhere(combined.col_data.index.isin(se4_sample_vals))
    beta_assay = combined.assays["beta"].toarray()
    non_se4_samples = np.delete(beta_assay, se4_sample_idxs, axis=1)

    assert not np.any(non_se4_samples)
    assert not np.isnan(beta_assay[:, se4_sample_idxs].any())

    # Scenario 4: null row name
    row_data_null_row_name = pd.DataFrame(
        {
            "seqnames": ["chr_5", "chr_3", "chr_2"],
            "start": [500, 300, 200],
            "end": [510, 310, 210],
        },
        index=[None, "BRCA1", "TPFK"],
    )
    se_null_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        row_data=row_data_null_row_name,
        col_data=summarized_experiments.col_data1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        summarized_experiments.se1.combine_cols(se_null_row_name, use_names=True)

    # Scenario 5: duplicated row name
    row_data_duplicated_row_name = pd.DataFrame(
        {
            "seqnames": ["chr_5", "chr_3", "chr_2"],
            "start": [500, 300, 200],
            "end": [510, 310, 210],
        },
        index=["HER2", "HER2", "TPFK"],
    )
    se_duplicated_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        row_data=row_data_duplicated_row_name,
        col_data=summarized_experiments.col_data1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        summarized_experiments.se1.combine_cols(se_duplicated_row_name, use_names=True)

    # Scenario 6: overlapping sample names
    combined = summarized_experiments.se4.combine_cols(
        summarized_experiments.se6, use_names=True
    )

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm", "beta"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_5", "chr_1", "chr_9", "chr_3"],
                "start": [700, 500, 100, 900, 300],
                "end": [710, 510, 110, 910, 310],
            },
            index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_10", "SAM_11", "SAM_12", "SAM_10", "SAM_11", "SAM_12"],
                "disease": ["True", "False", "False", "True", "False", "False"],
                "doublet_score": [0.15, 0.62, 0.18, np.nan, np.nan, np.nan],
                "qual": [np.nan, np.nan, np.nan, 0.95, 0.92, 0.98],
            },
            index=["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"],
        ),
    )

    # Scenario 7: empty row_data and col_data
    combined = summarized_experiments.se1.combine_cols(
        summarized_experiments.se_nonames, use_names=True
    )

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
                "sample": ["SAM_1", "SAM_2", "SAM_3", np.nan, np.nan, np.nan],
                "disease": ["True", "True", "True", np.nan, np.nan, np.nan],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        ),
    )


def test_SE_combine_cols_mix_sparse_and_dense(summarized_experiments):
    """Test case to verify combine_cols() when assays differ in dtype.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combine_cols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with both dense and sparse arrays.
    """

    # Scenario 1: both dense and sparse arrays
    combined = summarized_experiments.se3.combine_cols(
        summarized_experiments.se4, summarized_experiments.se_sparse, use_names=True
    )

    checkIdentical(
        se=combined,
        target_shape=(7, 9),
        target_assay_names=["counts", "lognorm", "beta"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_1", "chr_9", np.nan, np.nan, np.nan, np.nan],
                "start": [700.0, 100.0, 900.0, np.nan, np.nan, np.nan, np.nan],
                "end": [710.0, 110.0, 910.0, np.nan, np.nan, np.nan, np.nan],
            },
            index=["MYC", "BRCA2", "TPFK", "BRCA1", "GSS", "PIK3CA", "HRAS"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": [
                    "SAM_7",
                    "SAM_8",
                    "SAM_9",
                    "SAM_10",
                    "SAM_11",
                    "SAM_12",
                    "SAM_13",
                    "SAM_14",
                    "SAM_15",
                ],
                "disease": [
                    "True",
                    "False",
                    "False",
                    "True",
                    "False",
                    "False",
                    "True",
                    "True",
                    "True",
                ],
                "doublet_score": [0.15, 0.62, 0.18, 0.15, 0.62, 0.18, 0.32, 0.51, 0.09],
            },
            index=[
                "cell_7",
                "cell_8",
                "cell_9",
                "cell_10",
                "cell_11",
                "cell_12",
                "cell_13",
                "cell_14",
                "cell_15",
            ],
        ),
    )


def test_SE_combine_cols_not_all_SE(summarized_experiments):
    """Test case to verify combine_cols() throws an error if not all inputs are "SummarizedExperiment" objects.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combine_cols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with one object as a pandas DataFrame
    """

    # Scenario 1: one object as a pandas DataFrame
    with pytest.raises(TypeError):
        summarized_experiments.se1.combine_cols(pd.DataFrame({"dummy": [1, 2, 3]}))


def test_SE_combine_cols_biocframe(summarized_experiments):
    """Test case to verify combine_cols() correctly handles BiocFrames.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combine_cols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test when both `row_data` are of type `BiocFrame` and `use_names=True`.
    2. Test when both `row_data` are of type `BiocFrame` and `use_names=False`.
    3. Test when one `row_data` is a `pd.DataFrame` and the other a `BiocFrame`.
    """

    # Scenario 1: both `row_data` are of type `BiocFrame` and `use_names=True`
    combined = summarized_experiments.se_biocframe_1.combine_cols(
        summarized_experiments.se_biocframe_2, use_names=True
    )

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

    # Scenario 2: both `row_data` are of type `BiocFrame` and `use_names=False`
    combined = summarized_experiments.se_biocframe_1.combine_cols(
        summarized_experiments.se_biocframe_2, use_names=False
    )

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

    # Scenario 3: Test when one `row_data` is a `pd.DataFrame` and the other a `BiocFrame`.
    combined = summarized_experiments.se_biocframe_1.combine_cols(
        summarized_experiments.se3, use_names=True
    )

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm"],
        target_row_data=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2", np.nan, np.nan],
                "start": [500.0, 300.0, 200.0, np.nan, np.nan],
                "end": [510.0, 310.0, 210.0, np.nan, np.nan],
            },
            index=["HER2", "BRCA1", "TPFK", "MYC", "BRCA2"],
        ),
        target_col_data=pd.DataFrame(
            data={
                "sample": ["SAM_1", "SAM_2", "SAM_3", "SAM_7", "SAM_8", "SAM_9"],
                "disease": ["True", "True", "True", "True", "False", "False"],
                "doublet_score": [np.nan, np.nan, np.nan, 0.15, 0.62, 0.18],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_7", "cell_8", "cell_9"],
        ),
    )
