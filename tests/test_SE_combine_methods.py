from typing import Union
import numpy as np
import pandas as pd
import pytest
from itertools import chain

from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from biocframe import BiocFrame

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def make_assertions(
    combined, shape, assay_names, rownames, rowData_cols, colnames, colData_cols
):
    assert combined.shape == shape
    assert sorted(list(combined.assays)) == sorted(assay_names)
    assert sorted(combined.rownames) == sorted(rownames)
    assert sorted(combined.rowData.columns.tolist()) == sorted(rowData_cols)
    assert sorted(combined.colnames) == sorted(colnames)
    assert sorted(combined.colData.columns.tolist()) == sorted(colData_cols)


def as_pandas(df: Union[BiocFrame, pd.DataFrame]):
    """Converts a BiocFrame to a pandas DataFrame if necessary.

    Args:
        df (Union[BiocFrame, pd.DataFrame]): a BiocFrame or pd.DataFrame.

    Returns:
        pd.DataFrame: a pandas DataFrame.
    """
    if isinstance(df, BiocFrame):
        return df.to_pandas()
    return df


def repeat_range(r: range, n: int):
    """Repeat the elements in a range multiple times.

    Args:
        r (range): The range to be repeated.
        n (int): The number of times to repeat the range.

    Returns:
        list: A list containing the repeated elements.
    """
    return list(chain.from_iterable([r] * n))


def test_SE_combineCols_unnamed(test_data):
    """Test case to verify combineCols() when the inputs have unnamed rows.
    """
    combined = test_data.se_unnamed.combineCols(test_data.se_unnamed_2, useNames=False)

    # Column data is correctly combined
    assert as_pandas(combined.colData)["A"].equals(
        pd.Series(np.repeat([1, 2], 10), index=repeat_range(range(10), 2))
    )
    assert as_pandas(combined.colData)["B"].equals(
        pd.Series(np.repeat([np.nan, 3], 10), index=repeat_range(range(10), 2))
    )

    # Row data is correctly combined
    assert as_pandas(combined.rowData)["A"].equals(pd.Series([1] * 100))
    assert as_pandas(combined.rowData)["B"].equals(pd.Series(["B"] * 100))

    # Assay data is correctly combined
    assert sorted(list(combined.assays)) == sorted(["counts", "normalized"])
    assert combined.shape == (100, 20)


def test_SE_combineCols_useNames_false(test_data):
    """
    Test case to verify combineCols(..., useNames=False).

    This test case covers functionality of combineCols(..., useNames=False) by
    testing with "SummarizedExperiment" objects of varying shapes and properties.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with same number of rows and same row names.
    2. Test with same number of rows but different row names.
    3. Test with overlapping sample names.
    4. Test with empty rowData and colData.
    5. Test with different number of rows.
    """

    # Scenario 1: same number of rows and same row names
    combined = test_data.se1.combineCols(test_data.se2, useNames=False)

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 2: same number of rows but different row names
    combined = test_data.se2.combineCols(test_data.se3, useNames=False)

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 3: overlapping sample names
    combined = test_data.se4.combineCols(test_data.se6, useNames=False)

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm", "beta"],
        rownames=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"],
        colData_cols=["sample", "disease", "doublet_score", "qual"],
    )

    # Scenario 4: empty rowData and colData
    combined = test_data.se1.combineCols(test_data.se_nonames, useNames=False)

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        colData_cols=["sample", "disease"],
    )

    # Scenario 5: different number of rows
    with pytest.raises(ValueError):
        test_data.se3.combineCols(test_data.se4, useNames=False)


def test_SE_combineCols_useNames_true(test_data):
    """
    Test case to verify combineCols(..., useNames=True).

    This test case covers functionality of combineCols(..., useNames=True) by
    testing with "SummarizedExperiment" objects of varying shapes and properties.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with same number of rows and same row names.
    2. Test with same number of rows but different row names.
    3. Test with different number of rows.
    4. Test with null row name.
    5. Test with duplicated row name.
    6. Test with overlapping sample names.
    7. Test with empty rowData and colData.
    """

    # Scenario 1: same number of rows and same row names
    combined = test_data.se1.combineCols(test_data.se2, useNames=True)

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 2: same number of rows but different row names
    combined = test_data.se2.combineCols(test_data.se3, useNames=True)

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "BRCA2", "MYC", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 3: different number of rows
    combined = test_data.se3.combineCols(test_data.se4, useNames=True)

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm", "beta"],
        rownames=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_7", "cell_8", "cell_9", "cell_10", "cell_11", "cell_12"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # assert se4 samples are non-nan and other entries are 0 for 'beta' assay
    se4_sample_vals = test_data.se4.colnames
    se4_sample_idxs = np.argwhere(combined.colData.index.isin(se4_sample_vals))
    beta_assay = combined.assays["beta"].toarray()
    non_se4_samples = np.delete(beta_assay, se4_sample_idxs, axis=1)

    assert not np.any(non_se4_samples)
    assert not np.isnan(beta_assay[:, se4_sample_idxs].any())

    # Scenario 4: null row name
    with pytest.raises(ValueError):
        test_data.se1.combineCols(test_data.se_null_row_name, useNames=True)

    # Scenario 5: duplicated row name
    with pytest.raises(ValueError):
        test_data.se1.combineCols(test_data.se_duplicated_row_name, useNames=True)

    # Scenario 6: overlapping sample names
    combined = test_data.se4.combineCols(test_data.se6, useNames=True)

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm", "beta"],
        rownames=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"],
        colData_cols=["sample", "disease", "doublet_score", "qual"],
    )

    # Scenario 7: empty rowData and colData
    combined = test_data.se1.combineCols(test_data.se_nonames, useNames=True)

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        colData_cols=["sample", "disease"],
    )


def test_SE_combineCols_mix_sparse_and_dense(test_data):
    """
    Test case to verify combineCols() when assays differ in dtype.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with both dense and sparse arrays.
    """

    # Scenario 1: both dense and sparse arrays
    combined = test_data.se3.combineCols(
        test_data.se4, test_data.se_sparse, useNames=True
    )

    make_assertions(
        combined=combined,
        shape=(7, 9),
        assay_names=["counts", "lognorm", "beta"],
        rownames=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS", "PIK3CA", "HRAS"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=[
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
        colData_cols=["sample", "disease", "doublet_score"],
    )


def test_SE_combineCols_not_all_SE(test_data):
    """
    Test case to verify combineCols() throws an error if not all inputs are
    "SummarizedExperiment" objects.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with an invalid input type.
    """

    # Scenario 1: invalid input type
    with pytest.raises(TypeError):
        test_data.se1.combineCols(pd.DataFrame({"dummy": [1, 2, 3]}))


def test_SE_combineCols_biocframe(test_data):
    """
    Test case to verify combineCols() correctly handles BiocFrames.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test when both `rowData` are of type `BiocFrame` and `useNames=True`.
    2. Test when both `rowData` are of type `BiocFrame` and `useNames=False`.
    3. Test when one `rowData` is a `pd.DataFrame` and the other a `BiocFrame`.
    """

    # Scenario 1: both `rowData` are of type `BiocFrame` and `useNames=True`
    combined = test_data.se_biocframe_1.combineCols(
        test_data.se_biocframe_2, useNames=True
    )

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 2: both `rowData` are of type `BiocFrame` and `useNames=False`
    combined = test_data.se_biocframe_1.combineCols(
        test_data.se_biocframe_2, useNames=False
    )

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        colData_cols=["sample", "disease", "doublet_score"],
    )

    # Scenario 3: Test when one `rowData` is a `pd.DataFrame` and the other a `BiocFrame`.
    combined = test_data.se_biocframe_1.combineCols(test_data.se3, useNames=True)

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm"],
        rownames=["BRCA1", "BRCA2", "HER2", "MYC", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_7", "cell_8", "cell_9"],
        colData_cols=["sample", "disease", "doublet_score"],
    )
