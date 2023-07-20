import numpy as np
import pandas as pd
import pytest

from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def test_SE_combineCols_useNames_false(summarized_experiments):
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
    3. Test with different number of rows.
    4. Test with overlapping sample names.
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combineCols(
        summarized_experiments.se2, useNames=False
    )

    assert combined.shape == (3, 6)

    assert all(assay_name in combined.assays for assay_name in ["counts", "lognorm"])

    assert all(row_name in combined.rownames for row_name in ["HER2", "BRCA1", "TPFK"])

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combineCols(
        summarized_experiments.se3, useNames=False
    )

    assert combined.shape == (3, 6)

    assert all(assay_name in combined.assays for assay_name in ["counts", "lognorm"])

    assert all(row_name in combined.rownames for row_name in ["HER2", "BRCA1", "TPFK"])

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in ["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 3: different number of rows
    with pytest.raises(ValueError):
        summarized_experiments.se3.combineCols(
            summarized_experiments.se4, useNames=False
        )

    # Scenario 4: overlapping sample names
    combined = summarized_experiments.se4.combineCols(summarized_experiments.se6, useNames=True)

    assert combined.shape == (5, 6)

    assert all(
        assay_name in combined.assays for assay_name in ["counts", "lognorm", "beta"]
    )

    assert all(
        row_name in combined.rownames
        for row_name in ["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert sorted(combined.colnames) == sorted(
        ["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score", "qual"]
    )


def test_SE_combineCols_useNames_true(summarized_experiments):
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
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combineCols(
        summarized_experiments.se2, useNames=True
    )

    assert combined.shape == (3, 6)

    assert all(assay_name in combined.assays for assay_name in ["counts", "lognorm"])

    assert all(row_name in combined.rownames for row_name in ["HER2", "BRCA1", "TPFK"])

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combineCols(
        summarized_experiments.se3, useNames=True
    )

    assert combined.shape == (5, 6)

    assert all(assay_name in combined.assays for assay_name in ["counts", "lognorm"])

    assert all(
        row_name in combined.rownames
        for row_name in ["HER2", "BRCA1", "BRCA2", "MYC", "TPFK"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in ["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 3: different number of rows
    combined = summarized_experiments.se3.combineCols(
        summarized_experiments.se4, useNames=True
    )

    assert combined.shape == (5, 6)

    assert all(
        assay_name in combined.assays for assay_name in ["counts", "lognorm", "beta"]
    )

    # assert se4 samples are non-nan and other entries are 0 for 'beta' assay
    se4_sample_vals = summarized_experiments.se4.colnames
    se4_sample_idxs = np.argwhere(combined.colData.index.isin(se4_sample_vals))
    beta_assay = combined.assays["beta"].toarray()
    non_se4_samples = np.delete(beta_assay, se4_sample_idxs, axis=1)

    assert not np.any(non_se4_samples)
    assert not np.isnan(beta_assay[:, se4_sample_idxs].any())

    assert all(
        row_name in combined.rownames
        for row_name in ["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in ["cell_7", "cell_8", "cell_9", "cell_10", "cell_11", "cell_12"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 4: null row name
    rowData_null_row_name = pd.DataFrame(
        {
            "seqnames": ["chr_5", "chr_3", "chr_2"],
            "start": [10293804, 12098948, 20984392],
            "end": [28937947, 3872839, 329837492],
        },
        index=[None, "BRCA1", "TPFK"],
    )
    se_null_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        rowData=rowData_null_row_name,
        colData=summarized_experiments.colData1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        summarized_experiments.se1.combineCols(se_null_row_name, useNames=True)

    # Scenario 5: duplicated row name
    rowData_duplicated_row_name = pd.DataFrame(
        {
            "seqnames": ["chr_5", "chr_3", "chr_2"],
            "start": [10293804, 12098948, 20984392],
            "end": [28937947, 3872839, 329837492],
        },
        index=["HER2", "HER2", "TPFK"],
    )
    se_duplicated_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        rowData=rowData_duplicated_row_name,
        colData=summarized_experiments.colData1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        summarized_experiments.se1.combineCols(se_duplicated_row_name, useNames=True)


def test_SE_combineCols_mix_sparse_and_dense(summarized_experiments):
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
    combined = summarized_experiments.se3.combineCols(
        summarized_experiments.se4, summarized_experiments.se5, useNames=True
    )

    assert combined.shape == (7, 9)

    assert all(
        row_name in combined.rownames
        for row_name in ["MYC", "BRCA1", "BRCA2", "TPFK", "GSS", "PIK3CA", "HRAS"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colnames
        for col_name in [
            "cell_7",
            "cell_8",
            "cell_9",
            "cell_10",
            "cell_11",
            "cell_12",
            "cell_13",
            "cell_14",
            "cell_15",
        ]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )


def test_SE_combineCols_not_all_SE(summarized_experiments):
    """
    Test case to verify combineCols() throws an error if not all inputs are
    "SummarizedExperiment" objects.

    Test Steps:
    1. Set up the "SummarizedExperiment" inputs.
    2. Invoke combineCols() with the inputs.
    3. Assert the expected output.

    Test Scenarios:
    1. Test with one object as a pandas DataFrame
    """

    # Scenario 1: one object as a pandas DataFrame
    with pytest.raises(TypeError):
        summarized_experiments.se1.combineCols(pd.DataFrame({"dummy": [1, 2, 3]}))
