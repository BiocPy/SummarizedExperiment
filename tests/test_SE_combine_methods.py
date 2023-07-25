from typing import Union, MutableMapping, Sequence, Tuple
import numpy as np
import pandas as pd
import pytest
from itertools import chain

from summarizedexperiment.SummarizedExperiment import SummarizedExperiment
from biocframe import BiocFrame

__author__ = "keviny2, jkanche"
__copyright__ = "keviny2"
__license__ = "MIT"


def make_assertions(
    combined,
    shape,
    assay_names,
    rownames,
    rowData_cols,
    colnames,
    colData_cols,
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


def checkIdentical(
    se: SummarizedExperiment,
    target_shape: Tuple[int, int],
    target_assay_names: Sequence[str],
    target_rowData: pd.DataFrame,
    target_colData: pd.DataFrame,
):
    """Check if the SummarizedExperiment object matches the expected output.

    Args:
        se (SummarizedExperiment): the SummarizedExperiment object to be checked.
        target_shape (Tuple[int, int]): the expected shape of the SummarizedExperiment.
        target_assay_names (Sequence[str]): the expected assay names of the SummarizedExperiment.
        target_rowData pd.DataFrame: the expected rowData of the SummarizedExperiment.
        target_colData pd.DataFrame: the expected colData of the SummarizedExperiment.
    """
    assert se.shape == target_shape
    assert sorted(list(se.assays)) == sorted(target_assay_names)
    assert se.rowData.equals(target_rowData)
    assert se.colData.equals(target_colData)


def test_SE_combineCols_unnamed(summarized_experiments):
    """Test case to verify combineCols() when the inputs have unnamed rows."""
    combined = summarized_experiments.se_unnamed.combineCols(summarized_experiments.se_unnamed_2, useNames=False)

    checkIdentical(
        se=combined,
        target_shape=(100, 20),
        target_assay_names=["counts", "normalized"],
        target_rowData=pd.DataFrame(data={"A": [1] * 100, "B": ["B"] * 100}),
        target_colData=pd.DataFrame(
            data={
                "A": np.repeat([1, 2], 10),
                "B": np.repeat([np.nan, 3], 10),
            },
            index=repeat_range(range(10), 2),
        ),
    )


def test_SE_combineCols_useNames_false(summarized_experiments):
    """
    Test case to verify combineCols(..., useNames=False).

    Test Scenarios:
    1. Test with same number of rows and same row names.
    2. Test with same number of rows but different row names.
    3. Test with overlapping sample names.
    4. Test with empty rowData and colData.
    5. Test with different number of rows.
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combineCols(summarized_experiments.se2, useNames=False)

    checkIdentical(
        se=combined,
        target_shape=(3, 6),
        target_assay_names=["counts", "lognorm"],
        target_rowData=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2"],
                "start": [500, 300, 200],
                "end": [510, 310, 210],
            },
            index=["HER2", "BRCA1", "TPFK"],
        ),
        target_colData=pd.DataFrame(
            data={
                "sample": ["SAM_1", "SAM_2", "SAM_3", "SAM_4", "SAM_5", "SAM_6"],
                "disease": ["True", "True", "True", "True", "False", "True"],
                "doublet_score": [np.nan, np.nan, np.nan, 0.05, 0.23, 0.54],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"],
        ),
    )

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combineCols(summarized_experiments.se3, useNames=False)

    checkIdentical(
        se=combined,
        target_shape=(3, 6),
        target_assay_names=["counts", "lognorm"],
        target_rowData=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2"],
                "start": [500, 300, 200],
                "end": [510, 310, 210],
            },
            index=["HER2", "BRCA1", "TPFK"],
        ),
        target_colData=pd.DataFrame(
            data={
                "sample": ["SAM_4", "SAM_5", "SAM_6", "SAM_7", "SAM_8", "SAM_9"],
                "disease": ["True", "False", "True", "True", "False", "False"],
                "doublet_score": [0.05, 0.23, 0.54, 0.15, 0.62, 0.18],
            },
            index=["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"],
        ),
    )

    # Scenario 3: overlapping sample names
    combined = summarized_experiments.se4.combineCols(summarized_experiments.se6, useNames=False)

    checkIdentical(
        se=combined,
        target_shape=(5, 6),
        target_assay_names=["counts", "lognorm", "beta"],
        target_rowData=pd.DataFrame(
            data={
                "seqnames": ["chr_7", "chr_5", "chr_1", "chr_9", "chr_3"],
                "start": [700, 500, 100, 900, 300],
                "end": [710, 510, 110, 910, 310],
            },
            index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
        ),
        target_colData=pd.DataFrame(
            data={
                "sample": ["SAM_10", "SAM_11", "SAM_12", "SAM_10", "SAM_11", "SAM_12"],
                "disease": ["True", "False", "False", "True", "False", "False"],
                "doublet_score": [0.15, 0.62, 0.18, np.nan, np.nan, np.nan],
                "qual": [np.nan, np.nan, np.nan, 0.95, 0.92, 0.98],
            },
            index=["cell_10", "cell_11", "cell_12", "cell_10", "cell_11", "cell_12"],
        ),
    )

    # Scenario 4: empty rowData and colData
    combined = summarized_experiments.se1.combineCols(summarized_experiments.se_nonames, useNames=False)

    checkIdentical(
        se=combined,
        target_shape=(3, 6),
        target_assay_names=["counts", "lognorm"],
        target_rowData=pd.DataFrame(
            data={
                "seqnames": ["chr_5", "chr_3", "chr_2"],
                "start": [500, 300, 200],
                "end": [510, 310, 210],
            },
            index=["HER2", "BRCA1", "TPFK"],
        ),
        target_colData=pd.DataFrame(
            data={
                "sample": ["SAM_1", "SAM_2", "SAM_3", np.nan, np.nan, np.nan],
                "disease": ["True", "True", "True", np.nan, np.nan, np.nan],
            },
            index=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        ),
    )

    # Scenario 5: different number of rows
    with pytest.raises(ValueError):
        summarized_experiments.se3.combineCols(summarized_experiments.se4, useNames=False)


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
    6. Test with overlapping sample names.
    7. Test with empty rowData and colData.
    """

    # Scenario 1: same number of rows and same row names
    combined = summarized_experiments.se1.combineCols(
        summarized_experiments.se2, useNames=True
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

    # Scenario 2: same number of rows but different row names
    combined = summarized_experiments.se2.combineCols(
        summarized_experiments.se3, useNames=True
    )

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
    combined = summarized_experiments.se3.combineCols(
        summarized_experiments.se4, useNames=True
    )

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
    se4_sample_vals = summarized_experiments.se4.colnames
    se4_sample_idxs = np.argwhere(combined.colData.index.isin(se4_sample_vals))
    beta_assay = combined.assays["beta"].toarray()
    non_se4_samples = np.delete(beta_assay, se4_sample_idxs, axis=1)

    assert not np.any(non_se4_samples)
    assert not np.isnan(beta_assay[:, se4_sample_idxs].any())

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

    # Scenario 6: overlapping sample names
    combined = summarized_experiments.se4.combineCols(
        summarized_experiments.se6, useNames=True
    )

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
    combined = summarized_experiments.se1.combineCols(
        summarized_experiments.se_nonames, useNames=True
    )

    make_assertions(
        combined=combined,
        shape=(3, 6),
        assay_names=["counts", "lognorm"],
        rownames=["HER2", "BRCA1", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_1", "cell_2", "cell_3"],
        colData_cols=["sample", "disease"],
    )


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
        summarized_experiments.se4, summarized_experiments.se_sparse, useNames=True
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


def test_SE_combineCols_biocframe(summarized_experiments):
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
    combined = summarized_experiments.se_biocframe_1.combineCols(
        summarized_experiments.se_biocframe_2, useNames=True
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
    combined = summarized_experiments.se_biocframe_1.combineCols(
        summarized_experiments.se_biocframe_2, useNames=False
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
    combined = summarized_experiments.se_biocframe_1.combineCols(
        summarized_experiments.se3, useNames=True
    )

    make_assertions(
        combined=combined,
        shape=(5, 6),
        assay_names=["counts", "lognorm"],
        rownames=["BRCA1", "BRCA2", "HER2", "MYC", "TPFK"],
        rowData_cols=["seqnames", "start", "end"],
        colnames=["cell_1", "cell_2", "cell_3", "cell_7", "cell_8", "cell_9"],
        colData_cols=["sample", "disease", "doublet_score"],
    )
