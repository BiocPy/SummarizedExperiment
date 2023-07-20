import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


rowData1 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [10293804, 12098948, 20984392],
        "end": [28937947, 3872839, 329837492]
    },
    index=["HER2", "BRCA1", "TPFK"],
)
colData1 = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_3", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
se1 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData1,
    colData=colData1,
    metadata={"seq_type": "paired"},
)

rowData2 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [10293804, 12098948, 20984392],
        "end": [28937947, 3872839, 329837492]
    },
    index=["HER2", "BRCA1", "TPFK"],
)
colData2 = pd.DataFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [.05, .23, .54]
    },
    index=["cell_4", "cell_5", "cell_6"],
)
se2 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData2,
    colData=colData2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowData3 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_1", "chr_Y"],
        "start": [1084390, 1874937, 243879798],
        "end": [243895239, 358908298, 390820395]
    },
    index=["MYC", "BRCA2", "TPFK"],
)
colData3 = pd.DataFrame(
    {
        "sample": ["SAM_7", "SAM_8", "SAM_9"],
        "disease": ["True", "False", "False"],
        "doublet_score": [.15, .62, .18]
    },
    index=["cell_7", "cell_8", "cell_9"],
)
se3 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData3,
    colData=colData3,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowData4 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_1", "chr_Y", "chr_3"],
        "start": [1084390, 1273987, 1874937, 243879798, 2217981273],
        "end": [243895239, 128973192, 358908298, 390820395, 1987238927]
    },
    index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
)
colData4 = pd.DataFrame(
    {
        "sample": ["SAM_7", "SAM_8", "SAM_9"],
        "disease": ["True", "False", "False"],
        "doublet_score": [.15, .62, .18]
    },
    index=["cell_10", "cell_11", "cell_12"],
)
se4 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(5, 3))},
    rowData=rowData4,
    colData=colData4,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)


def test_SE_combineCols_useNames_false():
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
    """

    # Scenario 1: same number of rows and same row names
    combined = se1.combineCols(se2, useNames=False)

    assert combined.shape == (3, 6)

    assert all(
        row_name in combined.rowData.index.tolist()
        for row_name in ["HER2", "BRCA1", "TPFK"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colData.index.tolist()
        for col_name in ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 2: same number of rows but different row names 
    combined = se2.combineCols(se3, useNames=False)

    assert combined.shape == (3, 6)

    assert all(
        row_name in combined.rowData.index.tolist()
        for row_name in ["HER2", "BRCA1", "TPFK"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colData.index.tolist()
        for col_name in ["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 3: different number of rows
    with pytest.raises(ValueError):
        se3.combineCols(se4, useNames=False)


def test_SE_combineCols_useNames_true():
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
    combined = se1.combineCols(se2, useNames=True)

    assert combined.shape == (3, 6)

    assert all(
        row_name in combined.rowData.index.tolist()
        for row_name in ["HER2", "BRCA1", "TPFK"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colData.index.tolist()
        for col_name in ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 2: same number of rows but different row names
    combined = se2.combineCols(se3, useNames=True)

    assert combined.shape == (5, 6)

    assert all(
        row_name in combined.rowData.index.tolist()
        for row_name in ["HER2", "BRCA1", "BRCA2", "MYC", "TPFK"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colData.index.tolist()
        for col_name in ["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"]
    )

    assert all(
        col_name in combined.colData.columns.tolist()
        for col_name in ["sample", "disease", "doublet_score"]
    )

    # Scenario 3: different number of rows
    combined = se3.combineCols(se4, useNames=True)

    assert combined.shape == (5, 6)

    assert all(
        row_name in combined.rowData.index.tolist()
        for row_name in ["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"]
    )

    assert all(
        col_name in combined.rowData.columns.tolist()
        for col_name in ["seqnames", "start", "end"]
    )

    assert all(
        col_name in combined.colData.index.tolist()
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
            "end": [28937947, 3872839, 329837492]
        },
        index=[None, "BRCA1", "TPFK"],
    )
    se_null_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        rowData=rowData_null_row_name,
        colData=colData1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        se1.combineCols(se_null_row_name, useNames=True)

    # Scenario 5: duplicated row name
    rowData_duplicated_row_name = pd.DataFrame(
        {
            "seqnames": ["chr_5", "chr_3", "chr_2"],
            "start": [10293804, 12098948, 20984392],
            "end": [28937947, 3872839, 329837492]
        },
        index=["HER2", "HER2", "TPFK"],
    )
    se_duplicated_row_name = SummarizedExperiment(
        assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
        rowData=rowData_duplicated_row_name,
        colData=colData1,
        metadata={"seq_type": "paired"},
    )

    with pytest.raises(ValueError):
        se1.combineCols(se_duplicated_row_name, useNames=True)


def test_SE_combineCols_not_all_SE():
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
        se1.combineCols(pd.DataFrame({"dummy": [1, 2, 3]}))
