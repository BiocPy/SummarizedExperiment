import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


rowData1 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["elem_1", "elem_2", "elem_3"]},
    index=["HER2", "BRCA1", "TPFK"],
)
colData1 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["dat_1", "dat_2", "dat_3"]},
    index=["cell_1", "cell_2", "cell_3"],
)
se1 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData1,
    colData=colData1,
    metadata={"query": "SOME awesome QUERY"},
)

rowData2 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["elem_1", "elem_2", "elem_3"]},
    index=["MYC", "BRCA2", "GSS"],
)
colData2 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["dat_1", "dat_2", "dat_3"]},
    index=["cell_4", "cell_5", "cell_6"],
)
se2 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData2,
    colData=colData2,
    metadata={"gene_list": "MYC, BRCA2, GSS"},
)

rowData3 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["elem_1", "elem_2", "elem_3"]},
    index=["MYC", "BRCA2", "GSS"],
)
colData3 = pd.DataFrame(
    {"meta1": ["val_1", "val_2", "val_3"], "meta2": ["dat_1", "dat_2", "dat_3"]},
    index=["cell_7", "cell_8", "cell_9"],
)
se3 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    rowData=rowData3,
    colData=colData3,
)

rowData4 = pd.DataFrame(
    {
        "meta1": ["val_1", "val_2", "val_3", "val_4"],
        "meta2": ["elem_1", "elem_2", "elem_3", "elem_4"],
    },
    index=["MYC", "BRCA2", "GSS", "TPFK"],
)
colData4 = pd.DataFrame(
    {
        "meta1": ["val_1", "val_2", "val_3"],
        "meta2": ["dat_1", "dat_2", "dat_3"],
    },
    index=["cell_7", "cell_8", "cell_9"],
)
se4 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(4, 3))},
    rowData=rowData4,
    colData=colData4,
)

rowData5 = pd.DataFrame(
    {
        "meta1": ["val_1", "val_2", "val_3", "val_4"],
        "meta2": ["elem_1", "elem_2", "elem_3", "elem_4"],
    },
    index=["MYC", "BRCA2", "MYC", "TPFK"],
)
colData5 = pd.DataFrame(
    {
        "meta1": ["val_1", "val_2", "val_3"],
        "meta2": ["dat_1", "dat_2", "dat_3"],
    },
    index=["cell_7", "cell_8", "cell_9"],
)
se5 = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(4, 3))},
    rowData=rowData5,
    colData=colData5,
)


def test_SE_combineCols():
    combined_true = se1.combineCols(se2, use_names=True)

    assert combined_true.shape == (6, 6)

    assert all(
        row_name in combined_true.rowData.index.tolist()
        for row_name in ["HER2", "MYC", "GSS", "TPFK", "BRCA1", "BRCA2"]
    )

    assert all(
        col_name in combined_true.colData.index.tolist()
        for col_name in ["cell_1", "cell_2", "cell_3", "cell_4", "cell_5", "cell_6"]
    )

    combined_false = se2.combineCols(se3, use_names=False)

    assert combined_false.shape == (3, 6)

    assert all(
        row_name in combined_false.rowData.index.tolist()
        for row_name in ["MYC", "BRCA2", "GSS"]
    )

    assert all(
        col_name in combined_false.colData.index.tolist()
        for col_name in ["cell_4", "cell_5", "cell_6", "cell_7", "cell_8", "cell_9"]
    )

    with pytest.raises(TypeError):
        se1.combineCols(pd.DataFrame({"dummy": ["val_1", "val_2", "val_3"]}))

    with pytest.raises(ValueError):
        se1.combineCols(se4)

    with pytest.raises(ValueError):
        se4.combineCols(se5)
