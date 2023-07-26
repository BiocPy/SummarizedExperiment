from random import random

import genomicranges
import numpy as np
import pandas as pd
import pytest
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"

nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
df_gr = pd.DataFrame(
    {
        "seqnames": [
            "chr1",
            "chr2",
            "chr2",
            "chr2",
            "chr1",
            "chr1",
            "chr3",
            "chr3",
            "chr3",
            "chr3",
        ]
        * 20,
        "starts": range(100, 300),
        "ends": range(110, 310),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 20,
        "score": range(0, 200),
        "GC": [random() for _ in range(10)] * 20,
    }
)

gr = genomicranges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)


def test_SE_subset():
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    subset_tse = tse[0:10, 2:5]
    assert subset_tse is not None
    assert isinstance(subset_tse, SummarizedExperiment)

    assert len(subset_tse.rowData) == 10
    assert len(subset_tse.colData) == 3

    assert subset_tse.assay("counts").shape == (10, 3)


def test_SE_subset_by_name(summarized_experiments):
    # subset by name
    se = summarized_experiments.se1
    subset_se = se[["HER2", "BRCA1"], ["cell_1", "cell_3"]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 2

    assert subset_se.assay("counts").shape == (2, 2)

    # duplicate sample names
    se = summarized_experiments.se_duplicated_sample_name
    subset_se = se[:, ["cell_1"]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 3
    assert len(subset_se.colData) == 1

    assert subset_se.assay("counts").shape == (3, 1)

    # subset with non-existent sample name
    se = summarized_experiments.se1
    with pytest.raises(ValueError):
        subset_se = se[["HER2", "BRCA1", "something random"], ["cell_1", "cell_3"]]


def test_SE_subset_by_name_fails(summarized_experiments):
    # subset by name with some that do not exist
    se = summarized_experiments.se1
    with pytest.raises(Exception):
        subset_se = se[["HER2", "BRCA1", "RAND"], ["cell_1", "cell_3"]]


def test_SE_subset_with_biocframe(summarized_experiments):
    # subset BiocFrame
    se = summarized_experiments.se_biocframe_1
    subset_se = se[["HER2", "BRCA1"], ["cell_1", "cell_3"]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 2

    assert subset_se.assay("counts").shape == (2, 2)


def test_SE_subset_with_bools(summarized_experiments):
    se = summarized_experiments.se1
    subset_se = se[[True, False, True],]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 3

    assert list(subset_se.rowData.columns) == ["seqnames", "start", "end"]
    assert list(subset_se.colData.columns) == ["sample", "disease"]

    assert subset_se.assay("counts").shape == (2, 3)

    subset_se = se[[True, False, False], [True, False, False]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 1
    assert len(subset_se.colData) == 1

    assert subset_se.assay("counts").shape == (1, 1)


def test_SE_subset_biocframe_with_bools(summarized_experiments):
    se = summarized_experiments.se_biocframe_1
    subset_se = se[[True, False, True],]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 3

    assert subset_se.assay("counts").shape == (2, 3)

    subset_se = se[[True, False, False], [True, False, False]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 1
    assert len(subset_se.colData) == 1

    assert subset_se.assay("counts").shape == (1, 1)


def test_SE_subset_biocframe_with_bools_should_fail(summarized_experiments):
    se = summarized_experiments.se_biocframe_1
    with pytest.raises(Exception):
        subset_se = se[[True, False],]


def test_SE_subset_fails_with_indexes(summarized_experiments):
    # subset by invalid indexes
    se = summarized_experiments.se1
    with pytest.raises(Exception):
        subset_se = se["hello world", {"a": [1, 2, 3]}]

    # subset by name when index is not available
    tse = SummarizedExperiment(
        assays={"counts": counts}, rowData=df_gr, colData=colData
    )

    assert tse is not None
    assert isinstance(tse, SummarizedExperiment)

    with pytest.raises(ValueError):
        subset_tse = tse[["0", "1", "2"], ["2", "3"]]


def test_SE_subset_single_indexer_list(summarized_experiments):
    se = summarized_experiments.se1
    subset_se = se[[True, False, True]]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 3

    assert subset_se.assay("counts").shape == (2, 3)


def test_SE_subset_single_indexer_slicer(summarized_experiments):
    se = summarized_experiments.se1
    subset_se = se[0:2]
    assert subset_se is not None
    assert isinstance(subset_se, SummarizedExperiment)

    assert len(subset_se.rowData) == 2
    assert len(subset_se.colData) == 3

    assert subset_se.assay("counts").shape == (2, 3)
