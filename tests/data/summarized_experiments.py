import pandas as pd
import numpy as np
import scipy.sparse as sp
from biocframe import BiocFrame
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment


ncols = 10
nrows = 100
se_unnamed = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=10, size=(nrows, ncols))}
)
se_unnamed.col_data["A"] = [1] * ncols
se_unnamed.row_data["A"] = [1] * nrows

se_unnamed_2 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
        "normalized": np.random.normal(size=(nrows, ncols)),
    }
)
se_unnamed_2.col_data["A"] = [2] * ncols
se_unnamed_2.col_data["B"] = [3] * ncols
se_unnamed_2.row_data["B"] = ["B"] * nrows

row_data1 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
col_data1 = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_2", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
se1 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data1,
    col_data=col_data1,
    metadata={"seq_type": "paired"},
)

row_data2 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
col_data2 = pd.DataFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [0.05, 0.23, 0.54],
    },
    index=["cell_4", "cell_5", "cell_6"],
)
se2 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data2,
    col_data=col_data2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

row_data3 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_1", "chr_9"],
        "start": [700, 100, 900],
        "end": [710, 110, 910],
    },
    index=["MYC", "BRCA2", "TPFK"],
)
col_data3 = pd.DataFrame(
    {
        "sample": ["SAM_7", "SAM_8", "SAM_9"],
        "disease": ["True", "False", "False"],
        "doublet_score": [0.15, 0.62, 0.18],
    },
    index=["cell_7", "cell_8", "cell_9"],
)
se3 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data3,
    col_data=col_data3,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

row_data4 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_1", "chr_9", "chr_3"],
        "start": [700, 500, 100, 900, 300],
        "end": [710, 510, 110, 910, 310],
    },
    index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
)
col_data4 = pd.DataFrame(
    {
        "sample": ["SAM_10", "SAM_11", "SAM_12"],
        "disease": ["True", "False", "False"],
        "doublet_score": [0.15, 0.62, 0.18],
    },
    index=["cell_10", "cell_11", "cell_12"],
)
se4 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(5, 3)),
        "lognorm": np.random.lognormal(size=(5, 3)),
        "beta": np.random.beta(a=1, b=1, size=(5, 3)),
    },
    row_data=row_data4,
    col_data=col_data4,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

row_data5 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_4", "chr_9", "chr_8"],
        "start": [700, 500, 400, 900, 800],
        "end": [710, 510, 410, 910, 810],
    },
    index=["MYC", "BRCA1", "PIK3CA", "TPFK", "HRAS"],
)
col_data5 = pd.DataFrame(
    {
        "sample": ["SAM_13", "SAM_14", "SAM_15"],
        "disease": ["True", "True", "True"],
        "doublet_score": [0.32, 0.51, 0.09],
    },
    index=["cell_13", "cell_14", "cell_15"],
)
se_sparse = SummarizedExperiment(
    assays={
        "counts": sp.lil_matrix(np.random.poisson(lam=7, size=(5, 3))),
        "lognorm": sp.lil_matrix(np.random.lognormal(size=(5, 3))),
        "beta": sp.lil_matrix(np.random.beta(a=2, b=1, size=(5, 3))),
    },
    row_data=row_data5,
    col_data=col_data5,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

col_data6 = pd.DataFrame(
    {
        "sample": ["SAM_10", "SAM_11", "SAM_12"],
        "disease": ["True", "False", "False"],
        "qual": [0.95, 0.92, 0.98],
    },
    index=["cell_10", "cell_11", "cell_12"],
)
se6 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(5, 3)),
        "lognorm": np.random.lognormal(size=(5, 3)),
        "beta": np.random.beta(a=1, b=1, size=(5, 3)),
    },
    row_data=row_data4,
    col_data=col_data6,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

row_data_nonames = pd.DataFrame(
    {},
    index=["HER2", "BRCA1", "TPFK"],
)
col_data_nonames = pd.DataFrame(
    {},
    index=["cell_1", "cell_2", "cell_3"],
)
se_nonames = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data_nonames,
    col_data=col_data_nonames,
    metadata={},
)

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
    col_data=col_data1,
    metadata={"seq_type": "paired"},
)

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
    col_data=col_data1,
    metadata={"seq_type": "paired"},
)

col_data_duplicated_sample_name = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_1", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    index=["cell_1", "cell_1", "cell_2"],
)
se_duplicated_sample_name = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    row_data=row_data1,
    col_data=col_data_duplicated_sample_name,
    metadata={"seq_type": "paired"},
)

row_data_biocframe_1 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    row_names=["HER2", "BRCA1", "TPFK"],
)
col_data_biocframe_1 = BiocFrame(
    {
        "sample": ["SAM_1", "SAM_2", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    row_names=["cell_1", "cell_2", "cell_3"],
)
se_biocframe_1 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data_biocframe_1,
    col_data=col_data_biocframe_1,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

row_data_biocframe_2 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    row_names=["HER2", "BRCA1", "TPFK"],
)
col_data_biocframe_2 = BiocFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [0.05, 0.23, 0.54],
    },
    row_names=["cell_4", "cell_5", "cell_6"],
)
se_biocframe_2 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=7, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=row_data_biocframe_2,
    col_data=col_data_biocframe_2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)
