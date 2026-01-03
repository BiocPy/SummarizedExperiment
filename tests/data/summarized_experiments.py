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
se_unnamed.coldata["A"] = [1] * ncols
se_unnamed.rowdata["A"] = [1] * nrows

se_unnamed_2 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=10, size=(nrows, ncols)),
        "normalized": np.random.normal(size=(nrows, ncols)),
    }
)
se_unnamed_2.coldata["A"] = [2] * ncols
se_unnamed_2.coldata["B"] = [3] * ncols
se_unnamed_2.rowdata["B"] = ["B"] * nrows

rowdata1 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata1 = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_2", "SAM_3"],
        "disease": ["True", "True", "True"],
        "doublet_score": [0.15, 0.62, 0.18],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
se1 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata1,
    column_data=coldata1,
    row_names=["HER2", "BRCA1", "TPFK"],
    column_names=["cell_1", "cell_2", "cell_3"],
    metadata={"seq_type": "paired"},
)

rowdata2 = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "BRCA1", "TPFK"],
)
coldata2 = pd.DataFrame(
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
    row_data=rowdata2,
    column_data=coldata2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowdata3 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_1", "chr_9"],
        "start": [700, 100, 900],
        "end": [710, 110, 910],
    },
    index=["MYC", "BRCA2", "TPFK"],
)
coldata3 = pd.DataFrame(
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
    row_data=rowdata3,
    column_data=coldata3,
    row_names=["MYC", "BRCA2", "TPFK"],
    column_names=["cell_7", "cell_8", "cell_9"],
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowdata4 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_1", "chr_9", "chr_3"],
        "start": [700, 500, 100, 900, 300],
        "end": [710, 510, 110, 910, 310],
    },
    index=["MYC", "BRCA1", "BRCA2", "TPFK", "GSS"],
)
coldata4 = pd.DataFrame(
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
    row_data=rowdata4,
    column_data=coldata4,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowdata5 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_4", "chr_9", "chr_8"],
        "start": [700, 500, 400, 900, 800],
        "end": [710, 510, 410, 910, 810],
    },
    index=["MYC", "BRCA1", "PIK3CA", "TPFK", "HRAS"],
)
coldata5 = pd.DataFrame(
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
    row_data=rowdata5,
    column_data=coldata5,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

coldata6 = pd.DataFrame(
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
    row_data=rowdata4,
    column_data=coldata6,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowdata_nonames = pd.DataFrame(
    {},
    index=["HER2", "BRCA1", "TPFK"],
)
coldata_nonames = pd.DataFrame(
    {},
    index=["cell_1", "cell_2", "cell_3"],
)
se_nonames = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3)),
    },
    row_data=rowdata_nonames,
    column_data=coldata_nonames,
    metadata={},
)

rowdata_null_row_name = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=[None, "BRCA1", "TPFK"],
)
se_null_row_name = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    row_data=rowdata_null_row_name,
    column_data=coldata1,
    metadata={"seq_type": "paired"},
)

rowdata_duplicated_row_name = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    index=["HER2", "HER2", "TPFK"],
)
se_duplicated_row_name = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    row_data=rowdata_duplicated_row_name,
    column_data=coldata1,
    row_names=["HER2", "HER2", "TPFK"],
    column_names=["cell_1", "cell_2", "cell_3"],
    metadata={"seq_type": "paired"},
)

coldata_duplicated_sample_name = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_1", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    index=["cell_1", "cell_1", "cell_2"],
)
se_duplicated_sample_name = SummarizedExperiment(
    assays={"counts": np.random.poisson(lam=5, size=(3, 3))},
    row_data=rowdata1,
    column_data=coldata_duplicated_sample_name,
    row_names=["HER2", "HER2", "TPFK"],
    column_names=["cell_1", "cell_1", "cell_2"],
    metadata={"seq_type": "paired"},
)

rowdata_biocframe_1 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    row_names=["HER2", "BRCA1", "TPFK"],
)
coldata_biocframe_1 = BiocFrame(
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
    row_data=rowdata_biocframe_1,
    column_data=coldata_biocframe_1,
    column_names=["cell_1", "cell_2", "cell_3"],
    row_names=["HER2", "BRCA1", "TPFK"],
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowdata_biocframe_2 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [500, 300, 200],
        "end": [510, 310, 210],
    },
    row_names=["HER2", "BRCA1", "TPFK"],
)
coldata_biocframe_2 = BiocFrame(
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
    row_data=rowdata_biocframe_2,
    column_data=coldata_biocframe_2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)
