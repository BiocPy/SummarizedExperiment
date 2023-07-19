import pandas as pd
import numpy as np
import scipy.sparse as sp
from biocframe import BiocFrame
from summarizedexperiment.SummarizedExperiment import SummarizedExperiment

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
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
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
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
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
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
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
        "sample": ["SAM_10", "SAM_11", "SAM_12"],
        "disease": ["True", "False", "False"],
        "doublet_score": [.15, .62, .18]
    },
    index=["cell_10", "cell_11", "cell_12"],
)
se4 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(5, 3)),
        "lognorm": np.random.lognormal(size=(5, 3)),
        "beta": np.random.beta(a=1, b=1, size=(5, 3))
    },
    rowData=rowData4,
    colData=colData4,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowData5 = pd.DataFrame(
    {
        "seqnames": ["chr_7", "chr_5", "chr_4", "chr_Y", "chr_8"],
        "start": [1084390, 1273987, 18279843, 243879798, 127987239],
        "end": [243895239, 128973192, 290823094, 390820395, 238798237]
    },
    index=["MYC", "BRCA1", "PIK3CA", "TPFK", "HRAS"],
)
colData5 = pd.DataFrame(
    {
        "sample": ["SAM_13", "SAM_14", "SAM_15"],
        "disease": ["True", "True", "True"],
        "doublet_score": [.32, .51, .09]
    },
    index=["cell_13", "cell_14", "cell_15"],
)
se_sparse = SummarizedExperiment(
    assays={
        "counts": sp.lil_matrix(np.random.poisson(lam=7, size=(5, 3))),
        "lognorm": sp.lil_matrix(np.random.lognormal(size=(5, 3))),
        "beta": sp.lil_matrix(np.random.beta(a=2, b=1, size=(5, 3)))
    },
    rowData=rowData5,
    colData=colData5,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

colData6 = pd.DataFrame(
    {
        "sample": ["SAM_10", "SAM_11", "SAM_12"],
        "disease": ["True", "False", "False"],
        "qual": [.95, .92, .98]
    },
    index=["cell_10", "cell_11", "cell_12"],
)
se6 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(5, 3)),
        "lognorm": np.random.lognormal(size=(5, 3)),
        "beta": np.random.beta(a=1, b=1, size=(5, 3))
    },
    rowData=rowData4,
    colData=colData6,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowData_nonames = pd.DataFrame(
    {},
    index=["HER2", "BRCA1", "TPFK"],
)
colData_nonames = pd.DataFrame(
    {},
    index=["cell_1", "cell_2", "cell_3"],
)
se_nonames = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
    rowData=rowData_nonames,
    colData=colData_nonames,
    metadata={},
)

rowData_biocframe_1 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [10293804, 12098948, 20984392],
        "end": [28937947, 3872839, 329837492]
    },
    rowNames=["HER2", "BRCA1", "TPFK"],
)
colData_biocframe_1 = BiocFrame(
    {
        "sample": ["SAM_1", "SAM_3", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    rowNames=["cell_1", "cell_2", "cell_3"],
)
se_biocframe_1 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
    rowData=rowData_biocframe_1,
    colData=colData_biocframe_1,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)

rowData_biocframe_2 = BiocFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [10293804, 12098948, 20984392],
        "end": [28937947, 3872839, 329837492]
    },
    rowNames=["HER2", "BRCA1", "TPFK"],
)
colData_biocframe_2 = BiocFrame(
    {
        "sample": ["SAM_4", "SAM_5", "SAM_6"],
        "disease": ["True", "False", "True"],
        "doublet_score": [.05, .23, .54]
    },
    rowNames=["cell_4", "cell_5", "cell_6"],
)
se_biocframe_2 = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=7, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
    rowData=rowData_biocframe_2,
    colData=colData_biocframe_2,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)