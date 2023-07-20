# SummarizedExperiment

Container to represent data from genomic experiments, follows Bioconductor's [SummarizedExperiment](https://bioconductor.org/packages/release/bioc/html/SummarizedExperiment.html). It uses efficient structures already available in the Pandas/numpy eco-system & adds a familiar interface.


## Install

Package is deployed to [PyPI](https://pypi.org/project/summarizedexperiment/)

```shell
pip install summarizedexperiment
```

## Usage

Currently supports both `SummarizedExperiment` & `RangeSummarizedExperiment` objects

First create necessary sample data 

```python
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

gr = GenomicRanges.fromPandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)
```

To create a `SummarizedExperiment`,

```python
tse = SummarizedExperiment(
    assays={"counts": counts}, rowData=df_gr, colData=colData
)
```

To create a `RangeSummarizedExperiment`

```python
trse = SummarizedExperiment(
    assays={"counts": counts}, rowRanges=gr, colData=colData
)
```

For more use cases including subset, checkout the [documentation](https://biocpy.github.io/SummarizedExperiment/)

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
