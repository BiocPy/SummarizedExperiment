# Tutorial

Currently supports both `SummarizedExperiment` & `RangeSummarizedExperiment` objects

## Mock sample data 

we first create a mock dataset of 200 rows and 6 columns, also adding a few sample_data.

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

### `SummarizedExperiment`

`SummarizedExperiment` represents features as a Pandas DataFrame

```python
tse = SummarizedExperiment(
    assays={"counts": counts}, rowData=df_gr, colData=colData
)
```

###  `RangeSummarizedExperiment`

`RangeSummarizedExperiment` represents features as [`GenomicRanges`](https://github.com/BiocPy/GenomicRanges)

```python
trse = SummarizedExperiment(
    assays={"counts": counts}, rowRanges=gr, colData=colData
)
```

## Subset an experiment

Currently, the package provides methods to subset by indices

```python
# subset the first 10 rows and the first 3 samples
subset_tse = tse[0:10, 0:3]
```