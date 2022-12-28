# Tutorial

This package provides classes to represent genomic experiments. Currently supports both `SummarizedExperiment` & `RangeSummarizedExperiment` representations.

# Create a `SummarizedExperiment`

To create a `SummarizedExperiment`, we need

- `Assays`: a dictionary of matrices with keys specifying the assay name. 
- `rows`: feature information about the rows of the matrices.
- `cols`: sample information about the columns of the matrices.

Lets create these three objects

we first create a mock dataset of 200 rows and 6 columns, also adding a few sample data.

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

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)
```

Finally, create an appropriate summarized experiment class. 

## `SummarizedExperiment`

A `SummarizedExperiment` is a relaxed variant to represent genomic experiments. This class expects features (`rowData`) to be either a pandas `DataFrame` or any variant of `BiocFrame`.

```python
tse = SummarizedExperiment(
    assays={"counts": counts}, rowData=df_gr, colData=colData
)
```

##  `RangeSummarizedExperiment`

`RangeSummarizedExperiment` represents features as [`GenomicRanges`](https://github.com/BiocPy/GenomicRanges).

```python
gr = GenomicRanges.fromPandas(df_gr)

trse = SummarizedExperiment(
    assays={"counts": counts}, rowRanges=gr, colData=colData
)
```

## Accessors

Many properties can be accessed directly from the class instance

```python
tse.assays
tse.rowData or # tse.rowRanges
tse.colData

# Access the counts assay
tse.assay("counts")
```

# Subset an experiment

Use `[ ]` notation to subset a `SummarizedExperiment` object. 

```python
# subset the first 10 rows and the first 3 samples
subset_tse = tse[0:10, 0:3]
```

`RangeSummarizedExperiment` objects on the other hand support interval based operations.


```python
query = {"seqnames": ["chr2",], "starts": [4], "ends": [6], "strand": ["+"]}

query = GenomicRanges(query)

tse.subsetOverlaps(query)
```

Checkout the API docs or GenomicRanges for list of interval based operations.