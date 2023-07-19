# Tutorial

This package provides classes to represent genomic experiments. Currently supports both `SummarizedExperiment` & `RangeSummarizedExperiment` representations.

# Create a `SummarizedExperiment`

To create a `SummarizedExperiment`, we need

- `Assays`: a dictionary of matrices with keys specifying the assay name. 
- `rows`: feature information about the rows of the matrices.
- `cols`: sample information about the columns of the matrices.

Lets create these three objects. we first create a mock dataset of 200 rows and 6 columns, also adding a few sample data.

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

## File backed mode for large datasets

In addition to fully realized matrices in memory, SE/RSE also supports file backed arrays and matrices. [FileBackedArray](https://github.com/BiocPy/FileBackedArray) package provides lazy representation for matrices stored in hdf5 files.

```python
from filebackedarray import H5BackedSparseData

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
        * 100,
        "starts": range(0, 1000),
        "ends": range(0, 1000),
        "strand": ["-", "+", "+", "*", "*", "+", "+", "+", "-", "-"] * 100,
        "score": range(0, 1000),
        "GC": [random() for _ in range(10)] * 100,
    }
)

colData = pd.DataFrame({"treatment": ["ChIP"] * 3005,})

assay = H5BackedSparseData("tests/data/tenx.sub.h5", "matrix")

tse = SummarizedExperiment(
    assays={"counts_backed": assay},
    rowData=df_gr,
    colData=colData,
)
```


## Accessors

Many properties can be accessed directly from the class instance. Checkout the API for all available methods.

```python
tse.assays
tse.rowData or # tse.rowRanges
tse.colData

# Access the counts assay
tse.assay("counts")
```

# Subset an experiment

## `SummarizedExperiment`

Use `[ ]` notation to subset a `SummarizedExperiment` object.

```python
# subset the first 10 rows and the first 3 samples
subset_tse = tse[0:10, 0:3]
```

Alternatively, we can use a sequence of names or a boolean array. To show this, we create a `SummarizedExperiment` object with index names.

```python
rowData_with_index_names = pd.DataFrame(
    {
        "seqnames": ["chr_5", "chr_3", "chr_2"],
        "start": [100, 200, 300],
        "end": [110, 210, 310]
    },
    index=["HER2", "BRCA1", "TPFK"],
)
colData_with_index_names = pd.DataFrame(
    {
        "sample": ["SAM_1", "SAM_3", "SAM_3"],
        "disease": ["True", "True", "True"],
    },
    index=["cell_1", "cell_2", "cell_3"],
)
se_with_index_names = SummarizedExperiment(
    assays={
        "counts": np.random.poisson(lam=5, size=(3, 3)),
        "lognorm": np.random.lognormal(size=(3, 3))
    },
    rowData=rowData_with_index_names,
    colData=colData_with_index_names
)

# subset by name
subset_se_with_index_names = se_with_index_names[
    ["HER2", "BRCA1"], ["cell_1", "cell_3"]
]

# subset with boolean array
subset_se_with_bools = se_with_index_names[
    [True, True, False], [True, False, True]
]
```

In the case a name does not exist, an error will be thrown:

```python
# throws error because "RAND" does not exist
subset_se_with_index_names = se_with_index_names[
    ["HER2", "BRCA1", "RAND"], ["cell_1", "cell_3"]
]
```

## `RangedSummarizedExperiment`

`RangeSummarizedExperiment` objects on the other hand support interval based operations.


```python
query = {"seqnames": ["chr2",], "starts": [4], "ends": [6], "strand": ["+"]}

query = GenomicRanges(query)

tse.subsetOverlaps(query)
```

Checkout the API docs or GenomicRanges for list of interval based operations.
