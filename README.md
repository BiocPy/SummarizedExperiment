[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![PyPI-Server](https://img.shields.io/pypi/v/SummarizedExperiment.svg)](https://pypi.org/project/SummarizedExperiment/)
![Unit tests](https://github.com/BiocPy/SummarizedExperiment/actions/workflows/pypi-test.yml/badge.svg)

# SummarizedExperiment

Container to represent genomic experiments, follows Bioconductor's [SummarizedExperiment](https://bioconductor.org/packages/release/bioc/html/SummarizedExperiment.html).

## Install

Package is published to [PyPI](https://pypi.org/project/summarizedexperiment/),

```shell
pip install summarizedexperiment
```

## Usage

Currently supports `SummarizedExperiment` & `RangedSummarizedExperiment` classes

First create necessary sample data

```python
from random import random
import pandas as pd
import numpy as np
from biocframe import BiocFrame

nrows = 200
ncols = 6
counts = np.random.rand(nrows, ncols)
row_data = BiocFrame(
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

col_data = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)
```

To create a `SummarizedExperiment`,

```python
from summarizedexperiment import SummarizedExperiment

tse = SummarizedExperiment(
    assays={"counts": counts}, row_data=row_data, column_data=col_data,
    metadata={"seq_platform": "Illumina NovaSeq 6000"},
)
```

    ## output
    class: SummarizedExperiment
    dimensions: (200, 6)
    assays(1): ['counts']
    row_data columns(6): ['seqnames', 'starts', 'ends', 'strand', 'score', 'GC']
    row_names(0):
    column_data columns(1): ['treatment']
    column_names(0):
    metadata(1): seq_platform

To create a `RangedSummarizedExperiment`

```python
from summarizedexperiment import RangedSummarizedExperiment
from genomicranges import GenomicRanges

trse = RangedSummarizedExperiment(
    assays={"counts": counts}, row_data=row_data,
    row_ranges=GenomicRanges.from_pandas(row_data.to_pandas()), column_data=col_data
)
```

    ## output
    class: RangedSummarizedExperiment
    dimensions: (200, 6)
    assays(1): ['counts']
    row_data columns(6): ['seqnames', 'starts', 'ends', 'strand', 'score', 'GC']
    row_names(0):
    column_data columns(1): ['treatment']
    column_names(0):
    metadata(0):

For more examples, checkout the [documentation](https://biocpy.github.io/SummarizedExperiment/).

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
