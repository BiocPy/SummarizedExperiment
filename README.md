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
import pandas as pd
import numpy as np
from genomicranges import GenomicRanges

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

gr = genomicranges.from_pandas(df_gr)

colData = pd.DataFrame(
    {
        "treatment": ["ChIP", "Input"] * 3,
    }
)
```

To create a `SummarizedExperiment`,

```python
from summarizedexperiment import SummarizedExperiment

tse = SummarizedExperiment(
    assays={"counts": counts}, row_data=df_gr, col_data=colData
)
```

To create a `RangedSummarizedExperiment`

```python
from summarizedexperiment import RangedSummarizedExperiment

trse = RangedSummarizedExperiment(
    assays={"counts": counts}, row_ranges=gr, col_data=colData
)
```

For more examples, checkout the [documentation](https://biocpy.github.io/SummarizedExperiment/).

<!-- pyscaffold-notes -->

## Note

This project has been set up using PyScaffold 4.5. For details and usage
information on PyScaffold see https://pyscaffold.org/.
