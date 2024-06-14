# SummarizedExperiment

This package provides containers to represent genomic experimental data as 2-dimensional matrices, follows Bioconductor's [SummarizedExperiment](https://bioconductor.org/packages/release/bioc/html/SummarizedExperiment.html). In these matrices, the rows typically denote features or genomic regions of interest, while columns represent samples or cells.

The package currently includes representations for both `SummarizedExperiment` and `RangedSummarizedExperiment`. A distinction lies in the fact `RangedSummarizedExperiment` object provides an additional slot to store genomic regions for each feature and is expected to be `GenomicRanges` (more [here](https://github.com/BiocPy/GenomicRanges/)).

## Install

To get started, Install the package from [PyPI](https://pypi.org/project/summarizedexperiment/),

```shell
pip install summarizedexperiment
```

## Contents

```{toctree}
:maxdepth: 2

Overview <tutorial>
Extending Classes <extend_se>
Module Reference <api/modules>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[recommonmark]: https://recommonmark.readthedocs.io/en/latest
[autostructify]: https://recommonmark.readthedocs.io/en/latest/auto_structify.html
