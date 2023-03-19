import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import (  # type: ignore
        PackageNotFoundError,  # type: ignore
        version,  # type: ignore
    )
else:
    from importlib_metadata import (
        PackageNotFoundError,  # type: ignore
        version,  # type: ignore
    )

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "SummarizedExperiment"
    __version__: str = version(dist_name)  # type: ignore
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"  # type: ignore
finally:
    del version, PackageNotFoundError

from .RangeSummarizedExperiment import (
    RangeSummarizedExperiment as RangeSummarizedExperiment,
)
from .SummarizedExperiment import SummarizedExperiment as SummarizedExperiment
