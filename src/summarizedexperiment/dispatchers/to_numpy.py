from functools import singledispatch
from numpy import ndarray
from scipy import sparse as sp
from filebackedarray import H5BackedDenseData, H5BackedSparseData


@singledispatch
def to_numpy(x) -> ndarray:
    """Convert various objects to a numpy ndarray.
    
    Args:
        x (any): supported object.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        ndarray: object as a numpy ndarray
    """
    raise NotImplementedError(f"cannot convert class: {type(x)} to a numpy ndarray")

@to_numpy.register
def _(x: ndarray) -> ndarray:
    return x

@to_numpy.register
def _(x: sp.spmatrix) -> ndarray:
    return x.toarray()

@to_numpy.register
def _(x: H5BackedDenseData) -> ndarray:
    raise NotImplementedError(f"{type(x)} not supported yet")

@to_numpy.register
def _(x: H5BackedSparseData) -> ndarray:
    raise NotImplementedError(f"{type(x)} not supported yet")