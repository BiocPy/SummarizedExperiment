from collections import namedtuple
from typing import Optional, Sequence, Union

import h5py
from scipy import sparse as sp

H5DatasetInfo = namedtuple("H5DatasetInfo", ["shape", "dtype", "format"])


def infer_h5_dataset(dataset: h5py.Group, verbose: bool = False) -> H5DatasetInfo:
    """Infer type of matrix stored in H5 file.

    Args:
        dataset (h5py.Dataset): h5 dataset object to infer matrix.
        verbose (bool, optional): print info as we read the h5 file? Defaults to False.

    Raises:
        Exception: when shape is not present as either
            key or an attr (for sparse representations).

    Returns:
        H5DatasetInfo: info about the matrix.
    """
    if (
        hasattr(dataset, "keys")
        and len(set(dataset.keys()).intersection(["indptr", "data", "indices"])) == 3
    ):
        if verbose:
            print("dataset contains the following keys, ", dataset.keys())

        # data matrix stored as sparse (as in csr/csc format)
        shape = None
        if "shape" in dataset.keys():
            shape = dataset["shape"][:]
        elif "shape" in dataset.attrs.keys():
            shape = dataset.attrs["shape"][:]
        else:
            raise Exception("dataset: shape not found in either attrs")

        if verbose:
            print("shape is ", shape)

        shape = tuple(shape)

        format = "csc_matrix"
        print("length of indptr", dataset["indptr"], len(dataset["indptr"]))
        if shape[0] == len(dataset["indptr"]) - 1:
            format = "csr_matrix"
    else:
        # dense
        shape = dataset.shape
        format = "dense"

    dtype = dataset["data"].dtype.type
    return H5DatasetInfo(shape, dtype, format)


def check_indices(indices: Union[slice, Sequence[int]]):
    if isinstance(indices, slice):
        if indices.step is not None:
            raise NotImplementedError("step is not supported.")

        return indices
    elif isinstance(indices, Sequence):
        all_ints = all([isinstance(k, int) for k in indices])

        if all_ints:
            return indices

    raise Exception("indices is neither a slice nor a list of integers.")


def translate_slice(idx: slice) -> slice:
    start = idx.start
    stop = idx.stop

    if stop is not None and stop > 0:
        stop += 1
    if start is not None and start < 0:
        start -= 1

    return slice(start, stop)


def _extract_along_idx(
    h5: h5py.Group, h5info: H5DatasetInfo, start: int, stop: Optional[int] = None
) -> sp.spmatrix:
    idx = start
    if stop is not None:
        idx = slice(start, stop)

    indptr_slice = slice(idx.start, idx.stop)
    indptr = h5["indptr"][indptr_slice]
    data = h5["data"][indptr[0] : indptr[-1]]
    indices = h5["indices"][indptr[0] : indptr[-1]]
    indptr -= indptr[0]

    if h5info.format == "csr_matrix":
        shape = (indptr.size - 1, h5info.shape[1])
    elif h5info.format == "csc_matrix":
        shape = (h5info.shape[0], indptr.size - 1)

    mformat = _get_mat_class(h5info.format)
    return mformat((data, indices, indptr), shape=shape)


def _get_mat_class(sparse_format):
    if sparse_format == "csr_matrix":
        return sp.csc_matrix
    elif sparse_format == "csc_matrix":
        return sp.csc_matrix
    else:
        raise ValueError(f"sparse format {sparse_format} not supported")


def slice_h5_sparse(
    h5: h5py.Group, h5info: H5DatasetInfo, idx: Union[slice, Sequence[int]]
) -> sp.spmatrix:
    idx = check_indices(idx)

    if isinstance(idx, slice):
        t_rowIndices = translate_slice(idx)

        return _extract_along_idx(h5, h5info, t_rowIndices.start, t_rowIndices.stop)
    elif isinstance(idx, Sequence):
        all_mat_slices = []
        for i in idx:
            r_mat = _extract_along_idx(h5, h5info, t_rowIndices.start)
            all_mat_slices.append(r_mat)

        if h5info.format == "csr_matrix":
            return sp.vstack(all_mat_slices)
        elif h5info.format == "csc_matrix":
            return sp.hstack(all_mat_slices)
    else:
        raise Exception("provided slice is incorrect.")
