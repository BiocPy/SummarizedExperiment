from typing import Optional, Sequence, Tuple, Union

import h5py

from .utils import check_indices, infer_h5_dataset, slice_h5_sparse


class H5BackedAssay:
    def __init__(self, path: str, group: str) -> None:
        h5file = h5py.File(path, mode="r")
        self._dataset = h5file[group]

        self._dataset_info = infer_h5_dataset(self._dataset)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get shape of the dataset.

        Returns:
            Tuple[int, int]: number of rows by columns.
        """
        return self._dataset_info.shape

    @property
    def dtype(self) -> str:
        """Get type of values stored in the dataset.

        Returns:
            str: type of dataset, e.g. int8, float etc.
        """
        return self._dataset_info.dtype

    @property
    def mat_format(self) -> str:
        """Get matrix format of the dataset.

        either `csr_matrix`, `csc_matrix` or `dense`.

         Returns:
             str: matrix format.
        """
        return self._dataset_info.format

    def __getitem__(
        self,
        args: Tuple[Union[slice, Sequence[int]], Optional[Union[slice, Sequence[int]]]],
    ):
        if len(args) == 0:
            raise ValueError("Arguments must contain one slice")

        rowIndices = check_indices(args[0])
        colIndices = None

        if len(args) > 1:
            if args[1] is not None:
                colIndices = check_indices(args[1])
        elif len(args) > 2:
            raise ValueError("contains too many slices")

        if self.mat_format == "csr_matrix":
            mat = slice_h5_sparse(self._dataset, self._dataset_info, rowIndices)
            # now slice columns
            if colIndices is not None:
                mat = mat[:, colIndices]
            return mat
        elif self.mat_format == "csc_matrix":
            if colIndices is None:
                colIndices = slice(0)
            mat = slice_h5_sparse(self._dataset, self._dataset_info, colIndices)
            # now slice columns
            mat = mat[rowIndices, :]
            return mat
        elif self.mat_format == "dense":
            if colIndices is None:
                colIndices = slice(0)
            return self._dataset[rowIndices, colIndices]
        else:
            raise Exception("unknown matrix type in H5.")
