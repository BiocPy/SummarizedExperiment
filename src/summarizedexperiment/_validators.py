from typing import Sequence
import pandas as pd


def validate_row_names(rowDatas: Sequence[pd.DataFrame]):
    """Validate names across rowDatas.

    Args:
        rowDatas (pd.DataFrame): rowDatas to validate.

    Raises:
        ValueError: if there are null or duplicated row names.
    """

    def _validate_single_rowData(rowData: pd.DataFrame) -> bool:
        """Validate there are no null or duplicated row names.

        Args:
            rowData (pd.DataFrame): rowData to validate.

        Returns:
            bool: `True` if rowData does not have any null or duplicated row names.
                `False` otherwise.
        """
        any_null = rowData.index.isnull().any()
        any_duplicated = rowData.index.duplicated().any()
        return (not any_null) and (not any_duplicated)

    is_valid_row_names = all(
        [_validate_single_rowData(rowData) for rowData in rowDatas]
    )
    if not is_valid_row_names:
        raise ValueError(
            "at least one input `SummarizedExperiment` has null or duplicated row names"
        )


def validate_objects(objs, target_type):
    """Validate all provided objects are `target_type`.

    Args:
        objs: objects to validate.
        target_type (type): type to check objects against.

    Raises:
        TypeError: if any of the provided objects are not `target_type`. 
    """
    all_types = [isinstance(obj, target_type) for obj in objs]
    if not all(all_types):
        raise TypeError(f"not all provided objects are {str(target_type)} objects")


def validate_shapes(ses: Sequence["BaseSE"]):
    """Validate experiment shapes.

    Args:
        ses (BaseSE): experiments to validate.

    Raises:
        ValueError: if assays do not all have the same dimensions.
    """
    all_shapes = [se.shape for se in ses]
    is_all_shapes_same = all_shapes.count(all_shapes[0]) == len(all_shapes)
    if not is_all_shapes_same:
        raise ValueError("not all assays have the same dimensions")
