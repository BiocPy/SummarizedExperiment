from typing import Sequence, Literal
import pandas as pd

__author__ = "keviny2"
__copyright__ = "keviny2"
__license__ = "MIT"


def validate_names(
    ses: Sequence["BaseSE"], experiment_attribute: Literal["rowData", "colData"]
):
    """Validate names across experiments.

    Args:
        ses (Sequence[BaseSE]): SummarizedExperiment objects to validate.
        experiment_attribute (Literal["rowData", "colData"]): the experiment_attribute to validate.

    Raises:
        ValueError: if there are null or duplicated names.
    """

    def _validate_single_df(df: pd.DataFrame) -> bool:
        """Validate there are no null or duplicated names.

        Args:
            df (pd.DataFrame): DataFrame to validate.

        Returns:
            bool: `True` if df does not have any null or duplicated names.
                `False` otherwise.
        """
        any_null = any(name is None for name in df.index)
        any_duplicates = len(df.index) != len(set(df.index))
        return (not any_null) and (not any_duplicates)

    is_valid_names = all(
        [_validate_single_df(getattr(se, experiment_attribute)) for se in ses]
    )
    if not is_valid_names:
        raise ValueError(
            "at least one input `SummarizedExperiment` has null or duplicated row names"
        )


def validate_shapes(
    ses: Sequence["BaseSE"], experiment_attribute: Literal["rowData", "colData"]
):
    """Validate shapes across experiments.

    Args:
        ses (Sequence[BaseSE]): SummarizedExperiment objects to validate.
        experiment_attribute (Literal["rowData", "colData"]): the experiment_attribute to validate.

    Raises:
        ValueError: if all objects do not have the same shape of interest:
            - number of rows for cbind() and combineCols()
            - number of columns for rbind() and combineRows()
    """
    all_shapes = [getattr(se, experiment_attribute).shape[0] for se in ses]
    is_all_same_shape = all_shapes.count(all_shapes[0]) == len(all_shapes)
    if not is_all_same_shape:
        raise ValueError("not all objects have the same shape")


def validate_experiment_attribute(experiment_attribute: str):
    """Validate that `experiment_attribute` is either "rowData" or "colData".

    Args:
        experiment_attribute (str): the experiment attribute to validate.

    Raises:
        ValueError: if `experiment_attribute` is not "rowData" or "colData"
    """
    if experiment_attribute not in ["rowData", "colData"]:
        raise ValueError("`experiment_attribute` must be either 'rowData' or 'colData'")
