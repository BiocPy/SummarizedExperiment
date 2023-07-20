from typing import Sequence
import pandas as pd


def validate_names(dfs: Sequence[pd.DataFrame]):
    """Validate names across dataframes.

    Args:
        dfs (pd.DataFrame): DataFrames to validate.

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
        any_null = df.index.isnull().any()
        any_duplicated = df.index.duplicated().any()
        return (not any_null) and (not any_duplicated)

    is_valid_names = all(
        [_validate_single_df(df) for df in dfs]
    )
    if not is_valid_names:
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


def validate_num_rows(dfs: Sequence[pd.DataFrame]):
    """Validate number of rows across dataframes.

    Args:
        dfs (pd.DataFrame): DataFrames to validate.

    Raises:
        ValueError: if all objects do not have the same number of rows.
    """
    all_num_rows = [df.shape[0] for df in dfs]
    is_all_num_rows_same = all_num_rows.count(all_num_rows[0]) == len(all_num_rows)
    if not is_all_num_rows_same:
        raise ValueError("not all objects have the same number of rows")
