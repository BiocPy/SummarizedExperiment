# Changelog

## Version 0.4.0 to 0.4.1

This is a complete rewrite of the package, following the functional paradigm from our [developer notes](https://github.com/BiocPy/developer_guide#use-functional-discipline).

- Implements functional paradigm to access and set attributes on SE/RSE
- Can initialize SE/RSE without assays, also initialize an empty SE/RSE
- `row_data` and `column_data` are expected to be `BiocFrame` objects and will be converted if a pandas `DataFrame` is provided. This allows us to reduce complexity and implement consistent downstream operations.  If these are not provided, an empty `BiocFrame` is set as default.
- On RSE, if `row_ranges` is not provided, an empty `GenomicRangesList` is set as default.
- SE/RSE now contain `row_names` and `column_names` that are separate from row_data's and column_data's row names, also helps in simplifying subset operations.
- Printing SE/RSE objects now looks almost similar to R/Bioc's printing of these objects.
- Support combine operations, both strict and a flexible combine option when rows or columns do not exactly match between multiple objects.
- Streamlines subset operation for SE/RSE and probably downstream derivates; they only need to update the `slice` method.

In addition the following rules are set to access or update `row_names` and `column_names` either from the SE or the `row_data` or `column_data` slots.
- On construction, if `row_names` or `column_names` are not provided, these are automatically inferred from `row_data` and `column_data` objects.
- On extraction of these objects, the `row_names` in `row_data` and `column_data` are replaced by the equivalents from the SE level.
- On setting these objects, especially with the functional style (`set_row_data` and `set_column_data` methods), additional options are available to replace the names in the SE object.

Other changes
- Reduce dependency on a number of external packages.
- Update docstrings, tests and docs.

## Version 0.3.0

This release migrates the package to a more palatable Google's Python style guide. A major modification to the package is with casing, all `camelCase` methods, functions and parameters are now `snake_case`.

With respect to the classes, `BaseSE` is an intenal-only class. `RangedSummarizedExperiment` now extends `SummarizedExperiment` and so do all derivates. Typehints have been updated to reflect these changes.

In addition, docstrings and documentation has been updated to use sphinx's features of linking objects to their types. Sphinx now also documents private and special dunder methods (e.g. `__getitem__`, `__copy__` etc). Intersphinx has been updated to link to references from dependent packages.

Configuration for flake8, ruff and black has been added to pyproject.toml and setup.cfg to be less annoying.

Finally, pyscaffold has been updated to use "myst-parser" as the markdown compiler instead of recommonmark. As part of the pyscaffold setup, one may use pre-commits to run some of the routine tasks of linting and formatting before every commit. While this is sometimes annoying and can be ignored with `--no-verify`, it brings some consistency to the code base.

## Version 0.1.4
- rewriting Base classes, SE and RSE
- implement range methods specific to RSE
- tests
- docs

## Version 0.1.3
- more accessors to assays
- fix bug with defining shape of an experiment
- tests
- docs

## Version 0.1.2
- row/col data can be empty/None
- update tests and documentation

## Version 0.1.1
- Add subset, shape and custom print for the class
- Tests for subset
- Documentation update

## Version 0.1

- Initial creation of SE and RSE classes
- Tests
- Documentation
