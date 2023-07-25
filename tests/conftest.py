"""
    Dummy conftest.py for summarizedexperiment.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import data.summarized_experiments as ses


@pytest.fixture
def summarized_experiments():
    return ses

