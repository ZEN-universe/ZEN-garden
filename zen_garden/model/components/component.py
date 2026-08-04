"""Abstract base class for ZEN-model's components."""

import copy
import itertools
import logging
from typing import Sequence

import pandas as pd
import xarray as xr

from zen_garden.model.components.zen_set import ZenSet

logger = logging.getLogger(__name__)


class Component:
    """Class to prepare parameter, variable and constraint data to suit linopy."""

    def __init__(self):
        """Instantiate object of Component class."""
        self.docs: dict[str, str] = {}

    @staticmethod
    def compile_doc_string(
        doc: str, index_list: Sequence[str], name: str, domain: str | None = None
    ) -> str:
        """Compile docstring from doc and index_list.

        :param doc: docstring to be compiled
        :param index_list: list of indices
        :param name: name of parameter/variable/constraint
        :param domain: domain of parameter/variable/constraint
            (e.g., reals, non negative reals, ...)
        :return complete_doc: complete docstring composed of name, doc and dims
        """
        assert isinstance(doc, str), (
            f"Docstring {doc} has wrong format. Must be 'str' "
            f"but is '{type(doc).__name__}'"
        )
        # check for prohibited strings
        prohibited_strings = [",", ";", ":", "/", "name", "doc", "dims", "domain"]
        original_doc = copy.copy(doc)
        for string in prohibited_strings:
            if string in doc:
                logger.warning(
                    f"Docstring '{original_doc}' contains prohibited "
                    f"string '{string}'. Occurrences are dropped."
                )
                doc = doc.replace(string, "")
        # joined index names
        joined_index = ",".join(index_list)
        # complete doc string
        complete_doc = f"name:{name};doc:{doc};dims:{joined_index}"
        if domain:
            complete_doc += f";domain:{domain}"
        return complete_doc

    @staticmethod
    def get_index_names_data(
        index_list: (
            ZenSet | tuple[list | pd.Series, list[str]] | list[list] | xr.DataArray
        ),
    ) -> tuple[list | pd.Series, list[str]]:
        """Splits index_list in data and index names.

        :param index_list: list of indices (names and values)
        :return index_values: names of indices
        :return index_names:  values of indices
        """
        if isinstance(index_list, ZenSet):
            index_values = list(index_list)
            index_names = [index_list.name]
        elif isinstance(index_list, tuple):
            index_values, index_names = index_list
        elif isinstance(index_list, list):
            index_values = list(itertools.product(*index_list[0]))
            index_names = index_list[1]
        elif isinstance(index_list, xr.DataArray):
            index_values = index_list.to_series().dropna()
            index_names = [str(dim) for dim in index_list.coords.dims]
        else:
            raise TypeError(f"Type {type(index_list)} unknown to extract index names.")
        return index_values, index_names

    def __getattr__(self, name: str):
        """Fallback for dynamically-added parameters (see add_parameter).

        Only called when normal attribute lookup fails, so this covers
        parameters set via setattr(self, name, xr_data).
        """
        raise AttributeError(
            f"Parameter '{name}' does not exist. Did you forget to define it?"
        )
