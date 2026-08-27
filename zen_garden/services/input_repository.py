"""File IO helpers for input data."""

import json
import warnings
from pathlib import Path

import pandas as pd
import yaml


class InputRepository:
    """Handles reading raw input files and attribute definitions."""

    def __init__(self, folder_path: Path | str):
        self.folder_path = Path(folder_path)

    def read_csv(self, input_file_name: str) -> pd.DataFrame | None:
        """Reads a CSV file and returns a DataFrame or None if the file does not exist.

        Args:
            input_file_name (str): The name of the input file (without extension).

        Returns:
            pd.DataFrame | None: The DataFrame containing the CSV data,
                or None if the file does not exist.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file contains duplicate column names.
        """
        file_path = self.folder_path / f"{input_file_name}.csv"
        if not file_path.exists():
            return None

        df_input = pd.read_csv(
            file_path,
            header=0,
            index_col=None,
        )
        if any("." in col for col in df_input.columns):
            raise ValueError(
                f"The input data file {file_path.resolve()} "
                f"contains duplicate column names."
            )
        return df_input

    def read_csv_safe(self, input_file_name: str) -> pd.DataFrame:
        """Reads a CSV file and returns a DataFrame.
        If the file does not exist, raises a FileNotFoundError.

        Args:
            input_file_name (str): The name of the input file (without extension).

        Returns:
            pd.DataFrame: The DataFrame containing the CSV data.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            ValueError: If the CSV file contains duplicate column names.
        """
        df_input = self.read_csv(input_file_name)
        if df_input is None:
            raise FileNotFoundError(
                f"The input data file {input_file_name}.csv does not exist in "
                f"{self.folder_path.resolve()}."
            )
        return df_input

    def read_json(self, input_file_name: str) -> dict | None:
        """Reads a JSON file and returns a dictionary with its content.

        Args:
            input_file_name (str): The name of the input file (without extension).

        Returns:
            dict | None: The dictionary containing the JSON data,
                or None if the file does not exist.
        """
        file_path = self.folder_path / f"{input_file_name}.json"
        if not file_path.exists():
            return None
        with open(file_path, "r") as file:
            return json.load(file)

    def read_yaml(self, input_file_name: str) -> dict | None:
        """Reads a YAML file and returns a dictionary with its content.

        Accepts either a ``.yaml`` or ``.yml`` extension; ``.yaml`` is
        preferred if both are present.

        Args:
            input_file_name (str): The name of the input file (without extension).

        Returns:
            dict | None: The dictionary containing the YAML data,
                or None if the file does not exist.

        Raises:
            ValueError: If the YAML file cannot be parsed.
        """
        file_path = None
        for extension in (".yaml", ".yml"):
            candidate = self.folder_path / f"{input_file_name}{extension}"
            if candidate.exists():
                file_path = candidate
                break
        if file_path is None:
            return None
        with open(file_path, "r") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Failed to parse YAML file {file_path.resolve()}: {exc}"
                ) from exc

    def load_attribute_file(self, filename="attributes"):
        """Loads the attribute file, preferring JSON format over CSV.

        Args:
            filename (str): The name of the attribute file (without extension).

        Returns:
            dict: The dictionary containing the attribute data.

        Raises:
            NotImplementedError: If a CSV format is found, indicating deprecation.
            FileNotFoundError: If neither CSV nor JSON attribute files are found.
        """
        if (self.folder_path / f"{filename}.csv").exists():
            raise NotImplementedError(
                f"The .csv format for attributes is deprecated "
                f"({filename} of {Path(self.folder_path).name}). Use .json instead."
            )
        if not (self.folder_path / f"{filename}.json").exists():
            raise FileNotFoundError(
                f"Attributes file does not exist for {Path(self.folder_path).name}"
            )

        return self._load_attribute_file_json(filename=filename)

    def _load_attribute_file_json(self, filename: str):
        """Loads the attribute file in JSON format.

        Args:
            filename (str): The name of the attribute file (without extension).

        Returns:
            dict: The dictionary containing the attribute data.
        """
        file_path = self.folder_path / f"{filename}.json"
        with open(file_path, "r") as file:
            data = json.load(file)
        attribute_dict = {}
        if isinstance(data, list):
            warnings.warn(
                "The list format in attributes.json [{...}] is deprecated. "
                "Use a dict format instead {...}.",
                DeprecationWarning,
                stacklevel=2,
            )
            for item in data:
                for k, v in item.items():
                    if isinstance(v, list):
                        attribute_dict[k] = {sk: sv for d in v for sk, sv in d.items()}
                    else:
                        attribute_dict[k] = v
        else:
            for k, v in data.items():
                if isinstance(v, list):
                    attribute_dict[k] = {sk: sv for d in v for sk, sv in d.items()}
                else:
                    attribute_dict[k] = v
        return attribute_dict
