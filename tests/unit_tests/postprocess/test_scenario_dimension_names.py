import numpy as np
import xarray as xr

from zen_garden.config import HeaderDataInputs
from zen_garden.postprocess.results.scenario import (
    _build_query,
    _open_component_variable,
    _strip_dim_suffix,
)

UUID = "465a6136-4ced-49f1-b261-58be6bf5cfd3"


def _write_results_file(path):
    """Mimic a variables.nc of a dataset without transport technologies:
    the empty transport dims clash with the populated time dimension and are
    therefore stored with a ``_<uuid>`` suffix."""
    ds = xr.Dataset(
        {
            "flow_import": (
                ("set_time_steps_operation",),
                np.arange(3, dtype=float),
            ),
            "flow_transport_loss": (
                (
                    f"set_transport_technologies_{UUID}",
                    f"set_edges_{UUID}",
                    f"set_time_steps_operation_{UUID}",
                ),
                np.empty((0, 0, 0)),
            ),
        },
        coords={"set_time_steps_operation": [0, 1, 2]},
    )
    ds.to_netcdf(path)


def test_strip_dim_suffix():
    assert _strip_dim_suffix(f"set_edges_{UUID}") == "set_edges"
    assert _strip_dim_suffix("set_edges") == "set_edges"
    assert _strip_dim_suffix("set_time_steps_operation") == "set_time_steps_operation"


def test_open_component_variable_restores_set_names(tmp_path):
    path = tmp_path / "variables.nc"
    _write_results_file(path)

    da = _open_component_variable(path, "flow_transport_loss")
    assert da.dims == (
        "set_transport_technologies",
        "set_edges",
        "set_time_steps_operation",
    )
    # populated components are unaffected
    da = _open_component_variable(path, "flow_import")
    assert da.dims == ("set_time_steps_operation",)
    assert da.values.tolist() == [0.0, 1.0, 2.0]


def test_friendly_index_resolves_on_empty_component(tmp_path):
    path = tmp_path / "variables.nc"
    _write_results_file(path)
    da = _open_component_variable(path, "flow_transport_loss")

    query = _build_query(
        tuple(str(d) for d in da.dims),
        HeaderDataInputs().model_dump(),
        {"technology": ["battery"]},
    )
    assert list(query) == ["set_transport_technologies"]
    assert da.query(query).to_series().empty
