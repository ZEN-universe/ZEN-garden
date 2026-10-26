"""Shared alignment utilities for storage energy-capacity masks."""


def canonical_storage_array(array):
    """Rename storage technology and node dimensions to canonical mask names."""
    aliases = {
        "set_storage_technologies": "set_technologies",
        "set_nodes": "set_location",
    }
    return array.rename(
        {source: target for source, target in aliases.items() if source in array.dims}
    )
