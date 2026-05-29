"""Plugin loader.

Import plugin modules from a package and update their config dictionaries
according to the user's configuration (as defined in config.json).

Plugins are discovered via the ``zen_garden.plugins`` entry-point group.
Any package that declares an entry point in that group is automatically
available::

    # in the plugin's pyproject.toml:
    [project.entry-points."zen_garden.plugins"]
    my_plugin_name = "my_package.plugin"
"""

import importlib
import logging
from importlib.metadata import entry_points
from types import ModuleType

from zen_garden.plugin_system.events import EventPublisher

logger = logging.getLogger(__name__)

_ENTRYPOINT_GROUP = "zen_garden.plugins"


def _discover_entrypoints() -> dict[str, str]:
    """Return a mapping of plugin name → dotted module path from installed entry points.

    Returns:
        dict[str, str]: Plugin name to module path mapping.
    """
    eps = entry_points(group=_ENTRYPOINT_GROUP)
    return {ep.name: ep.value for ep in eps}


def register_plugins(plugins_config: dict[str, dict]) -> dict[str, ModuleType]:
    """Import plugin modules and apply provided configurations.

    Plugins are discovered exclusively via installed entry points in the
    ``zen_garden.plugins`` group. Install a plugin package (``pip install``)
    before referencing it in ``config.json``.

    Upon import all callables are registered to their respective events. The
    plugin's ``config`` dictionary is updated with the provided configuration.

    Args:
        plugins_config (dict[str, dict]): Mapping of plugin name to config dict.

    Returns:
        dict[str, ModuleType]: Mapping of plugin name to the imported module.

    Raises:
        ModuleNotFoundError: If a requested plugin is not found among installed
            entry points.

    Example:

        >>> plugins_config = {"my_plugin": {"param1": "value1"}}
        >>> register_plugins(plugins_config)
    """
    discovered = _discover_entrypoints()
    output = {}

    for plugin, config in plugins_config.items():
        if plugin not in discovered:
            available = list(discovered.keys())
            raise ModuleNotFoundError(
                f"Plugin '{plugin}' not found among installed entry points. "
                f"Available plugins: {available}. "
                f"Install the plugin package and ensure it declares an entry point "
                f"in the '{_ENTRYPOINT_GROUP}' group."
            )
        module_path = discovered[plugin]
        logger.debug("Loading plugin '%s' from entry point '%s'.", plugin, module_path)
        module = importlib.import_module(module_path)
        module.config.update(config)
        output[plugin] = module

    return output


def deregister_plugins():
    EventPublisher.deregister_all()
