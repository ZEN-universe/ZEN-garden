"""Plugin loader.

Import plugin modules from a package and update their config dictionaries
according to the user's configuration (as defined in config.json).
"""

import importlib
from importlib.metadata import EntryPoint, entry_points
from types import ModuleType

from zen_garden.plugin_system.events import EventPublisher
from zen_garden.workflow_step import workflow_step

ENTRY_POINT_GROUP = "zen_garden.plugins"


def _get_installed_plugins() -> dict[str, EntryPoint]:
    """Return installed ZEN-garden plugin entry points keyed by plugin name."""
    installed_plugins: dict[str, EntryPoint] = {}

    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name in installed_plugins:
            existing = installed_plugins[entry_point.name]
            raise RuntimeError(
                f"Multiple plugins are registered with the name "
                f"{entry_point.name!r}: {existing.value!r} and "
                f"{entry_point.value!r}."
            )

        installed_plugins[entry_point.name] = entry_point

    return installed_plugins


def _load_installed_plugin(
    plugin_name: str,
    installed_plugins: dict[str, EntryPoint],
) -> ModuleType:
    """Load one plugin from its installed package entry point."""
    try:
        entry_point = installed_plugins[plugin_name]
    except KeyError:
        available_plugins = ", ".join(sorted(installed_plugins)) or "none"

        raise ModuleNotFoundError(
            f"ZEN-garden plugin {plugin_name!r} is not installed. "
            f"Available plugins: {available_plugins}. Install the package "
            f"that provides the plugin in the same Python environment as "
            f"ZEN-garden."
        ) from None

    plugin_module = entry_point.load()

    if not isinstance(plugin_module, ModuleType):
        raise TypeError(
            f"Entry point {plugin_name!r} must resolve to a module, but "
            f"{entry_point.value!r} resolved to "
            f"{type(plugin_module).__name__}."
        )

    return plugin_module


@workflow_step(
    order=3,
    phase="Setup",
    label="Register plugins; notify via after_model_schema_creation event",
)
def register_plugins(
    plugins_config: dict[str, dict],
    source_package: str | None = None,
) -> dict[str, ModuleType]:
    """Import configured plugins and apply their configuration.

    Installed plugins are normally discovered through entry points in the
    ``zen_garden.plugins`` group.

    ``source_package`` provides backward compatibility and supports internal
    tests or bundled plugins that should be imported from a specific package
    instead of through installed entry points.

    Args:
        plugins_config: Mapping of plugin names to configuration dictionaries.
        source_package: Optional package containing plugin subpackages.

    Returns:
        Mapping of configured plugin names to their imported modules.

    Raises:
        ModuleNotFoundError: If a configured plugin is not installed.
        AttributeError: If a plugin module does not expose ``config``.
        TypeError: If an entry point does not resolve to a module.
    """
    loaded_plugins: dict[str, ModuleType] = {}

    installed_plugins = _get_installed_plugins() if source_package is None else {}

    for plugin_name, plugin_config in plugins_config.items():
        if source_package is None:
            plugin_module = _load_installed_plugin(
                plugin_name,
                installed_plugins,
            )
        else:
            plugin_module = importlib.import_module(
                f"{source_package}.{plugin_name}.plugin"
            )

        if not hasattr(plugin_module, "config"):
            raise AttributeError(
                f"Plugin {plugin_name!r} does not expose the required "
                f"module-level 'config' dictionary."
            )

        if not isinstance(plugin_module.config, dict):
            raise TypeError(
                f"Plugin {plugin_name!r} exposes 'config', but it is not "
                f"a dictionary."
            )

        plugin_module.config.update(plugin_config)
        loaded_plugins[plugin_name] = plugin_module

    return loaded_plugins


def deregister_plugins() -> None:
    """Deregister all plugin event callbacks."""
    EventPublisher.deregister_all()
