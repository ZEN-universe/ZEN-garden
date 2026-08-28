"""Service container for dependency injection."""

from typing import Any, TypeVar

T = TypeVar("T")


class ServiceContainer:
    """A simple service container for dependency injection."""

    def __init__(self, name: str):
        """Initialize the service container."""
        self._services: dict[str, Any] = {}
        self.register(name, self)

    def register(self, name: str, service: Any):
        """Register a service with a given name.

        :param name: The name of the service.
        :param service: The service instance to register.
        """
        self._services[name] = service

    def get(self, name: str) -> Any | None:
        """Retrieve a service by name.

        :param name: The name of the service.
        :return: The service instance if found, else None.
        """
        return self._services.get(name)

    def build(self, obj: type[T], **kwargs: Any) -> T:
        """Build an object by injecting services into its constructor.

        :param obj: The object to inject services into.
        :param kwargs: Additional keyword arguments to pass to the object's constructor.
        :return: An instance of the object with services injected.
        """
        for param_name in obj.__init__.__annotations__:
            if param_name in kwargs:
                # User-provided argument takes precedence over service injection
                pass
            elif param_name in self._services:
                kwargs[param_name] = self._services[param_name]
            else:
                raise ValueError(
                    f"Service '{param_name}' is required for {obj.__name__} "
                    "but not provided."
                )
        return obj(**kwargs)

    def build_and_register(self, name: str, obj: type[T], **kwargs: Any) -> T:
        """Build an object by injecting services into its constructor and register it.

        :param name: The name to register the service under.
        :param obj: The object to inject services into.
        :param kwargs: Additional keyword arguments to pass to the object's constructor.
        :return: An instance of the object with services injected.
        """
        instance = self.build(obj, **kwargs)
        self.register(name, instance)
        return instance
