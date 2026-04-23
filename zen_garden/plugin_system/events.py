"""Event utilities for the plugin system.

Provides a base `Event` enum and an `EventPublisher` to register and trigger
events.
"""

from enum import Enum, auto
from typing import Callable


class Event(Enum):
    """Base enum for plugin events.

    Extend this class to declare specific events used by plugins.

    Example:

        >>> class Event(Enum):
        >>>     after_model_construction = auto()

        The event definition can be used as a decorator as:

        >>> @EventPublisher.register(Event.after_model_construction)
        >>> def any_function(any_arg):
        >>>     pass

        The event can be triggered in the core as:

        >>> EventPublisher.trigger(Event.after_model_construction, any_arg)


    Todo:
        - define event locations in core

    """

    test_event1 = auto()
    test_event2 = auto()


class EventPublisher:
    """Class to register and trigger events.

    Observers are callables that can be registered to a specific event.
    When an event is triggered, all registered observers are
    invoked in registration order with the provided arguments.

    Example:
        An observer is defined in a plugin module as a function decorated
        with `EventPublisher.register`:

        >>> @EventPublisher.register(Event.after_model_construction)
        >>> def any_function(any_arg):
        >>>     pass

    Todo:
        - call deregistration in core
    """

    __observers: dict[Event, list[Callable]] = {}

    @classmethod
    def observers(cls):
        """Returns all observers registered."""
        return {k: list(v) for k, v in cls.__observers.items()}

    @classmethod
    def register(cls, event: Event):
        """Decorator that registers a callable as an observer for `event`.

        Args:
            event (Event): The event value to register the observer for.

        Returns:
            Callable: A decorator which takes a function and registers it.
        """

        def decorator(observer, **kwargs):
            if event not in cls.__observers:
                cls.__observers[event] = []
            cls.__observers[event].append(observer)
            return observer

        return decorator

    @classmethod
    def deregister_all(cls):
        """Deregister all observers for all events."""
        cls.__observers = {}

    @classmethod
    def trigger(cls, event: Event, *args, **kwargs):
        """Trigger `event` and call all registered observers.

        Args:
            event (Event): The event to trigger.
            *args: Positional arguments passed to observer callables.
            **kwargs: Keyword arguments passed to observer callables.

        Todo:
            - Add logging
        """
        for observer in cls.__observers.get(event, []):
            observer(*args, **kwargs)
