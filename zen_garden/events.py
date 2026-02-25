from enum import Enum, auto
from typing import Callable


class Event(Enum):
    before_optimization_construction = auto()
    after_optimization_construction = auto()


class Events:
    __observers: dict[Event, list[Callable]] = {}

    @classmethod
    def register(cls, event: Event):
        def decorator(observer, **kwargs):
            if event not in cls.__observers:
                cls.__observers[event] = []
            cls.__observers[event].append(observer)
            return observer
        return decorator

    @classmethod
    def trigger(cls, event: Event, *args, **kwargs):
        # We could add logging here
        for observer in cls.__observers.get(event, []):
            observer(*args, **kwargs)