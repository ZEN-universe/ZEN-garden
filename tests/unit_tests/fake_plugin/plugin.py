from tests.unit_tests.test_events import TestEvent
from zen_garden.plugin_system.events import EventPublisher

config = {}


@EventPublisher.register(TestEvent.test_event1)
def first_method(**kwargs):
    """
    Testing method.

    Is registered to test_event1 for testing.
    """
    pass


@EventPublisher.register(TestEvent.test_event2)
def second_method(**kwargs):
    """
    Testing method.

    Is registered to test_event2 for testing.
    """
    pass
