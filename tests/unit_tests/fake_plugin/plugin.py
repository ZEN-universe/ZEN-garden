from zen_garden.plugin_system.events import EventPublisher
from tests.unit_tests.test_events import TestEvent

config = {}


@EventPublisher.register(TestEvent.test_event1)
def first_method(**kwargs):
    pass
